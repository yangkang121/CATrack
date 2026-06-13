import torch
import torch.nn as nn


# -------------------------- 基础卷积工具函数 对应结构图黄色Conv模块 --------------------------
def autopad(k, p=None):
    """自动计算padding，保证卷积输出尺寸与输入一致"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """
    标准CBS卷积块：Conv + BatchNorm + SiLU
    对应结构图中所有黄色Conv模块，是CGMM的基础特征变换单元
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -------------------------- CGMM核心模块 对应结构图Cross-modal Global Modeling Module全流程 --------------------------
class NLFEM(nn.Module):
    """
    CGMM: Cross-modal Global Modeling Module 跨模态全局建模模块（代码中命名为NLFEM）
    完整对应结构图全流程，核心功能：实现RGB与红外双模态的双向通道-空间全局交互，同步增强两个模态的特征表达
    输入：[RGB特征, 红外特征]，输出：[增强后的RGB特征, 增强后的红外特征]
    """
    def __init__(self, in_channels):
        super(NLFEM, self).__init__()
        # 输入特征1x1卷积预处理，统一通道维度，对应结构图最左侧Conv模块
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        # 空间Key生成卷积：生成空间维度的注意力权重，对应结构图中生成空间Key的Conv
        self.key = Conv(in_channels, 1, 1, 1)
        # 特征Value生成卷积：生成全局交互的Value特征，对应结构图中生成Value的Conv
        self.value = Conv(in_channels, in_channels, 1, 1)
        # 通道权重变换卷积，对应结构图中通道分支的Conv
        self.convb = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        # 空间特征融合卷积：融合平均+最大池化的空间特征，对应结构图中空间分支的Conv
        self.conv_half = Conv(2, 1, 1, 1)
        # 全局池化模块 对应结构图GAP/GMP
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # GAP 全局平均池化，捕捉通道全局统计特性
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # GMP 全局最大池化，捕捉通道显著特性
        # 层归一化，保证数值稳定性
        self.layernorm = nn.LayerNorm([in_channels, 1, 1])

    def forward(self, x):
        # 输入双模态特征：rgb_fea=可见光特征E^L_RGB，ir_fea=红外特征E^L_IR
        rgb_fea = x[0]
        ir_fea = x[1]
        N, C, H, W = rgb_fea.shape

        # -------------------------- 步骤1：双模态特征预处理 对应结构图最左侧Conv --------------------------
        rgb_fea_conv = self.conv(rgb_fea)
        ir_fea_conv = self.conv(ir_fea)

        # -------------------------- 步骤2：RGB模态全局特征提取 对应结构图上半部分RGB分支 --------------------------
        # 通道维度全局统计：GAP/GMP + Softmax归一化，生成通道权重向量 [N, 1, 1, C]
        rgb_avg = self.avg_pool(rgb_fea_conv).softmax(1).contiguous().view(N, 1, 1, C)
        rgb_max = self.max_pool(rgb_fea_conv).softmax(1).contiguous().view(N, 1, 1, C)
        # 空间维度Key生成：全局空间权重 [N, 1, HW, 1]
        rgb_key = self.key(rgb_fea).contiguous().view(N, 1, -1, 1).softmax(2)
        # 全局Value特征生成：通道-空间全局特征 [N, 1, C, HW]
        rgb_value = self.value(rgb_fea).contiguous().view(N, 1, C, -1)

        # -------------------------- 步骤3：红外模态全局特征提取 对应结构图下半部分IR分支 --------------------------
        # 通道维度全局统计：GAP/GMP + Softmax归一化，生成通道权重向量 [N, 1, 1, C]
        ir_avg = self.avg_pool(ir_fea_conv).softmax(1).contiguous().view(N, 1, 1, C)
        ir_max = self.max_pool(ir_fea_conv).softmax(1).contiguous().view(N, 1, 1, C)
        # 空间维度Key生成：全局空间权重 [N, 1, HW, 1]
        ir_key = self.key(ir_fea).contiguous().view(N, 1, -1, 1).softmax(2)
        # 全局Value特征生成：通道-空间全局特征 [N, 1, C, HW]
        ir_value = self.value(ir_fea).contiguous().view(N, 1, C, -1)

        # -------------------------- 步骤4：红外引导RGB的跨模态全局交互 对应结构图RGB增强分支 --------------------------
        # 红外Value x RGB Key -> 红外引导的RGB通道权重 [N, C, 1, 1]
        rgb_fea_ = torch.matmul(ir_value, rgb_key).contiguous().view(N, C, 1, 1)
        # RGB通道权重 x 红外Value -> RGB引导的红外空间特征 [N, 1, H, W]
        rgb_fea_avg = torch.matmul(rgb_avg, ir_value).contiguous().view(N, 1, H, W)
        rgb_fea_max = torch.matmul(rgb_max, ir_value).contiguous().view(N, 1, H, W)
        # 平均+最大空间特征拼接融合
        rgb_fea_cat = torch.cat([rgb_fea_avg, rgb_fea_max], 1)
        # 通道权重与空间特征哈达玛积融合，生成增强特征
        out_rgb_fea = self.layernorm(self.convb(rgb_fea_)).sigmoid() * self.conv_half(rgb_fea_cat)
        # 残差连接，输出增强后的RGB特征E^G_RGB
        out_rgb = out_rgb_fea + rgb_fea

        # -------------------------- 步骤5：RGB引导红外的跨模态全局交互 对应结构图IR增强分支 --------------------------
        # RGB Value x 红外Key -> RGB引导的红外通道权重 [N, C, 1, 1]
        ir_fea_ = torch.matmul(rgb_value, ir_key).contiguous().view(N, C, 1, 1)
        # 红外通道权重 x RGB Value -> 红外引导的RGB空间特征 [N, 1, H, W]
        ir_fea_avg = torch.matmul(ir_avg, rgb_value).contiguous().view(N, 1, H, W)
        ir_fea_max = torch.matmul(ir_max, rgb_value).contiguous().view(N, 1, H, W)
        # 平均+最大空间特征拼接融合
        ir_fea_cat = torch.cat([ir_fea_avg, ir_fea_max], 1)
        # 通道权重与空间特征哈达玛积融合，生成增强特征
        out_ir_fea = self.layernorm(self.convb(ir_fea_)).sigmoid() * self.conv_half(ir_fea_cat)
        # 残差连接，输出增强后的红外特征E^G_IR
        out_ir = out_ir_fea + ir_fea

        return [out_rgb, out_ir]


# 模块测试代码
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rgb = torch.randn(1, 768, 16, 16).to(device)
    ir = torch.randn(1, 768, 16, 16).to(device)
    x = [rgb, ir]
    model = NLFEM(768).to(device)
    y = model(x)

    print("输入RGB特征维度：", x[0].shape)
    print("输入IR特征维度：", x[1].shape)
    print("输出增强后RGB特征维度：", y[0].shape)
    print("输出增强后IR特征维度：", y[1].shape)
