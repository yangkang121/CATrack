import torch
import torch.nn as nn
import torch.nn.init as init


def initialize_weights(net_l, scale=1):
    """
    网络权重初始化函数
    针对卷积/线性/BN层做Kaiming初始化，残差块输出卷积做权重缩放，保证训练初始阶段的恒等映射，提升稳定性
    """
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # 残差块输出卷积缩放，避免初始值破坏原有特征
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class SELayer(nn.Module):
    """
    SE通道注意力模块 对应结构图Attention unit中「全局平均池化+全连接层+Sigmoid」分支
    核心功能：自适应学习通道间的重要性权重，强化包含低频结构的关键通道，抑制无效通道
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 通道压缩-激活-还原的全连接分支，生成通道权重
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # 全局平均池化聚合空间信息，压缩为通道级特征
        y = self.avg_pool(x).view(b, c)
        # 生成通道权重并恢复维度
        y = self.fc(y).view(b, c, 1, 1)
        # 通道加权，强化关键通道
        return x * y.expand_as(x)


class AttentionBlock(nn.Module):
    """
    Attention unit 注意力单元 对应结构图上半部分完整流程
    核心功能：从原始输入中提取全局低频显著结构信息，生成低频引导注意力图imp_map
    流程：密集卷积多尺度特征提取→通道拼接→SE通道注意力加权→输出注意力图
    """
    def __init__(self, input=3, output=3, bias=True):
        super(AttentionBlock, self).__init__()
        # 密集连接卷积层，对应结构图中的Dense layers，多尺度提取输入特征
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # SE通道注意力，自适应加权特征通道
        self.senet = SELayer(channel=input + 2 * 32)
        # 输出卷积零初始化，保证初始状态为恒等映射，不破坏原始特征
        initialize_weights([self.conv3], 0.)

    def forward(self, x):
        # 密集连接特征提取，逐步融合多尺度特征
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        # 拼接原始输入+多尺度特征，完整保留输入的低频结构信息
        x = torch.cat((x, x1, x2), 1)
        # SE通道注意力加权，强化低频结构相关通道
        x = self.senet(x)
        # 生成最终低频引导注意力图
        x3 = self.conv3(x)
        return x3


class LPM(nn.Module):
    """
    LPM (Low-frequency Preservation Module) 低频保留模块 完整对应结构图全流程
    双分支架构：Attention unit（低频引导图生成）+ Trunk unit（特征提取+低频全程引导）
    核心功能：全程通过低频注意力图引导特征提取，在增强高频细节的同时，完整保留图像全局低频结构、亮度、颜色信息
    """
    def __init__(self, in_channel=3, att_channel=3, width=16, bias=True):
        super(LPM, self).__init__()
        # Attention unit：生成低频引导注意力图
        self.attn_block = AttentionBlock(input=in_channel, output=att_channel)
        # Trunk unit：特征提取主干，对应结构图下半部分完整流程
        self.conv1 = nn.Conv2d(in_channel, width, 3, 1, 1, bias=bias)  # 输入特征升维
        self.conv2 = nn.Conv2d(width + att_channel, width, 3, 1, 1, bias=bias)  # 融合注意力图的特征变换
        self.prelu1 = nn.PReLU()
        self.conv3 = nn.Conv2d(width + att_channel, width, 3, 1, 1, bias=bias)  # 第二次融合注意力图
        self.prelu2 = nn.PReLU()
        self.conv4 = nn.Conv2d(width, width, 3, 1, 1, bias=bias)  # 最终特征变换
        self.conv5 = nn.Conv2d(width, in_channel, 1, 1, 0, bias=bias)  # 输出维度还原

    def forward(self, x):
        # 步骤1：Attention unit生成低频引导注意力图imp_map
        imp_map = self.attn_block(x)
        # 步骤2：Trunk unit输入特征初始变换
        x1 = self.conv1(x)
        # 步骤3：第一次融合注意力图，残差连接保留低频信息
        x2 = self.prelu1(self.conv2(torch.cat((x1, imp_map), 1)))
        x2 = x2 + x1
        # 步骤4：第二次融合注意力图，再次强化低频引导
        x3 = self.prelu2(self.conv3(torch.cat((x2, imp_map), 1)))
        x3 = x3 + x1
        # 步骤5：最终特征变换，残差连接保证信息不丢失
        x4 = self.conv4(x3)
        x4 = x4 + x1
        # 步骤6：维度还原，输出最终增强特征
        x5 = self.conv5(x4)
        return x5


# 模块测试代码
if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 3, 32, 32).to(device)
    model = LPM(3, 3, 64).to(device)
    y = model(x)

    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)
