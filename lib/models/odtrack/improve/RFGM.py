import torch
import torch.nn as nn
import torch.nn.functional as F


class MagGuidedFusion(nn.Module):
    """
    幅值引导融合模块 对应结构图中【Amp Residual Guidance】核心分支
    核心功能：通过通道间相似度匹配生成自适应加权权重，对增强幅值做残差引导融合
    流程：通道相似度计算→Top1关键通道提取→权重生成→残差加权融合
    """
    def __init__(self, channels):
        super(MagGuidedFusion, self).__init__()
        self.channels = channels
        # 将Top1单通道幅值扩展到全通道，匹配输入特征维度
        self.expand_conv = nn.Conv2d(1, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, mag0, mag1):
        # mag0: 原始幅值特征A_{i-1} | mag1: 增强后幅值特征A_i，shape均为[B, C, H, W]
        B, C, H, W = mag0.shape
        # 展平空间维度，为通道相似度计算做准备
        mag0_flat = mag0.view(B, C, -1)  # [B, C, H*W]
        mag1_flat = mag1.view(B, C, -1)  # [B, C, H*W]

        # L2归一化，计算通道间余弦相似度
        mag0_norm = F.normalize(mag0_flat, dim=-1)
        mag1_norm = F.normalize(mag1_flat, dim=-1)
        # 批量矩阵乘法，生成通道间相似度矩阵 [B, C, C]
        similarity_matrix = torch.bmm(mag0_norm, mag1_norm.transpose(1, 2))
        # 单通道相似度取均值，得到全局通道相似度得分 [B, C]
        similarity_scores = similarity_matrix.mean(dim=-1)
        # 筛选相似度最高的Top1通道，定位最具代表性的结构通道
        top1_indices = torch.argmax(similarity_scores, dim=-1)  # [B,]

        # 提取每个样本的Top1幅值特征图 [B, 1, H, W]
        mag0_top1 = torch.stack([mag0[b, top1_indices[b]] for b in range(B)], dim=0).unsqueeze(1)
        # 单通道扩展到全通道，生成全局加权权重
        mag0_expanded = self.expand_conv(mag0_top1)
        # Sigmoid归一化到0-1，得到最终幅值加权权重
        mag0_weight = torch.sigmoid(mag0_expanded)

        # 残差加权融合：用引导权重对增强幅值加权，叠加原始特征避免信息丢失
        fused_features = mag1 * mag0_weight + mag1
        return fused_features, mag0_weight


class LightTopKFreBlock(nn.Module):
    """
    RFGM (Residual Fourier-Guided Module) 残差傅里叶引导核心模块
    完整对应结构图RFGM全流程：空域→FFT频域→幅值-相位双分支解耦处理→频域重构→iFFT空域
    双分支设计：上分支Amp Residual Guidance（幅值残差引导）、下分支Pha Residual Compensate（相位残差补偿）
    """
    def __init__(self, nc):
        super(LightTopKFreBlock, self).__init__()
        self.nc = nc
        # 输入特征预卷积，完成通道维度特征变换
        self.conv0 = nn.Conv2d(nc, nc, 1, 1, 0)

        # -------------------------- 幅值分支（Amp Residual Guidance）卷积层 --------------------------
        # 原始幅值特征增强，对应结构图A_{i-1}→Conv→ReLU→Conv→A_i
        self.process1_mag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0)
        )

        # -------------------------- 相位分支（Pha Residual Compensate）卷积层 --------------------------
        # 原始相位特征增强，对应结构图P_{i-1}→Conv→ReLU→Conv→P_i
        self.process1_pha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0)
        )
        # 原始相位+增强相位拼接融合，完成相位残差补偿，对应结构图Concat→Conv→ReLU→Conv
        self.process2_pha = nn.Sequential(
            nn.Conv2d(nc * 2, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0)
        )

        # 幅值引导融合模块，实现幅值残差引导加权
        self.magGuideFusion = MagGuidedFusion(channels=nc)
        # 输出卷积（预留融合接口，代码中未启用）
        self.conv_out = nn.Conv2d(nc * 2, nc, 1, 1, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        # 步骤1：输入特征预变换
        x_conv0 = self.conv0(x)
        # 步骤2：空域→频域RFFT变换，对应结构图FFT模块
        x_freq = torch.fft.rfft2(x_conv0, norm='backward')
        # 解耦频域特征为幅值mag0（A_{i-1}）和相位pha0（P_{i-1}）
        mag0 = torch.abs(x_freq)
        pha0 = torch.angle(x_freq)

        # -------------------------- 幅值残差引导分支 对应结构图上分支 --------------------------
        # 原始幅值特征增强
        mag1 = self.process1_mag(mag0)
        # 幅值引导融合，得到最终增强幅值
        mag_out, mag0_weight = self.magGuideFusion(mag0, mag1)

        # -------------------------- 相位残差补偿分支 对应结构图下分支 --------------------------
        # 原始相位特征增强
        pha1 = self.process1_pha(pha0)
        # 原始相位+增强相位拼接，完成残差补偿融合
        pha_cat = torch.cat((pha0, pha1), dim=1)
        pha_out = self.process2_pha(pha_cat)

        # 步骤3：频域特征重构，用增强幅值和补偿相位还原复数频域特征
        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        x_out_freq = torch.complex(real, imag)

        # 步骤4：频域→空域IRFFT逆变换，对应结构图iFFT模块，输出增强特征
        x_out = torch.fft.irfft2(x_out_freq, s=(H, W), norm='backward')
        return x_out


if __name__ == "__main__":
    device = torch.device('cuda:0'if torch.cuda.is_available() else'cpu')

    x = torch.randn(1, 64, 32, 32).to(device)
    model = LightTopKFreBlock(64).to(device)

    y = model(x)


    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)