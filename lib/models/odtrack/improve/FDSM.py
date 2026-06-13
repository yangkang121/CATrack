"""
FDSM - Frequency Dynamic Selection Module
根据论文架构图实现，输入可见光(RGB)特征和红外(NIR)特征，输出增强后的两路特征。

架构说明：
  - FDSM 左侧主流：
      RGB/NIR 各经过 Conv → 拼接(Concat) → Conv+PReLU+Conv+SiLU → 两路分别送入 FDS
  - FDS（Frequency Dynamic Selection）子模块：
      Aggregation 分支：GlobalPooling → MLP → Softmax → Dynamic Weight
      Feature    分支：2D DFT → 与 (DynamicWeight × LearnableFilter) 做频域加权 → 2D IDFT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 工具：基本卷积块
# ─────────────────────────────────────────────────────────────
def conv_bn(in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1,
            padding: int = 1) -> nn.Sequential:
    """Conv2d + BatchNorm2d"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
    )


# ─────────────────────────────────────────────────────────────
# FDS：频率动态选择子模块
# ─────────────────────────────────────────────────────────────
class FDS(nn.Module):
    """
    Frequency Dynamic Selection (FDS) Module

    流程：
        输入 x  (B, C, H, W)
        ├─ Aggregation 分支
        │    GlobalAvgPool → 展平 → MLP(两层 FC+ReLU+FC) → Softmax
        │    输出 dynamic_weight  (B, num_filters, 1, 1)
        └─ Feature 分支
             2D DFT（rfft2）→ 频域特征 X_freq  (B, C, H, W//2+1, 复数)
             LearnableFilter (num_filters 组可学习频域滤波器)  (num_filters, C, H, W//2+1)
             dynamic_weight × LearnableFilter → 加权融合滤波器  (B, C, H, W//2+1)
             X_freq × 融合滤波器 → 滤波后频域特征
             2D IDFT（irfft2）→ 输出  (B, C, H, W)
    """

    def __init__(self, channels: int, height: int, width: int,
                 num_filters: int = 4, mlp_ratio: int = 4):
        """
        Args:
            channels   : 输入/输出特征通道数
            height     : 特征图高度
            width      : 特征图宽度
            num_filters: 可学习滤波器组数量（即 Dynamic Weight 的维度）
            mlp_ratio  : MLP 隐层扩展比例
        """
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.num_filters = num_filters

        freq_w = width // 2 + 1   # rfft2 后频域宽度

        # ── Aggregation 分支 ──────────────────────────────────
        hidden_dim = max(channels * mlp_ratio, num_filters * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_filters),
        )
        # Softmax 归一化生成动态权重
        self.softmax = nn.Softmax(dim=1)

        # ── Feature 分支：可学习频域滤波器组 ─────────────────────
        # shape: (num_filters, channels, height, freq_w)  实部+虚部分开存储
        self.learnable_filter_real = nn.Parameter(
            torch.randn(num_filters, channels, height, freq_w) * 0.02
        )
        self.learnable_filter_imag = nn.Parameter(
            torch.randn(num_filters, channels, height, freq_w) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # ── Aggregation 分支 ──────────────────────────────────
        # GlobalAvgPool → (B, C)
        gap = x.mean(dim=[2, 3])                        # (B, C)
        weight = self.mlp(gap)                          # (B, num_filters)
        weight = self.softmax(weight)                   # (B, num_filters)
        weight = weight.view(B, self.num_filters, 1, 1, 1)  # 广播维度

        # ── Feature 分支 ──────────────────────────────────────
        # 2D DFT
        x_freq = torch.fft.rfft2(x, norm='ortho')      # (B, C, H, W//2+1) 复数张量

        # 组合可学习滤波器为复数形式
        lf_complex = torch.complex(
            self.learnable_filter_real,
            self.learnable_filter_imag,
        )   # (num_filters, C, H, freq_w)

        # dynamic_weight × learnable_filter → 加权融合
        # weight: (B, num_filters, 1, 1, 1)
        # lf_complex: (num_filters, C, H, freq_w)  → 扩展为 (1, num_filters, C, H, freq_w)
        lf = lf_complex.unsqueeze(0)                   # (1, num_filters, C, H, freq_w)
        fused_filter = (weight * lf).sum(dim=1)        # (B, C, H, freq_w)

        # 频域特征 × 融合滤波器（逐元素点乘）
        x_freq_filtered = x_freq * fused_filter        # (B, C, H, freq_w)

        # 2D IDFT
        out = torch.fft.irfft2(x_freq_filtered, s=(H, W), norm='ortho')  # (B, C, H, W)

        return out


# ─────────────────────────────────────────────────────────────
# FDSM：完整模块
# ─────────────────────────────────────────────────────────────
class FDSM(nn.Module):
    """
    FDSM - Frequency Dynamic Selection Module

    输入：
        rgb_feat  (B, C_in, H, W)  —— 可见光特征
        nir_feat  (B, C_in, H, W)  —— 红外特征

    输出：
        F_R  (B, C_out, H, W)  —— 增强后的可见光特征
        F_N  (B, C_out, H, W)  —— 增强后的红外特征

    流程（对应架构图左侧主流）：
        1. rgb_feat → Conv_rgb (→ C_mid)
        2. nir_feat → Conv_nir (→ C_mid)
        3. Concat([rgb_feat_conv, nir_feat_conv])  → C_mid*2
        4. 共享融合分支：Conv + PReLU + Conv + SiLU (→ C_out)
        5. F_R = FDS_R(fused)
        6. F_N = FDS_N(fused)
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        height: int,
        width: int,
        num_filters: int = 4,
    ):
        """
        Args:
            in_channels  : 输入 RGB/NIR 特征通道数
            mid_channels : 各路 Conv 后中间通道数
            out_channels : 融合分支输出通道数（也是 FDS 的通道数）
            height       : 特征图高度
            width        : 特征图宽度
            num_filters  : FDS 内可学习滤波器组数量
        """
        super().__init__()

        # 1. 各路独立卷积
        self.conv_rgb = nn.Sequential(
            conv_bn(in_channels, mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_nir = nn.Sequential(
            conv_bn(in_channels, mid_channels),
            nn.ReLU(inplace=True),
        )

        # 2. 拼接后融合分支：Conv + PReLU + Conv + SiLU
        self.fusion = nn.Sequential(
            conv_bn(mid_channels * 2, out_channels),
            nn.PReLU(num_parameters=out_channels),
            conv_bn(out_channels, out_channels),
            nn.SiLU(inplace=True),
        )

        # 3. 两路 FDS（分别作用于 RGB 路和 NIR 路）
        self.fds_r = FDS(out_channels, height, width, num_filters)
        self.fds_n = FDS(out_channels, height, width, num_filters)

    def forward(
        self,
        rgb_feat: torch.Tensor,
        nir_feat: torch.Tensor,
    ):
        """
        Args:
            rgb_feat: (B, in_channels, H, W)
            nir_feat: (B, in_channels, H, W)
        Returns:
            F_R: (B, out_channels, H, W)
            F_N: (B, out_channels, H, W)
        """
        # 各路卷积
        rgb_conv = self.conv_rgb(rgb_feat)   # (B, mid_channels, H, W)
        nir_conv = self.conv_nir(nir_feat)   # (B, mid_channels, H, W)

        # 拼接并融合
        concat   = torch.cat([rgb_conv, nir_conv], dim=1)  # (B, mid_channels*2, H, W)
        fused    = self.fusion(concat)                      # (B, out_channels, H, W)

        # 两路 FDS
        F_R = self.fds_r(fused)   # 可见光输出
        F_N = self.fds_n(fused)   # 红外输出

        return F_R, F_N


# ─────────────────────────────────────────────────────────────
# 快速验证
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, C_in, H, W = 2, 64, 32, 32

    model = FDSM(
        in_channels=C_in,
        mid_channels=64,
        out_channels=64,
        height=H,
        width=W,
        num_filters=4,
    )

    rgb = torch.randn(B, C_in, H, W)
    nir = torch.randn(B, C_in, H, W)

    F_R, F_N = model(rgb, nir)

    print("=" * 50)
    print("FDSM 前向传播验证")
    print("=" * 50)
    print(f"输入 RGB 特征 shape : {rgb.shape}")
    print(f"输入 NIR 特征 shape : {nir.shape}")
    print(f"输出 F_R shape      : {F_R.shape}")
    print(f"输出 F_N shape      : {F_N.shape}")
    print("=" * 50)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")