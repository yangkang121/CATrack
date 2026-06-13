# 双提示特征融合专家模块（Dual-Prompt Feature Fusion Expert Module, DP-FFEM）
# 核心设计：整合频域提示生成、噪声门控混合专家（MoE）与双特征融合，通过"频域提示引导→动态专家分配→特征融合增强"的流程，
# 基于参考特征生成频域提示，自适应分配专家网络处理目标特征，实现精准高效的双特征融合，提升特征表达的针对性与泛化性

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np


class SparseDispatcher(object):
    """
    稀疏调度器：混合专家（MoE）的核心辅助模块，实现输入分配与输出融合

    功能：
        - dispatch：根据门控权重，将批量输入分配给对应专家网络
        - combine：聚合各专家输出，按门控权重加权融合

    核心设计：
        - 稀疏优化：仅将非零门控对应的输入分配给专家，降低计算成本
        - 权重加权：融合时按门控权重缩放专家输出，保证融合针对性

    Args:
        num_experts: 专家网络数量
        gates: 门控权重矩阵，shape=[batch_size, num_experts]
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.

        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.

        Args:
            inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`

        Returns:
            a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.

        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values. If `multiply_by_gates` is set to False, the
        gate values are ignored.

        Args:
            expert_out: a list of `num_experts` `Tensor`s, each with shape
                `[expert_batch_size_i, <extra_output_dims>]`.
            multiply_by_gates: a boolean

        Returns:
            a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)  # .exp()
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(1).unsqueeze(1))

        zeros = torch.zeros(
            (self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), expert_out[-1].size(3)),
            requires_grad=True,
            device=stitched.device,
        )

        combined = zeros.index_add(0, self._batch_index, stitched.float())

        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined  # .log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.

        Returns:
            a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
            and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class DecoderLayer(nn.Module):
    """
    专家网络：MoE中的单个特征处理单元，轻量化卷积解码器

    功能：对分配的输入特征进行卷积增强，输出与输入维度一致的特征
    结构：5层卷积+ReLU激活，逐步降维再升维，强化特征表达

    Args:
        input_size: 输入通道数
        output_size: 输出通道数
        hidden_size: 中间通道数
        kernel_size: 卷积核大小（默认3）
        stride: 步长（默认1）
        padding: 填充量（默认1）
    """

    def __init__(self, input_size, output_size, hidden_size, kernel_size=3, stride=1, padding=1):
        super(DecoderLayer, self).__init__()
        self.decoder_Conv2 = nn.Conv2d(input_size, 64, 3, 1, 1)
        self.decoder_Act2 = nn.ReLU(inplace=True)
        self.decoder_Conv3 = nn.Conv2d(64, 32, 3, 1, 1)
        self.decoder_Act3 = nn.ReLU(inplace=True)
        self.decoder_Conv4 = nn.Conv2d(32, hidden_size, 3, 1, 1)
        self.decoder_Act4 = nn.ReLU(inplace=True)
        self.decoder_Conv5 = nn.Conv2d(hidden_size, output_size, 3, 1, 1)

    def forward(self, x):
        de_x = self.decoder_Conv2(x)
        de_x = self.decoder_Act2(de_x)
        de_x = self.decoder_Conv3(de_x)
        de_x = self.decoder_Act3(de_x)
        de_x = self.decoder_Conv4(de_x)
        de_x = self.decoder_Act4(de_x)
        de_x = self.decoder_Conv5(de_x)
        return de_x


class FrePromptMoE(nn.Module):
    """
    双提示特征融合专家模块（Dual-Prompt Feature Fusion Expert Module, DP-FFEM）

    功能：基于参考特征生成频域提示，通过MoE动态处理目标特征，实现双特征融合增强

    核心设计：
        - 频域提示生成：参考特征经FFT转换至频域，生成调制提示，引导目标特征优化
        - 噪声门控MoE：带噪声的Top-K门控，动态分配专家网络，提升泛化性
        - 双特征融合：频域提示调制目标特征，专家网络增强后聚合输出
        - 负载均衡损失：约束专家负载与重要性分布，避免单专家过载

    Args:
        img_size: 输入图像/特征尺寸
        input_size: 输入通道数
        output_size: 输出通道数
        num_experts: 专家网络数量（默认4）
        hidden_size: 专家网络中间通道数
        noisy_gating: 是否启用噪声门控（默认True）
        k: Top-K专家选择数（默认4）
        trainingmode: 训练/推理模式（默认True）
    """

    def __init__(
        self,
        img_size,
        input_size,
        output_size,
        num_experts,
        hidden_size,
        noisy_gating=True,
        k=4,
        trainingmode=True,
    ):
        super(FrePromptMoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.training = trainingmode
        self.k = k
        self.freq_channels = 2 * input_size  # Real + Imag

        # Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((img_size, img_size // 2 + 1))
        self.avgpool1x1 = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((img_size, img_size // 2 + 1))

        # Frequency Modulator
        self.freq_modulator = nn.Sequential(
            nn.Conv2d(self.freq_channels * 2, self.freq_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.freq_channels, self.freq_channels, kernel_size=1),
        )
        self.freq_enhance = nn.Sequential(
            nn.Conv2d(self.freq_channels, self.freq_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        # instantiate experts
        self.experts = nn.ModuleList(
            [DecoderLayer(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)]
        )
        self.w_gate = nn.Parameter(torch.zeros(img_size**2 * input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(img_size**2 * input_size, num_experts), requires_grad=True)
        self.fre_prompt = nn.Parameter(
            torch.zeros(self.freq_channels, img_size, img_size // 2 + 1), requires_grad=True
        )

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def frequency_prompt(self, x):
        # is_image = x.ndim == 4
        # Prompt Generation
        B, C, H, W = x.shape

        # FFT to frequency domain
        fft = torch.fft.rfft2(x, norm="ortho")  # Shape: (B, C, H, W//2 + 1)
        real = fft.real
        imag = fft.imag
        freq = torch.cat([real, imag], dim=1)  # Shape: (B, 2C, H, W//2+1)

        pooled = torch.cat([self.avgpool(freq), self.maxpool(freq)], dim=1)  # (B, 4C, H, W//2+1)

        # Frequency Modulator
        modulated = self.freq_modulator(pooled)  # (B, 2C, 1, 1)
        prompt = self.fre_prompt  # (2C, 1, 1)
        prompt = prompt.unsqueeze(0)  # (1, 2C, 1, 1)
        enhanced_freq = modulated * prompt

        # Enhance and project back
        enhanced_freq = self.freq_enhance(enhanced_freq)  # Conv + ReLU
        real, imag = torch.chunk(enhanced_freq, 2, dim=1)
        complex_freq = torch.complex(real, imag)  # Reconstruct complex tensor

        # Inverse FFT
        out = torch.fft.irfft2(complex_freq, s=(H, W), norm="ortho")  # Back to spatial domain

        return F.softmax(self.avgpool1x1(out), dim=1)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, ref, loss_coef=1e-2):
        fre_prompt = self.frequency_prompt(ref)  # b c h w
        x_ds = (x * fre_prompt + x).contiguous().view(x.shape[0], -1)
        gates, load = self.noisy_top_k_gating(x_ds, self.training)
        # calculate importance loss
        importance = gates.sum(0)

        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs[i]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        return y, loss


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    f_t = torch.randn(2, 768, 16, 16).to(device)
    f_r = torch.randn(2, 768, 16, 16).to(device)
    model = FrePromptMoE(16, 768, 768, 8, 64, True, 2, True).to(device)

    y, _ = model(f_t, f_r)


    print("输入Fr特征维度：", f_t.shape)
    print("输入Ft特征维度：", f_r.shape)
    print("输出特征维度：", y.shape)