# 记忆专家补偿模块（Memory Expert Compensation Module, MECM）
# 核心设计：融合门控混合专家（MoE）与双流记忆网络，通过"门控动态选专家→记忆网络特征补偿→多专家加权融合"的流程，
# 为不同样本分配专属记忆专家，利用记忆库的原型特征实现特征的全局-空间双维度补偿增强，兼顾模型容量与特征表达的针对性

import torch
import torch.nn as nn
import torch.nn.functional as F


class GateNetwork(nn.Module):
    def __init__(self, input_size, num_experts, top_k):
        super(GateNetwork, self).__init__()
        # Global pooling layers to capture global context
        self.gap = nn.AdaptiveMaxPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        self.input_size = input_size
        self.num_experts = num_experts
        self.top_k = top_k
        # Two fully connected layers for scoring
        self.fc0 = nn.Linear(input_size, num_experts)
        self.fc1 = nn.Linear(input_size, num_experts)
        self.relu1 = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)
        # Initialize fc1 weights to zero to stabilize training at start
        torch.nn.init.zeros_(self.fc1.weight)
        # Softplus ensures noise is positive and smooth
        self.sp = nn.Softplus()

    def forward(self, x):
        """
        Args:
            x: Input feature map of shape [B, C, H, W].
        Returns:
            gating_coeffs: Tensor of shape [B, num_experts],
                           softmax-normalized gating weights per expert.
        """
        # Step 1. Global pooling → squeeze spatial dimension
        x = self.gap(x) + self.gap2(x)
        x = x.view(-1, self.input_size)  # (batch_size, C)
        inp = x

        # Step 2. Compute raw expert scores from fc1
        x = self.fc1(x)
        x = self.relu1(x)

        # Step 3. Compute smooth noise term (exploration)
        noise = self.sp(self.fc0(inp))  # (batch_size, num_experts)

        # Normalize noise to zero mean and unit variance (per sample)
        noise_mean = torch.mean(noise, dim=1)
        noise_mean = noise_mean.view(-1, 1)
        std = torch.std(noise, dim=1)
        std = std.view(-1, 1)
        noram_noise = (noise - noise_mean) / std
        # Step 4. Add noise to scores and select Top-K experts
        topk_values, topk_indices = torch.topk(x + noram_noise, k=self.top_k, dim=1)

        # Step 5. Build mask for Top-K positions
        mask = torch.zeros_like(x).scatter_(dim=1, index=topk_indices, value=1.0)

        # Suppress all non-topK scores by setting them to -inf
        x[~mask.bool()] = float('-inf')

        # Step 6. Apply softmax across experts → gating distribution
        gating_coeffs = self.softmax(x)  # (batch_size, num_experts)
        # Each row sums to 1, only Top-K experts get non-zero weights

        return gating_coeffs


class Memory(nn.Module):
    def __init__(self, channel_dim, dilation):
        """
        Memory module for expert networks.
        Each memory stores prototypical features and interacts with input queries
        through both global pattern matching and spatial refinement streams.

        Args:
            channel_dim (int): Dimensionality of input feature channels (D).
            dilation (int): Dilation rate for depth-wise convolution (controls
                            receptive field expansion for spatial refinement).
        """
        super(Memory, self).__init__()
        # Fusion conv: combines outputs of global and spatial streams
        self.fusion = nn.Sequential(nn.Conv2d(2 * channel_dim, channel_dim, kernel_size=1, stride=1, padding=0), nn.LeakyReLU(0.2))

        # Depth-wise conv: enlarges receptive field while keeping channel structure
        self.depth_conv = nn.Sequential(nn.Conv2d(channel_dim, channel_dim, kernel_size=3, stride=1, padding=dilation, groups=channel_dim, dilation=dilation), nn.LeakyReLU(0.2))

        # Number of top memory slots to attend in spatial refinement
        self.topk = 2

    def forward(self, image_feature, memory, train):
        """
        Forward pass of Memory module.

        Args:
            image_feature (Tensor): Input query feature map (B, D, H, W).
            memory (Tensor): Memory bank (M, D), where M = number of slots.
            train (bool): If True, update memory and compute losses.

        Returns:
            updated_image (Tensor): Updated features after memory interaction (B, D, H, W).
            updated_memory (Tensor): Updated memory slots (M, D).
            loss_mem_align (float): Memory alignment loss (0 if train=False).
            loss_mem_triplet (float): Memory triplet loss (0 if train=False).
        """

        global_compensation, updated_memory, loss_mem_align, loss_mem_triplet = self.Global_Pattern_Interaction_Stream(image_feature, memory, train)
        memory_feat = self.Spatial_Context_Refinement_Stream(image_feature, memory)

        # 卷积融合：通过 fusion_conv 将 (B, 2*D, H, W) 转换为 (B, D, H, W)
        updated_image = self.fusion(torch.cat([global_compensation, memory_feat], dim=1))  # (B, D, H, W)
        updated_image = self.depth_conv(updated_image)  # (B, D, H, W)
        updated_image = updated_image.contiguous()

        return updated_image, updated_memory, loss_mem_align, loss_mem_triplet

    def Spatial_Context_Refinement_Stream(self, image_feature, memory):
        B, D, H, W = image_feature.shape
        M, _ = memory.shape

        # Treat memory as conv kernels: measure similarity (B, M, H, W)
        memory_kernels = memory.view(M, D, 1, 1)
        score_maps = F.conv2d(image_feature, weight=memory_kernels)  # (B, M, H, W)

        # Flatten similarity and select top-K memory slots
        score_maps_flat = score_maps.view(B, M, -1)  # (B, M, HW)
        topk_scores, topk_indices = torch.topk(score_maps_flat, k=self.topk, dim=1)  # (B, K, HW)

        # Normalize attention across top-K slots
        attn_weights = F.softmax(topk_scores, dim=1)  # (B, K, HW)

        # Gather top-K memory embeddings
        idx_flat = topk_indices.permute(0, 2, 1).reshape(B, -1)  # (B, HW*K)

        # gather: keys_expanded ∈ (1, M, D) → index ∈ (B, HW*K) → output ∈ (B, HW*K, D)
        keys_exp = memory.unsqueeze(0).expand(B, -1, -1)  # (B, M, D)
        gathered_keys = torch.gather(keys_exp, dim=1, index=idx_flat.unsqueeze(-1).expand(-1, -1, D))  # (B, HW*K, D)

        #  reshape back to (B, HW, K, D) for attention
        gathered_keys = gathered_keys.view(B, H * W, self.topk, D).permute(0, 2, 1, 3)  # (B, K, HW, D)

        # attention weights: (B, K, HW, 1)
        attn_weights = attn_weights.unsqueeze(-1)  # (B, K, HW, 1)

        # Weighted sum of memory
        memory_feat_flat = torch.sum(attn_weights * gathered_keys, dim=1)  # (B, HW, D)

        # reshape back
        memory_feat = memory_feat_flat.transpose(1, 2).reshape(B, D, H, W)
        return memory_feat

    def Global_Pattern_Interaction_Stream(self, image_feature, memory, train):
        if train:
            # Step 1: Perform global pattern adjustment between image features and memory
            I_G, global_compensation, score_memory, score_image = self.Global_Pattern_Adjustment(image_feature, memory)

            # Step 2: Update memory using the adjusted features and similarity scores
            updated_memory, gathering_indices = self.Memory_Evolution(I_G, memory, score_memory, score_image, train)

            # Step 3: Compute memory alignment loss based on updated memory and gathered indices
            loss_mem_align = self.loss_mem_align(I_G, memory, gathering_indices)

            # Step 4: Compute triplet loss to enforce memory discrimination
            loss_mem_triplet = self.loss_mem_triplet(I_G, memory, score_image)

            # Return compensated features, updated memory, and two loss terms
            return global_compensation, updated_memory, loss_mem_align, loss_mem_triplet
        else:
            # Step 1: Perform global pattern adjustment (no memory update during inference)
            I_G, global_compensation, score_memory, score_image = self.Global_Pattern_Adjustment(image_feature, memory)

            # Return compensated features, original memory, and zero losses
            return global_compensation, memory, 0, 0

    def Global_Pattern_Adjustment(self, image_feature, memory):
        """
        Args:
            image_feature: (B, D, H, W).
            memory: (M, D).

        Returns:
            I_G: Global pooled query (B, 1, 1, D).
            global_compensation: Compensated global feature (B, D, H, W).
            score_memory: Similarity scores normalized over queries (B*HW, M).
            score_image: Similarity scores normalized over memory (B*HW, M).
        """

        I_G = F.adaptive_avg_pool2d(image_feature, output_size=(1, 1))  # (B, D, 1, 1)
        I_G = I_G.permute(0, 2, 3, 1)  # (B, 1, 1, D)
        b, h, w, d = I_G.size()
        score_memory, score_image = self.get_score(memory, I_G)  # (B*h*w, M)
        I_G_flat = I_G.contiguous().view(b * h * w, d)  # (B*h*w, d)

        memory_response = torch.matmul(score_image.detach(), memory)  # (B*h*w, d)
        memory_response = I_G_flat + memory_response  # (B*h*w, d)
        memory_response = memory_response.view(b, h, w, d).permute(0, 3, 1, 2)  # (B, d, h, w)
        global_compensation = torch.sigmoid(memory_response) * image_feature

        return I_G, global_compensation, score_memory, score_image

    def get_score(self, memory, I_G):
        """
               Compute query-to-memory similarity.

               Args:
                   memory: (M, D).
                   I_G: (B, h, w, D).

               Returns:
                   score_memory: Normalized over queries (B*h*w, M).
                   score_image: Normalized over memory (B*h*w, M).
               """
        b, h, w, d = I_G.size()
        m, d = memory.size()
        score = torch.matmul(I_G, torch.t(memory))  # (B, h, w, M)
        score = score.view(b * h * w, m)  # (B*h*w, M)
        score_memory = F.softmax(score, dim=0)
        score_image = F.softmax(score, dim=1)
        return score_memory, score_image

    def Memory_Evolution(self, I_G, memory, score_memory, score_image, train):
        """
              Update memory by aggregating query features.

              Args:
                  I_G: Global query (B, 1, 1, D).
                  memory: (M, D).
                  score_memory: (B*h*w, M).
                  score_image: (B*h*w, M).

              Returns:
                  updated_memory: (M, D).
                  gathering_indices: (B*h*w, 1).
              """
        b, h, w, d = I_G.size()
        I_G_flat = I_G.contiguous().view(b * h * w, d)  # (B*h*w, d)
        _, gathering_indices = torch.topk(score_image, 1, dim=1)  # (B*h*w, 1)
        weights = score_memory.gather(1, gathering_indices)  # (N, 1)
        update_vector = I_G_flat * weights  # (N, d)
        N, d = I_G_flat.shape

        # Aggregate into memory slots
        M = score_memory.size(1)
        output = torch.zeros((M, d), device=I_G_flat.device, dtype=I_G_flat.dtype)
        gathering_indices_exp = gathering_indices.expand(-1, d)  # (N, d)
        memory_increment = output.scatter_add_(0, gathering_indices_exp, update_vector)  # 聚合到 (M, d)
        updated_memory = F.normalize(memory_increment + memory, dim=1)  # (M, d)
        return updated_memory.detach(), gathering_indices

    def loss_mem_triplet(self, I_G, memory, score_image):
        """
        Triplet loss: encourage query closer to best-matching memory
        and farther from second-best.

        Args:
            I_G: (B, 1, 1, D).
            memory: (M, D).
            score_image: (B*h*w, M).

        Returns:
            loss: scalar.
        """
        b, h, w, d = I_G.size()
        if score_image.size(1) < 2:
            # If the number of memory items is insufficient, directly return zero loss
            return torch.tensor(0.0, device=I_G.device)
        loss_fn = nn.TripletMarginLoss(margin=1.0)
        I_G_flat = I_G.contiguous().view(b * h * w, d)  # (B*h*w, d)
        _, gathering_indices = torch.topk(score_image, 2, dim=1)  # (B*h*w, 2)
        pos = memory[gathering_indices[:, 0]]  # (B*h*w, d)
        neg = memory[gathering_indices[:, 1]]  # (B*h*w, d)
        loss = loss_fn(I_G_flat, pos.detach(), neg.detach())
        return loss

    def loss_mem_align(self, I_G, memory, gathering_indices):
        """
              Alignment loss: encourage query and its closest memory to align.

              Args:
                  I_G: (B, 1, 1, D).
                  memory: (M, D).
                  gathering_indices: (B*h*w, 1).

              Returns:
                  loss: scalar.
              """
        b, h, w, d = I_G.size()
        loss_fn = nn.MSELoss()
        I_G_flat = I_G.contiguous().view(b * h * w, d)  # (B*h*w, d)
        loss = loss_fn(I_G_flat, memory[gathering_indices].squeeze(1).detach())
        return loss


class Expert(nn.Module):
    def __init__(self, memory_module):
        super(Expert, self).__init__()
        # Each expert has its own memory module
        # memory_module should implement the forward method:
        #     (x, memory, train) → (updated_x, updated_memory, loss_align, loss_triplet)
        self.memory_module = memory_module

    def forward(self, x, memory, train):
        """
        Args:
            x: Input features for this expert. Shape: [B_expert, C, H, W],
               where B_expert is the number of samples routed to this expert.
            memory: The expert's private memory bank (tensor or dict).
            train: Boolean flag. If True, compute memory alignment and triplet losses.

        Returns:
            out: Processed features after memory interaction. Shape same as x.
            updated_memory: Updated memory bank for this expert.
            loss_mem_align: Memory alignment loss (encourages feature consistency).
            loss_mem_triplet: Triplet loss for discriminative memory usage.
        """
        # Each expert processes input using its own memory module
        out, updated_memory, loss_mem_align, loss_mem_triplet = self.memory_module(x, memory, train)
        return out, updated_memory, loss_mem_align, loss_mem_triplet


class Memory_Expert_Compensation_Module(nn.Module):
    def __init__(self, channels, num_experts, top_k, memory_size=64):
        super(Memory_Expert_Compensation_Module, self).__init__()
        # Gate network generates soft assignment (coefficients) for selecting experts
        self.gate = GateNetwork(channels, num_experts, top_k)
        self.channels = channels
        self.num_experts = num_experts
        self.memory_size = memory_size

        # Each expert receives a unique dilation factor to diversify receptive fields
        dilations = [2 * i + 1 for i in range(num_experts)]
        assert num_experts == len(dilations)

        # Construct a pool of experts. Each expert has its own memory module.
        # Memory module is initialized with the corresponding dilation factor.
        self.Memory_Experts = nn.ModuleList([
            Expert(Memory(channels, dilation=d)) for d in dilations
        ])

        # 方式一：将 memory 定义为 nn.Parameter，作为模型的可学习参数
        # 初始化为小型随机值，使用零初始化可能会有问题（对称性）
        self.memory_list = nn.ParameterList([
            nn.Parameter(torch.randn(memory_size, channels) * 0.02)
            for _ in range(num_experts)
        ])

    def forward(self, x, train):
        """
               Forward pass of the Memory Expert Compensation Module (MECM).
               方式一：memory 作为 nn.Parameter 内部维护，无需外部传入

               Args:
                   x: Input features [B, C, H, W].
                   train: Boolean flag indicating whether in training mode.

               Returns:
                   out: Expert-augmented features [B, C, H, W].
                   loss_mem_aligns: Memory alignment loss (averaged across experts).
                   loss_mem_triplets: Memory triplet loss (averaged across experts).
                   cof: Gating coefficients [B, num_experts].
        """
        # Step 1. Compute gating coefficients for each expert
        cof = self.gate(x)  # (batch_size, num_experts)

        # Step 2. Initialize output feature (same shape as input)
        out = torch.zeros_like(x).to(x.device)

        # Step 3. Prepare containers for memory losses
        loss_mem_aligns = []
        loss_mem_triplets = []

        # Step 4. Iterate through all experts
        for idx in range(len(self.Memory_Experts)):
            # Check if any sample selects this expert (differentiable way)
            has_selection = cof[:, idx].sum() > 0

            # Use detached value for Python-level control flow, but keep graph for backward
            if not has_selection.item():
                continue

            # Otherwise, find batch indices where this expert is selected
            mask = torch.where(cof[:, idx] > 0)[0]

            # Each expert processes its own subset of data and updates its memory
            expert_out, updated_memory, loss_mem_align, loss_mem_triplet = \
                self.Memory_Experts[idx](x[mask], self.memory_list[idx], train)

            # 原地更新 memory 参数（保持可求导）
            self.memory_list[idx].data = updated_memory

            loss_mem_aligns.append(loss_mem_align)
            loss_mem_triplets.append(loss_mem_align)

            # Step 5. Apply expert contribution to the output, scaled by gate coefficient
            cof_k = cof[mask, idx].view(-1, 1, 1, 1)
            out[mask] += expert_out * cof_k

        # Step 6. Aggregate losses across all experts
        # if train and len(loss_mem_aligns) > 0:
        #     # Stack all experts' losses and average
        #     loss_mem_aligns = torch.stack(loss_mem_aligns).mean()
        #     loss_mem_triplets = torch.stack(loss_mem_triplets).mean()
        # else:
        #     loss_mem_aligns = torch.tensor(0.0, device=x.device)
        #     loss_mem_triplets = torch.tensor(0.0, device=x.device)

        return out, loss_mem_aligns, loss_mem_triplets, cof


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    x = torch.randn(1, 64, 32, 32).to(device)
    # 方式一：memory 作为 nn.Parameter，无需外部定义
    model = Memory_Expert_Compensation_Module(
        channels=64, 
        num_experts=4, 
        top_k=2,
        memory_size=64  # 每个 expert 的 memory 槽数量
    ).to(device)

    # forward 不再需要传入 memory_list
    y, loss_align, loss_triplet, cof = model(x, train=True)

    print("输入特征维度：", x.shape)
    print("输出特征维度：", y.shape)
    print("Memory 参数数量：", sum(p.numel() for p in model.parameters()))
    print("Memory 参数是否可训练：", all(p.requires_grad for p in model.memory_list))
    print("Gating 系数维度：", cof.shape)
