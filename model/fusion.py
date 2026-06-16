import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Highway



class DualViewCoModulation(nn.Module):
    """
    负责在进入 DualInteract 之前，利用对方视图的背景知识修正特征
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 提取上下文 (Self-Attention)
        self.ctx_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm_ctx = nn.LayerNorm(dim)

        # FiLM 生成器 (Input: Context -> Output: Gamma, Beta)
        self.film_gen = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU())

        self.gate_gen = nn.Sequential(nn.Linear(dim * 2, 1), nn.Sigmoid())  # 门控

    def get_context(self, triplet):
        # triplet: [Batch, 3, Dim]
        attn_out, _ = self.ctx_attn(triplet, triplet, triplet)
        context = self.norm_ctx(triplet + attn_out).mean(dim=1)  # [Batch, Dim]

        return context

    def apply_film(self, x_triplet, context_other):
        # x_triplet: [Batch, 3, Dim] (要被修正的特征)
        # context_other: [Batch, Dim] (对方的上下文)

        B, N, D = x_triplet.shape
        # 展平处理: [Batch*3, Dim]
        x_flat = x_triplet.view(-1, D)
        ctx_expand = context_other.repeat_interleave(N, dim=0)  # 对齐

        # 生成 FiLM 参数
        params = self.film_gen(ctx_expand)
        gamma, beta = torch.split(params, self.dim, dim=1)
        gamma = 1.0 + torch.tanh(gamma)


        # 计算门控
        combined = torch.cat([x_flat, ctx_expand], dim=1)
        alpha = self.gate_gen(combined)

        # 修正
        x_corrected = (gamma * x_flat) + beta
        out = (1 - alpha) * x_flat + alpha * x_corrected

        return out.view(B, N, D)  # 变回 [Batch, 3, Dim]

    def forward(self, h_kg_triplet, h_fd_triplet):
        # 1. 各自计算上下文
        ctx_kg = self.get_context(h_kg_triplet)
        ctx_fd = self.get_context(h_fd_triplet)

        # 2. 互相修正
        h_kg_mod = self.apply_film(h_kg_triplet, ctx_fd)  # FD 修正 KG
        h_fd_mod = self.apply_film(h_fd_triplet, ctx_kg)  # KG 修正 FD

        return h_kg_mod, h_fd_mod




class DualInteract(nn.Module):
    def __init__(self, field_dim, embed_size, dropout=0.5, layers=2):
        super(DualInteract, self).__init__()
        self.bit_wise_net = Highway(input_size=field_dim * embed_size,
                                    num_highway_layers=layers)

        hidden_dim = 1024
        self.trans_bit_nn = nn.Sequential(
            nn.LayerNorm(field_dim * embed_size),
            nn.Linear(field_dim * embed_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, field_dim * embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
            x : batch, field_dim, embed_dim
        """
        b, f, e = x.shape  # batch_size, 5, 256
        bit_wise_x = self.bit_wise_net(x.reshape(b, f * e))  # batch_size, 1280
        m_bit = self.trans_bit_nn(bit_wise_x)  # batch_size, 1280
        m_x = m_bit + x.reshape(b, f * e)
        return m_x
