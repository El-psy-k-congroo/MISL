import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Highway

class CrossCoAttention(nn.Module):
    def __init__(self, input_dim, heads=4, dropout_rate=0.5):
        super(CrossCoAttention, self).__init__()
        self.attn1 = nn.MultiheadAttention(embed_dim=input_dim, num_heads=heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim=input_dim, num_heads=heads, batch_first=True)

        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)

        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(input_dim)
        )

        # 使用 LayerNorm 替换 BatchNorm (Transformer 类结构通常首选 LayerNorm)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.ReLU()

    def forward(self, A, B):
        """
        输入:
        A: (batch_size, input_dim)
        B: (batch_size, input_dim)

        输出:
        output: (batch_size, input_dim) -> 融合了 A-B 交互信息的特征
        """
        # 增加序列维度: (Batch, Dim) -> (Batch, 1, Dim)
        a_seq = A.unsqueeze(1)
        b_seq = B.unsqueeze(1)

        # 1. Cross-attention (A 关注 B)
        # Query=A, Key=B, Value=B
        attn_output1, _ = self.attn1(a_seq, b_seq, b_seq)
        attn_output1 = self.dropout(attn_output1)
        attn_output1 = self.norm1(attn_output1 + a_seq)  # 残差
        attn_output1 = self.act(self.linear1(attn_output1))

        # 2. Cross-attention (B 关注 A)
        # Query=B, Key=A, Value=A
        attn_output2, _ = self.attn2(b_seq, a_seq, a_seq)
        attn_output2 = self.dropout(attn_output2)
        attn_output2 = self.norm2(attn_output2 + b_seq)  # 残差
        attn_output2 = self.act(self.linear2(attn_output2))

        # 3. 拼接与融合
        # cat dim=2: (Batch, 1, Dim) + (Batch, 1, Dim) -> (Batch, 1, 2*Dim)
        combined = torch.cat([attn_output1, attn_output2], dim=2)

        # 降维并去掉序列维度: (Batch, 1, 2*Dim) -> (Batch, 1, Dim) -> (Batch, Dim)
        output = self.fusion(combined).squeeze(1)

        return output


class CrossInteractionAttention(nn.Module):
    def __init__(self, input_dim, heads=4, dropout_rate=0.2):
        super(CrossInteractionAttention, self).__init__()

        # --- 1. 注意力机制 ---
        self.attn_context1 = nn.MultiheadAttention(embed_dim=input_dim, num_heads=heads, batch_first=True)
        self.attn_context2 = nn.MultiheadAttention(embed_dim=input_dim, num_heads=heads, batch_first=True)

        # --- 2. 分支内的非线性变换 (Feed-Forward) ---
        # 作用：处理从 Context 抓取来的特征，增加非线性表达能力
        self.ffn_branch1 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),  # 关键的非线性激活
            nn.Dropout(dropout_rate)
        )

        self.ffn_branch2 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),  # 关键的非线性激活
            nn.Dropout(dropout_rate)
        )

        # --- 3. Norm 层 ---
        # 这里的 Norm 放在残差连接之后
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        # --- 4. 最终融合层 ---
        # 将 [原始A, 处理后的分支1, 处理后的分支2] 融合
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim),
            nn.ReLU(),  # 融合后的非线性激活
            nn.Dropout(dropout_rate),
            nn.LayerNorm(input_dim)
        )

    def forward(self, target, context1, context2):
        """
        target: (Batch, Dim) -> A
        context1: (Batch, Dim) -> B
        context2: (Batch, Dim) -> C
        """
        # (Batch, Dim) -> (Batch, 1, Dim)
        q = target.unsqueeze(1)
        k1 = v1 = context1.unsqueeze(1)
        k2 = v2 = context2.unsqueeze(1)

        # -------------------------------------------------------
        # 分支 1: Target 查询 Context1 (A 关注 B)
        # -------------------------------------------------------
        attn_out1, _ = self.attn_context1(q, k1, v1)
        feat_1 = self.norm1(attn_out1 + q)
        feat_1 = self.ffn_branch1(feat_1)

        # -------------------------------------------------------
        # 分支 2: Target 查询 Context2 (A 关注 C)
        # -------------------------------------------------------
        attn_out2, _ = self.attn_context2(q, k2, v2)
        feat_2 = self.norm2(attn_out2 + q)
        feat_2 = self.ffn_branch2(feat_2)

        # -------------------------------------------------------
        # 3. 拼接与融合
        # -------------------------------------------------------
        # 拼接: [原始Target, 分支1特征, 分支2特征]
        # 注意：这里的 feat_1 和 feat_2 已经包含了一部分 target 的信息(通过残差)，
        # 但再次显式拼接原始 q 有助于保留最纯粹的自身特征，防止过平滑。
        combined = torch.cat([q, feat_1, feat_2], dim=2)

        # (Batch, 1, 3*Dim) -> (Batch, Dim)
        updated_target = self.fusion(combined).squeeze(1)

        return updated_target


class BiContextModulation(nn.Module):
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
