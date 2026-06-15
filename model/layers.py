import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat

class Highway(nn.Module):
    r"""Highway Layers
    Args:
        - num_highway_layers(int): number of highway layers.
        - input_size(int): size of highway input.
    """

    def __init__(self, num_highway_layers, input_size):
        super(Highway, self).__init__()
        self.num_highway_layers = num_highway_layers
        self.non_linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):  # 256 * 1280
        for layer in range(self.num_highway_layers):
            gate = torch.sigmoid(self.gate[layer](x))  # 经过两次线性映射，用sigmoid激活
            # Compute percentage of non linear information to be allowed for each element in x
            non_linear = F.relu(self.non_linear[layer](x))  # 经过两次线性映射，用relu激活
            # Compute non linear information
            linear = self.linear[layer](x)  # 经过两次线性映射
            # Compute linear information
            x = gate * non_linear + (1 - gate) * linear
            # Combine non linear and linear information according to gate
            x = self.dropout(x)
        return x




class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)




class PostNormForward(nn.Module):
    def __init__(self, d_model, fn):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))


class PostNormAttention(nn.Module):
    def __init__(self, d_model, fn):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        return self.norm(self.fn(q, k, v))


class Attention(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.scale = d_model ** -0.5
        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(d_model, d_model)
        self.to_value = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values, mask=None):
        # b: batch_size、l,n: sequence_length, _: embedding_dim，h: attention heads
        b, l, _, h = *queries.shape, self.n_heads
        _, n, _, = keys.shape
        queries = self.to_query(queries)
        keys = self.to_key(keys)
        values = self.to_value(values)
        # (batch_size, sequence_length, num_heads, dim_per_head)
        queries = queries.view(b, l, h, -1).transpose(1, 2)
        keys = keys.view(b, n, h, -1).transpose(1, 2)
        values = values.view(b, n, h, -1).transpose(1, 2)
        # (batch_size, num_heads, sequence_length, sequence_length)
        dots = torch.einsum('bhid,bhjd->bhij', queries, keys) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'
            mask = mask[:, None, :].expand(-1, l, -1)
            dots.masked_fill_(~mask, float('-inf'))

        attn = dots.softmax(dim=-1)

        self.saved_attn = attn.detach().cpu()

        out = torch.einsum('bhij,bhjd->bhid', attn, values)
        out = out.transpose(1, 2).contiguous().view(b, l, -1)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, depth, n_heads, ff_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # PreNormAttention(d_model, Attention(d_model, n_heads = n_heads)),
                # PreNormForward(d_model, FeedForward(d_model, ff_dim, dropout = dropout))
                PostNormAttention(d_model, Attention(d_model, n_heads=n_heads)),
                PostNormForward(d_model, FeedForward(d_model, ff_dim, dropout=dropout))
            ]))

    def forward(self, x, save_hidden=False):
        if save_hidden == True:
            hidden_list = []
            hidden_list.append(x)
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
                hidden_list.append(x)
            return hidden_list
        else:
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
            return x


class CrossTransformerEncoder(nn.Module):
    def __init__(self, d_model, depth, n_heads, ff_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # PreNormAttention(d_model, Attention(d_model, n_heads = n_heads)),
                # PreNormForward(d_model, FeedForward(d_model, ff_dim, dropout = dropout))
                PostNormAttention(d_model, Attention(d_model, n_heads=n_heads)),
                PostNormForward(d_model, FeedForward(d_model, ff_dim, dropout=dropout))
            ]))

    def forward(self, target_x, source_x, ):
        for attn, ff in self.layers:
            target_x_tmp = attn(target_x, source_x, source_x)
            target_x = target_x_tmp + target_x
            target_x = ff(target_x) + target_x
        return target_x




class Multi_CA(nn.Module):
    def __init__(self, dim, ff_dim, heads=8, dropout=0.):
        super().__init__()

        self.heads = heads
        dim_head = int(dim / heads)
        self.scale = dim_head ** -0.5

        self.softmax = nn.Softmax(dim=-1)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k_t = nn.Linear(dim, dim, bias=False)
        self.to_k_a = nn.Linear(dim, dim, bias=False)
        self.to_k_v = nn.Linear(dim, dim, bias=False)
        self.to_v_t = nn.Linear(dim, dim, bias=False)
        self.to_v_a = nn.Linear(dim, dim, bias=False)
        self.to_v_v = nn.Linear(dim, dim, bias=False)

        self.ffn = FeedForward(dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    # def forward(self, query, h_t, h_a, h_v):
    #     b, n, _, h = *h_t.shape, self.heads
    #
    #     q = self.to_q(query)
    #     k_t = self.to_k_t(h_t)
    #     k_a = self.to_k_a(h_a)
    #     k_v = self.to_k_v(h_v)
    #     v_t = self.to_v_t(h_t)
    #     v_a = self.to_v_a(h_a)
    #     v_v = self.to_v_v(h_v)
    #
    #     q, k_t, k_a, k_v, v_t, v_a, v_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
    #                                           (q, k_t, k_a, k_v, v_t, v_a, v_v))
    #
    #     dots_qt = einsum('b h i d, b h j d -> b h i j', q, k_t) * self.scale
    #
    #     # attention
    #     attn_qt = self.softmax(dots_qt)
    #     out_qt = einsum('b h i j, b h j d -> b h i d', attn_qt, v_t)
    #     out_qt = rearrange(out_qt, 'b h n d -> b n (h d)')
    #
    #     dots_qa = einsum('b h i d, b h j d -> b h i j', q, k_a) * self.scale
    #     attn_qa = self.softmax(dots_qa)
    #     out_qa = einsum('b h i j, b h j d -> b h i d', attn_qa, v_a)
    #     out_qa = rearrange(out_qa, 'b h n d -> b n (h d)')
    #
    #     dots_qv = einsum('b h i d, b h j d -> b h i j', q, k_v) * self.scale
    #     attn_qv = self.softmax(dots_qv)
    #     out_qv = einsum('b h i j, b h j d -> b h i d', attn_qv, v_v)
    #     out_qv = rearrange(out_qv, 'b h n d -> b n (h d)')
    #
    #     out_tav = query + out_qt + out_qa + out_qv
    #     out_tav = self.norm1(out_tav)
    #
    #     out_tav = out_tav + self.ffn(out_tav)
    #     out_tav = self.norm2(out_tav)
    #
    #     return out_tav

    def forward(self, query, h_t, h_a, h_v):
        # 输入维度:
        # query: [B, num_slots, D]
        # h_t, h_a, h_v: [B, 1, D]
        h = self.heads

        # 1. 映射 Query
        q = self.to_q(query)

        # 2. 映射 Key，并在序列维度(dim=1)拼接
        k_t = self.to_k_t(h_t)
        k_a = self.to_k_a(h_a)
        k_v = self.to_k_v(h_v)
        K = torch.cat([k_t, k_a, k_v], dim=1)  # 拼接后形状: [B, 3, D]

        # 3. 映射 Value，并在序列维度(dim=1)拼接
        v_t = self.to_v_t(h_t)
        v_a = self.to_v_a(h_a)
        v_v = self.to_v_v(h_v)
        V = torch.cat([v_t, v_a, v_v], dim=1)  # 拼接后形状: [B, 3, D]

        # 4. 拆分多头注意力维度 (einops神器自动处理所有形状)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        K = rearrange(K, 'b n (h d) -> b h n d', h=h)
        V = rearrange(V, 'b n (h d) -> b h n d', h=h)

        # 5. 计算 Attention 权重
        # i 代表 query 的序列长度 (num_slots)
        # j 代表 Key 的序列长度 (3)
        dots = einsum('b h i d, b h j d -> b h i j', q, K) * self.scale
        attn = self.softmax(dots)  # 在包含 DrugA, DrugB, Cell 这3个元素的维度上进行全局 Softmax 竞争

        # 6. 加权求和
        out = einsum('b h i j, b h j d -> b h i d', attn, V)
        out = rearrange(out, 'b h n d -> b n (h d)') # 恢复形状 [B, num_slots, D]

        # 7. 残差连接与前馈网络 (和原来的逻辑保持一致)
        out_tav = self.norm1(query + out)
        out_tav = out_tav + self.ffn(out_tav)
        out_tav = self.norm2(out_tav)

        return out_tav




class HierarchicalMechanismBottleneckFusion(nn.Module):
    def __init__(self, d_model=256, n_heads=4, ff_dim=512, drop_out=0.5, depth=2, num_slots=4):
        super(HierarchicalMechanismBottleneckFusion, self).__init__()
        assert depth >= 1, "depth must be >= 1"

        self.depth = depth
        # 【修改 1】这里定义初始槽位的数量
        self.initial_query_len = num_slots

        self.encoder_q = nn.ModuleList()
        self.encoder_q2tav = nn.ModuleList()
        self.encoder_t = nn.ModuleList()
        self.encoder_a = nn.ModuleList()
        self.encoder_v = nn.ModuleList()


        # 形状: [1, num_slots, d_model]
        self.mechanism_slots = nn.Parameter(torch.randn(1, self.initial_query_len, d_model))
        # 使用正交初始化，保证槽位的多样性
        nn.init.orthogonal_(self.mechanism_slots)

        for i in range(depth):
            self.encoder_q.append(TransformerEncoder(d_model, 1, n_heads, ff_dim, drop_out))
            self.encoder_q2tav.append(Multi_CA(d_model, ff_dim, n_heads, drop_out))
            self.encoder_t.append(CrossTransformerEncoder(d_model, 1, n_heads, ff_dim, drop_out))
            self.encoder_a.append(CrossTransformerEncoder(d_model, 1, n_heads, ff_dim, drop_out))
            self.encoder_v.append(CrossTransformerEncoder(d_model, 1, n_heads, ff_dim, drop_out))


    def forward(self, x_t, x_a, x_v):
        # 【修改 5】移除了 x_m 参数，现在只需要三个视图/模态的输入
        # x_t: Drug A, x_a: Drug B, x_v: Cell (对应关系由你决定)

        # 1. 维度处理：确保输入是序列形式 [B, 1, D]
        if x_t.dim() == 2:
            x_t = x_t.unsqueeze(1)
        if x_a.dim() == 2:
            x_a = x_a.unsqueeze(1)
        if x_v.dim() == 2:
            x_v = x_v.unsqueeze(1)

        m_t, m_a, m_v = x_t, x_a, x_v

        # 【修改 6】初始化 Query (机制槽)
        B = x_t.size(0)
        # 将静态的参数复制到当前 Batch
        query = self.mechanism_slots.repeat(B, 1, 1)  # [B, num_slots, D]

        keep = self.initial_query_len

        for level in range(self.depth):
            # A. 槽位自更新 (Self-Refinement)
            # 在第一层，这是槽位之间的自注意力；在后续层，这是融合信息后的整理
            query = self.encoder_q[level](query)

            # B. 瓶颈压缩 (Bottleneck Compression)
            # 这一步体现了 HBF 的精髓：每层只保留前 keep 个槽位
            # 强迫模型把最重要的协同模式压缩到前面的槽位中
            # query = query[:, :keep]

            # if level != 0:
            #     query = query[:, keep:]

            # perm = torch.randperm(query.size(1))
            # query = query[:, perm]
            # query = query[:, :keep]

            # C. 槽位从三方获取信息 (Gather / Extraction)
            # Query=Slots, Key/Value=Drugs/Cell
            # 槽位主动去"看"药物和细胞，提取相关特征
            query = self.encoder_q2tav[level](query, m_t, m_a, m_v)

            # D. 三方从槽位获取反馈 (Distribute / Update)
            # Query=Drugs/Cell, Key/Value=Slots
            # 药物和细胞根据槽位提取出的"协同模式"来更新自己
            m_t = self.encoder_t[level](m_t, query)
            m_a = self.encoder_a[level](m_a, query)
            m_v = self.encoder_v[level](m_v, query)

            # E. 减少下一层的槽位数量
            #######[消融 实验]#############
            keep = max(1, keep // 2)

        # 返回更新后的 DrugA, DrugB, Cell 特征
        # [:, 0] 是因为输入被 unsqueeze 成了长度 1，这里 squeeze 回去
        return m_t[:, 0], m_a[:, 0], m_v[:, 0]

