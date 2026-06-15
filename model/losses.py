import torch
import torch.nn as nn
import torch.nn.functional as F


class CGSCL_Loss(nn.Module):
    """
    Confidence-Gated Asymmetric Supervised Contrastive Loss
    创新点: 结合了 MMLoss 的非对称梯度思想 + Supervised Contrastive Learning
    """

    def __init__(self, temperature=0.1):
        super(CGSCL_Loss, self).__init__()
        self.temperature = temperature

    def forward(self, z_view1, z_view2, p_view1, p_view2, labels):
        """
        :param z_view1: [B, D] 视图1的特征 (Teacher Candidate)
        :param z_view2: [B, D] 视图2的特征 (Student)
        :param p_view1: [B, C] 视图1的预测 Logits
        :param p_view2: [B, C] 视图2的预测 Logits
        :param labels:  [B] 真实标签
        """
        device = z_view1.device
        batch_size = z_view1.shape[0]

        # --- 0. 预处理特征 (提前定义 z_student) ---
        # 归一化
        z_view1 = F.normalize(z_view1, dim=1)
        z_view2 = F.normalize(z_view2, dim=1)

        # 定义角色：View2 (Student) 向 View1 (Teacher) 学习
        # Teacher 需要 detach，不更新梯度；Student 保留梯度
        z_teacher = z_view1.detach()
        z_student = z_view2

        # --- 1. 计算置信度权重 (Confidence Weighting) ---
        probs1 = F.softmax(p_view1, dim=1)
        probs2 = F.softmax(p_view2, dim=1)

        # gather: 取出真实标签对应的概率值
        conf1 = probs1.gather(1, labels.view(-1, 1)).squeeze()
        conf2 = probs2.gather(1, labels.view(-1, 1)).squeeze()

        # --- 2. 动态门控掩码 (Dynamic Gating Mask) ---
        # gap > 0 表示 Teacher 比 Student 更好
        gap = conf1 - conf2

        # 权重设计：
        # Teacher 必须自信 (conf1 > 0.5) 且 比 Student 准 (gap > 0)
        weights = torch.clamp(gap, min=0) * (conf1 > 0.5).float()

        # --- 3. 归一化权重与零梯度处理 ---
        if weights.sum() > 0:
            weights = weights / weights.sum() * batch_size
        else:
            # 如果没有一个样本满足条件 (权重全为0)，返回带梯度的 0
            # 使用 z_student 确保计算图不断裂 (虽然梯度是0)
            return 0.0 * z_student.sum()

        # --- 4. 构建有监督对比矩阵 ---
        # 相似度: [B, B]
        sim_matrix = torch.matmul(z_student, z_teacher.T) / self.temperature

        # 正样本 Mask: 标签相同的即为正样本
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # --- 5. 计算加权 Loss ---
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # mean_log_prob_pos: 每个样本 i 与所有同类样本 j 的平均相似度
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        # Apply Weights: 加权平均
        loss = - (weights * mean_log_prob_pos).mean()

        return loss



def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    p_loss = p_loss.mean()
    q_loss = q_loss.mean()
    loss = (p_loss + q_loss) / 2
    return loss


