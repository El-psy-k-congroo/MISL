# -*- coding: utf-8 -*-
# @Time    : 2025/3/30 10:33
# @Author  : jxpeng
# @FileName: Synergy.py
# @Software: PyCharm
# @Email   : 1367523296@qq.com
import math
import os
import sys

from torch_geometric.nn import global_mean_pool

from model.layers import HierarchicalMechanismBottleneckFusion

BASEDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEDIR)
import torch
from torch import nn
from encoder import RGCN
from fusion import BiContextModulation, DualInteract, CrossCoAttention, CrossInteractionAttention
from losses import CGSCL_Loss
import torch



class Synergy(nn.Module):
    def __init__(self, max_mol_rel,
                 used_drug_dict, used_cell_dict, input_dim, hidden_dim, output_dim, num_relations=6, proj_dim=256, depth=2, num_slots=4, dropout=0.5,
                 device='cuda:0'):
        super(Synergy, self).__init__()

        self.device = device


        self.used_drug_dict = used_drug_dict
        self.used_cell_dict = used_cell_dict

        # 1. KG Encoder
        self.kg_rgcn_encoder = RGCN(input_dim, hidden_dim, output_dim, num_relations=num_relations)

        # 2. Molecule Graph Encoder
        self.mol_rgcn_encoder = RGCN(78, hidden_dim, proj_dim, num_relations=max_mol_rel)
        # self.get_drug_target_embs = GraphTransformer(78, hid_feats, proj_dim)

        self.kg_drug_dense = nn.Sequential(nn.Linear(output_dim, proj_dim), nn.ReLU(), nn.LayerNorm(proj_dim),
                                                nn.Dropout(dropout))
        self.kg_cell_dense = nn.Sequential(nn.Linear(output_dim, proj_dim), nn.ReLU(), nn.LayerNorm(proj_dim),
                                                   nn.Dropout(dropout))
        self.fp_dense = nn.Sequential(nn.Linear(1024, proj_dim), nn.ReLU(), nn.LayerNorm(proj_dim),
                                                nn.Dropout(dropout))

        self.cell_dense = nn.Sequential(nn.Linear(16383, proj_dim), nn.ReLU(), nn.LayerNorm(proj_dim),
                                                   nn.Dropout(dropout))


        self.projection_graph_fp_context = nn.Sequential(nn.Linear(proj_dim * 2, proj_dim), nn.ReLU(), nn.LayerNorm(proj_dim),
                                                   nn.Dropout(dropout))


        self.hbf = HierarchicalMechanismBottleneckFusion(d_model=proj_dim, depth=depth, num_slots=num_slots)


        self.context_modulation = BiContextModulation(proj_dim)

        self.feature_interact_kg = DualInteract(field_dim=3, embed_size=proj_dim, dropout=dropout, layers=2)  # 0.5
        self.feature_interact_fd = DualInteract(field_dim=3, embed_size=proj_dim, dropout=dropout, layers=2)  # 0.5


        self.transform_modality_kg = nn.Sequential(nn.LayerNorm(proj_dim * 3), nn.Linear(proj_dim * 3, 2))
        self.transform_modality_fd = nn.Sequential(nn.LayerNorm(proj_dim * 3), nn.Linear(proj_dim * 3, 2))


        self.transform_fusion = nn.Sequential(nn.LayerNorm(proj_dim * 6), nn.Linear(proj_dim * 6, 2))


    def to(self, device):
        super().to(device)  # 这一句通常能处理大部分子模块

    def reset_parameters(self):

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                # Xavier/Glorot初始化（适合线性层）
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                # BN层初始化
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(_init_weights)

        for m in self.modules():
            if m is self:
                continue
            # 如果子模块实现了 reset_parameters 且不是我们已经通过 apply 覆盖的基础模块
            if hasattr(m, 'reset_parameters') and not isinstance(m, (nn.Linear, nn.BatchNorm1d)):
                try:
                    # 有些 reset_parameters 可能需要额外参数或会抛异常，故用 try/except 包裹
                    m.reset_parameters()
                except TypeError:
                    # 少见：当 reset_parameters 需要参数时，可选择忽略或记录
                    pass
                except Exception:
                    # 若某子模块的 reset_parameters 出错，可打印以便调试
                    print(f"Warning: reset_parameters failed for module {m.__class__.__name__}")



    def forward(self, drugA_ids, drugB_ids, cell_ids, labels, mol_data, kg_data, cell_feature, drug_fp_feature, infer=False, return_features=False):

        # 1. 获取 KG Embeddings
        kg_rgcn_embedding = self.kg_rgcn_encoder(kg_data.x, kg_data.edge_index, kg_data.edge_attr)

        drug_tg_1 = self.kg_drug_dense(kg_rgcn_embedding[drugA_ids])
        drug_tg_2 = self.kg_drug_dense(kg_rgcn_embedding[drugB_ids])
        cline_tg = self.kg_cell_dense(kg_rgcn_embedding[cell_ids])

        # KG Triplet
        h_kg_triplet = torch.stack([drug_tg_1, drug_tg_2, cline_tg], dim=1)

        # 2. 获取特征域 Embeddings
        cline_fea_proj = self.cell_dense(cell_feature)
        drug_fp_fea_proj = self.fp_dense(drug_fp_feature)
        drug_graph_fea_proj = self.mol_rgcn_encoder(mol_data.x, mol_data.edge_index, mol_data.edge_attr)
        drug_graph_fea_proj = global_mean_pool(drug_graph_fea_proj, mol_data.batch)

        fused_drug_embs = self.projection_graph_fp_context(torch.cat([drug_graph_fea_proj, drug_fp_fea_proj], dim=1))

        drugA_map_ids = [self.used_drug_dict[id] for id in drugA_ids.tolist()]
        drugB_map_ids = [self.used_drug_dict[id] for id in drugB_ids.tolist()]
        cell_map_ids = [self.used_cell_dict[id] for id in cell_ids.tolist()]

        drug_fp_1, drug_fp_2 = fused_drug_embs[drugA_map_ids], fused_drug_embs[drugB_map_ids]
        cline_fp = cline_fea_proj[cell_map_ids]


        update_d1, update_d2, update_c = self.hbf(drug_fp_1, drug_fp_2, cline_fp)
        h_fd_triplet = torch.stack([update_d1, update_d2, update_c], dim=1)

        ###########【消融实验】####################

        # h_fd_triplet = torch.stack([drug_fp_1, drug_fp_2, cline_fp], dim=1)

        ###########【消融实验】####################


        h_kg_mod, h_fd_mod = self.context_modulation(h_kg_triplet, h_fd_triplet)
        # h_kg_mod = h_kg_triplet
        # h_fd_mod = h_fd_triplet

        modality_kg = self.feature_interact_kg(h_kg_mod)
        modality_fd = self.feature_interact_fd(h_fd_mod)

        ###########【消融实验】####################

        # b, f, e = h_kg_mod.shape
        #
        # modality_kg = h_kg_mod.reshape(b, f * e)
        # modality_fd = h_fd_mod.reshape(b, f * e)

        ###########【消融实验】####################

        # --- 4. 预测与融合 ---
        output_kg = self.transform_modality_kg(modality_kg)
        output_fd = self.transform_modality_fd(modality_fd)


        # 3. 融合与输出
        fusion_modality = torch.cat([modality_kg, modality_fd], dim=1)
        output_mm = self.transform_fusion(fusion_modality)

        #########[消融实验]###########################

        if return_features:
            # 阶段 (a): 初始拼接特征 (HMBF 之前)
            initial_triplet = torch.stack([drug_fp_1, drug_fp_2, cline_fp], dim=1)
            b_size = initial_triplet.size(0)

            # 将三元组张量展平为一维向量以供 t-SNE 使用
            features_dict = {
                'initial': h_fd_triplet.view(b_size, -1).detach().cpu().numpy(),
                'macro_kg': h_kg_mod.view(b_size, -1).detach().cpu().numpy(),
                'micro_fd': h_fd_mod.view(b_size, -1).detach().cpu().numpy(),
                'final_joint': fusion_modality.detach().cpu().numpy()
            }
            # 返回预测结果和特征字典
            return output_mm, features_dict


        if infer:
            return output_mm

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        MMLoss_m = torch.mean(criterion(output_mm, labels))
        MMLoss_m0 = torch.mean(criterion(output_kg, labels))
        MMLoss_m1 = torch.mean(criterion(output_fd, labels))
        MMLoss_sum = MMLoss_m + MMLoss_m0 + MMLoss_m1

        return MMLoss_sum, output_mm

    def infer(self, drugA_ids, drugB_ids, cell_ids, labels, mol_data,  kg_data, cell_feature, drug_feature,return_features=False):
        MMlogit = self.forward(drugA_ids, drugB_ids, cell_ids, labels, mol_data, kg_data, cell_feature, drug_feature, infer=True, return_features=return_features)
        return MMlogit



