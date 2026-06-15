import json
import os

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
import logging
from rdkit.Chem import rdMolDescriptors

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

e_map = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'QUADRUPLE',
        'QUINTUPLE',
        'HEXTUPLE',
        'ONEANDAHALF',
        'TWOANDAHALF',
        'THREEANDAHALF',
        'FOURANDAHALF',
        'FIVEANDAHALF',
        'AROMATIC',
        'IONIC',
        'HYDROGEN',
        'THREECENTER',
        'DATIVEONE',
        'DATIVE',
        'DATIVEL',
        'DATIVER',
        'OTHER',
        'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


def atom_features(atom):
    # 44 + 11 + 11 + 11 + 1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()]), atom.GetDegree()


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# mol smile to mol graph edge index
def single_smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    degrees = []
    for atom in mol.GetAtoms():
        feature, degree = atom_features(atom)
        features.append((feature.astype(np.float32)).tolist())
        degrees.append(degree)

    mol_index = []  ##begin, end, rel
    for bond in mol.GetBonds():
        mol_index.append(
            [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), e_map['bond_type'].index(str(bond.GetBondType()))])
        mol_index.append(
            [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx(), e_map['bond_type'].index(str(bond.GetBondType()))])

    if len(mol_index) == 0:
        return 0, 0, 0, 0

    mol_index = np.array(sorted(mol_index))
    mol_edge_index = mol_index[:, :2]
    mol_rel_index = mol_index[:, 2]

    # c_size:原子的个数
    # features:每个原子的特征 c_size * 78
    # edge_index:边 n_edges * 2
    return c_size, features, mol_edge_index.tolist(), mol_rel_index.tolist()


def smile_to_graph(datapath, drug_smiles, entity_id, device='cuda:0'):
    smile_graph = {}

    paths = datapath + f"/smile2graph.json"

    if os.path.exists(paths):
        with open(paths, 'r') as f:
            smile_graph = json.load(f)

        max_rel = 0
        max_deg = 0
        for s in smile_graph.keys():
            max_rel = max(smile_graph[s][3]) if max(smile_graph[s][3]) > max_rel else max_rel
            # max_deg = max(smile_graph[s][8]) if max(smile_graph[s][8]) > max_deg else max_deg
        return smile_graph, max_rel + 1, max_deg + 1

    max_rel = 0
    max_deg = 0
    for d in drug_smiles.keys():
        smiles = drug_smiles[d]

        mol = Chem.MolFromSmiles(smiles)
        lg = Chem.MolToSmiles(mol)

        c_size, features, edge_index, rel_index = single_smile_to_graph(lg)
        if c_size == 0:
            continue
        # abs_pe, rel_pe_idx, rel_pe_val, log_deg, deg = add_full_rrwp(edge_index, c_size, walk_length=walk_length, attr_name_abs=attr_name_abs, device=device)
        # abs_pe, rel_pe_idx, rel_pe_val, log_deg, deg = abs_pe.cpu().tolist(), rel_pe_idx.cpu().tolist(), rel_pe_val.cpu().tolist(), log_deg.cpu().tolist(), deg.cpu().tolist()
        max_rel = max(max_rel, max(rel_index))
        # max_deg = max(max_deg, max(deg))
        smile_graph[entity_id[d]] = c_size, features, edge_index, rel_index

    with open(paths, 'w') as f:
        json.dump(smile_graph, f)

    return smile_graph, max_rel + 1, max_deg + 1


def load_drug_data_features(datapath: str, drug_smiles: dict, entity_id, device):
    smile_graph, max_rel, max_deg = smile_to_graph(datapath, drug_smiles, entity_id, device)

    return smile_graph, max_rel, max_deg


def get_Fingerprint(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    nbits = 1024
    fpFunc_dict = {}
    fpFunc_dict['hashap'] = lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nbits)
    fp = fpFunc_dict['hashap'](mol)
    return np.asarray(fp)


def build_aligned_features(
        entity_mapping: dict,
        used_cell_dict: dict,
        used_drug_dict: dict,
        base_path: str,
        logger,
        device):
    drug_smiles_path = os.path.join(base_path, 'drug_smiles.csv')
    cline_fea_path = os.path.join(base_path, 'Expression.csv')

    # 2. 读取原始特征文件
    logger.info("Loading raw feature files...")
    # 假设第一列是索引(name)
    cline_feature_df = pd.read_csv(cline_fea_path, index_col=0)
    drug_features_df = pd.read_csv(drug_smiles_path, index_col=0)

    # 3. 建立反向映射 (Global ID -> Name)
    # 因为 used_dict 给的是 ID，我们需要知道 ID 对应的名字才能去 CSV 查特征
    id_to_name = {v: k for k, v in entity_mapping.items()}

    # ==========================================
    # 4. 构建 Cell 特征矩阵
    # ==========================================
    # 将 used_cell_dict 按局部索引(value)排序，确保矩阵的第 i 行对应局部索引 i
    # sorted_cells: list of (Global_ID, Local_Index) ordered by Local_Index
    sorted_cells = sorted(used_cell_dict.items(), key=lambda x: x[1])

    num_cells = len(sorted_cells)
    expr_dim = cline_feature_df.shape[1]

    # 初始化全0矩阵 (Rows, Features)
    cell_matrix = np.zeros((num_cells, expr_dim))

    logger.info(f"Building Cell Matrix for {num_cells} cells...")

    for global_id, local_idx in sorted_cells:
        name = id_to_name.get(global_id)

        if name and name in cline_feature_df.index:
            cell_matrix[local_idx] = cline_feature_df.loc[name].values
        else:
            # 如果找不到名字或名字不在CSV中，保持为0 (Zero Padding)
            # print(f"Warning: Cell {name} (ID: {global_id}) feature missing.")
            pass

            # ==========================================
    # 5. 构建 Drug 特征矩阵
    # ==========================================
    sorted_drugs = sorted(used_drug_dict.items(), key=lambda x: x[1])
    num_drugs = len(sorted_drugs)

    # 动态确定指纹维度 (尝试找一个存在的药物计算长度)
    fp_dim = 167  # 默认 fallback
    for global_id, _ in sorted_drugs:
        name = id_to_name.get(global_id)
        if name and name in drug_features_df.index:
            try:
                smiles = str(drug_features_df.loc[name].iloc[0])
                # 注意：这里需要确保你引入了 get_Fingerprint 函数
                temp_fp = get_Fingerprint(smiles)
                fp_dim = len(temp_fp)
                break
            except:
                continue

    # 初始化全0矩阵
    drug_matrix = np.zeros((num_drugs, fp_dim))

    logger.info(f"Building Drug Matrix for {num_drugs} drugs (dim={fp_dim})...")

    for global_id, local_idx in sorted_drugs:
        name = id_to_name.get(global_id)

        if name and name in drug_features_df.index:
            try:
                smiles = str(drug_features_df.loc[name].iloc[0])
                drug_matrix[local_idx] = get_Fingerprint(smiles)
            except:
                # 解析失败，保持为0
                pass
        else:
            # 缺失，保持为0
            pass

    # 6. 转为 Tensor 并移动到设备
    expression_tensor = torch.tensor(cell_matrix, dtype=torch.float32).to(device)
    fingerprint_tensor = torch.tensor(drug_matrix, dtype=torch.float32).to(device)

    logger.info(f"Done. Cell Tensor: {expression_tensor.shape}, Drug Tensor: {fingerprint_tensor.shape}")

    return expression_tensor, fingerprint_tensor
