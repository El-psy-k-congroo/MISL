import json
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold

def k_fold(data, skf, y, internal_val_ratio=0.125, random_state=2025):
    all_train_indices = []
    all_val_indices = []
    all_test_indices = []

    if len(y):
        split_iter = skf.split(torch.zeros(len(data)), y)
    else:
        split_iter = skf.split(data)

    for train_idx, test_idx in split_iter:
        # 从训练集内部划分一部分做内部验证
        internal_train_idx, internal_val_idx = train_test_split(
            train_idx,
            test_size=internal_val_ratio,
            stratify=y[train_idx] if len(y) else None,
            random_state=random_state
        )

        all_train_indices.append(torch.tensor(internal_train_idx, dtype=torch.long))
        all_val_indices.append(torch.tensor(internal_val_idx, dtype=torch.long))
        all_test_indices.append(torch.tensor(test_idx, dtype=torch.long))

    return all_train_indices, all_val_indices, all_test_indices


def split_fold(folds, dataset, labels, scenario_type='random'):
    test_indices, train_indices, val_indices = [], [], []

    if scenario_type == 'random':
        skf = StratifiedKFold(folds, shuffle=True, random_state=2025)
        train_indices, val_indices, test_indices = k_fold(dataset, skf, labels)

    return train_indices, val_indices, test_indices





def get_split_generator(strategy, folds, datasets, seed=2025):
    """
    根据策略生成 (train_indices, val_indices, test_indices) 的迭代器
    datasets: numpy array, shape (N, 4), columns: [drug1_id, drug2_id, cell_id, label]
    """
    num_samples = len(datasets)
    indices = np.arange(num_samples)
    labels = datasets[:, 3]

    # 1. Random Split (原有的分层随机划分)
    if strategy == 'random':
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        # 这里只生成 train/test，需要在外面进一步划分 valid
        # 为了兼容你原来的 split_fold 逻辑，我们这里稍微调整一下
        # 原 split_fold 可能是直接返回三份索引，或者我们利用 KFold 生成两份，再手动切分验证集

        # 假设我们用两层切分：外层 5-Fold 切分 Test，内层切分 Valid
        for train_val_idx, test_idx in skf.split(np.zeros(num_samples), labels):
            # 内部再划分 10% 做验证集 (从 train_val 中)
            # 这里的 random_state 可以变化或者固定
            inner_splitter = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)
            train_idx, val_idx = next(inner_splitter.split(train_val_idx, labels[train_val_idx]))

            # 映射回原始索引
            train_indices = train_val_idx[train_idx]
            valid_indices = train_val_idx[val_idx]

            yield train_indices, valid_indices, test_idx

    # 2. Leave-Drug-Out (Cold Start Drug)
    elif strategy == 'cold_drug':
        # 获取所有唯一的药物 ID
        # drug1 和 drug2 都在第 0 和 1 列
        all_drugs = np.unique(np.concatenate([datasets[:, 0], datasets[:, 1]]))
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

        for train_val_drug_idx, test_drug_idx in kf.split(all_drugs):
            test_drugs = all_drugs[test_drug_idx]

            # 找出包含测试药物的所有样本索引
            # 只要 drug1 或 drug2 在 test_drugs 里，就算测试集样本
            # isin 返回布尔掩码
            mask_test = np.isin(datasets[:, 0], test_drugs) | np.isin(datasets[:, 1], test_drugs)
            test_indices = indices[mask_test]

            # 剩下的就是训练+验证
            train_val_indices = indices[~mask_test]

            # 再次切分验证集 (这里简单随机切分，或者按 Drug 切分也可以)
            # 为了简单，这里对训练集做随机切分验证集
            inner_splitter = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)
            # 注意：这里 labels 需要索引对齐
            t_idx, v_idx = next(inner_splitter.split(train_val_indices, labels[train_val_indices]))

            train_indices = train_val_indices[t_idx]
            valid_indices = train_val_indices[v_idx]

            yield train_indices, valid_indices, test_indices

    # 3. Leave-Cell-Out (Cold Start Cell)
    elif strategy == 'cold_cell':
        all_cells = np.unique(datasets[:, 2])
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

        for train_val_cell_idx, test_cell_idx in kf.split(all_cells):
            test_cells = all_cells[test_cell_idx]

            mask_test = np.isin(datasets[:, 2], test_cells)
            test_indices = indices[mask_test]
            train_val_indices = indices[~mask_test]

            # 内部切分验证集
            inner_splitter = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)
            t_idx, v_idx = next(inner_splitter.split(train_val_indices, labels[train_val_indices]))

            train_indices = train_val_indices[t_idx]
            valid_indices = train_val_indices[v_idx]

            yield train_indices, valid_indices, test_indices

    # 4. Leave-Combination-Out (Cold Start Combination)
    elif strategy == 'cold_comb':
        # 这里的组合指的是 (Drug1, Drug2) 对，不考虑 Cell
        # 为了方便，我们将 drug1_id 和 drug2_id 拼接成字符串或者元组来唯一标识
        # 假设 ID 是整数，可以用 "id1_id2" (确保 id1 < id2 以处理对称性)

        # 确保 drug1 < drug2 以处理无序对 (A,B) == (B,A)
        d1 = np.minimum(datasets[:, 0], datasets[:, 1])
        d2 = np.maximum(datasets[:, 0], datasets[:, 1])

        # 创建唯一标识，例如使用复数 d1 + d2*1j 或者字符串
        # 使用字符串比较稳妥
        comb_ids = np.array([f"{x}_{y}" for x, y in zip(d1, d2)])
        unique_combs = np.unique(comb_ids)

        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)

        for train_val_c_idx, test_c_idx in kf.split(unique_combs):
            test_combs = unique_combs[test_c_idx]

            mask_test = np.isin(comb_ids, test_combs)
            test_indices = indices[mask_test]
            train_val_indices = indices[~mask_test]

            # 内部切分验证集
            inner_splitter = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)
            t_idx, v_idx = next(inner_splitter.split(train_val_indices, labels[train_val_indices]))

            train_indices = train_val_indices[t_idx]
            valid_indices = train_val_indices[v_idx]

            yield train_indices, valid_indices, test_indices

    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

def get_data_from_pickle(
        filename: str,
):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def get_dict_from_json_file(
        filename: str
) -> dict:
    with open(filename, 'r', encoding='utf-8') as file:
        # 读取文件内容并解析 JSON
        return json.load(file)

def get_dict_from_df(
    df: pd.DataFrame,
    key_index: int,
    val_index: int,
) -> dict:
    dict = {}
    for index, row in df.iterrows():
        dict[row[key_index]] = row[val_index]
    return dict


def entity_to_id(dir_name: str, data_type: str) -> None:
    id_dict_filename = os.path.join(dir_name, data_type, 'id_dict.json')
    id_dict = get_dict_from_json_file(id_dict_filename)

    entity_id_dict = {key: idx for idx, key in enumerate(id_dict)}

    save_filename = os.path.join(dir_name, data_type, 'entity_id.json')
    with open(save_filename, 'w') as file:
        json.dump(entity_id_dict, file, indent=4)

    return


def id_to_entity(dir_name: str, data_type: str) -> None:
    entity_id_filename = os.path.join(dir_name, data_type, 'entity_id.json')
    entity_id = get_dict_from_json_file(entity_id_filename)

    id_entity_dict = {value: key for key, value in entity_id.items()}

    save_filename = os.path.join(dir_name, data_type, 'id_entity.json')
    with open(save_filename, 'w') as file:
        json.dump(id_entity_dict, file, indent=4)

    return



def load_datasets(datapath: str, entity_id: dict, logger):
    datasets = []
    used_drug = set()
    used_cell = set()
    df = pd.read_csv(datapath)

    # sorted_data = np.sort(np.array(df.iloc[:, 5]))
    # threshold1 = sorted_data[17359]
    # threshold2 = sorted_data[52077]
    column_data = df.iloc[:, 5]
    threshold1 = column_data.quantile(0.25)
    threshold2 = column_data.quantile(0.75)

    logger.info(f'threshold1 is:{threshold1}')
    logger.info(f'threshold2 is:{threshold2}')

    for index, row in tqdm(df.iterrows(), total=len(df)):
        drug1, drug2, cell, label = row.iloc[3], row.iloc[4], row.iloc[2], row.iloc[5]

        # if cell in ['MOLT4', 'COLO205', 'SNB19', 'TC32']:
        #     continue

        drug_id1, drug_id2, cell_id = entity_id[drug1], entity_id[drug2], entity_id[cell]
        if float(label) > threshold2:
            datasets.append([drug_id1, drug_id2, cell_id, 1])
            used_drug.add(drug_id1)
            used_drug.add(drug_id2)
            used_cell.add(cell_id)
        elif float(label) < threshold1:
            datasets.append([drug_id1, drug_id2, cell_id, 0])
            used_drug.add(drug_id1)
            used_drug.add(drug_id2)
            used_cell.add(cell_id)

    used_drug_dict = {v: i for i, v in enumerate(sorted(used_drug))}
    used_cell_dict = {v: i for i, v in enumerate(sorted(used_cell))}


    # id_entity = {v: k for k, v in entity_id.items()}
    #
    # cell_line = [id_entity[id] for id in used_cell]
    #
    # print(cell_line)


    datasets = np.array(datasets)

    # 正负样本
    pos_num = np.sum(datasets[:, 3] == 1)
    neg_num = np.sum(datasets[:, 3] == 0)

    # 药物和细胞系数量
    drug_num = len(used_drug)
    cell_num = len(used_cell)

    logger.info(f"Drug number: {drug_num}")
    logger.info(f"Cell line number: {cell_num}")

    logger.info(f"Positive samples: {pos_num}")
    logger.info(f"Negative samples: {neg_num}")

    logger.info(f"Pos/Neg ratio: {pos_num / neg_num:.3f}")
    logger.info(f"Positive ratio: {pos_num / (pos_num + neg_num):.3f}")

    logger.info('The length of datasets is:{}'.format(len(datasets)))

    return np.array(datasets), used_drug_dict, used_cell_dict








if __name__ == '__main__':
    # entity_to_id('./data', 'drugcombdb')
    # id_to_entity('./data', 'drugcombdb')
    # print(construct_kg('./data', 'drugcombdb'))
    pass
