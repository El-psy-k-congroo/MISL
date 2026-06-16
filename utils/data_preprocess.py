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
