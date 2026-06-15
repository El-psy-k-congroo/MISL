
import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm



def read_kg(path: str):

    kg_df = pd.read_csv(path)
    edge_index = kg_df.iloc[:, [0, 1]].values
    rel_index = kg_df.iloc[:, 2].values

    num_node = np.max((np.array(edge_index))) + 1
    num_rel = max(rel_index) + 1

    return num_node, torch.tensor(edge_index, dtype=torch.long), torch.tensor(rel_index, dtype=torch.long), num_rel


def get_drug_1_protein_edge(entity_id: dict, drug_protein_filename: str, logger):
    logger.info("init drug protein edge")

    df = pd.read_csv(drug_protein_filename, sep=',')
    triples = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        drug, protein = row.iloc[0], row.iloc[1]
        drug_id, protein_id = entity_id[drug], entity_id[str(protein)]

        triples.append([drug_id, protein_id, 1])

    logger.info(f'number of drug 1 protein edges is:{len(triples)}')
    return torch.tensor(triples)


def get_cell_2_protein_and_cell_3_tissue_edge(entity_id, cell_protein_tissue, logger):
    logger.info("init cell protein tissue edge")

    df = pd.read_csv(cell_protein_tissue)

    cell_protein_triples = []
    cell_tissue_triples = []

    cell_protein_set = set()
    cell_tissue_set = set()
    for _, row in tqdm(df.iterrows(), total=len(df)):
        cell, protein, tissue = row.iloc[3], row.iloc[2], row.iloc[4]
        cell_id, protein_id, tissue_id = entity_id[cell], entity_id[str(protein)], entity_id[tissue]

        cell_protein_triple = (cell_id, protein_id, 2)
        cell_tissue_triple = (cell_id, tissue_id, 3)

        if cell_protein_triple not in cell_protein_set:
            cell_protein_triples.append(list(cell_protein_triple))
            cell_protein_set.add(cell_protein_triple)
        if cell_tissue_triple not in cell_tissue_set:
            cell_tissue_triples.append(list(cell_tissue_triple))
            cell_tissue_set.add(cell_tissue_triple)

    logger.info(f'number of cell 2 protein edges is:{len(cell_protein_triples)}')
    logger.info(f'number of cell 3 tissue edges is:{len(cell_tissue_triples)}')
    return torch.tensor(cell_protein_triples), torch.tensor(cell_tissue_triples)


def get_protein_4_protein_edge(
        entity_id: dict,
        protein_protein_filename: str,
        logger
):
    logger.info("init protein protein edge")

    df = pd.read_excel(protein_protein_filename)
    triples = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        protein1, protein2 = row.iloc[0], row.iloc[1]
        protein_id1, protein_id2 = entity_id[str(protein1)], entity_id[str(protein2)]
        triples.append([protein_id1, protein_id2, 4])

    logger.info(f'number of protein 4 protein edges is:{len(triples)}')
    return torch.tensor(triples)


def construct_kg(dir_filename: str, data_type: str, entity_id: dict, logger):
    save_filename = os.path.join(dir_filename, data_type, 'kg.csv')
    logger.info(f'try to get the kg data:{save_filename}')
    if os.path.exists(save_filename):
        logger.info('kg data already exists')
        return read_kg(save_filename)


    '''
        1: drug_protein association
        2: cell_protein association
        3: cell_tissue association
        4: protein_protein association
    '''

    drug_protein_filename = os.path.join(dir_filename, data_type, 'drug_protein.csv')
    drug_1_protein = get_drug_1_protein_edge(entity_id, drug_protein_filename, logger)

    cell_protein_tissue = os.path.join(dir_filename, data_type, 'cell_protein.csv')
    cell_2_protein, cell_3_tissue = get_cell_2_protein_and_cell_3_tissue_edge(entity_id, cell_protein_tissue, logger)

    protein_protein_filename = os.path.join(dir_filename, data_type, 'protein-protein_network.xlsx')
    protein_4_protein = get_protein_4_protein_edge(entity_id, protein_protein_filename, logger)

    graph_edges = torch.concat([drug_1_protein, cell_2_protein, cell_3_tissue, protein_4_protein], dim=0)

    # 添加反向边
    reversed_edges = graph_edges[:, [1, 0, 2]]
    all_edges = torch.cat([graph_edges, reversed_edges], dim=0)

    # 去重，仅去除完全相同的边 (a, b, r)
    edge_set = set(tuple(row.tolist()) for row in all_edges)
    kg_edges = torch.tensor(list(edge_set), dtype=graph_edges.dtype)

    try:
        # 将张量移到 CPU 上，并转换为 numpy 数组
        edges_np = kg_edges.cpu().numpy()
        # 将每一行元素转换为字符串并用空格连接，然后用换行符连接所有行
        lines = '\n'.join([','.join(map(str, row)) for row in edges_np])
        with open(save_filename, 'w') as f:
            # 一次性将所有内容写入文件
            f.write(lines)
        logger.info(f"成功将张量写入 {save_filename}")
    except Exception as e:
        logger.info(f"写入文件时出现错误: {e}")
    return read_kg(save_filename)