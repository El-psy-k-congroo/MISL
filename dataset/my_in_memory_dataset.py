import os
import sys

BASEDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASEDIR)

import time
import torch
from torch_geometric.data import Data, Batch, InMemoryDataset
from torch.utils.data import Dataset
from typing import Sequence, Tuple, Optional, Dict, Final, Any



class SynergyDataset(Dataset):
    def  __init__(self, data_items, args=None):
        super(SynergyDataset, self).__init__()

        self.args = args
        self.data_items = data_items


    def __len__(self):

        return len(self.data_items)

    def __getitem__(self, idx):
        start_time = time.time()

        drugA, drugB, cell, label = self.data_items[idx]

        label = int(label)

        return drugA, drugB, cell, label

def collate1(
        data_list: Sequence[Tuple[int, int, int, int]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # 解包数据列表
    drugA_ids = torch.tensor([data[0] for data in data_list])
    drugB_ids = torch.tensor([data[1] for data in data_list])
    cell_ids = torch.tensor([data[2] for data in data_list])
    labels = torch.tensor([data[3] for data in data_list])

    return drugA_ids, drugB_ids, cell_ids, labels


class GraphDataset(InMemoryDataset):
    def  __init__(self, data_items, smile_graph, args=None, transform=None):
        super(GraphDataset, self).__init__()

        self.args = args
        self.data_items = data_items
        self.smile_graph = smile_graph
        self.transform = transform
        self.pe_dim = 8


    def __len__(self):

        return len(self.data_items)


    def read_graph_info(self, drug_id):
        c_size, features, edge_index, rel_index = self.smile_graph[str(drug_id)]



        data = Data(x=torch.FloatTensor(features),
                    edge_index=torch.LongTensor(edge_index).T.contiguous(),
                    )
        if rel_index is not None:
            data.edge_attr = torch.LongTensor(rel_index)  # 作为标准边属性
        data.c_size = torch.tensor([c_size], dtype=torch.long)
        return data

    def __getitem__(self, idx):
        start_time = time.time()

        drug_id = self.data_items[idx]

        data = self.read_graph_info(drug_id)

        if self.transform is not None:
            data = self.transform(data)

        return data


def collate2(
        data_list: Sequence[Data]
) -> Batch:
    # 解包数据列表
    batchA = Batch.from_data_list(data_list)

    return batchA


class KGDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        parameter_dict: Dict[str, Any],
        transform: Optional[callable] = None,
        pre_transform: Optional[callable] = None,
        rebuild: bool = False,
    ):
        self.parameter_dict = parameter_dict
        self._rebuild = rebuild
        super().__init__(root, transform=transform, pre_transform=pre_transform)
        # load processed data
        if os.path.exists(self.processed_paths[0]):
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # we don't rely on raw files on disk; building from parameter_dict
        return []

    @property
    def processed_file_names(self):
        return ['data_nbw.pt']

    def read_graph_info(self):

        x = torch.randn(self.parameter_dict['num_kg_nodes'], 300)


        kg_data = Data(
            x=x,
            edge_index=torch.LongTensor(self.parameter_dict['kg_edge_index']).transpose(1, 0),
            edge_attr=torch.LongTensor(self.parameter_dict['kg_rel_index']),
            c_size=torch.tensor([self.parameter_dict['num_kg_nodes']], dtype=torch.long),

        )

        return kg_data


    def process(self):

        data = self.read_graph_info()

        # save single-element dataset
        data_list = [data]
        data, slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])


    # override exists check to support rebuild flag
    def processed_exists(self):
        processed_path = self.processed_paths[0]
        if self._rebuild and os.path.exists(processed_path):
            os.remove(processed_path)
            return False
        return os.path.exists(processed_path)


