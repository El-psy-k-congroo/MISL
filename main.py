import os
import sys
import warnings

import networkx as nx
from PIL.ImageChops import offset
from jinja2.compiler import generate

# warnings.filterwarnings("ignore")  # 忽略所有警告
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['TORCH_HOME'] = 'C:\\Users\\30662\\PyCharm\\MyProject\\SynergyPredictionProject2\\pretrain_model' # TODO
import argparse
import torch

import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from torch.utils.data import DataLoader
from dataset.my_in_memory_dataset import collate1, collate2, SynergyDataset, GraphDataset, KGDataset
from utils.data_preprocess import get_dict_from_json_file, load_datasets, get_dict_from_df, split_fold
from utils.kg_utils import construct_kg
from utils.feature_preprocess import build_aligned_features,  load_drug_data_features
from utils.utils import setup_logger, set_random_seed, try_gpu, EarlyStopping, save, save_results
from utils.trainer import train, valid
from model.Synergy import MISL
import torch.nn.functional as F




def init_args(user_args=None):
    parser = argparse.ArgumentParser(description='Synergy prediction project')

    parser.add_argument('--model_name', type=str, default='MISL')

    parser.add_argument('--seed', type=int, default=2025,
                        help='seed')
    parser.add_argument('--dataset', type=str, default="OncologyScreen", help="drugcombdb or OncologyScreen")

    parser.add_argument('--folds', type=int, default=5)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=768)
    parser.add_argument('--num_epochs', type=int, default=600,
                        help='maximum number of epochs (default: 600)')
    parser.add_argument('--patience', type=int, default=70,
                        help='patience for early stopping (default: 70)')

    parser.add_argument("--depth", type=int, default=2,
                        help="depth of slots [2, 3]")
    parser.add_argument("--num_slots", type=int, default=4,
                        help="number of slots [2, 4, 8, 16]")

    parser.add_argument("--input_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=600)
    parser.add_argument("--output_dim", type=int, default=300)
    parser.add_argument("--project_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)

    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--alpha", type=float, default=5)


    parser.add_argument('--saved-model', type=str,
                        help='the path of trained_model', default='./saved_model/0_fold_SynergyX.pth')

    parser.add_argument('--log', type=str, help='记录主要发生时的改变',
                        default=''
                               )

    args = parser.parse_args()

    return args



def load_data(args, device, logger):
    dir_name = './data'
    data_type = args.dataset

    logger.info('laod drug smiles')
    drug_seq_df = pd.read_csv(os.path.join(dir_name, data_type, 'drug_smiles.csv'))
    drug_smiles = get_dict_from_df(drug_seq_df, 0, 1)
    #
    entity_id = get_dict_from_json_file(os.path.join(dir_name, data_type, 'entity_id.json'))


    logger.info('load drug data features')
    smile_graph, max_mol_rel, max_mol_deg = load_drug_data_features(os.path.join(dir_name, data_type), drug_smiles, entity_id, device)


    logger.info('load drugA drugB cell_line label data')
    datasets, used_drug_dict, used_cell_dict = load_datasets(os.path.join(dir_name, data_type, 'drug_combinations.csv'), entity_id, logger)
    datasets_without_labels = datasets[:, :3]
    labels = datasets[:, 3]

    cell_feature, drug_feature = build_aligned_features(
        entity_mapping=entity_id,
        used_cell_dict=used_cell_dict,
        used_drug_dict=used_drug_dict,
        base_path=os.path.join(dir_name, data_type),
        logger=logger,
        device=device
    )


    logger.info('construct kg')
    num_kg_node, kg_edge_index, kg_rel_index, num_kg_rel = construct_kg(dir_name, data_type, entity_id, logger)


    parameter_dict = {
        'max_mol_rel': max_mol_rel,
        'cell_feature':cell_feature,
        'drug_feature':drug_feature,
        'used_drug_dict': used_drug_dict,
        'used_cell_dict': used_cell_dict,
        'num_kg_nodes': num_kg_node,
        'kg_edge_index': kg_edge_index,
        'kg_rel_index': kg_rel_index,
        'num_kg_rel': num_kg_rel,
    }

    return datasets, datasets_without_labels, labels, smile_graph, used_drug_dict, parameter_dict



def load_dataloader(datasets, train_indices, valid_indices, test_indices, used_drug_dict, data_root, smile_graph,
                    batch_size, parameter_dict):
    if not isinstance(datasets, (torch.Tensor, np.ndarray)):
        raise ValueError("datasets 必须是 torch.Tensor 或 numpy.ndarray 类型，以支持整数数组索引。")


    train_data = SynergyDataset(data_items=datasets[train_indices])
    valid_data = SynergyDataset(data_items=datasets[valid_indices])
    test_data = SynergyDataset(data_items=datasets[test_indices])

    kg_data = KGDataset(root=data_root, parameter_dict=parameter_dict)
    drug_ids = list(used_drug_dict.keys())
    mol_data = GraphDataset(data_items=drug_ids, smile_graph=smile_graph)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate1, num_workers=4,
                                  pin_memory=True, persistent_workers=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=collate1, num_workers=4,
                                  pin_memory=True, persistent_workers=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate1, num_workers=4,
                                 pin_memory=True, persistent_workers=True)

    drug_dataloader = DataLoader(mol_data, batch_size=len(used_drug_dict), shuffle=False, collate_fn=collate2,
                                 num_workers=0, pin_memory=True)
    kg_dataloader = DataLoader(kg_data, batch_size=1, shuffle=False, collate_fn=collate2, num_workers=0,
                               pin_memory=True)

    return train_dataloader, valid_dataloader, test_dataloader, drug_dataloader, kg_dataloader



def init_model(args, parameter_dict, learning_rate, epochs, trainLoader, device):
    # 假设 Synergy 模型已导入
    model = MISL(
        max_mol_rel=parameter_dict['max_mol_rel'],
        input_dim=args.input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim, num_relations=6, proj_dim=args.project_dim,
        depth=args.depth, num_slots=args.num_slots,
        used_drug_dict=parameter_dict['used_drug_dict'],
        used_cell_dict=parameter_dict['used_cell_dict'],
    )

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, epochs=epochs,
                                                    steps_per_epoch=len(trainLoader))

    return model, optimizer, scheduler


# =============================================================================
# 4. 主程序 Main
# =============================================================================

def main():
    timestamp = time.strftime('%Y%m%d_%H%M', time.localtime())

    args = init_args()
    set_random_seed(args.seed)
    device = try_gpu()

    dir_name = './data'
    data_type = args.dataset
    data_root = os.path.join(dir_name, data_type)
    batch_size = args.batch_size
    patience = args.patience
    num_epochs = args.num_epochs

    log_folder = os.path.join('logs/', data_type + f'{timestamp}' + '.log')
    logger = setup_logger(log_folder)
    logger.info(f'Title ===> {args.log}')


    experiment_folder = os.path.join('experiment/', data_type,  f'{timestamp}')
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)

    # 加载数据
    datasets, datasets_without_labels, labels, smile_graph, used_drug_dict, parameter_dict = load_data(args, device, logger)

    cell_feature, drug_feature = parameter_dict['cell_feature'], parameter_dict['drug_feature']

    folds = args.folds
    results_of_each_fold = []


    for i, (train_indices, val_indices, test_indices) in tqdm(enumerate(
            zip(*split_fold(folds, datasets_without_labels, labels))), desc='CV===============>>'):
        logger.info(f'=================={i} / {folds}===============================')

        # 打印一下数据量，确保划分逻辑正确
        logger.info(f"Train Size: {len(train_indices)}, Valid Size: {len(val_indices)}, Test Size: {len(test_indices)}")



        train_dataloader, valid_dataloader, test_dataloader, drug_dataloader, kg_dataloader = \
            load_dataloader(datasets, train_indices, val_indices, test_indices, used_drug_dict, data_root,
                            smile_graph, batch_size, parameter_dict)

        model, optimizer, scheduler = init_model(args, parameter_dict, args.lr, args.num_epochs, train_dataloader,
                                                 device)
        model.to(device)


        start_time = time.time()
        stopper = EarlyStopping(mode='higher', metric='accuracy', patience=patience, n_fold=i, folder=experiment_folder)

        train_log = {'train_acc': [], 'train_prec': [], 'train_rec': [], 'train_f1': [], 'train_bacc': [],
                     'train_auc_roc': [], 'train_mcc': [], 'train_kap': [], 'train_ap': [], 'train_aupr': [],
                     'train_loss': [],
                     'valid_acc': [], 'valid_prec': [], 'valid_rec': [], 'valid_f1': [], 'valid_bacc': [],
                     'valid_auc_roc': [], 'valid_mcc': [], 'valid_kap': [], 'valid_ap': [], 'valid_aupr': [],
                     'valid_loss': []}

        test_log = {'acc': [], 'prec': [], 'rec': [], 'f1': [], 'bacc': [],
                    'auc_roc': [], 'mcc': [], 'kap': [], 'ap': [], 'aupr': [], 'loss': []}

        mol_data = next(iter(drug_dataloader))
        kg_data = next(iter(kg_dataloader))

        for epoch in tqdm(range(num_epochs), desc="Epochs", ncols=80):

            train_acc, train_prec, train_rec, train_f1, train_bacc, train_auc_roc, train_mcc, train_kap, train_ap, train_aupr, \
                train_loss = train(train_dataloader, mol_data, kg_data, model, optimizer, device, cell_feature, drug_feature,
                                   scheduler, args.alpha)

            valid_acc, valid_prec, valid_rec, valid_f1, valid_bacc, valid_auc_roc, valid_mcc, valid_kap, valid_ap, valid_aupr, \
                valid_loss = valid(valid_dataloader, mol_data, kg_data, model, device, cell_feature, drug_feature)

            if epoch % 10 == 0:
                logger.info('Epoch %d, train_acc %f, valid_acc %f' % (epoch, train_acc, valid_acc))

            # 训练指标
            train_log['train_acc'].append(train_acc)
            train_log['train_prec'].append(train_prec)
            train_log['train_rec'].append(train_rec)
            train_log['train_f1'].append(train_f1)
            train_log['train_bacc'].append(train_bacc)
            train_log['train_auc_roc'].append(train_auc_roc)
            train_log['train_mcc'].append(train_mcc)
            train_log['train_kap'].append(train_kap)
            train_log['train_ap'].append(train_ap)
            train_log['train_aupr'].append(train_aupr)
            train_log['train_loss'].append(train_loss)

            # 验证指标
            train_log['valid_acc'].append(valid_acc)
            train_log['valid_prec'].append(valid_prec)
            train_log['valid_rec'].append(valid_rec)
            train_log['valid_f1'].append(valid_f1)
            train_log['valid_bacc'].append(valid_bacc)
            train_log['valid_auc_roc'].append(valid_auc_roc)
            train_log['valid_mcc'].append(valid_mcc)
            train_log['valid_kap'].append(valid_kap)
            train_log['valid_ap'].append(valid_ap)
            train_log['valid_aupr'].append(valid_aupr)
            train_log['valid_loss'].append(valid_loss)

            early_stop = stopper.step(valid_acc, model)
            if early_stop:
                logger.info('EarlyStopping! Finish training!')
                break

        logger.info(f'{i}_fold training is done! Training_time:{(time.time() - start_time) / 60}min')
        logger.info('Start testing ... ')

        stopper.load_checkpoint(model)
        model.to(device)

        kg_data = next(iter(kg_dataloader))
        test_acc, test_prec, test_rec, test_f1, test_bacc, test_auc_roc, test_mcc, test_kap, test_ap, test_aupr, \
            test_loss = valid(test_dataloader, mol_data, kg_data, model, device, cell_feature, drug_feature)

        test_log['acc'].append(test_acc)
        test_log['prec'].append(test_prec)
        test_log['rec'].append(test_rec)
        test_log['f1'].append(test_f1)
        test_log['bacc'].append(test_bacc)
        test_log['auc_roc'].append(test_auc_roc)
        test_log['mcc'].append(test_mcc)
        test_log['kap'].append(test_kap)
        test_log['ap'].append(test_ap)
        test_log['aupr'].append(test_aupr)
        test_log['loss'].append(test_loss)

        save(experiment_folder, args, train_log, test_log, i)
        logger.info(f"save to {experiment_folder}")
        results_of_each_fold.append(test_log)

    save_results(experiment_folder, args, results_of_each_fold)


if __name__ == '__main__':
    main()
