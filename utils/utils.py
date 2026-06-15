import torch
import os
import numpy as np
import json
import logging
import time
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def setup_logger(log_file=r"train.log"):
    # 创建日志器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 可设为 DEBUG/INFO/WARNING/ERROR

    # 清除旧处理器（防止重复打印）
    if logger.hasHandlers():
        logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # 添加两个处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False




def save(save_dir, args, train_log, test_log, fold):
    # args.device = 0 # 可能会导致 json 序列化问题，建议注释
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_dir + "/args.json", 'w', encoding='utf-8') as f:
        json.dump(args.__dict__, f, indent=4)
    with open(save_dir + f'/test_results_fold_{fold}.json', 'w') as f:
        json.dump(test_log, f, indent=4)
    with open(save_dir + f'/train_log_fold_{fold}.json', 'w') as f:
        json.dump(train_log, f, indent=4)


def save_results(save_dir, args, results_list):
    acc = []
    prec = []
    rec = []
    f1 = []
    bacc = []
    auc_roc = []
    mcc = []
    kap = []
    ap = []
    aupr = []

    for result in results_list:
        acc.append(result['acc'])
        prec.append(result['prec'])
        rec.append(result['rec'])
        f1.append(result['f1'])
        bacc.append(result['bacc'])
        auc_roc.append(result['auc_roc'])
        mcc.append(result['mcc'])
        kap.append(result['kap'])
        ap.append(result['ap'])
        aupr.append(result['aupr'])

    results = {
        'acc': [np.mean(acc), np.std(acc)],
        'prec': [np.mean(prec), np.std(prec)],
        'rec': [np.mean(rec), np.std(rec)],
        'f1': [np.mean(f1), np.std(f1)],
        'bacc': [np.mean(bacc), np.std(bacc)],
        'auc_roc': [np.mean(auc_roc), np.std(auc_roc)],
        'mcc': [np.mean(mcc), np.std(mcc)],
        'kap': [np.mean(kap), np.std(kap)],
        'ap': [np.mean(ap), np.std(ap)],
        'aupr': [np.mean(aupr), np.std(aupr)],
    }

    final_results = vars(args).copy()
    final_results.update(results)

    with open(os.path.join(save_dir, 'all_results.json'), 'a+', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)



class EarlyStopping():
    def __init__(self, mode='higher', patience=50, filename=None, metric=None, n_fold=None, folder=None):
        """
        Initialize EarlyStopping object.

        Args:
            mode (str): 'higher' if a higher score is better, 'lower' if a lower score is better.
            patience (int): Number of epochs to wait for improvement before early stopping.
            filename (str): Name of the checkpoint file to save the model state.
            metric (str): Metric to monitor for early stopping. Can be 'accuracy', 'precision', 'recall', 'f1',
                          'bacc', 'auc_roc', 'mcc', 'kap', 'ap'.
            n_fold (int): Fold number used for naming checkpoint file.
            folder (str): Folder path to save checkpoint file.
        """

        if filename is None:
            filename = os.path.join(folder, '{}_fold_early_stop.pth'.format(n_fold))

        if metric is not None:
            supported_metrics = ['accuracy', 'precision', 'recall', 'f1', 'bacc', 'auc_roc', 'mcc', 'kap', 'ap', 'loss']
            assert metric in supported_metrics, \
                f"Expect metric to be one of {supported_metrics}, got {metric}"
            if metric in ['accuracy', 'precision', 'recall', 'f1', 'bacc', 'auc_roc', 'mcc', 'kap', 'ap']:
                logging.info(f'For metric {metric}, the higher the better')
                mode = 'higher'
            if metric in ['loss']:
                logging.info(f'For metric {metric}, the lower the better')
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False
        self.metric = metric

    def _check_higher(self, score, prev_best_score):
        """
        Check if the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """
        Check if the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
            logging.info(f'New best {self.metric}: {self.best_score}. Saved model to {self.filename}')
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            logging.info(f'New best {self.metric}: {self.best_score}. Saved model to {self.filename}')
        else:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                logging.info(f'Early stopping triggered. Best {self.metric}: {self.best_score}')
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.filename))




def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
               for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


if __name__ == '__main__':
    X = torch.ones(2, 3, device=try_gpu())
    print(X)
    pass
