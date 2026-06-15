import torch
from tqdm import tqdm
import torch.nn.functional as F
from model.losses import compute_kl_loss
from utils.evaluation import accuracy, precision, recall, f1_score, bacc_score, roc_auc, mcc_score, kappa, \
    ap_score, aupr_score

def metrics(y_prob, y_test):
    y_pred = [1 if prob >= 0.5 else 0 for prob in y_prob]

    acc = accuracy(y_pred, y_test)
    prec = precision(y_pred, y_test)
    rec = recall(y_pred, y_test)
    f1 = f1_score(y_pred, y_test)
    bacc = bacc_score(y_pred, y_test)
    auc_roc = roc_auc(y_prob, y_test)
    mcc = mcc_score(y_pred, y_test)
    kap = kappa(y_pred, y_test)
    ap = ap_score(y_prob, y_test)
    aupr = aupr_score(y_pred, y_test)

    return acc, prec, rec, f1, bacc, auc_roc, mcc, kap, ap, aupr



def train(dataloader, mol_data, kg_data, model, optimizer, device, cell_feature, drug_feature, scheduler, alpha):

    total_loss = 0

    pre_label, true_label = [], []


    model.train()

    kg_data = kg_data.to(device)
    mol_data = mol_data.to(device)
    cell_feature = cell_feature.to(device)
    drug_feature = drug_feature.to(device)

    for drugA_ids, drugB_ids, cell_ids, labels in tqdm(dataloader, desc='train dataloader...'):
        drugA_ids, drugB_ids, cell_ids, labels = (drugA_ids.to(device), drugB_ids.to(device), cell_ids.to(device),
                                                  labels.to(device))

        optimizer.zero_grad(set_to_none=True)

        output0 = model(drugA_ids, drugB_ids, cell_ids, labels, mol_data, kg_data, cell_feature, drug_feature)
        output1 = model(drugA_ids, drugB_ids, cell_ids, labels, mol_data, kg_data, cell_feature, drug_feature)

        ce_loss = output0[0]
        kl_loss = compute_kl_loss(output0[1], output1[1])
        loss = ce_loss + alpha * kl_loss
        # loss = ce_loss

        y = labels
        total_loss += loss.item()

        pred = F.softmax(output0[1].cpu().detach(), dim=1)[:, 1]

        pre_label.append(pred)
        true_label.append(y)

        loss.backward()
        optimizer.step()
        scheduler.step()

    train_loss = total_loss / len(dataloader)
    pre_label = torch.cat(pre_label).cpu().detach().numpy()
    true_label = torch.cat(true_label).cpu().detach().numpy()

    # 假设 metrics 函数已导入
    acc, prec, rec, f1, bacc, auc_roc, mcc, kap, ap, aupr = metrics(pre_label, true_label)

    return acc, prec, rec, f1, bacc, auc_roc, mcc, kap, ap, aupr, train_loss


def valid(dataloader, mol_data, kg_data, model, device, cell_feature, drug_feature):
    model.eval()
    total_loss = 0
    kg_data = kg_data.to(device)
    mol_data = mol_data.to(device)
    cell_feature = cell_feature.to(device)
    drug_feature = drug_feature.to(device)
    pre_label, true_label = [], []

    with torch.no_grad():
        for drugA_ids, drugB_ids, cell_ids, labels in tqdm(dataloader, desc='valid or test dataloader...'):
            drugA_ids, drugB_ids, cell_ids, labels = (drugA_ids.to(device), drugB_ids.to(device), cell_ids.to(device),
                                                      labels.to(device))

            pred = model.infer(drugA_ids, drugB_ids, cell_ids, labels, mol_data, kg_data, cell_feature, drug_feature)
            y = labels
            pred = F.softmax(pred.cpu().detach(), dim=1)[:, 1]

            # total_loss += loss.item() # 如果 infer 不返回 loss，这里无法计算 loss

            pre_label.append(pred)
            true_label.append(y)

    valid_loss = total_loss / len(dataloader)
    pre_label = torch.cat(pre_label).cpu().detach().numpy()
    true_label = torch.cat(true_label).cpu().detach().numpy()

    acc, prec, rec, f1, bacc, auc_roc, mcc, kap, ap, aupr = metrics(pre_label, true_label)

    return acc, prec, rec, f1, bacc, auc_roc, mcc, kap, ap, aupr, valid_loss