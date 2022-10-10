from tabnanny import verbose
from dataset import *
import torch.nn as nn 
import torch 
from sklearn.metrics import roc_curve, auc
from model import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch 
import argparse
import os 
from lifelines.utils import concordance_index
from utils import *
import matplotlib.pyplot as plt 
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter
import shutil 
from copy import deepcopy
import thop

# ema_model = None
# global_step = 0

def train(args, epoch, model, train_loader_lst, criterion, optimizer, printer, 
            time_key_to_id, time_pe, lesion_key_to_id, lesion_pe, 
            clinical_key_to_id, clinical_pe):
    # global ema_model
    # global global_step

    deepsurv_criterion = DeepSurvLoss().to(args.device)
    model.train()
    # ema_model.train()
    records = {'auc': [], 'loss': [], 'loss_ce': [], 
                # 'loss_risk': [], 
                #'loss_surv': []
                }
    y_true, y_pred = [], []
    #optimizer.zero_grad()
    train_loader = np.random.choice(train_loader_lst, 1)[0]
    for idx, data in enumerate(train_loader):
        X = data['X'].float().to(args.device)
        M = data['M'].float().to(args.device)
        S = data['S'].float().to(args.device)
        y = data['y'].long().to(args.device)
        OS = data['OS'].long().to(args.device)
        event = data['event'].long().to(args.device)
        query_time_key_lst = data['time_key_lst']
        cur_time_pe = torch.stack([query_position_embedding(time_key_to_id, query_time_key_lst, time_pe) 
                            for query_time_key_lst in data['time_key_lst']], dim=1).float().to(args.device) # (B, T, C)
        cur_lesion_pe = torch.stack([query_position_embedding(lesion_key_to_id, query_lesion_key_lst, lesion_pe) 
                            for query_lesion_key_lst in data['lesion_key_lst']], dim=1).float().to(args.device) # (B, N, C)

        if epoch == 1 and idx == 0:
            flops, params = thop.profile(model,inputs=(X[0:1], M[0:1], S[0:1], cur_time_pe[0:1], cur_lesion_pe[0:1]))
            printer(f'Flops={flops/1e9:.2f} G\tParams={params/1e6:.2f} M')

        p = model(X, M, S, cur_time_pe, cur_lesion_pe)
        loss_ce = criterion(p, y)
        loss_risk, _ = deepsurv_criterion(torch.softmax(p, dim=1)[:, 0],
                                                    torch.softmax(p, dim=1)[:, 1], 
                                                    OS, event)
        loss = loss_ce + loss_risk
        #loss = loss_risk
        #loss = loss_ce
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        p = torch.softmax(p, dim=1)
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(p[:, 1].detach().cpu().numpy())
        records['loss'].append(loss.item())
        records['loss_ce'].append(loss_ce.item())
        # records['loss_risk'].append(loss_risk.item())
        # records['loss_surv'].append(loss_surv.item())

        # update_ema_variables(model, ema_model, global_step)
        # ema_model(X, M, S, cur_time_pe, cur_lesion_pe)

    fpr, tpr, thresholds = roc_curve(np.concatenate(y_true, axis=0), 
                                    np.concatenate(y_pred, axis=0), pos_label=1)
    records['auc'] = auc(fpr, tpr)
    context = f'[TRAIN]\tEpoch: {epoch}\t'
    for key, value in records.items():
        context += f'{key}: {np.mean(value):.4f}\t'
    printer(context)
    return records

def evaluate(args, model, eval_loader_lst, name, printer, mode, 
            time_key_to_id, time_pe, lesion_key_to_id, lesion_pe, 
            clinical_key_to_id, clinical_pe, is_show=False):
    model.eval()
    if mode == 'cindex':
        event_times, predicted_scores, event_observed = [], [], []
    elif mode == 'auc':
        y_true, y_pred = [], []
    else:
        raise NotImplementedError
    show_data = []
    for _, data_lst in enumerate(zip(*eval_loader_lst)):
        with torch.no_grad():
            p = []
            for data in data_lst:
                hos = data['hos']
                pid = data['pid']
                X = data['X'].float().to(args.device)
                M = data['M'].float().to(args.device)
                S = data['S'].float().to(args.device)
                y = data['y'].long().to(args.device)
                OS = data['OS'].long().to(args.device)
                event = data['event'].long().to(args.device)
                cur_time_pe = torch.stack([query_position_embedding(time_key_to_id, query_time_key_lst, time_pe) 
                                    for query_time_key_lst in data['time_key_lst']], dim=1).float().to(args.device) # (B, T, C)
                cur_lesion_pe = torch.stack([query_position_embedding(lesion_key_to_id, query_lesion_key_lst, lesion_pe) 
                                    for query_lesion_key_lst in data['lesion_key_lst']], dim=1).float().to(args.device) # (B, N, C)
                p_ = model(X, M, S, cur_time_pe, cur_lesion_pe)
                p_ = torch.softmax(p_, dim=1)
                p.append(p_)
            p = torch.mean(torch.stack(p, dim=0), dim=0)
            if mode == 'cindex':
                event_times.append(OS.detach().cpu().numpy())
                predicted_scores.append(p[:, 1].detach().cpu().numpy())
                event_observed.append(event.detach().cpu().numpy())
                for h, pi, t, p, e in zip(hos, pid, event_times[-1], predicted_scores[-1], event_observed[-1]):
                    show_data.append([h, pi, p, t, e])
            elif mode == 'auc':
                y_true.append(y.detach().cpu().numpy())
                y_pred.append(p[:, 1].detach().cpu().numpy())
                for h, pi, t, p, o in zip(hos, pid, y_true[-1], y_pred[-1], OS.detach().cpu().numpy()):
                    show_data.append([h, pi, p, t, o])

    records = {}
    if mode == 'cindex':
        records['cindex'] = concordance_index(np.concatenate(event_times, axis=0), 
                                np.concatenate(predicted_scores, axis=0), 
                                event_observed=np.concatenate(event_observed, axis=0))
        if is_show:
            show_data = sorted(show_data, key=lambda x: (x[2], x[3]))
            show_table = PrettyTable()
            show_table.field_names = ['Hospital', 'PID', 'Surv score', 'OS', 'Censor']
            for h, pi, p, t, e in show_data:
                show_table.add_row([h, pi, p, t, e])
            printer(show_table)
    elif mode == 'auc':
        fpr, tpr, thresholds = roc_curve(np.concatenate(y_true, axis=0), 
                                    np.concatenate(y_pred, axis=0), pos_label=1)
        records['auc'] = auc(fpr, tpr)
        if is_show:
            show_data = sorted(show_data, key=lambda x: (x[2], x[3]))
            show_table = PrettyTable()
            show_table.field_names = ['Hospital', 'PID', 'Surv score', 'Response', 'OS']
            for h, pi, p, t, o in show_data:
                show_table.add_row([h, pi, p, t, o])
            printer(show_table)
    context = "\t".join([f"{key}: {value}" for key, value in records.items()])
    printer(f'[{name}]\t{context}')
    records['show_data'] = show_data
    return records

def build_dl(args, split, discard, shuffle, batch_size, prefix_lst, printer, strong_aug):
    dl_lst = []
    prefix_lst = ['0']
    for prefix in prefix_lst:
        if split in ['train', 'valid',]:
            data_dir = args.data_dir 
            split_file = eval(f'args.{split}_split_file')
            #num_time = eval(f'args.{split}_num_time')
            ds = SurvDataset(
                    data_dir=data_dir,
                    split_file=split_file,
                    split=split,
                    anno_file=args.anno_file,
                    printer=printer,
                    median=args.median,
                    num_time=args.eval_num_time,
                    num_lesion=args.train_num_lesion,
                    discard=discard,
                    target_size=args.target_size,
                    black_lst=['a_target_Liver1.jpg', 'a_target_Liver2.jpg'],
                    prefix=prefix,
                    strong_aug=strong_aug) 
        elif split in ['test']:
            ds = SurvDataset(
                    data_dir=args.test_data_dir,
                    split_file=args.test_split_file,
                    split=split,
                    anno_file=args.test_anno_file,
                    printer=printer,
                    median=args.median,
                    num_time=args.eval_num_time,
                    num_lesion=args.train_num_lesion,
                    discard=discard,
                    target_size=args.target_size,
                    black_lst=['a_target_Liver1.jpg', 'a_target_Liver2.jpg'],
                    prefix=prefix,
                    strong_aug=strong_aug)
        elif split in ['hpd1']:
            ds = SurvDataset(
                    data_dir=args.hpd1_data_dir,
                    split_file=args.hpd1_split_file,
                    split=split,
                    anno_file=args.hpd1_anno_file,
                    printer=printer,
                    median=args.median,
                    num_time=args.eval_num_time,
                    num_lesion=args.train_num_lesion,
                    discard=discard,
                    target_size=args.target_size,
                    black_lst=['a_target_Liver1.jpg', 'a_target_Liver2.jpg'],
                    prefix=prefix,
                    strong_aug=strong_aug)
        else:
            raise NotImplementedError
        dl = DataLoader(ds, shuffle=shuffle, pin_memory=False, num_workers=args.num_workers, batch_size=batch_size)
        dl_lst.append(dl)
    return dl_lst

def main(args, printer):
    # global ema_model
    # global global_step

    model = eval(args.model)(args, pretrained=args.pretrained)
    if os.path.isfile(args.resume):
        model_state_dict = model.state_dict()
        suc = 0
        checkpoint = torch.load(args.resume)
        for key, value in checkpoint.items():
            new_key = key
            if new_key in model_state_dict and model_state_dict[new_key].shape == value.shape:
                model_state_dict[new_key] = value 
                suc += 1
        model.load_state_dict(model_state_dict)
        printer(f'Loaded weight from {args.resume}: {suc}/{len(model_state_dict)}')
    #model = torch.nn.DataParallel(model, device_ids=args.device_ids)
    model = model.to(args.device)

    # ema_model = deepcopy(model).to(args.device)
    # global_step += 1

    # optimizer = torch.optim.AdamW([{'params': model.cnn.parameters(), 'lr': args.cnn_lr, 'weight_decay': args.weight_decay},
    #                                 {'params': model.ttrans.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
    #                                 {'params': model.ltrans.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
    #                                 {'params': model.fc.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},], 
    #                                  lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, last_epoch=-1, verbose=True)
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs, T_mult=1, eta_min=5e-4, last_epoch=-1, verbose=True)
    criterion = nn.CrossEntropyLoss().to(args.device)

    auc_train_dl_lst = build_dl(args, split='train', discard=True, shuffle=True, batch_size=args.batch_size, prefix_lst=['-1', '0', '1'], printer=printer, strong_aug=True)
    
    auc_trainT_dl_lst = build_dl(args, split='train', discard=True, shuffle=True, batch_size=1, prefix_lst=['-1', '0', '1'], printer=printer, strong_aug=False)
    auc_valid_dl_lst = build_dl(args, split='valid', discard=True, shuffle=False, batch_size=1, prefix_lst=['-1', '0', '1'], printer=printer, strong_aug=False)
    auc_test_dl_lst  = build_dl(args, split='test', discard=True, shuffle=False, batch_size=1, prefix_lst=['-1', '0', '1'], printer=printer, strong_aug=False)
    auc_hpd1_dl_lst  = build_dl(args, split='hpd1', discard=True, shuffle=False, batch_size=1, prefix_lst=['-1', '0', '1'], printer=printer, strong_aug=False)

    cindex_train_dl_lst = build_dl(args, split='train', discard=False, shuffle=False, batch_size=1, prefix_lst=['-1', '0', '1'], printer=printer, strong_aug=False)
    cindex_valid_dl_lst = build_dl(args, split='valid', discard=False, shuffle=False, batch_size=1, prefix_lst=['-1', '0', '1'], printer=printer, strong_aug=False)
    cindex_test_dl_lst  = build_dl(args, split='test', discard=False, shuffle=False, batch_size=1, prefix_lst=['-1', '0', '1'], printer=printer, strong_aug=False)
    cindex_hpd1_dl_lst  = build_dl(args, split='hpd1', discard=False, shuffle=False, batch_size=1, prefix_lst=['-1', '0', '1'], printer=printer, strong_aug=False)

    time_key_to_id, time_pe = position_embedding(args.d_model, mode='time')
    lesion_key_to_id, lesion_pe = position_embedding(args.d_model, mode='lesion')
    clinical_key_to_id, clinical_pe = position_embedding(args.d_model, mode='clinical')

    # writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
    save_records = None
    save_test_records = None
    records = {}
    plot_keys = ['AUC', 'C-index']
    for epoch in range(1, args.epochs+1):
        _ = train(args, epoch, model, auc_train_dl_lst, criterion, optimizer, printer, 
                                time_key_to_id, time_pe, lesion_key_to_id, lesion_pe, clinical_key_to_id, clinical_pe)
                                
        auc_train_records = evaluate(args, model, auc_trainT_dl_lst, 'TRAIN', printer, 'auc', 
                                time_key_to_id, time_pe, lesion_key_to_id, lesion_pe, clinical_key_to_id, clinical_pe)
        auc_valid_records = evaluate(args, model, auc_valid_dl_lst, 'VALID', printer, 'auc', 
                                time_key_to_id, time_pe, lesion_key_to_id, lesion_pe, clinical_key_to_id, clinical_pe)
        auc_test_records = evaluate(args, model, auc_test_dl_lst, 'TEST', printer, 'auc', 
                                time_key_to_id, time_pe, lesion_key_to_id, lesion_pe, clinical_key_to_id, clinical_pe)
        auc_hpd1_records = evaluate(args, model, auc_hpd1_dl_lst, 'HPD1', printer, 'auc', 
                                time_key_to_id, time_pe, lesion_key_to_id, lesion_pe, clinical_key_to_id, clinical_pe)

        cindex_train_records = evaluate(args, model, cindex_train_dl_lst, 'TRAIN', printer, 'cindex', 
                                time_key_to_id, time_pe, lesion_key_to_id, lesion_pe, clinical_key_to_id, clinical_pe)
        cindex_valid_records = evaluate(args, model, cindex_valid_dl_lst, 'VALID', printer, 'cindex', 
                                time_key_to_id, time_pe, lesion_key_to_id, lesion_pe, clinical_key_to_id, clinical_pe)
        cindex_test_records = evaluate(args, model, cindex_test_dl_lst, 'TEST', printer, 'cindex', 
                                time_key_to_id, time_pe, lesion_key_to_id, lesion_pe, clinical_key_to_id, clinical_pe)
        cindex_hpd1_records = evaluate(args, model, cindex_hpd1_dl_lst, 'HPD1', printer, 'cindex', 
                                time_key_to_id, time_pe, lesion_key_to_id, lesion_pe, clinical_key_to_id, clinical_pe)

        records['C-index/train'] = [*records.get('C-index/train', []), cindex_train_records['cindex']]
        records['C-index/valid'] = [*records.get('C-index/valid', []), cindex_valid_records['cindex']]
        records['C-index/test'] = [*records.get('C-index/test', []), cindex_test_records['cindex']]
        records['C-index/hpd1'] = [*records.get('C-index/hpd1', []), cindex_hpd1_records['cindex']]

        records['AUC/train'] = [*records.get('AUC/train', []), auc_train_records['auc']]
        records['AUC/valid'] = [*records.get('AUC/valid', []), auc_valid_records['auc']]
        records['AUC/test'] = [*records.get('AUC/test', []), auc_test_records['auc']]
        records['AUC/hpd1'] = [*records.get('AUC/hpd1', []), auc_hpd1_records['auc']]

        _, axarr = plt.subplots(1, 2, figsize=(14, 5))
        for i, key in enumerate(plot_keys):
            for rec_key in records:
                if key in rec_key:
                    axarr[i].plot(range(len(records[rec_key])), records[rec_key], label=rec_key)
            axarr[i].legend()
        plt.savefig(os.path.join(args.output_dir, 'plot.png'))
        plt.close()

        # writer.add_scalar('C-index/train', cindex_train_records['cindex'], epoch)
        # writer.add_scalar('C-index/valid', cindex_valid_records['cindex'], epoch)
        # writer.add_scalar('C-index/test', cindex_test_records['cindex'], epoch)
        # writer.add_scalar('C-index/hpd1', cindex_hpd1_records['cindex'], epoch)

        # writer.add_scalar('AUC/train', auc_train_records['auc'], epoch)
        # writer.add_scalar('AUC/valid', auc_valid_records['auc'], epoch)
        # writer.add_scalar('AUC/test', auc_test_records['auc'], epoch)
        # writer.add_scalar('AUC/hpd1', auc_hpd1_records['auc'], epoch)

        if save_records is None or save_records['cindex'] < cindex_valid_records['cindex']:
            save_records = auc_valid_records
            save_records.update(cindex_valid_records)
            save_test_records = auc_test_records
            save_test_records.update(cindex_test_records)
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best.pth'))
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'last.pth'))
        # if epoch % 1 == 0:
        #     torch.save(model.state_dict(), os.path.join(args.output_dir, f'{epoch:03d}.pth'))
        # printer(str(save_records))
        # printer(str(save_test_records))
        printer('')
        
        #lr_scheduler.step()

    return save_records

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='%s_%s_%s_%s_%s_%s_%s_%s')
    parser.add_argument('--postfix', type=str, default='')
    parser.add_argument('--model', type=str, default='MyModel')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_head', type=int, default=4)
    parser.add_argument('--init_seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='Results')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--milestones', type=int, nargs='+')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--cnn_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_num_time', type=int, default=3)
    parser.add_argument('--train_num_lesion', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default='SurvData/PKCancerCohortCropData')
    parser.add_argument('--test_data_dir', type=str, default='SurvData/OthersCohortCropData')
    parser.add_argument('--hpd1_data_dir', type=str, default='SurvData/HPD1CohortCropData')
    parser.add_argument('--train_split_file', type=str, default='SurvData/Annotations/train_hos.txt')
    parser.add_argument('--valid_split_file', type=str, default='SurvData/Annotations/valid_hos.txt')
    parser.add_argument('--test_split_file', type=str, default='SurvData/Annotations/test_hos.txt')
    parser.add_argument('--hpd1_split_file', type=str, default='SurvData/Annotations/hpd1_hos.txt')
    parser.add_argument('--anno_file', type=str, default='SurvData/Annotations/PKCancerCohort.csv')
    parser.add_argument('--test_anno_file', type=str, default='SurvData/Annotations/OthersCohort.csv')
    parser.add_argument('--hpd1_anno_file', type=str, default='SurvData/Annotations/HPD1Cohort.csv')
    parser.add_argument('--target_size', type=int, default=224)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--median', type=float, default=365)
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = get_args()
    args.exp = args.exp % (args.model, args.backbone, args.pretrained, args.d_model, args.num_head, args.dropout, args.init_seed, args.postfix)
    args.output_dir = os.path.join(args.output_dir, args.exp)
    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = build_logging(os.path.join(args.output_dir, 'log.log'))
    printer = logger.info
    print_args(args, printer)
    setup_seed(args.init_seed)
    main(args, printer)