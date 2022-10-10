from turtle import forward
import scipy
import scipy.stats
import numpy as np
import logging 
import torch 
import random 
import torch.nn as nn 
import math
from sklearn import metrics

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point
    
def ROC(label, y_prob):
    """
    Receiver_Operating_Characteristic, ROC
    :param label: (n, )
    :param y_prob: (n, )
    :return: fpr, tpr, roc_auc, optimal_th, optimal_point
    """
    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return fpr, tpr, roc_auc, optimal_th, optimal_point

def compute_youden_index(y_true, y_pred):
    fpr, tpr, roc_auc, optimal_th, optimal_point = ROC(y_true, y_pred)
    return optimal_th

def update_ema_variables(model, ema_model, global_step, alpha=0.99):
    # Use the true average until the exponential average is more correct
    # alpha: 0.99
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        ema_param.data.mul_(alpha).add_(param.data, alpha=1.-alpha)

def print_args(args, printer):
    for arg in vars(args):
        printer(format(arg, '<20')+'\t'+format(str(getattr(args, arg)), '<')) 

def build_logging(filename):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=filename,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging

def setup_seed(seed):
    #torch.backends.cudnn.enabled = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def mean_interval(mean=None, std=None, sig=None, n=None, confidence=0.95):
    alpha = 1 - confidence
    z_score = scipy.stats.norm.isf(alpha / 2)  # z分布临界值
    t_score = scipy.stats.t.isf(alpha / 2, df = (n-1) )  # t分布临界值

    if n >= 30 and sig != None:
        me = z_score*sig / np.sqrt(n)  # 误差
        lower_limit = mean - me
        upper_limit = mean + me
        
    if n >= 30 and sig == None:
        me = z_score*std / np.sqrt(n)
        lower_limit = mean - me
        upper_limit = mean + me
        
    if n < 30 and sig == None:
        me = t_score*std / np.sqrt(n)
        lower_limit = mean - me
        upper_limit = mean + me
    
    return lower_limit, upper_limit

class DeepSurvLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def _compute_loss(self, P, T, E, M, mode):
        P_exp = torch.exp(P) # (B,)
        P_exp_B = torch.stack([P_exp for _ in range(P.shape[0])], dim=0) # (B, B)
        if mode == 'risk':
            E = E.float() * (M.sum(dim=1) > 0).float()
        elif mode == 'surv':
            E = (M.sum(dim=1) > 0).float()
        else:
            raise NotImplementedError
        P_exp_sum = (P_exp_B * M.float()).sum(dim=1)
        loss = -torch.sum(torch.log(P_exp / (P_exp_sum+1e-6)) * E) / torch.sum(E)
        return loss

    def forward(self, P_risk, P_surv, T, E):
        # P: (B,)
        # T: (B,)
        # E: (B,) \in {0, 1}
        M_risk = T.unsqueeze(dim=1) < T.unsqueeze(dim=0) # (B, B)
        M_surv = T.unsqueeze(dim=1) > T.unsqueeze(dim=0) # (B, B)
        M_surv = M_surv.float() * torch.stack([E for _ in range(P_surv.shape[0])], dim=0).float()
        loss_risk = self._compute_loss(P_risk, T, E, M_risk, mode='risk')
        loss_surv = self._compute_loss(P_surv, T, E, M_surv, mode='surv')
        return loss_risk, loss_surv

def position_embedding(d_model, mode):
    if mode == 'time':
        #key_lst = [f'{i}F' for i in range(20)] + ['NONE',]
        key_lst = list(map(str, range(731)))
    elif mode == 'lesion':
        key_lst = ['a_target_Bone1', 'l_target_LN1', 'v_target_Spleen', 'a_target_R_Aden', 
                        'a_target_Bone2', 'v_target_L_Aden', 'v_target_Soft', 'v_target_Liver1', 'a_target_Spleen', 
                        'v_target_Liver2', 'a_target_Soft', 'v_target_LN2', 'v_target_LN1', 'v_target_Other', 
                        'v_target_Peritoneum', 'a_target_Peritoneum2', 'l_target_LN2', 'a_target_Liver2', 
                        'v_target_R_Aden', 'a_target_Peritoneum', 'v_target_Bone2', 'a_source', 'a_target_LN1', 
                        'v_source', 'v_target_Bone1', 'a_target_LN2', 'v_target_Peritoneum2', 'a_target_Liver1', 
                        'a_target_L_Aden', 'a_target_Other']
    elif mode == 'clinical':
        key_lst = ['LDH', 'NSE', 'CEA', 'CA125', 'CA199', 'CA724', 'AFP']
    elif mode == 'clinical_time':
        key_lst = list(map(str, range(731)))
    else:
        raise NotImplementedError

    max_len = len(key_lst)
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(dim=1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * 
                        -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return {key:i for i, key in enumerate(key_lst)}, pe

def query_position_embedding(key_to_id, query_key_lst, pe):
    id_lst = [key_to_id[key] for key in query_key_lst]
    return pe[id_lst]

if __name__ == '__main__':
    # criterion = DeepSurvLoss()
    # P = torch.Tensor([0.9, 0.5, 0.98, 0.99])
    # T = torch.Tensor([1, 2, 3, 4])
    # E = torch.Tensor([0, 1, 1, 1])
    # criterion(P, T, E)
    key_to_id, position_embedding = position_embedding(16, 'time')

    query_key_lst = ['0F', '1F']
    id_lst = [key_to_id[key] for key in query_key_lst]
    print(position_embedding[id_lst], id_lst)