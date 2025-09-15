import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from torchmetrics import Precision, Recall


def compute_max_recall(pred_tensor, gt_tensor):
    """
    compute the maximum recall at precision 1
    """
    recall = Recall(task="binary").to(pred_tensor.device)
    precision = Precision(task="binary").to(pred_tensor.device)
    req_prec = 1.0
    mr = 0
    for thresh in np.linspace(0, 1, 100):
        binary_preds = (pred_tensor > thresh).float()
        rec = recall(binary_preds, gt_tensor)
        prec = precision(binary_preds, gt_tensor)
        if rec >= mr and prec == req_prec:
            mr = rec
        else:
            break
    return mr


def eval(scores, labels):
    '''
    compute average precision and maximum recall at precision 1
    Based on: https://github.com/jarvisyjw/GV-Bench/blob/a3df94638d706bbdb9d98e9e24f0e3638aa6fab2/main.py
    Args:
        scores: np.array
        labels: np.array
        
    Return:
        precision: np.array
        recall: np.array
    
    '''
    # mAP
    average_precision = average_precision_score(labels, scores)
    precision, recall, TH = precision_recall_curve(labels, scores)
    # max recall @ 100% precision
    idx = np.where(precision == 1.0)
    recall_max = np.max(recall[idx])
    return average_precision, recall_max