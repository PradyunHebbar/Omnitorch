import os
import json
import numpy as np

def log_training_progress(log_file, epoch, train_loss, val_loss, full_train_acc=None, full_val_acc=None, train_auc=None, val_auc=None, train_acc=None, val_acc=None):
    """
    Log training progress to a file.
    
    :param log_file: Path to the log file
    :param epoch: Current epoch number
    :param train_loss: Training loss for the current epoch
    :param val_loss: Validation loss for the current epoch
    :param train_acc: Training accuracy for the current epoch (optional)
    :param val_acc: Validation accuracy for the current epoch (optional)
    :param train_auc: Training AUC score for the current epoch (optional)
    :param val_auc: Validation AUC score for the current epoch (optional)
    """
    log_entry = {
        'epoch': epoch,
        'train_loss': float(train_loss),
        'val_loss': float(val_loss)
    }
    
    if full_train_acc is not None:
        log_entry['full_train_acc'] = float(train_acc)
    if full_val_acc is not None:
        log_entry['full_val_acc'] = float(val_acc)
    if train_acc is not None:
        log_entry['train_acc'] = float(train_acc)
    if val_acc is not None:
        log_entry['val_acc'] = float(val_acc)
    if train_auc is not None:
        log_entry['train_auc'] = float(train_auc)
    if val_auc is not None:
        log_entry['val_auc'] = float(val_auc)
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def log_roc_data(roc_file, fpr, tpr, thresholds, epoch):
    """
    Log ROC curve data to a file.
    
    :param roc_file: Path to the ROC data file
    :param fpr: False Positive Rate array
    :param tpr: True Positive Rate array
    :param thresholds: Thresholds array
    :param epoch: Current epoch number
    """
    roc_data = {
        'epoch': epoch,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist()
    }
    
    with open(roc_file, 'a') as f:
        f.write(json.dumps(roc_data) + '\n')