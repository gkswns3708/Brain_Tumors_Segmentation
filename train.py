import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import gc
import nibabel as nib
import tqdm as tqdm


from utils.augment import DataAugmenter
from metrics.metrics import AverageMeter

from monai.data import decollate_batch

import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.makedirs('logger', exist_ok=True)
file_handler = logging.FileHandler(filename='logeer/train_logger.log')
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="%(asctime)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

class Solver:
    """list of optimizers for training NN"""
    def __init__(self, model: nn.Module, lr: float = 1e-4, weight_decay: float = 1e-5):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def select_solver(self, name):
        optimizers = {
            "Adam": torch.optim.Adam(self.model.parameters(), lr=self.lr, 
                                     weight_decay=self.weight_decay, 
                                     amsgrad=True), 
            "AdamW": torch.optim.AdamW(self.model.parameters(), lr=self.lr, 
                                       weight_decay=self.weight_decay, 
                                       amsgrad=True),
            "SGD": torch.optim.SGD(self.model.parameters(), lr=self.lr, 
                                   weight_decay=self.weight_decay),
        }
        return optimizers[name]

def train_epoch(model, loader, optimizer, loss_func):
    """
    Training NN for Brain Tumor Segmentation Task with 3D MRI Voxel Images.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    augmenter = DataAugmenter().to(device)
    torch.cuda.empty_cache()
    gc.gollect()
    model.train() 
    run_loss = AverageMeter() # TODO: 왜 AverageMeter를 사용하는가?
    # TODO: 혹은 tqdm을 사용해야 할 수 도 있음
    for idx, batch_data in enumerate(loader):
        image, label = batch_data["image"].to(device), batch_data["label"].to(device)
        image, label = augmenter(image, label)
        logits = model(image)
        loss = loss_func(logits, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        run_loss.update(loss.item(), n = batch_data["image"].shape[0])
    torch.cuda.empty_cache()
    return run_loss.avg

def val(model, loader, acc_func, model_inferer = None,
        post_sigmoid = None, post_pred = None, post_label = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    run_acc = AverageMeter()
    with torch.no_grad():
        # TODO: 혹은 tqdm을 사용해야 할 수 도 있음
        for idx, batch_data in enumerate(loader):
            logits = model_inferer(batch_data["image"].to(device))
            masks = decollate_batch(batch_data["label"].to(device))
            prediction_lists = decollate_batch(logits)
            predictions = [post_pred(post_sigmoid(prediction)) for prediction in prediction_lists]
            # masks = [post_label(mask) for mask in masks]
            # TODO: What is acc_func's parent class?
            acc_func.reset()
            acc_func(y_pred=predictions, y=masks)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())
    return run_acc.avg
            
            
            
    
        


