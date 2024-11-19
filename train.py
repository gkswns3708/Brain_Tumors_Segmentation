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
    run_loss = AverageMeter()


