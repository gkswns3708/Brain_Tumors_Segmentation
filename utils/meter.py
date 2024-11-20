import torch
import numpy as np
import sys
import os
from pathlib import Path

# 상위 경로의 폴더를 import 하기 위해 
FILE = Path(__file__).resolve()  # script.py의 절대 경로
ROOT = FILE.parents[0].parents[0]  # 상위 2단계 디렉토리(Brain_Tumors_Segmentation)로 설정
if ROOT not in sys.path:  # ROOT 디렉토리가 sys.path에 없다면 추가
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 현재 작업 디렉토리 기준 상대 경로로 ROOT 설정

from metrics.metrics import dice_metric, jaccard_metric

class Meter:
    "Accumulate iou and dice scores"
    def __init__(self, threshold: float = 0.5):
        self.threshold: float = threshold
        self.dice_scores: list = [] # dice_scores
        self.iou_scores: list = [] # jaccard_scores
        
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        # TODO: logits, targets의 shape 확인
        probs = torch.sigmoid(logits)
        dice = dice_metric(probs, targets, self.threshold)
        iou = jaccard_metric(probs, targets, self. threshold)
        
        self.dice_scores.append(dice)
        self.iou_scores.append(iou)
    
    def get_metrics(self) -> np.ndarray:
        dice = np.mean(self.dice_scores)
        iou = np.mean(self.iou_scores)
        return dice, iou

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # TODO: what is val? 
        # [expected answer] value?
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)