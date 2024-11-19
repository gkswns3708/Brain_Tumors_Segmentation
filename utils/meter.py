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

class Meter