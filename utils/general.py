import torch
import monai
import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
# from config.configs import*


def load_pretrained_model(model,
                        state_path):
    '''
    Load a pretraiend model, it is sometimes important to leverage the knowlege 
    from the pretrained model when the dataset is limited

    Parameters
    ----------
    model: nn.Module
    state_path: str
    '''
    model.load_state_dict(torch.load(state_path, map_location=torch.device('cpu'))["state_dict"])
    print("Pretrained model loaded")
    return model