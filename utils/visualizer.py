import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.util import montage
from pathlib import Path
import os
import cv2
import sys
from IPython.display import Image
import imageio


def channel_last(image, label):
    """
    convert to channel last as expected by some functions 
    
    Parameters
    ----------
    image: torch.Tensor (an MRI scan consists of 4 channels)
    label: torch.Tensor (1개의 label에 대한 scan) 

    Returns
    -------
    data: tuple(np.ndarray, np.ndarray) a processed sample
    """
    # input으로 단일 채널의 label만 들어와야 함. 즉, (1, 1, W, H, D) 형태여야 함.

    #remove batch dim
    if len(image.shape) == 5:
        image = image.squeeze(0)
        label = label.squeeze(0)

    #conver to last channel from (c, d, h, w) --> (h, w, d, c)
    #label (d, h, w) --> (h, w, d)
    image = image.permute(2, 3, 1, 0)
    label = label.permute(1, 2, 0)
    image = image.numpy()
    label = label.numpy()
    return (image, label)


def get_labelled_image(image, label, is_categorical = False):
    """
    get an MRI scan labelled with annotated label by a radiologist
    And highlight three different kind of tumors by three different 
    color schemes.
    
    Parameters
    ----------
    image: torch.Tensor (an MRI scan of a patient)
    label: torch.Tensor (label of the MRI scan)
    categorical: bool   (whether to convert to categorical or not)

    Returns
    -------
    labeled_image: np.ndarray (a labelled image for visualization)
    """
    #convert to channel last 
    image, label = channel_last(image, label)

    # convert to one hot encoding
    if not is_categorical:
        label = to_categorical(label.astype(np.uint8), num_classes= 4)
    
    # normalize the image
    image = cv2.normalize(image[:, :, :, 0], None, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
    
    labeled_image = np.zeros_like(label[:, :, :, 1:])

    #remove tumor part from image
    labeled_image[:, :, :, 0] = image * (label[:, :, :, 0])
    labeled_image[:, :, :, 1] = image * (label[:, :, :, 0])
    labeled_image[:, :, :, 2] = image * (label[:, :, :, 0])

     # color labels
    labeled_image += label[:, :, :, 1:] * 255
    return labeled_image


# TODO: prediction image와 label image를 함께 보여주어 차이를 보는 Voxel을 생성하는 함수.
def get_prediction_label_image(image, label, prediction):
    # image     : [B, 4, W, H, D] -> 4개의 modality
    # label     : [B, 3, W, H, D] -> Background를 제외한 3개의 image
    # prediction: [B, 3, W, H, D] -> label과 동일한 형태
    raise NotImplementedError


def visualize_data_gif(data_):
    images = []
    for i in range(data_.shape[0]):
        x = data_[min(i, data_.shape[0] - 1), :, :]
        y = data_[:, min(i, data_.shape[1] - 1), :]
        z = data_[:, :, min(i, data_.shape[2] - 1)]
        img = np.concatenate((x, y, z), axis=1)
        images.append(img)
    os.makedirs(f"../runs", exist_ok=True)
    try:
        imageio.mimsave(f"../runs/gif.gif", images, duration=0.5, format='GIF', loop = 0)
        return Image(filename=f"../runs/gif.gif", format='png')
    except FileNotFoundError:
        root = Path(__file__).resolve().parents[1]
        imageio.mimsave(f"{root}/runs/gif.gif", images, duration=0.5, format='GIF', loop = 0)
        return Image(filename=f"{root}/runs/gif.gif", format='png')