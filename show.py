"""
==============================================================
script to show the image and label, or image with labeled mask
==============================================================

Author: Muhammad Faizan
Date: 13 May 2023
Copywrite (c) Muhammad Faizan
==============================================================
"""
import matplotlib.pyplot as plt
import logging
import argparse
import numpy as np
import hydra
from omegaconf import DictConfig
import sys

import torch
from brats import get_datasets
from utils.visualizer import visualize_abnormal_area, get_labelled_image, visualize_data_gif, get_prediction_label_image
from utils.general import visualize_data_sample

# from networks.models.nnformer.nnformer 

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s: %(name)s: %(message)s")
file_handler = logging.FileHandler(filename= "logger/show.log")
stream_handler = logging.StreamHandler()
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


@hydra.main(config_name='configs', config_path='conf', version_base=None)
def show_result(cfg: DictConfig):
    """
    Visualize labelled brain scan on a patient case, three options are available
    1 - create brain scan slices and label them
    2 - create a .gif format file to visualize part of brain (labelled)
    3 - visualize a scan with it's label in a subplot format
    """
    # Hydra에서 type 값을 가져오기
    # TODO: Show 를 위한 Config 설정
    # ./conf/configs.yaml의 'type' key를 확인해 값이 있으면 가져오고 없으면 "get-gif"로 사용
    visualization_type = cfg.get("type", "label_get-gif") 

    # Load data
    dataset = get_datasets(cfg.dataset.dataset_folder, "test")
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              shuffle=False, num_workers=8,
                                              pin_memory=True)

    # Batch of data
    batch = next(iter(data_loader))
    image, label = batch["image"], batch['label'][:, 0]
    logger.info('visualizing an image with label')

    # Visualize
    if visualization_type == "show-abnormal-image":
        visualize_abnormal_area(image, label)
    elif visualization_type == "label_get-gif":
        print(image.shape, label.shape, "- get_labelled_image 입력전 image, label shape")
        labelled_img = get_labelled_image(image, label)
        visualize_data_gif(labelled_img)
    elif visualization_type == "prediction_label-get-gif":
        # TODO: prediction image와 label을 한번에 비교할 수 있도록 하는 gif 생성.
        # print(image.shape, label.shape, "- get_labelled_image 입력전 image, label shape")
        # TODO: Model 생성 부분 config로 수정할 수 있도록 함.
        model = nnFormer(crop_size=np.array([128, 128, 128]), 
                         embedding_dim=96, 
                         input_channels=in_channels, 
                         num_classes=num_classes, 
                         depths=[2, 2, 2, 2], 
                         num_heads=[3, 6, 12, 24], 
                         deep_supervision=False,
                         conv_op=nn.Conv3d,
                         patch_size= [4,4,4], 
                         window_size=[4,4,8,4]).to(device)
        labelled_img = get_prediction_label_image(image, label, prediction_image)
        visualize_data_gif(labelled_img)
    elif visualization_type == "show-case":
        visualize_data_sample(cfg.paths.test_patient)
    else:
        logger.info('No option selected')
        sys.exit()


if __name__ == "__main__":
    show_result()
    print('Done!!!')

