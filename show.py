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
import torch.nn as nn
from brats import get_datasets
# from utils.visualizer import visualize_abnormal_area
from utils.visualizer import get_labelled_image, visualize_data_gif, get_show_image
# from utils.general import visualize_data_sample

from networks.models.nnformer.nnFormer_tumor import nnFormer

from monai.inferers import sliding_window_inference

from utils.all_utils import pad_to_original_shape

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference(model, input, batch_size, overlap):
    """inference on input with trained model"""
    def _compute(input):
        return sliding_window_inference(inputs=input, roi_size=(128, 128, 128), sw_batch_size=batch_size, predictor=model, overlap=overlap)
    return _compute(input)

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
    visualization_type = cfg.show.type
    patient_id = cfg.show.patient_id
    print(visualization_type, "- visualization_type")

    # Load data
    dataset = get_datasets(cfg.show.dataset_folder, "inference")
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              shuffle=False, num_workers=8,
                                              pin_memory=True)

    # Batch of data
    if 0 <= patient_id < len(data_loader):
        batch = data_loader.dataset[patient_id]
        image, label, id, nonzero_indexes = batch["image"].unsqueeze(0), batch['label'].unsqueeze(0), batch['patient_id'], batch['nonzero_indexes']

    # batch = next(iter(data_loader))
    # image, label = batch["image"], batch['label']
    print("현재 처리중인 patent_id: ", id)
    print(image.shape, label.shape, "- get_labelled_image 입력전 image, label shape")
    logger.info('visualizing an image with label')

    # Visualize
    # if visualization_type == "show-abnormal-image":
    #     visualize_abnormal_area(image, label)
    if visualization_type == "label_get-gif":
        print(image.shape, label.shape, "- get_labelled_image 입력전 image, label shape")
        labelled_img = get_labelled_image(image, label)
        visualize_data_gif(labelled_img)
    elif visualization_type == "prediction_label-get-gif":
        # TODO: prediction image와 label을 한번에 비교할 수 있도록 하는 gif 생성.
        # print(image.shape, label.shape, "- get_labelled_image 입력전 image, label shape")
        # TODO: Model 생성 부분 config로 수정할 수 있도록 함.
        # nnFormer = nnFormer_tumor.py
        model = nnFormer(crop_size=np.array([128, 128, 128]), 
                        embedding_dim=96, 
                        input_channels=4, 
                        num_classes=3, 
                        depths=[2, 2, 2, 2], 
                        num_heads=[3, 6, 12, 24], 
                        deep_supervision=False,
                        conv_op=nn.Conv3d,
                        patch_size=[4,4,4], 
                        window_size=[4,4,8,4]).to(device)
        checkpoint = torch.load(cfg.show.model_path)
        model.load_state_dict(checkpoint['model'])
        model.eval()  # 모델을 평가 모드로 전환
        model.to(device)
        sw_bs = cfg.test.sw_batch
        infer_overlap = cfg.test.infer_overlap
        # Inference with no_grad
        with torch.no_grad():
            image = image.to(device)
            # prediction_image = model(image).int()  # 예측 수행
            prediction_image = torch.sigmoid(inference(model, image, batch_size=sw_bs, overlap=infer_overlap))
            prediction_image = (prediction_image > 0.5)  # Thresholding to binary
            image = image.to('cpu')
            label = label.to('cpu')
            prediction_image = prediction_image.to('cpu')

        print(prediction_image.shape, "- prediction_image shape")
        # 시각화를 위한 데이터 준비
        image = pad_to_original_shape(image, nonzero_indexes, (155, 240, 240))
        label = pad_to_original_shape(label, nonzero_indexes, (155, 240, 240))
        prediction_image = pad_to_original_shape(prediction_image, nonzero_indexes, (155, 240, 240))

        labelled_imgs, predicted_imgs, overlay_predicted_imgs = get_show_image(image, label, prediction_image, cfg, is_categorical=True)
        visualize_data_gif(labelled_imgs, predicted_imgs, overlay_predicted_imgs, cfg)

    else:
        logger.info('No option selected')
        sys.exit()


if __name__ == "__main__":
    show_result()
    print('Done!!!')

