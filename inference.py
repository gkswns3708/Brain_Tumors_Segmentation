"""
==========================================
A script to evaluate the model performance
test set evaluation on BraTS23 dataset.
==========================================

Author: Muhammad Faizan
Date: 16.09.2024
==========================================
"""
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from monai.data import decollate_batch
from monai.handlers.utils import from_engine
from monai.metrics import DiceMetric
from utils.general import load_pretrained_model
from utils.all_utils import save_seg_csv
from utils.all_utils import cal_confuse, cal_dice
from utils.all_utils import pad_to_original_shape
from brats import get_datasets
from utils.meter import AverageMeter

import nibabel as nib
from monai.metrics import DiceMetric
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.utils.enums import MetricReduction
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    AsDiscrete,
    Activations,
)

from monai.networks.nets import SwinUNETR, SegResNet, VNet, BasicUNetPlusPlus, AttentionUnet, DynUNet, UNETR
# from networks.models.ResUNetpp.model import ResUnetPlusPlus
# from networks.models.UNet.model import UNet3D
# from networks.models.UX_Net.network_backbone import UXNET
from networks.models.nnformer.nnFormer_tumor import nnFormer
# try:
#     from thesis.models.SegUXNet.model import SegUXNet
#     from thesis.models.v2.model import SegSCNet
# except ModuleNotFoundError:
#     print('model not available, please train with other models')
#     sys.exit(1)

from functools import partial

import hydra
from omegaconf import OmegaConf, DictConfig
import logging
import os
from tqdm import tqdm

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.makedirs("logger", exist_ok= True)
file_handler = logging.FileHandler(filename= "logger/logger_test.log")
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt= "%(asctime)s: %(message)s", datefmt= '%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Stream and file logging
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def get_value(value):
    """proprecess value to scaler"""
    if torch.is_tensor(value):
        return value.item()
    return value
  
def reconstruct_label(image):
    """reconstruct image label"""
    if type(image) == torch.Tensor:
        image = image.cpu().numpy()
    c1, c2, c3 = image[0], image[1], image[2]
    image = (c3 > 0).astype(np.uint8)
    image[(c2 == False)*(c3 == True)] = 2
    image[(c1 == True)*(c3 == True)] = 4
    return image

def inference(model, input, batch_size, overlap):
    """inference on input with trained model"""
    def _compute(input):
        return sliding_window_inference(inputs=input, roi_size=(128, 128, 128), sw_batch_size=batch_size, predictor=model, overlap=overlap)
    return _compute(input)


def test(args, data_loader, model, mode="test"):
    """test the model on the test dataset"""
    metrics_dict = []
    haussdor = HausdorffDistanceMetric(include_background=True, percentile=95)
    meandice = DiceMetric(include_background=True)
    # TODO: sw_batch: sliding_window batch
    sw_bs = args.test.sw_batch
    infer_overlap = args.test.infer_overlap

    save_dir = "./stored_prediction/nn_former/"
    os.makedirs(save_dir, exist_ok=True)
    result = []
    
    for data in tqdm(data_loader):
        patient_id = data["patient_id"][0]
        inputs = data["image"]
        # targets = data["label"].cuda()
        pad_list = data["pad_list"]
        inputs = inputs.cuda()
        model.cuda()
        # print(args.test.tta, "- args.test.tta")
        with torch.no_grad():  
            if args.test.tta:
                predict = torch.sigmoid(inference(model, inputs, batch_size=sw_bs, overlap=infer_overlap))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(2,)).flip(dims=(2,)), batch_size=sw_bs, overlap=infer_overlap))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(3,)).flip(dims=(3,)), batch_size=sw_bs, overlap=infer_overlap))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(4,)).flip(dims=(4,)), batch_size=sw_bs, overlap=infer_overlap))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(2, 3)).flip(dims=(2, 3)), batch_size=sw_bs, overlap=infer_overlap))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(2, 4)).flip(dims=(2, 4)), batch_size=sw_bs, overlap=infer_overlap))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(3, 4)).flip(dims=(3, 4)), batch_size=sw_bs, overlap=infer_overlap))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(2, 3, 4)).flip(dims=(2, 3, 4)), batch_size=sw_bs, overlap=infer_overlap))
                predict = predict / 8.0 
            else:
                predict = torch.sigmoid(inference(model, inputs, batch_size=sw_bs, overlap=infer_overlap))
                
        # targets = targets[:, :, pad_list[-4]:targets.shape[2]-pad_list[-3], pad_list[-6]:targets.shape[3]-pad_list[-5], pad_list[-8]:targets.shape[4]-pad_list[-7]]
        if mode == "inference":
            # TODO: 원래 (240, 240, 155)인지 (155, 240, 240인지 확인.)
            # print(predict.shape, "- predict.shape")
            # print(data['label'].shape, "- label.shape")
            nonzero_indexes = data['nonzero_indexes']
            # print(nonzero_indexes, "- nonzero_indexes")
            predict = pad_to_original_shape(predict, nonzero_indexes, (155, 240, 240)) 
            label = pad_to_original_shape(data['label'], nonzero_indexes, (155, 240, 240))
        else:
            predict = predict[:, :, pad_list[-4]:predict.shape[2]-pad_list[-3], pad_list[-6]:predict.shape[3]-pad_list[-5], pad_list[-8]:predict.shape[4]-pad_list[-7]]
        predict = (predict>0.5).squeeze().to(torch.int8)
        
        # print(targets.shape, "- targets.shape")
        print(predict.shape, "- predict.shape")

        # Save the prediction to .nii.gz
        prediction_save_path = os.path.join(save_dir, f"prediction_{patient_id}.nii.gz")
        label_save_path = os.path.join(save_dir, f"label_{patient_id}.nii.gz")
        affine = np.eye(4)  # Identity matrix as the default affine
        if predict.is_cuda:
            np_predict = predict.cpu().numpy().astype(np.int32)
            np_label = label.cpu().numpy().astype(np.int32)
        else:
            np_predict = predict.numpy().astype(np.int32)
            np_label = label.numpy().astype(np.int32)

        prediction_nifti_img = nib.Nifti1Image(np_predict, affine)
        label_nifiti_img = nib.Nifti1Image(np_label, affine)

        nib.save(prediction_nifti_img, prediction_save_path)
        nib.save(label_nifiti_img, label_save_path)

        # print(label.shape, predict.shape, type(label), type(predict), "- label, predict shape")

        print(f"Saved prediction for {patient_id} at {prediction_save_path}")
        label = label.squeeze()
        dice_metrics = cal_dice(predict, label, haussdor, meandice)
        confuse_metric = cal_confuse(predict, label, patient_id)
        et_dice, tc_dice, wt_dice = dice_metrics[0], dice_metrics[1], dice_metrics[2]
        et_hd, tc_hd, wt_hd = dice_metrics[3], dice_metrics[4], dice_metrics[5]
        et_sens, tc_sens, wt_sens = get_value(confuse_metric[0][0]), get_value(confuse_metric[1][0]), get_value(confuse_metric[2][0])
        et_spec, tc_spec, wt_spec = get_value(confuse_metric[0][1]), get_value(confuse_metric[1][1]), get_value(confuse_metric[2][1])
        metrics_dict.append(dict(id=patient_id,
            et_dice=et_dice, tc_dice=tc_dice, wt_dice=wt_dice, 
            et_hd=et_hd, tc_hd=tc_hd, wt_hd=wt_hd,
            et_sens=et_sens, tc_sens=tc_sens, wt_sens=wt_sens,
            et_spec=et_spec, tc_spec=tc_spec, wt_spec=wt_spec))

    result_df = pd.DataFrame(metrics_dict)
    result_df.to_csv("result_custom.csv", index=False)
    save_seg_csv(metrics_dict, args)


@hydra.main(config_name='configs', config_path= 'conf', version_base=None)
def main(cfg: DictConfig):
    # Select model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # TODO: 아마 inference에는 삭제해도 될 거 같긴한데, 혹시 모르니 남겨두기.
    # Efficient training
    torch.backends.cudnn.benchmark = True

    # BraTS configs
    num_classes = 3
    in_channels = 4
    spatial_size = 3

    # Select Network architecture for inference
    # TODO: Add other architecture
    # nnFormer
    if cfg.model.architecture == "nn_former":
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


        
    print('Chosen Network Architecture: {}'.format(cfg.model.architecture))
 
    
    # Hyperparameters
    batch_size = cfg.test.batch
    workers = cfg.test.workers
    dataset_folder = cfg.dataset.train_val_folder
    # torch.serialization.add_safe_globals([complex])
    # Load checkpoints
    checkpoint = torch.load(cfg.test.weights)
    model.load_state_dict(checkpoint['model'])
    # model.load_state_dict(torch.load(cfg.test.weights, weights_only=True))
    model.eval()

    # Load dataset
    # TODO: 
    test_loader = get_datasets(dataset_folder=dataset_folder, mode="inference", target_size=(128, 128, 128))
    test_loader = torch.utils.data.DataLoader(test_loader, 
                                            batch_size=batch_size, 
                                            shuffle=False, num_workers=workers, 
                                            pin_memory=True) 
    
    # mode=cfg.inference.mode
    mode = "inference" # TODO: 여기 바꾸기, cfg로 조절할 수 있도록. args나

    # Evaluate
    print("start test")
    test(cfg, test_loader, model, mode=mode)

    print('done!!')

if __name__ == '__main__':
    main()
