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
from tqdm import tqdm

def to_categorical(mask, num_classes):
    """ 
    1-hot encoding of mask into number of classes mentioned.
    
    Parameters
    ----------
    mask: np.ndarray (label of the image)
    num_classes: int (number of tumor classes)

    Returns
    -------
    encoded: np.ndarray (one hot encoded numpy array)
    """
    # print(f'mask data type: {mask[:, :, 78].max(), mask[:, :, 78].min()}')
    # create one hot encoding of the label...
    # print(mask.shape, "- mask shape")
    return np.eye(num_classes, dtype= np.uint8)[mask]

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


def show_channel_last(image, label, prediction):
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
        image      = image.squeeze(0)
        label      = label.squeeze(0)
        prediction = prediction.squeeze(0)

    #conver to last channel from (c, d, h, w) --> (h, w, d, c)
    #label (d, h, w) --> (h, w, d)
    image      = image.permute(2, 3, 1, 0)
    label      = label.permute(2, 3, 1, 0)
    prediction = prediction.permute(2, 3, 1, 0)
    image      = image.numpy()
    label      = label.numpy()
    prediction = prediction.numpy()
    return image, label, prediction


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
    image, label = show_channel_last(image, label)

    # convert to one hot encoding
    # if not is_categorical:
    #     label = to_categorical(label.astype(np.uint8), num_classes= 4)
    
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

def get_show_image(image, labels, predictions, cfg, is_categorical = False):
    modality = cfg.show.modality
    index2modality = {0: "T1n", 1: "T1c", 2: "T2w", 3: "T2f"}
    modality2index = {value:key for key, value in index2modality.items()}
    
    image, labels, predictions = show_channel_last(image, labels, predictions)

    if not is_categorical:
        labels = to_categorical(labels.astype(np.uint8), num_classes= 4)

    image = cv2.normalize(image[:, :, :, modality2index[modality]], None, alpha=0, beta=255,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

    labeled_image_list = [image.copy()]
    predicted_images_list = [image.copy()]
    overlay_images_list = [image.copy()]
    class_colors = {
        0: [255, 0, 0],    # Class 1 (Red)
        1: [0, 255, 0],    # Class 2 (Green)
        2: [0, 0, 255]     # Class 3 (Blue)
    }
    alpha = 0.5
    for idx in tqdm(range(labels.shape[-1]), desc="Processing images"):
        mask = (labels[:, :, :, idx] == 1)  # 특정 클래스에 해당하는 마스크
        prediction = (predictions[:, :, :, idx] == 1)  
        labeled_image =  np.stack([image] * 3, axis=-1)  
        predicted_image = np.stack([image] * 3, axis=-1) 
        overlay_predicted_image = np.stack([image] * 3, axis=-1)  
        for channel in range(3):
            labeled_image[..., channel] = labeled_image[..., channel] * (1 - alpha * mask) \
                + alpha * mask * class_colors[idx][channel]
            # non-overlay
            predicted_image[..., channel] = predicted_image[..., channel] * (1 - alpha * prediction) \
                + alpha * prediction * class_colors[(idx + 1) % 3][channel]     
            # overlay
            overlay_predicted_image[..., channel] = overlay_predicted_image[..., channel] * (1 - alpha * prediction) \
                + (alpha / 2) * prediction * class_colors[(idx + 1) % 3][channel] + (alpha / 2) * mask * class_colors[idx][channel]
            
        labeled_image_list.append(labeled_image)
        predicted_images_list.append(predicted_image)
        overlay_images_list.append(overlay_predicted_image)

    return labeled_image_list, predicted_images_list, overlay_images_list

def process_images(image_list, processed_images):
        """
        이미지를 처리하고 결과를 저장합니다.
        
        Args:
            image_list (list): 처리할 이미지들의 리스트.
            processed_images (list): 처리된 이미지를 저장할 리스트.
        """
        for idx, data_ in enumerate(tqdm(image_list, desc="Concatenating images")):
            for i in range(data_.shape[0]):
                x = data_[min(i, data_.shape[0] - 1), :, :]
                y = data_[:, min(i, data_.shape[1] - 1), :]
                z = data_[:, :, min(i, data_.shape[2] - 1)]
                img = np.concatenate((x, y, z), axis=1)
                processed_images[idx].append(img)

# 각 이미지 리스트에 대해 처리 함수 호출

def save_gif(images, folder_path, name, cfg, mapping_table, label_type):
    """
    GIF 파일을 저장하는 함수.

    Args:
        images (list): 저장할 이미지 리스트.
        folder_path (str): 저장할 폴더 경로.
        name (str): 파일 이름에 포함될 이름.
        patient_id (str): 환자 ID.
        mapping_table (list): 매핑 테이블.
        label_type (str): 라벨 유형 ('label', 'prediction', 'overlay').
    """
    patient_id = cfg.show.patient_id
    modality = cfg.show.modality
    for idx, img in enumerate(tqdm(images, desc="Saving GIFs")):
        file_path = f"{folder_path}/{name}_{label_type}_{patient_id}_{modality}_{mapping_table[idx]}.gif"
        imageio.mimsave(file_path, img, duration=0.5, format='GIF', loop=0)

def visualize_data_gif(labeld_img_list, predicted_img_list, overlay_predicted_img_list, cfg, name="gif"):
    mapping_table = {0: "BG", 1: "ET", 2: "TC", 3: "WT"}
    labeled_images = [[] for _ in range(len(labeld_img_list))]
    predicted_images = [[] for _ in range(len(predicted_img_list))]
    overlay_predicted_images = [[] for _ in range(len(overlay_predicted_img_list))]

    process_images(labeld_img_list, labeled_images)
    process_images(predicted_img_list, predicted_images)
    process_images(overlay_predicted_img_list, overlay_predicted_images)
    os.makedirs(f"./runs/label", exist_ok=True)
    os.makedirs(f"./runs/overlay", exist_ok=True)
    os.makedirs(f"./runs/prediction", exist_ok=True)

    save_gif(labeled_images, "./runs/label", name, cfg, mapping_table, "label")
    save_gif(predicted_images, "./runs/prediction", name, cfg, mapping_table, "prediction")
    save_gif(overlay_predicted_images, "./runs/overlay", name, cfg, mapping_table, "overlay")

        
    return 
        

def visualize_data_list_gif(data_, name="gif"):
    images = []
    for label in range(3):
        temp_img = []
        for i in range(data_.shape[0]):
            x = data_[min(i, data_.shape[0] - 1), :, :]
            y = data_[:, min(i, data_.shape[1] - 1), :]
            z = data_[:, :, min(i, data_.shape[2] - 1)]
            temp_img.extend([x, y, z])
        
        img = np.concatenate((x, y, z), axis=1)
        images.append(img)
    os.makedirs(f"../runs", exist_ok=True)
    try:
        imageio.mimsave(f"../runs/{name}.gif", images, duration=0.5, format='GIF', loop = 0)
        return Image(filename=f"../runs/{name}.gif", format='png')
    except FileNotFoundError:
        root = Path(__file__).resolve().parents[1]
        imageio.mimsave(f"{root}/runs/{name}.gif", images, duration=0.5, format='GIF', loop = 0)
        return Image(filename=f"{root}/runs/{name}.gif", format='png')