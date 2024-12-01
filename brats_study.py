import torch
import os
from torch.utils.data.dataset import Dataset
from utils.all_utils import pad_or_crop_image, minmax, load_nii, pad_image_and_label, listdir, get_brats_folder

class BraTS(Dataset):
    def __init__(self, patients_dir, patient_ids, mode, target_size=(128, 128, 128)):
        super(BraTS, self).__init__()
        self.patients_dir = patients_dir # Dataset Path ~~~/brats/(train/val/text)/{patient_ids}
        self.patent_ids = patient_ids # {patient_ids}
        self.mode = mode
        self.target_size = target_size
        self.datas = []
        self.patterns = ["-t1n", "-t1c", "-t2w", "-t2f"]
        if mode == "train" or mode == "train_val" or mode == "test":
            self.patterns += ["-seg"]
        for patient_id in patient_ids:
            paths = [f"{patient_id}{pattern}.nii.gz" for pattern in self.patterns]
            # patient = dict(
            #     id=patient_id, t1=paths[0], t1ce=paths[1],
            #     t2=paths[2], flair=paths[3], seg=paths[4] if mode == "train" or mode == "train_val" or mode == "val" or mode == "test" or mode == "visualize" else None
            # )
            patient = {
                'id': patient_id,
                't1': paths[0],
                't1ce': paths[1],
                't2': paths[2],
                'flair': paths[3],
                'seg': paths[4] if mode == "train" or mode == "train_val" or mode == "val" or mode == "test" or mode == "visualize" else None
            }
            self.datas.append(patient)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (image, label) where label is the ground-truth of the image.
        
        input:
            inital_label shape: (3, 155, 240, 240)
            inital_image shape: (4, 155, 240, 240)
        processed_image:
            shape: (4, 128, 128, 128)
                because of the padding or cropping
                # TODO: 설명 추가.
        output:
            patient_label shape: (3, 128, 128, 128)

        """
        patient = self.datas[idx]
        patient_id = patient['id']
        crop_list = []
        pad_list = []
        patient_image_dict = {key:torch.tensor(load_nii(f"{self.patients_dir}/{patient_id}/{patient[key]}")) for key in patient if key not in ["id", "seg"]}
        patient_label = torch.tensor(load_nii(f"{self.patients_dir}/{patient_id}/{patient['seg']}").astype("int8"))
        patient_image = torch.stack([patient_image_dict[key] for key in patient_image_dict])

        if self.mode == "train" or self.mode == "train_val" or self.mode == "test":
            # original_Label
                # 0: Background
                # 1: NCR/NET(Necrotic and Non-Enhancing Tumor, 괴사 또는 비증강 종양)
                # 2: ED(Peritumoral Edema, 종양 주위 부종)
                # 3: ET(Enhancing Tumor, 증강되는 종양)
            # Task_label
                # 0: Background
                # 1: TC(Tumor Core) -> 1 + 3
                # 2: WT(Whole Tumor) -> 1 + 2 + 3
                # 3: ET(Enhancing Tumor) -> 3
            et = patient_label == 3
            tc = torch.logical_or(patient_label == 1, patient_label == 3)
            wt = torch.logical_or(tc, patient_label == 2)
            patient_label = torch.stack([et, tc, wt])
        # Removing black area from the edge of the MRI
        # 모든 Modality에서 값이 없는(=0)인 부분을 얻음
        nonzero_index = torch.nonzero(torch.sum(patient_image, axis=0) != 0)
        z_indexes, y_indexes, x_indexes = nonzero_index[:, 0], nonzero_index[:, 1], nonzero_index[:, 2]
        z_min, y_min, x_min = [max(0, int(torch.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
        z_max, y_max, x_max = [int(torch.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
        # ex) (4, 155, 240, 240) -> (4, 133, 167, 137)
        patient_image = patient_image[:, z_min:z_max, y_min:y_max, x_min:x_max].float() # 없는 부분 제거.

        # TODO: 왜 test시에도 random crop하면서 진행하는가?
        if self.mode == "train" or self.mode == "train_val" or self.mode == "test":
            patient_image, patient_label, pad_list, crop_list = pad_or_crop_image(patient_image, patient_label, target_size=self.target_size)
        elif self.mode == "test_pad":
            # 1개의 Modality에서 128이 안되는 경우를 패딩으로 해결하기 위함
            # 다만 TEST라서 inference시에만 사용.
            d, h, w = patient_image.shape[1:]
            pad_d = (128-d) if 128-d > 0 else 0
            pad_h = (128-h) if 128-h > 0 else 0
            pad_w = (128-w) if 128-w > 0 else 0
            patient_image, patient_label, pad_list = pad_image_and_label(patient_image, patient_label, target_size=(d+pad_d, h+pad_h, w+pad_w))
        return dict(
            patient_id = patient["id"],
            image = patient_image.to(dtype=torch.float32),
            label = patient_label.to(dtype=torch.float32),
            nonzero_indexes = ((z_min, z_max), (y_min, y_max), (x_min, x_max)),
            box_slice = crop_list,
            pad_list = pad_list
        )

    def __len__(self):
        return len(self.datas)

def get_datasets(dataset_folder, mode, target_size = (128, 128, 128)):
    dataset_folder = get_brats_folder(dataset_folder, mode)
    assert os.path.exists(dataset_folder), "Dataset Folder Does Not Exist1"
    patients_ids = [x for x in listdir(dataset_folder)]
    return BraTS(dataset_folder, patients_ids, mode, target_size=target_size)