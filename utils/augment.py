import torch
from torch import nn
from random import random, uniform
from monai.transforms.spatial.array import Zoom
from monai.transforms.intensity.array import RandGaussianNoise, GaussianSharpen, AdjustContrast



class DataAugmenter(nn.Module):
    def __init__(self):
        super(DataAugmenter, self).__init__()
        self.flip_dim = []
        self.zoom_rate = uniform(0.7, 1.0)
        self.sigma_1 = uniform(0.5, 1.5)
        self.sigma_2 = uniform(0.5, 1.5)
        self.image_zoom = Zoom(zoom=self.zoom_rate, mode='trilinear', padding_mode='constant')
        self.label_zoom = Zoom(zoom=self.zoom_rate, mode="nearest", padding_mode="constant")
        self.noisy = RandGaussianNoise(prob=1, mean=0, std=uniform(0, 0.33))
        self.blur = GaussianSharpen(sigma1=self.sigma_1, sigma2=self.sigma_2)
        self.contrast = AdjustContrast(gamma=uniform(0.65, 1.5))
    def forward(self, images, lables):
        # torch.no_grad() 사용하여 자동 미분을 비활성화합니다. 
        # 이 경우 데이터 변형만 수행되고, 연산 그래프가 저장되지 않습니다.
        with torch.no_grad():
            # 배치 내 각 이미지 및 레이블에 대해 반복 수행
            for b in range(images.shape[0]):
                image = images[b].squeeze(0)  # 배치 차원을 제거하여 이미지 차원 조정
                lable = lables[b].squeeze(0)  # 배치 차원을 제거하여 레이블 차원 조정

                # 15% 확률로 이미지 및 레이블에 대해 줌(확대/축소) 변환 적용
                if random() < 0.15:
                    image = self.image_zoom(image)
                    lable = self.label_zoom(lable)

                # 50% 확률로 이미지 및 레이블을 y축 (dims=(1,))에 대해 플립(좌우 대칭)
                if random() < 0.5:
                    image = torch.flip(image, dims=(1,))
                    lable = torch.flip(lable, dims=(1,))

                # 50% 확률로 이미지 및 레이블을 x축 (dims=(2,))에 대해 플립(상하 대칭)
                if random() < 0.5:
                    image = torch.flip(image, dims=(2,))
                    lable = torch.flip(lable, dims=(2,))

                # 50% 확률로 이미지 및 레이블을 z축 (dims=(3,))에 대해 플립(깊이 방향 대칭)
                if random() < 0.5:
                    image = torch.flip(image, dims=(3,))
                    lable = torch.flip(lable, dims=(3,))

                # 15% 확률로 이미지에 가우시안 노이즈 추가
                if random() < 0.15:
                    image = self.noisy(image)

                # 15% 확률로 이미지에 가우시안 샤프닝(선명화) 적용
                if random() < 0.15:
                    image = self.blur(image)

                # 15% 확률로 이미지 대비 조정 (Contrast Adjustment)
                if random() < 0.15:
                    image = self.contrast(image)

                # 변형된 이미지를 배치 차원으로 다시 추가하여 원래 위치에 할당
                images[b] = image.unsqueeze(0)
                # 변형된 레이블을 배치 차원으로 다시 추가하여 원래 위치에 할당
                lables[b] = lable.unsqueeze(0)
            
            # 변형된 이미지와 레이블 반환
            return images, lables

    