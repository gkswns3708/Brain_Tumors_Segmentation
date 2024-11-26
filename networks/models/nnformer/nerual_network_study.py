from networks.models.nnformer.utilities.random_stuff import no_op

import torch
import torch.nn as nn

import numpy as np
import torch.nn as nn

from typing import Union, Tuple, List

from torch.cuda.amp import autocast


class NerualNetwork(nn.Module):
    def __init__(self):
        super(NerualNetwork, self).__init__()
        
    def get_device(self):
        if next(self.parameters()).device == "cpu":
            return "cpu"
        else:
            return next(self.parameters()).device.index
    
    def set_device(self, device):
        if device == "cpu":
            self.cpu()
        else:
            self.cuda(device)
    
    def forward(self, x):
        raise NotImplementedError

class SegmentationNetwork(NerualNetwork):
    def __init__(self):
        # nn.Module의 __init__ 메소드를 호출하기 위해 아래와 같이 사용함.
        super(NerualNetwork, self).__init__()
        
        # if 5개의 pooling layer가 있다면 우리의 patch는 2**5 = 32로 나누어져야 함.
        self.input_shape_must_be_divisible_by = None
        
        # Input image가 2D 혹은 3D이냐에 따라 Convolution Operation이 달라지므로, 이를 사전에 정의해줘야 함.        
        self.conv_op = None

        # output의 channel의 갯수가 무엇인지를 알려줘야함.
        # Important for preallocation in inference.
        self.num_classes = None

        # 손실 함수에따라 non-linear layer를 architecture에 적용해야 하므로, 사전에 정의해주어야 합니다.
        # 일반적으로 Softmax를 사용할 것으로 생각됩니다.
        self.inference_apply_nonlin = lambda x: x # softmax_helper

        # Inference시에 Gaussian importance map을 저장하기 위한 것입니다.
        # 중심에 가까운 복셀일수록 더 높은 가중치를 부여합니다.
        # 일반적으로 border(경계)의 prediction은 정확하지 않기에 중요도를 낮춥니다. 
        # 하지만 이러한 Gaussian은 Computation-Cost가 많이 들 수 있으므로, 저장하고 재사용하는 것이 좋습니다.
        # TODO: 왜 Gaussian으로 Operation을 하는 것이 Cost가 만이 드는가?
        self._gaussian_3d = self._patch_size_for_gaussian_3d = None
        self._gaussian_2d = self._patch_size_for_gaussian_2d = None
    
    def predict_2D(self, x: np.ndarray, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
                   use_sliding_window: bool = False,
                   step_size: float=0.5, patch_size: Tuple[int, ...] = None, regions_class_order: Tuple[int, ...]=None,
                   use_gaussian: bool=False, pad_border_mode: str="constant",
                   pad_kwargs: dict=None, all_in_gpu:bool=False,
                   verbose:bool=True, mixed_precision:bool=True) -> Tuple[np.ndarray, np.ndarray]:
        """
        # TODO: 그래서 해당 함수는 3D input으로 들어오는 Voxel Iamge를 처리할 수 있는지 궁금함.
        이 함수는 2D image를 prediction하기 위한 function입니다. 
        네트워크가 2D 또는 3D U-Net인지에 관련없이 적절한 코드를 수행하도록 설계되었습니다.

        Prediction할 때, Fully Convolution 혹은 sliding window기반의 inference를 실행할 지 결정해야 합니다.
        Base option으로는 sliding window 방식을 추천합니다.

        사용자 network가 올바른 mode(ex: inference를 위한 eval mode)에 있는지 확인해야 합니다. 
        Network가 eval mode가 아니라면 warning을 출력합니다.

        # TODO: mirroring을 하는 이유 파악 -> (TTA, 축을 기반으로 뒤집어서 inference 한 형태(mirror)를 학습 및 추론)
        """
        torch.cuda.empty_cache()

        assert step_size <= 1, 'step_size must be smaller than 1, Othersize there will be a gap between consecutive prediction'

        if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        assert self.get_device() != "cpu", "CPU not implemented"

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # 옛날에는 mirror axes는 (2, 3, 4)를 3D Network에 사용했습니다.
        # TODO: 현대적 네트워크 구현에서는 보통 mirror_axes가 (0, 1, 2)로 설정하는 듯 합니다.
        # 아래의 코드는 해당 convention을 사용했던 코드를 intercept(가져와) 수행하기 위한 코드입니다.

        if len(mirror_axes):
            if self.conv_op == nn.Conv2d:
                if max(mirror_axes) > 1:
                    raise ValueError("mirror axes error")
            if self.conv_op == nn.Conv3d:
                if max(mirror_axes) > 2:
                    raise ValueError("mirror axes error")
        
        # assert self.training, "WARNING! Network is in train mode during inference. This may be intended, or not..."
        if self.training: 
            print("WARNING! Network is in train mode during inference. This may be intended, or not...")
        
        assert len(x.shape) == 4, "data must have shape (c, x, y, z)"

        if mixed_precision:
            context = autocast
        else:
            context = no_op

        with context():
            with torch.no_grad():
                if self.conv_op == nn.Conv2d:
                    if use_sliding_window:
                        res = self._internal_predict_2D_2Dconv_tiled(x, step_size, do_mirroring, mirror_axes, patch_size,
                                                                     regions_class_order, use_gaussian, pad_border_mode,
                                                                     pad_kwargs, all_in_gpu, verbose)
                    else:
                        res = self._internal_predict_2D_2DConv(x, patch_size, do_mirroring, mirror_axes, regions_class_order, 
                                                               pad_border_mode, pad_kwargs, verbose)
                else:
                    raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")
        
        return res

    
        