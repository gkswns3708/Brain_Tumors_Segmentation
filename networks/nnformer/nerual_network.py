import numpy as np
import torch.nn as nn



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
        
        # 
        self.conv_op = None
        