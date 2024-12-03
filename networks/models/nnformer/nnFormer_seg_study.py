from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import to_3tuple

# 원본은 networks.nnformer.nerual_network
from networks.models.nnformer.nerual_network import SegmentationNetwork

class ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.contiguous()
    

class Mlp(nn.Module):
    """Multi-Layer Perceptrion"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features 
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Drop(drop)
    
    def forward(self, x):
        # 중복 제거: helper 메서드 사용
        x = self._apply_layer(self.fc1, x)
        x = self._apply_layer(self.fc2, x)
        return x

    def _apply_layer(self, layer, x):
        """레이어, 활성화 및 Dropout 적용"""
        x = layer(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # TODO: 왜 relative_position_bias_table의 shape이 
        # (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))
        
        # get pair-wise relative position index for each token inside the window
        # S는 Sequence Length
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w]))
        # TODO: [B, S, H, W] -> [B, S, H * W] 맞는지 확인
        # relative coordinate 해당 과정 완벽하게 이해하기.
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:,:,None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= 3 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1) 


class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # 만약 window size가 input resolution보다 크다면, window를 더 이상 partition 하지 않음.
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
    
    
        assert 0 <= self.shift_size < self.window_size, "sfhit_size must smaller than window_size"

        self.norm1 = norm_layer(dim)

        self.attn = WindowAttention(

        )



class BasicLayer(nn.Module)

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=True
                 ):
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth

        self.block = nn.ModuleList([
            SwinTransformerBlock(

            )
        ])


class Projection(nn.Module):
    # *는 단순히 함수 호출을 가독성을 위해 keyword argument 형태로 호출하기 위해 사용됨.
    # 실행시에는 들어가는 인자는 없을 것.
    def __init__(self, *, in_dim, out_dim, stride, padding, activate, norm, last=False):
        super().__init__()
        self.out_dim=out_dim
        self.conv1=nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=stride, padding=padding)
        self.conv2=nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.activate=activate()
        self.norm1=norm(out_dim)
        self.last=last
        if not last:
            self.norm2=norm(out_dim)
    
    # TODO: 변수명으로 (C, X, Y, Z) 크기를 결정하기.
    # Init shape-> (C, X, Y, Z)
    def forward(self, x):
        x=self.conv1(x)
        x=self.activate(x)
        # norm1
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm1(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)

        x = self.conv2(x)
        if not self.last:
            x=x.activate(x)
            # norm2
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm2(x)
            x = x.transpose(1, 2).contigous().view(-1, self.out_dim, Ws, Wh, Ww)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_channels=4, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size) # patch_size = (4, 4, 4)
        self.patch_size = patch_size

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        stride1=[patch_size[0], patch_size[1]//2, patch_size[2]//2] # stride1 = [4, 2, 2]
        stride2=[patch_size[0]//2, patch_size[1]//2, patch_size[2]//2] # tride2 = [2, 2, 2]
        self.proj1 = Projection(
            in_dim=in_channels,
            out_dim=embed_dim // 2,
            stride=stride1,
            padding=1,
            activate=nn.GELU,
            norm=nn.LayerNorm,
            last=False
        )

        self.proj2 = Projection(
            in_dim=embed_dim // 2,
            out_dim=embed_dim,
            stride=stride2,
            padding=1,
            activate=nn.GELU,
            norm=nn.LayerNorm,
            last=True
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
    
    def forward(self, x):
        # 혹여나 patch_size로 정확하게 나누어 떨어지지 않으면 잘리는 부분이 생기므로, 미리 자름.
        _, _, S, H, W = x.size()
        # Width
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        # Height
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        # Sequence, Patch Embedding 후 Data가 Sequence화 되었음.
        if S % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - S % self.patch_size[0]))
        x = self.proj1(x) # B C Ws Wh Ww
        x = self.proj2(x) # B C Ws Wh Ww
        if self.norm is not None:
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contguous()
            x = x.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, Ws, Wh, Ww)
        return x



class Encoder(nn.Module):
    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_channels=1,
                 embed_dim=96,
                 depths=[2,2,2,2],
                 num_heads=[4,8,16,32],
                 window_size=7,
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3)
                 ):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth
        drop_path_rate = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(

            )



class nnFormer(SegmentationNetwork):

    def __init__(self, crop_size=[96, 96, 96],
                 embedding_dim=192, 
                 input_channels=1,
                 num_classes=14,
                 conv_op=nn.Conv3d,
                 depths=[2,2,2,2],
                 num_heads=[6,12,24,48],
                 patch_size=[2,4,4],
                 window_size=[4,4,8,4],
                 deep_supervision=False):
        super(nnFormer, self).__init__()

        # TODO: _deep_supervision 변수가 의미하는 것.
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.num_classes = num_classes
        self.conv_op = conv_op # nn.Conv3d

        self.upscale_logits_ops = []

        self.upscale_logits_ops.append(lambda x:x)

        embed_dim = embedding_dim
        depths = depths
        num_heads=num_heads
        patch_size=patch_size
        window_size=window_size
        self.model_down=Encoder(pretrain_img_size=crop_size,window_size=window_size,
                                embed_dim=embed_dim,patch_size=patch_size,depths=depths,
                                num_heads=num_heads,in_chans=input_channels)





