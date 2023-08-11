# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).to(device)).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            shortcut = x
            x = self.downsample(x)
        else:
            shortcut = None
        return x, shortcut, self.input_resolution[0], self.input_resolution[1]

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformerV2_Backbone(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],gvi_num=8, **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.eps=1e-6
        self.alpha1=nn.Linear(3,gvi_num,bias=qkv_bias)
        self.alpha2=nn.Linear(3,gvi_num,bias=qkv_bias)
        self.gvi_norm=norm_layer(gvi_num)
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans+gvi_num, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer])
            self.layers.append(layer)

        self.num_features_per_layer = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        for i_layer in range(self.num_layers-1):
            layer = norm_layer(self.num_features_per_layer[i_layer])
            layer_name = f'norm{i_layer}_prevDs'
            self.add_module(layer_name, layer)
            layer = norm_layer(self.num_features_per_layer[i_layer+1])
            layer_name = f'norm{i_layer}_postDs'
            self.add_module(layer_name, layer)
        layer = norm_layer(self.num_features_per_layer[self.num_layers-1])
        layer_name = f'norm{self.num_layers-1}'
        self.add_module(layer_name, layer)

        #self.norm = norm_layer(self.num_features)
        #self.avgpool = nn.AdaptiveAvgPool1d(64)

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        gvis=x.transpose(1,3)
        x1=self.alpha1(gvis)
        x2=torch.clamp(self.alpha2(gvis),min=1+self.eps)
        gvis=x1/x2
        gvis=self.gvi_norm(gvis)
        gvis=gvis.transpose(1,3)
        x=torch.concat([gvis,x],1)
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        outs=[x.transpose(1,2)]
        i=0
        for layer in self.layers:
            x,shortcut,H,W = layer(x)
            if(i<self.num_layers-1):
                norm_layer = getattr(self, f'norm{i}_prevDs')
                x_out = norm_layer(shortcut)
                out = x_out.transpose(1,2).contiguous()
                outs.append(out)
                norm_layer = getattr(self, f'norm{i}_postDs')
                x_out = norm_layer(x)
                out = x_out.transpose(1,2).contiguous()
                outs.append(out)
            else:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                out = x_out.transpose(1,2).contiguous() #.view(-1, H, W, self.num_features_per_layer[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
            i+=1
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        #flops += self.num_features * self.num_classes
        return flops

class ShiftWindowAttentionBlock(nn.Module):
        def __init__(self,img_size=64,feature_size=256,num_head=6,
                            drop=0,
                            norm_layer=nn.LayerNorm):
            super().__init__()
            self.num_heads=num_head
            self.softmax=nn.Softmax(dim=1)
            self.img_size=img_size
            self.mlp=Mlp(feature_size**2,self.img_size**2,self.img_size**2,nn.ReLU,drop)
            self.norm=norm_layer(img_size**2)
            self.drop = nn.Dropout(drop)
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((1,1,img_size**2))), requires_grad=True)

        def forward(self,x,feature):
            """
            x: B C H*W
            feature: B C H*W
            """
            #print(feature.shape)
            feature=self.mlp(feature).transpose(1,2)
            logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).to(device)).exp()
            #print(feature.shape, x.shape, logit_scale.shape)
            temp=torch.bmm(feature,x)*logit_scale #//q*k
            temp=self.softmax(temp)
            x=torch.bmm(x,temp)
            x=self.norm(x)
            x=self.drop(x)
            return x

        def flops(self):
            flops = 0
            # flops+=self.conv.flops()
            # flops+=self.softmax.flops()
            # flops+=self.norm.flops()
            # if self.dropout is not None:
            #     flops+=self.dropout.flops()
            return flops

class ChannelAttentionBlock(nn.Module):
        def __init__(self,img_size=64,feature_size=256,num_head=6,
                            drop=0,
                            norm_layer=nn.LayerNorm):
            super().__init__()
            self.num_heads=num_head
            self.softmax=nn.Softmax(dim=1)
            self.img_size=img_size
            self.mlp=Mlp(feature_size**2,feature_size**2,self.img_size**2,nn.ReLU,drop)
            self.norm=norm_layer(num_head)
            self.drop = nn.Dropout(drop)
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((1,1,img_size**2))), requires_grad=True)

        def forward(self,x,feature):
            """
            x: B C H*W
            feature: B C H*W
            """
            #print(feature.shape)
            x=x.transpose(1,2)
            feature=self.mlp(feature)
            logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).to(device)).exp()
            #print(feature.shape, x.shape, logit_scale.shape)
            temp=torch.bmm(x,feature)*logit_scale #//q*k
            temp=self.softmax(temp)
            x=torch.bmm(temp,x)
            x=self.norm(x).transpose(1,2)
            x=self.drop(x)
            return x

        def flops(self):
            flops = 0
            # flops+=self.conv.flops()
            # flops+=self.softmax.flops()
            # flops+=self.norm.flops()
            # if self.dropout is not None:
            #     flops+=self.dropout.flops()
            return flops

class UnNamedBlock(nn.Module):

        def __init__(self,depth=2,feature_nums=3,img_size=256,embed_dim=96,
                            drop=0.1, attn_drop=0.1,
                            norm_layer=nn.LayerNorm,upsample_size=128):
            super().__init__()
            self.feature_nums=feature_nums
            self.img_size = img_size
            self.depth=depth
            #self.swablock=ShiftWindowAttentionBlock(img_size=img_size,feature_size=img_size,num_head=feature_nums,drop=attn_drop)
            self.aspp=ASPP(feature_nums,embed_dim,[1,4,6])
            self.upsample=PSPUpsample(embed_dim,embed_dim*2,upsample_size//self.img_size)
            #self.conv=nn.Conv2d(in_channels=feature_nums*4//(2**depth),out_channels=feature_nums*2//(2**depth),kernel_size=3,padding=1)


        def forward(self,x,feature):
            #x=x.view(-1,self.feature_nums,self.img_size/self.img_size)
            #x=self.swablock(x,feature)+x #SWA
            x=x.view(-1,self.feature_nums,self.img_size,self.img_size)
            #print(x.shape)
            x=self.aspp(x)
            x=self.upsample(x)
            #x=self.conv(x)
            #print(x.shape)
            return x #B H W C

        def flops(self):
            flops=0
            # flops+=self.alpha1.flops()
            # flops+=self.alpha2.flops()
            # flops+=self.test_conv.flops()
            # for layer in self.embed_layers:
            #     flops+=layer.flops()
            # flops+=self.conv.flops()
            return flops

class Decoder(nn.Module):
        def __init__(self,img_size=224,num_classes=15,patch_size=4,depth_nums=[ 2, 4, 8], gvi_nums=[1,2,3],feature_nums=[ 16, 4, 2],
                         embed_dim=96,
                        qkv_bias=True,
                        drop_rate=0.1, attn_drop=0.1,
                        norm_layer=nn.LayerNorm):
            super().__init__()
            self.img_size = img_size
            self.patch_size = patch_size
            self.num_layers = 4
            self.num_classes = num_classes+1
            self.embed_dim = embed_dim
            self.layers=nn.ModuleList()
            for i in range(self.num_layers):
                layer=UnNamedBlock(depth=i+1,feature_nums=embed_dim*2**i,img_size=(self.img_size//self.patch_size)//2**i,embed_dim=embed_dim,
                                        drop=drop_rate, attn_drop=attn_drop,
                                        norm_layer=norm_layer)
                self.layers.append(layer)
            self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,norm_layer=norm_layer)

            self.pspmodule=PSPModule(features=embed_dim,out_features=embed_dim)
            #self.upsample1=PSPUpsample(embed_dim, embed_dim*6, 8)
            #self.drop1=nn.Dropout2d(drop_rate)

            self.conv2d_1=nn.Conv2d(embed_dim*2,embed_dim*2,kernel_size=3,padding=1,stride=1)
            self.conv2d_2=nn.Conv2d(embed_dim*2,3,kernel_size=1,stride=1)
            self.norm=nn.BatchNorm2d(embed_dim*2)
            self.relu=nn.ReLU()
            self.upsample3=PSPUpsample(embed_dim*10,embed_dim*4)
            self.drop3=nn.Dropout2d(drop_rate)
            self.upsample4=PSPUpsample(embed_dim*4,embed_dim*2)
            self.drop4=nn.Dropout2d(drop_rate)
            #self.drop=nn.Dropout(drop_rate)
            #self.mlp_classifier=Mlp(in_features=self.num_layers,hidden_features=128,out_features=1,drop=drop_rate)
            #self.softmax=nn.LogSoftmax(1)

        def forward(self,x,feature):
            res=None
            cnt=0
            for i in range(self.num_layers):
                temp=self.layers[i](feature[2*i+1],feature[2*i]) #Block
                if res is not None:
                    res=torch.cat([res,temp],dim=1)#B embed_dim 128 128
                else:
                    res=temp
                cnt+=1
            psp=self.patch_embed(x).transpose(1,2).view(-1,self.embed_dim, self.img_size//self.patch_size, self.img_size//self.patch_size)
            #print(psp.shape)
            psp=self.pspmodule(psp)
            res=torch.cat([feature[1].view(-1,self.embed_dim,self.img_size//self.patch_size,self.img_size//self.patch_size),psp,res],dim=1)

            output=self.upsample3(res)
            output=self.drop3(output)
            output=self.upsample4(output)
            output=self.drop4(output)

            output=self.conv2d_1(output)
            output=self.norm(output)
            output=self.relu(output)
            output=self.conv2d_2(output)
            #output=self.softmax(output)

            return output

        @torch.no_grad()
        def forward_test(self,feature):
            B,C,H,W = feature[0].shape
            channels=self.embed_dim
            res=torch.zeros((B,channels,self.img_size,self.img_size,self.num_layers)).to(device)
            output2=torch.zeros((x.shape[0],self.num_classes,self.img_size,self.img_size,self.num_layers)).to(device)
            cnt=0
            for i in range(self.num_layers):
                temp=self.layers[i](x,feature)
                res[:,:,:,:,cnt]=temp
                temp=self.conv2d_1(temp)
                temp=self.relu(temp)
                #temp=self.mlp_classifier(temp).transpose(1,3)
                temp=self.conv2d_2(temp)
                temp=self.softmax(temp)
                output2[:,:,:,:,cnt]=temp
                cnt+=1
            res=res.transpose(1,4) #B CNT H W C
            res=self.conv3d_1(res).transpose(1,4).squeeze() #0 4 2 3 1 -> 0 1 2 3
            res=self.mlp_classifier(res).transpose(1,4).squeeze()
            output=self.conv2d_1(res) #B W H C
            output=self.relu(output)
            #output=self.mlp_classifier(output).transpose(1,3) #B C H W
            output=self.conv2d_2(output)
            output=self.softmax(output)
            return output,output2

        def flops(self):
            flops=0
            # for layer in self.layers:
            #     flops+=layer.flops()
            # flops+=self.conv3d.flops()
            # flops+=self.conv2d.flops()
            # flops+=self.norm.flops()
            return flops

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project( feature['low_level'] )
        output_feature = self.aspp(feature['out'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class SARNet(nn.Module):
    r"""
        Args:
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=15,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0.1, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0],
                 decoder_depth=[1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1],
                 gvi_nums=[8, 4, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3],
                 decoder_features=[16, 8, 8, 8, 4, 8, 16, 16, 16, 16, 8, 8, 8, 4, 4],**kwargs):
        super().__init__()
        self.swin_backbone=SwinTransformerV2_Backbone(img_size, patch_size, in_chans,
                 embed_dim, depths, num_heads,
                 window_size, mlp_ratio, qkv_bias,
                 drop_rate, attn_drop_rate, drop_path_rate,
                 norm_layer, ape, patch_norm,
                 use_checkpoint, pretrained_window_sizes)
        self.decoder=Decoder(img_size=img_size,num_classes=num_classes, patch_size=patch_size,
                            depth_nums=decoder_depth,
                            gvi_nums=gvi_nums,
                            feature_nums=decoder_features,embed_dim=embed_dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(6, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.PReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(2, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.PReLU())

    def _sar_res(self,x,sar_images):
        B1,C1,H1,W1 = x.shape
        B2,C2,H2,W2 = sar_images.shape
        out = torch.zeros((B1,C1*C2,H2,W2)).cuda()
        for i in range(C1*C2):
            out[:,i,:,:] = x[:,i//C2,:,:]-sar_images[:,i%C2,:,:]
        out = self.conv1(out)
        sar_images = self.conv2(sar_images)
        out = out + sar_images
        return out

    def forward(self,x,sar_images):
        feature=self.swin_backbone(x)
        x=self._sar_res(x,sar_images)
        x=self.decoder(x, feature)
        return x

    @torch.no_grad()
    def forward_test(self,x):
        feature=self.swin_backbone(x)
        x=self.decoder.forward_test(feature)
        return x

    def flops(self):
        flops=0
        flops+=self.swin_backbone.flops()
        flops+=self.decoder.flops()
        return flops

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'swin_backbone.absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"swin_backbone.cpb_mlp", "swin_backbone.logit_scale", 'swin_backbone.relative_position_bias_table'}

# class ResBlock(nn.Module):
#         def __init__(self,channels,drop=0.1):
#             super().__init__()

#             self.conv1_1_1x1=nn.Conv1d(in_channels=channels,out_channels=channels//2,kernel_size=1,padding=0,stride=1)
#             self.conv2_1_1x1=nn.Conv1d(in_channels=channels,out_channels=2*channels,kernel_size=1,padding=0,stride=1)
#             self.bn2_2=nn.BatchNorm1d(2*channels)
#             self.relu2_3=nn.ReLU()
#             self.conv2_4_3x3=nn.Conv1d(in_channels=2*channels,out_channels=2*channels,kernel_size=3,padding=1,stride=1)
#             self.bn2_5=nn.BatchNorm1d(2*channels)
#             self.relu2_6=nn.ReLU()
#             self.conv2_7_1x1=nn.Conv1d(in_channels=2*channels,out_channels=channels//2,kernel_size=1,padding=0,stride=1)
#             self.drop=nn.Dropout(drop)

#         def forward(self,x):
#             x=x.transpose(1,2)
#             x1=self.conv1_1_1x1(x)
#             x2=self.conv2_1_1x1(x)
#             x2=self.bn2_2(x2)
#             x2=self.relu2_3(x2)
#             x2=self.conv2_4_3x3(x2)
#             x2=self.bn2_5(x2)
#             x2=self.relu2_6(x2)
#             x2=self.conv2_7_1x1(x2)
#             x=torch.cat([x1,x2],dim=1)
#             x=x.transpose(1,2)
#             x=self.drop(x)

#             return x

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear',align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, multi=2):
        super().__init__()
        self.multi = multi
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = self.multi * x.size(2), self.multi * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear',align_corners=True)
        return self.conv(p)

# class PSPNet(nn.Module):
#     def __init__(self, n_classes=18, sizes=(1, 2, 3, 6), psp_size=2048):#, deep_features_size=1024, backend='resnet34', pretrained=True):
#         super().__init__()
#         self.psp = PSPModule(psp_size, 1024, sizes)
#         self.drop_1 = nn.Dropout2d(p=0.3)

#         self.up_1 = PSPUpsample(1024, 256)
#         self.up_2 = PSPUpsample(256, 64)
#         self.up_3 = PSPUpsample(64, 3)

#         self.drop_2 = nn.Dropout2d(p=0.1)

#         # self.classifier = nn.Sequential(
#         #     nn.Linear(deep_features_size, 256),
#         #     nn.ReLU(),
#         #     nn.Linear(256, n_classes)
#         # )

#     def forward(self, x):
#         f = x
#         p = self.psp(f)
#         p = self.drop_1(p)

#         p = self.up_1(p)
#         p = self.drop_2(p)

#         p = self.up_2(p)
#         p = self.drop_2(p)

#         p = self.up_3(p)
#         p = self.drop_2(p)

#         #auxiliary = F.adaptive_max_pool2d(input=class_f, output_size=(1, 1)).view(-1, class_f.size(1))

#         return p#, self.classifier(auxiliary)