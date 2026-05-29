from typing import Sequence, Type, Union

import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.unetr_block import UnetrBasicBlock
from monai.networks.layers import Conv
from monai.utils import ensure_tuple_rep
from torch.nn import LayerNorm


class PatchEmbed(nn.Module):
    """
    Patch embedding block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    Unlike ViT patch embedding block: (1) input is padded to satisfy window size requirements (2) normalized if
    specified (3) position embedding is not used.
    """

    def __init__(
        self,
        patch_size: Union[Sequence[int], int] = 2,
        in_chans: int = 1,
        embed_dim: int = 48,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,  # type: ignore
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            patch_size: dimension of patch size.
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            norm_layer: normalization layer.
            spatial_dims: spatial dimension.
        """

        super().__init__()

        if not (spatial_dims == 2 or spatial_dims == 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = Conv[Conv.CONV, spatial_dims](
            in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            _, _, d, h, w = x_shape
            if w % self.patch_size[2] != 0:
                x = F.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))
            if h % self.patch_size[1] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))
            if d % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - d % self.patch_size[0]))

        elif len(x_shape) == 4:
            _, _, h, w = x.size()
            if w % self.patch_size[1] != 0:
                x = F.pad(x, (0, self.patch_size[1] - w % self.patch_size[1]))
            if h % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[0] - h % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            x_shape = x.size()
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            if len(x_shape) == 5:
                d, wh, ww = x_shape[2], x_shape[3], x_shape[4]
                x = x.transpose(1, 2).view(-1, self.embed_dim, d, wh, ww)
            elif len(x_shape) == 4:
                wh, ww = x_shape[2], x_shape[3]
                x = x.transpose(1, 2).view(-1, self.embed_dim, wh, ww)
        return x


class PatchEmbed3D(nn.Module):
    """ Volume to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,2,2).
        in_chans (int): Number of input volume channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 48.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, vol_size=(96, 96, 96), patch_size=(2, 2, 2), in_chans=1, embed_dim=48, norm_layer=None):
        super().__init__()
        vol_size = ensure_tuple_rep(vol_size, 3)
        patch_size = ensure_tuple_rep(patch_size, 3)
        patches_resolution = [vol_size[0] // patch_size[0], vol_size[1] // patch_size[1], vol_size[2] // patch_size[2]]
        self.vol_size = vol_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, D, Wh, Ww)

        return x

class PatchEmbedDeep(nn.Module):
    def __init__(self, vol_size=(96, 96, 96), patch_size=(2, 2, 2), in_chans=1, embed_dim=48, spatial_dims=3, norm_name="batch", norm_layer=None):
        super().__init__()

        vol_size = ensure_tuple_rep(vol_size, 3)
        patch_size = ensure_tuple_rep(patch_size, 3)
        patches_resolution = [vol_size[0] // patch_size[0], vol_size[1] // patch_size[1], vol_size[2] // patch_size[2]]
        self.vol_size = vol_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.block1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_chans,
            out_channels=embed_dim // 4,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=False,
        )
        self.block2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=embed_dim // 4,
            out_channels=embed_dim // 2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=False,
        )

        self.proj = nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        # Creating patch embedding
        x = self.block1(x)
        x = self.block2(x)
        x = self.proj(x)  # B C D Wh Ww

        # Normalization if flag set
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, D, Wh, Ww)

        return x


class PatchEmbedGlobal(nn.Module):

    def __init__(self, vol_size=(96, 96, 96), in_chans=1, embed_dim=48, norm_layer=None):
        super().__init__()

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        ksize = (vol_size[0] // 4, vol_size[1] // 4, vol_size[2] // 4)
        self.down1 = nn.Conv3d(in_chans, self.in_chans * 2, kernel_size=2, stride=2)
        self.down2 = nn.Conv3d(self.in_chans * 2, self.in_chans * 4, kernel_size=2, stride=2)
        self.proj = nn.Conv3d(self.in_chans * 4, embed_dim, kernel_size=ksize, stride=ksize)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        x = self.down1(x)
        x = self.down2(x)
        x = self.proj(x)

        if self.norm is not None:
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, Ws, Wh, Ww)

        return x


class PatchEmbedRegion(nn.Module):

    def __init__(self, region_size=(32, 32, 32), in_chans=1, embed_dim=48, norm_layer=None):
        super().__init__()

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        ksize = (region_size[0] // 2, region_size[1] // 2, region_size[2] // 2)
        self.down = nn.Conv3d(in_chans, self.in_chans * 2, kernel_size=2, stride=2)
        self.proj = nn.Conv3d(self.in_chans * 2, embed_dim, kernel_size=ksize, stride=ksize)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        x = self.down(x)
        x = self.proj(x)

        if self.norm is not None:
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, Ws, Wh, Ww)

        return x
