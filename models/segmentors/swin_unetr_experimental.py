# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, Tuple, Union

import torch.nn as nn
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep


class SwInceptionDecoder(nn.Module):
    """
    UNETR-style decoder for SwInception. Accepts the hierarchical feature maps
    produced by the SwInception encoder and reconstructs the segmentation mask
    through a sequence of residual up-blocks with skip connections.

    Args:
        encoder: SwInception backbone instance.
        in_channels: number of input image channels.
        out_channels: number of segmentation classes.
        img_size: spatial dimensions of the input volume.
        hidden_size: base feature dimension (must match encoder embed_dim).
        patch_size: patch size used by the encoder.
        norm_name: normalisation type passed to MONAI blocks.
        dropout_rate: dropout probability (0–1).
        spatial_dims: number of spatial dimensions (3 for volumetric).
        decode_hs_ratio: scales decoder hidden size relative to hidden_size.
    """

    def __init__(
        self,
        encoder,
        in_channels: int,
        out_channels: int,
        img_size: Union[Sequence[int], int] = [96, 96, 96],
        hidden_size: int = 48,
        patch_size: Union[Sequence[int], int] = [2, 2, 2],
        norm_name: Union[Tuple, str] = "instance",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        decode_hs_ratio: float = 1.0
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")


        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.img_size = img_size
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.hidden_size = hidden_size
        self.decode_hidden_size = int(hidden_size * decode_hs_ratio)
        self.classification = False
        self.encoder = encoder
        self.fl_out_size = tuple(img_d // (p_d*(2**4)) for img_d, p_d in zip(self.img_size, self.patch_size))

        self.unet_encoders = nn.ModuleList()
        self.unet_decoders = nn.ModuleList()

        encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=self.decode_hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True
        )

        encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=self.decode_hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=self.decode_hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        decoder0 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.decode_hidden_size,
            out_channels=self.decode_hidden_size,
            kernel_size=3,
            upsample_kernel_size=self.patch_size,
            norm_name=norm_name,
            res_block=True,
        )

        decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.decode_hidden_size,
            out_channels=self.decode_hidden_size,
            kernel_size=3,
            upsample_kernel_size=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.unet_encoders.append(encoder0)
        self.unet_encoders.append(encoder1)
        self.unet_encoders.append(encoder2)
        self.unet_decoders.append(decoder0)
        self.unet_decoders.append(decoder1)

        for i_layer in range(1, self.encoder.num_layers):
            enc = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=hidden_size * 2 ** i_layer,
                out_channels=self.decode_hidden_size * 2 ** i_layer,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=True,
            )
            self.unet_encoders.append(enc)

            dec = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.decode_hidden_size * 2 ** i_layer,
                out_channels=self.decode_hidden_size * 2 ** (i_layer-1),
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )
            self.unet_decoders.append(dec)

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=self.decode_hidden_size, out_channels=out_channels)

    def forward(self, x_in):
        z = self.encoder(x_in)

        x = self.unet_decoders[-1](self.unet_encoders[-1](z[-1]), self.unet_encoders[-2](z[-2]))
        for i in range(1, self.encoder.num_layers):
            enc = self.unet_encoders[-(i+2)]
            dec = self.unet_decoders[-(i+1)]
            x = dec(x, enc(z[-(i+2)]))
        x = self.unet_decoders[0](x, self.unet_encoders[0](x_in))
        return self.out(x)
