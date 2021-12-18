import torch
import torch.nn as nn

from typing import List

from models.commons.layers import Conv2dBlock
from models.commons.layers import Deconv2dBlock


class RedNet(nn.Module):
    decoder_layers: List[nn.Module]
    encoder_layers: List[nn.Module]

    def __init__(self, num_layers: int, input_channels: int = 3, mid_channels: int = 64, skip_period: int = 2):
        super().__init__()
        self.num_layers = num_layers
        self.half_layers = num_layers // 2
        self.input_channels = input_channels
        self.mid_channels = mid_channels
        self.skip_period = skip_period

        self._build_encoder()
        self._build_decoder()

    def _build_encoder(self):
        self.encoder_layers = []
        self.encoder_layers.append(Conv2dBlock(input_channels=self.input_channels,
                                               output_channels=self.mid_channels,
                                               stride=3, padding=1))
        for _ in range(self.half_layers - 1):
            self.encoder_layers.append(Conv2dBlock(input_channels=self.mid_channels,
                                                   output_channels=self.mid_channels,
                                                   padding=1))

    def _build_decoder(self):
        self.decoder_layers = []
        for _ in range(self.half_layers - 1):
            self.decoder_layers.append(Deconv2dBlock(input_channels=self.mid_channels,
                                                     output_channels=self.mid_channels,
                                                     padding=1))
        self.decoder_layers.append(Deconv2dBlock(input_channels=self.mid_channels,
                                                 output_channels=self.input_channels,
                                                 stride=3, padding=1))

    def forward(self, x):
        skip_data: List[torch.tensor] = []

        for num_layer, layer in enumerate(self.encoder_layers):
            x = layer(x)
            if not (num_layer % self.skip_period) and self.skip_period:
                skip_data.append(x)

        for num_layer, layer in enumerate(self.decoder_layers):
            x = layer(x)
            if not (num_layer % self.skip_period) and self.skip_period:
                x += skip_data.pop()
        return x

