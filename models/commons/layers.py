import torch
import torch.nn as nn

from typing import List
from typing import Tuple
from typing import Union


class BaseBlock(nn.Module):
    layers: List[nn.Module]
    network: nn.Module

    def __init__(self, output_channels: int, input_channels: int = 3, kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0,
                 add_batch_norm: bool = True):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.add_batch_norm = add_batch_norm

        self._build_network()

    def _build_network(self):
        self.layers = []
        self._add_conv(output_channels=self.output_channels, input_channels=self.input_channels,
                       kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        if self.add_batch_norm:
            self._add_batch_norm(num_channels=self.output_channels)
        self._add_relu()
        self.network = nn.Sequential(*self.layers)

    def _add_conv(self, output_channels: int, input_channels: int, kernel_size: Union[int, Tuple[int, int]],
                  stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]]):
        raise NotImplementedError()

    def _add_relu(self):
        self.layers.append(nn.ReLU())

    def _add_batch_norm(self, num_channels):
        self.layers.append(nn.BatchNorm2d(num_channels))

    def forward(self, x):
        return self.network(x)


class Conv2dBlock(BaseBlock):
    def _add_conv(self, output_channels: int, input_channels: int, kernel_size: Union[int, Tuple[int, int]],
                  stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]]):
        self.layers.append(nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                                     kernel_size=kernel_size, stride=stride, padding=padding))


class Deconv2dBlock(BaseBlock):
    def _add_conv(self, output_channels: int, input_channels: int, kernel_size: Union[int, Tuple[int, int]],
                  stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]]):
        self.layers.append(nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels,
                                              kernel_size=kernel_size, stride=stride, padding=padding))
