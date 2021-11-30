"""
Copyright 2021 by Sergei Belousov
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalize(nn.Module):
    """ Implementation of differentiable normalization of the input tensor
    Arguments:
        mean (list<float>): per channel mean.
        std (list<float>): per channel mean.
        input_range (tuple<float, float>): range of the input values.

    """
    def __init__(
            self,
            mean: typing.List[float],
            std: typing.List[float],
            input_range: typing.Tuple[float, float] = (-1.0, 1.0)
    ):
        super().__init__()
        # input range
        self.input_range = input_range
        # project input_range -> [0, 1]
        range_shift = input_range[0]
        range_scale = input_range[1] - input_range[0] + 1e-5
        # prepare mean
        mean = torch.Tensor(mean).view(1, -1, 1, 1)
        mean = range_shift + range_scale * mean
        self.register_buffer("mean", mean)
        # prepare std
        std = torch.Tensor(std).view(1, -1, 1, 1)
        std = range_scale * std
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor.
        Arguments:
            x (torch.Tensor): input tensor.
        Returns:
            y (torch.Tensor): normalized tensor.
        """
        # clamp range
        x = torch.clamp(x, self.input_range[0], self.input_range[1])
        # normalize
        x = (x - self.mean) / (self.std + 1e-5)
        return x


class Resize(nn.Module):
    """ Implementation of differentiable resize of the input tensor
    Arguments:
        size (int or tuple<int, int>): output size.
        mode (str): interpolation mode (default = "bilinear").

    """
    def __init__(self, size: int or typing.Tuple[int, int], mode: str = "bilinear"):
        super().__init__()
        self.size = size if isinstance(size, tuple) else (size, size)
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Resize input tensor.
        Arguments:
            x (torch.Tensor): input tensor.
        Returns:
            y (torch.Tensor): resized tensor.
        """
        if x.size(2) == self.size[0] and x.size(3) == self.size[1]:
            return x
        return F.interpolate(x, size=self.size, mode=self.mode)
