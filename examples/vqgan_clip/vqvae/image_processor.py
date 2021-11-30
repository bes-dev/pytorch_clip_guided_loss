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
import cv2
import torch
import numpy as np
from .utils import img_to_tensor, tensor_to_img


class OpenCVImageProcessor:
    """ Implementation of image encode/decode for VQVAE. """
    @staticmethod
    def encode(
            img: np.array,
            size: typing.Optional[int] = None,
            normalize: bool = True,
            input_range: typing.Tuple[float, float] = (0.0, 255.0),
            bgr2rgb: bool = True
    ) -> torch.Tensor:
        """ Encode input image.
        Arguments:
            img (np.array): input image.
            size (typing.Optional[int]): target size of the image.
            normalize (bool): normalize input image.
            input_range (typing.Tuple[float, float]): input range.
            bgr2rgb (bool): convert input image from BGR to RGB.
        Returns:
            tensor (torch.Tensor): encoded image.
        """
        if size is not None:
            img = cv2.resize(img, size)
        tensor = img_to_tensor(img, normalize, input_range, bgr2rgb)
        return tensor

    @staticmethod
    def decode(
            tensor: torch.Tensor,
            normalize: bool = True,
            input_range: typing.Tuple[float, float] = (-1, 1),
            to_numpy: bool = True,
            rgb2bgr: bool = True
    ) -> typing.List[np.array or torch.Tensor]:
        """ Encode input tensor (output of the VQVAE decoder).
        Arguments:
            tensor (torch.Tensor): input tensor.
            normalize (bool): normalize input tensor.
            input_range (typing.Tuple[float, float]): input range.
            to_numpy (bool): convert outputs to np.array.
            rgb2bgr (bool): convert output from RGB to BGR.
        Returns:
            imgs (typing.List[np.array or torch.Tensor]): decoded images.
        """
        imgs = []
        for i in range(tensor.size(0)):
            img = tensor_to_img(tensor[i], normalize, input_range, to_numpy, rgb2bgr)
            imgs.append(img)
        return imgs
