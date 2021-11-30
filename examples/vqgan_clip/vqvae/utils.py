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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def tensor_to_img(
        tensor: torch.Tensor,
        normalize: bool = True,
        input_range: typing.Tuple[float, float] = (-1, 1),
        to_numpy: bool = True,
        rgb2bgr: bool = True
) -> np.array or torch.Tensor:
    """ Decode torch.Tensor to np.array.
    Arguments:
        tensor (torch.Tensor): input tensor.
        normalize (bool): normalize input tensor.
        input_range (typing.Tuple[float, float]): input range.
        to_numpy (bool): convert outputs to np.array.
        rgb2bgr (bool): convert output from RGB to BGR.
    Returns:
        img (np.array or torch.Tensor): decoded image.
    """
    if normalize:
        tensor = torch.clamp(tensor, min=input_range[0], max=input_range[1])
        tensor = (tensor - input_range[0]) / (input_range[1] - input_range[0] + 1e-5)
    img = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    if to_numpy:
        img = img.to('cpu', torch.uint8).numpy()
    if rgb2bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def img_to_tensor(
        img: np.array,
        normalize: bool = True,
        input_range: typing.Tuple[float, float] = (0.0, 255.0),
        bgr2rgb: bool = True
) -> torch.Tensor:
    """ Encode np.array to torch.Tensor.
    Arguments:
        img (np.array): input image.
        size (typing.Optional[int]): target size of the image.
        normalize (bool): normalize input image.
        input_range (typing.Tuple[float, float]): input range.
        bgr2rgb (bool): convert input image from BGR to RGB.
    Returns:
        tensor (torch.Tensor): encoded image.
    """
    if bgr2rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    if normalize:
        tensor = torch.clamp(tensor, min=input_range[0], max=input_range[1])
        tensor = (tensor - input_range[0]) / (input_range[1] - input_range[0] + 1e-5)
    tensor = 2.0 * tensor - 1.0
    return tensor


class STEQuantize(torch.autograd.Function):
    """ Quantize VQVAE embeddings to VQVAE codebook with
    gradients in style of Straight-Through Estimators.
    """
    @staticmethod
    def forward(ctx, embs: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """ Forward path.
        Arguments:
            embs (torch.Tensor): input embeddings.
            codebook (torch.Tensor): VQVAE codebook.
        Returns:
            embs_q (torch.Tensor): quantized embeddings
        """
        d = embs.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * embs @ codebook.T
        indices = d.argmin(-1)
        embs_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
        return embs_q

    @staticmethod
    def backward(ctx, grad_in: torch.Tensor) -> typing.Tuple[torch.Tensor, None]:
        """ Backward path like Straight-Through Estimators.
        Arguments:
            grad_in (torch.Tensor): input gradients.
        Returns:
            grad_out (torch.Tensor): STE gradients.
        """
        return grad_in, None


def ste_quantize(x: torch.Tensor, codebook: torch.tensor) -> torch.Tensor:
    """ Quantize VQVAE embeddings to VQVAE codebook with
    gradients in style of Straight-Through Estimators.
    Arguments:
        embs (torch.Tensor): input embeddings.
        codebook (torch.Tensor): VQVAE codebook.
    Returns:
        embs_q (torch.Tensor): quantized embeddings
    """
    return STEQuantize.apply(x, codebook)
