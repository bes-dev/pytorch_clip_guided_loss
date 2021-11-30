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
import torch
import numpy as np


def tokens_to_tensor(tokens: np.array, target_length: int, pad_id: int) -> torch.Tensor:
    """Convert tokens to torch.Tensor.
    Arguments:
        tokens (np.array): input tokens.
        target_length (int): target length.
        pad_id (int): padding symbol.
    Returns:
        tokens (torch.Tensor): tokens padded if need.
    """
    pad_size = target_length - len(tokens)
    if pad_size > 0:
        tokens = np.hstack((tokens, np.full(pad_size, pad_id)))
    if len(tokens) > target_length:
        tokens = tokens[:target_length]
    return torch.Tensor(tokens).long()
