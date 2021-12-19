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
import torch.nn as nn
from omegaconf import OmegaConf
from .utils import res_path
from .clip_guided_loss import CLIPGuidedLoss
from pytorch_clip import get_clip_model, get_models_list


def get_clip_guided_loss(
        clip_type: str = "clip",
        input_range: typing.Tuple[float, float] = (-1.0, 1.0),
        cache_dir: str = "/tmp/"
) -> nn.Module:
    """get CLIPGuidedloss model.
    Arguments:
        clip_typle (str): type of the CLIP model ("clip" or "ruclip").
        input_range (tuple[float, float]): input range.
        cache_dir (str): path to cache dir.
    Returns:
        model (nn.Module): CLIPGuidedloss model.
    """
    clip, tokenizer, transforms, cfg = get_clip_model(clip_type, input_range, cache_dir)
    return CLIPGuidedLoss(clip, tokenizer, transforms)
