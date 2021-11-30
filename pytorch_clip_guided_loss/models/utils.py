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
from pytorch_clip_guided_loss.processor import TextProcessor
from omegaconf import OmegaConf
from .clip import CLIP
from .ruclip import ruCLIP


def get_clip_model(cfg: OmegaConf, input_range: typing.Tuple[float, float] = (-1.0, 1.0), cache_dir: str = "/tmp/") -> typing.Tuple[nn.Module, TextProcessor, nn.Module]:
    """Build CLIP model from config file.
    Arguments:
        cfg (OmegaConf): configuration of the model.
        input_range (tuple[float, float]): input range.
        cache_dir (str): path to cache dir.
    Returns:
        model (nn.Module): CLIP model.
        text_processor (TextProcessor): text processor.
        image_processor (nn.Module): image processor.
    """
    if cfg.type == "ruclip":
        return ruCLIP.from_pretrained(cfg, input_range, cache_dir=cache_dir)
    elif cfg.type == "clip":
        return CLIP.from_pretrained(cfg, input_range, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown model type: {cfg.type}. Available model types: [clip, ruclip]")
