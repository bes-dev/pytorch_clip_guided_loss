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
from transformers import CLIPModel, CLIPProcessor
from pytorch_clip_guided_loss.processor import HFCLIPTextProcessor, TextProcessor, Resize, Normalize
# utils
from omegaconf import OmegaConf


class CLIP(nn.Module):
    """ Implementation of CLIP model by OpenAI."""
    @staticmethod
    def from_pretrained(
            cfg: OmegaConf,
            input_range: typing.Tuple[float, float] = (-1.0, 1.0),
            cache_dir: str = "/tmp/"
    ) -> typing.Tuple[nn.Module, TextProcessor, nn.Module]:
        """Build model from pre-trained checkpoint.
        Arguments:
            cfg (OmegaConf): configuration of the model.
            input_range (tuple[float, float]): input range.
            cache_dir (str): path to cache dir.
        Returns:
            model (nn.Module): CLIP model.
            text_processor (TextProcessor): text processor.
            image_processor (nn.Module): image processor.
        """
        model = CLIPModel.from_pretrained(cfg.model.config)
        processor = CLIPProcessor.from_pretrained(cfg.processor.config)
        text_processor = HFCLIPTextProcessor(processor.tokenizer, target_lang=cfg.tokenizer.target_lang)
        # load image transforms
        image_processor = nn.Sequential(
            Resize(size=processor.feature_extractor.size),
            Normalize(
                mean=processor.feature_extractor.image_mean,
                std=processor.feature_extractor.image_std,
                input_range=input_range
            )
        )
        return model, text_processor, image_processor
