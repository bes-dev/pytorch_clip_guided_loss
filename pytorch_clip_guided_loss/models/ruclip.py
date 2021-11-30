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
import os
import typing
import torch
import torch.nn as nn
from transformers import CLIPModel
from huggingface_hub import hf_hub_url, cached_download
from pytorch_clip_guided_loss.processor import YTTMTokenizerTextProcessor, TextProcessor, Resize, Normalize
# utils
from omegaconf import OmegaConf


class ruCLIP(nn.Module):
    """ Implementation of CLIP model by SberAI."""
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
        # load ckpt
        for filename in cfg.ckpt.files:
            fileurl = hf_hub_url(repo_id=cfg.ckpt.repo_id, filename=f"{cfg.name}/{filename}")
            cached_download(fileurl, cache_dir=os.path.join(cache_dir, "ruclip"), force_filename=filename)
        model = CLIPModel.from_pretrained(os.path.join(cache_dir, "ruclip"))
        # load text tokenizer
        fileurl = hf_hub_url(repo_id=cfg.tokenizer.repo_id, filename=f"{cfg.name}/{cfg.tokenizer.filename}")
        cached_download(fileurl, cache_dir=os.path.join(cache_dir, "ruclip"), force_filename=cfg.tokenizer.filename)
        text_processor = YTTMTokenizerTextProcessor(
            model_path=os.path.join(cache_dir, "ruclip", cfg.tokenizer.filename),
            target_lang=cfg.tokenizer.target_lang,
            target_length=cfg.tokenizer.target_length
        )
        # load image transforms
        image_processor = nn.Sequential(
            Resize(size=cfg.image.size),
            Normalize(mean=cfg.image.mean, std=cfg.image.std, input_range=input_range)
        )
        return model, text_processor, image_processor
