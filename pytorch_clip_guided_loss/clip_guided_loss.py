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
# clip model
from pytorch_clip import get_clip_model
from pytorch_clip.processor.text_processor import TextProcessor
# utils
from omegaconf import OmegaConf


class CLIPPrompt(nn.Module):
    """ Implementation of CLIP prompt
    Arguments:
        embed (torch.Tensor): input embedding.
        weight (float): importance of the prompt.
        src (torch.Tensor or str): source data of the prompt.
    """
    def __init__(
            self,
            embed: torch.Tensor,
            weight: float,
            src: typing.Optional[torch.Tensor or str] = None
    ):
        super().__init__()
        self.register_buffer("embed", embed)
        self.register_buffer("weight", torch.as_tensor(weight))
        if isinstance(src, torch.Tensor):
            self.register_buffer("src", src)
        else:
            self.src = src

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spherical distance loss between prompt and input embedding.
        Arguments:
            x (torch.Tensor): input embedding.
        Returns:
            loss (torch.Tensor): output spherical loss.
        """
        return self.weight * x.sub(self.embed).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


class CLIPGuidedLoss(nn.Module):
    """ Implementation of CLIP guided loss function.
    Arguments:
        model (nn.Module): CLIP model.
        text_processor (TextProcessor): text processor.
        image_processor (nn.Module): image processor.
    """
    def __init__(self, model: nn.Module, text_processor: TextProcessor, image_processor: nn.Module):
        super().__init__()
        # clip model
        self.model = model
        self.text_processor = text_processor
        self.image_processor = image_processor
        # prompts
        self.prompts = nn.ModuleDict()
        # device info
        self.register_buffer("device_info", torch.tensor(1))

    def add_prompt(
            self,
            image: typing.Optional[torch.Tensor] = None,
            text: typing.Optional[str] = None,
            weight: float = 1.0,
            label: typing.Optional[str] = None,
            store_src: bool = True
    ) -> str:
        """Add prompt to loss function.
        Arguments:
            image (torch.Tensor): input image [Optional].
            text (str): input text [Optional].
            weight (float): importance of the prompt.
            label (str): label of the prompt [Optional].
            store_src (bool): store source data of the prompt.
        Returns:
            label (src): label of the prompt.
        """
        if text is None and image is None:
            return
        embed, src = self._get_embed(image, text)
        if label is None:
            label = str(len(self.prompts))
        self.prompts[label] = CLIPPrompt(
            embed = embed.detach(),
            weight = weight,
            src = src if store_src else None
        ).to(self.device_info.device)
        return label

    def delete_prompt(self, label: typing.Optional[str] = None) -> None:
        """Add prompt to loss function.
        Arguments:
            label (str): label of the prompt to delete [Optional].
        """
        if label in self.prompts:
            self.prompts.pop(label)

    def clear_prompts(self) -> None:
        """Delete all available prompts."""
        self.prompts.clear()

    def get_prompts_list(self) -> typing.List[str]:
        """Get list of all available prompts.
        Returns:
            prompts (list<str>): list of prompts labels.
        """
        return list(self.prompts.keys())

    def get_prompt(self, label: str) -> typing.Optional[CLIPPrompt]:
        """Get prompt if available.
        Arguments:
            label (str): label of the prompt [Optional].
        Returns:
            prompt (CLIPPrompt or None): prompt [Optional].
        """
        if label in self.prompts:
            return self.prompts[label]

    def forward(self, embed: typing.Optional[torch.Tensor], reduce: str = "mean") -> torch.Tensor:
        """Compute CLIP guided loss between input image/text and all available prompts.
        Arguments:
            embed (torch.Tensor): input embedding.
            reduce (str): reduce mode ("mean", "sum" or None)
        Returns:
            loss (torch.Tensor): CLIP guided loss.
        """
        assert reduce in ["mean", "sum", None]
        loss = {}
        for key, prompt in self.prompts.items():
            loss[key] = prompt(embed)
            if reduce is not None:
                loss[key] = loss[key].mean() if reduce == "mean" else loss[key].sum()
        loss["loss"] = sum(loss.values()) if len(loss) else 0
        return loss

    def image_loss(self, image: typing.Optional[torch.Tensor], reduce: str="mean") -> torch.Tensor:
        """Compute CLIP guided loss between input image and all available prompts.
        Arguments:
            image (torch.Tensor): input image.
            reduce (str): reduce mode ("mean", "sum" or None)
        Returns:
            loss (torch.Tensor): CLIP guided loss.
        """
        embed, _ = self._get_embed(image=image)
        return self(embed, reduce=reduce)

    def text_loss(self,
                  input_ids: typing.Optional[torch.Tensor],
                  embed: typing.Optional[torch.Tensor],
                  attention_mask: typing.Optional[torch.Tensor] = None,
                  reduce: str = "mean"
    ) -> torch.Tensor:
        """Compute CLIP guided loss between input text and all available prompts.
        Arguments:
            input_ids (torch.Tensor): input text tokens indices.
            embed (torch.Tensor): input text embeddings.
            attention_mask (torch.Tensor): attention mask [Optional].
            reduce (str): reduce mode ("mean", "sum" or None)
        Returns:
            loss (torch.Tensor): CLIP guided loss.
        """
        embed = self.model.get_text_features(
            input_ids = input_ids,
            attention_mask = attention_mask,
            inputs_embeds = embed
        )
        return self(embed, reduce=reduce)

    def _get_embed(
            self,
            image: typing.Optional[torch.Tensor] = None,
            text: typing.Optional[typing.Dict[str, torch.Tensor] or str] = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor or typing.Dict[str, torch.Tensor] or str]:
        """Compute CLIP embedding for input data.
        Arguments:
            image (torch.Tensor): input image [Optional].
            text (str): input text [Optional].
        Returns:
            embed (torch.Tensor): output embedding.
        """
        if image is None and text is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")
        if image is not None:
            embed = self.model.get_image_features(self.image_processor(image))
            src = image
        else:
            if isinstance(text, str):
                text = self.text_processor.encode(text, return_mask=True)
                text["input_ids"] = text["input_ids"].view(1, -1).to(self.device_info.device)
                if "attention_mask" in text:
                    text["attention_mask"] = text["attention_mask"].view(1, -1).to(self.device_info.device)
            embed = self.model.get_text_features(**text)
            src = text
        embed = F.normalize(embed, dim=-1)
        return embed, src
