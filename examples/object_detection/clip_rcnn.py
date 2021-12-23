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
from typing import List, Dict, Tuple, Optional
import cv2
import selectivesearch
import torch
import torch.nn as nn
import numpy as np
from pytorch_clip_guided_loss import get_clip_guided_loss


class ClipRCNN(nn.Module):
    """ Implementation of the CLIP guided object detection model.
    Model is based on Selective Search region proposals and CLIP
    guided loss to make text/image driven object detection.
    Arguments:
        scale (int): Free parameter. Higher means larger clusters in felzenszwalb segmentation.
        sigma (float): Width of Gaussian kernel for felzenszwalb segmentation.
        min_size (int): Minimum component size for felzenszwalb segmentation.
        aspect_ratio (Tuple[float, float]): valid range of aspect ratios for region proposals.
        clip_type (str): type of the CLIP model.
        batch_size (int): batch size.
        top_k (int): top k predictions will be return.
    """
    def __init__(
            self,
            scale: int = 500,
            sigma: float = 0.9,
            min_size: float = 0.1,
            aspect_ratio: Tuple[float, float] = (0.5, 1.5),
            clip_type: str = "ruclip",
            batch_size: int = 128,
            top_k: int = 1
    ):
        super().__init__()
        # selective search parameters
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size
        self.aspect_ratio = aspect_ratio
        # inference params
        self.batch_size = batch_size
        # output params
        self.top_k = top_k
        # CLIP guided loss
        self.clip_loss = get_clip_guided_loss(clip_type, input_range=(0.0, 1.0))
        self.input_size = self.clip_loss.image_processor[0].size
        # utils
        self.register_buffer("device_info", torch.tensor(0))

    def add_prompt(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[str] = None,
            weight: float = 1.0,
            label: Optional[str] = None,
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
        return self.clip_loss.add_prompt(image, text, weight, label, store_src)

    def clear_prompts(self) -> None:
        """Delete all available prompts."""
        return self.clip_loss.clear_prompts()

    @torch.no_grad()
    def detect(self, img: np.array) -> List[Dict]:
        """ Detect objects on the input image using CLIP guided prompts.
        Argument:
            img (np.array): input image.
        Returns:
            outputs (List[Dict]): predicts in format:
                                  [{"rect": [x, y, w, h], "loss": loss_val}]
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # generate proposals by selective search
        proposals = self._generate_proposals(img_rgb)
        if not len(proposals):
            return []
        batch = self._prepare_batch(img_rgb, proposals).to(self.device_info.device)
        # predict CLIP loss
        loss = self._predict_clip_loss(batch)
        outputs = self._generate_output(proposals, loss)
        return outputs

    def _generate_proposals(self, img: np.array) -> List[Tuple[int, int, int, int]]:
        """ Generate region proposals using selective search algorithm.
        Argument:
            img (np.array): input image.
        Returns:
            proposals (List[Tuple[int, int, int, int]]): output proposals in format [(x, y, w, h)]
        """
        min_size = int(img.shape[0] * img.shape[1] * self.min_size)
        # generate proposals
        img_lbl, regions = selectivesearch.selective_search(
            img, scale=self.scale, sigma=self.sigma, min_size=min_size
        )
        # filter by aspect ratio
        proposals = []
        for region in regions:
            x, y, w, h = region["rect"]
            aspect_ratio = float(w) / float(h)
            if aspect_ratio > self.aspect_ratio[0] and aspect_ratio < self.aspect_ratio[1]:
                proposals.append([x, y, w, h])
        return proposals

    def _prepare_batch(self, img: np.array, proposals: List[Tuple[int, int, int, int]]) -> torch.Tensor:
        """ Crop region proposals and generate batch
        Argument:
            img (np.array): input image.
            proposals (List[Tuple[int, int, int, int]]): output proposals in format [(x, y, w, h)]
        Returns:
            batch (torch.Tensor): output batch (B, C, H, W).
        """
        batch = []
        for x, y, w, h in proposals:
            crop = cv2.resize(img[y:y+h, x:x+w], self.input_size)
            batch.append(torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0))
        batch = torch.cat(batch, dim=0)
        # normalize batch
        batch = batch / 255.0
        return batch

    def _predict_clip_loss(self, batch_full: torch.Tensor) -> torch.Tensor:
        """ Predict CLIP loss for region proposals using user's prompts.
        Argument:
            batch_full (torch.Tensor): input batch (B, C, H, W).
        Returns:
            loss (torch.Tensor): output batch (B, ).
        """
        loss = []
        id_start = 0
        while id_start < batch_full.size(0):
            id_stop = min(id_start + self.batch_size, batch_full.size(0))
            batch = batch_full[id_start:id_stop]
            loss.append(self.clip_loss.image_loss(image=batch, reduce=None)["loss"].cpu())
            id_start = id_stop
        loss = torch.cat(loss, dim=0)
        return loss

    def _generate_output(self, proposals: List[Tuple[int, int, int, int]], loss: torch.Tensor) -> List[Dict]:
        """ Generate top_k predictions as an output of the model.
        Argument:
            proposals (List[Tuple[int, int, int, int]]): output proposals in format [(x, y, w, h)]
            loss (torch.Tensor): output batch (B, ).
        Returns:
            outputs (List[Dict]): predicts in format:
                                  [{"rect": [x, y, w, h], "loss": loss_val}]
        """
        output = []
        vals, ids = loss.sort()
        top_k = min(self.top_k, len(proposals))
        for i in range(top_k):
            output.append({
                "rect": proposals[ids[i]],
                "loss": vals[i]
            })
        return output
