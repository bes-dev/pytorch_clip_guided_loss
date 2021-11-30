# -*- coding: utf-8 -*-
from math import sqrt, log
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from taming.modules.diffusionmodules.model import Encoder, Decoder
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from .image_processor import OpenCVImageProcessor


class GumbelQuantize(nn.Module):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, num_hiddens, embedding_dim, n_embed, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed
        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight
        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(self.n_embed, self.embedding_dim)
        self.use_vqinterface = use_vqinterface

    def forward(self, z, temp=None, return_logits=False):
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp
        logits = self.proj(z)
        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)
        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()
        ind = soft_one_hot.argmax(dim=1)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind


class GumbelVQ(nn.Module):
    def __init__(self, ddconfig, n_embed, embed_dim, kl_weight=1e-8, **kwargs):
        super().__init__()
        z_channels = ddconfig['z_channels']
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = GumbelQuantize(z_channels, embed_dim, n_embed=n_embed, kl_weight=kl_weight, temp_init=1.0)
        self.quant_conv = torch.nn.Conv2d(ddconfig['z_channels'], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig['z_channels'], 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def img_to_ids(self, img):
        _, _, [_, _, indices] = self.encode(img)
        return rearrange(indices, 'b h w -> b (h w)')

    def ids_to_embs(self, ids):
        b, n = ids.shape
        one_hot = F.one_hot(ids, num_classes=self.quantize.n_embed).float()
        embs = (one_hot @ self.quantize.embed.weight)
        embs = rearrange(embs, 'b (h w) c -> b c h w', h = int(sqrt(n)))
        return embs

    def embs_to_img(self, embs):
        img = self.decode(embs)
        return img

    @classmethod
    def from_pretrained(cls, cfg):
        print(f"[{cls.__name__}]: create model")
        model = cls(**cfg.params.model.params)
        print(f"[{cls.__name__}]: load checkpoint {cfg.ckpt}")
        model.load_state_dict(
            torch.load(hf_hub_download(**cfg.ckpt), map_location="cpu")["state_dict"],
            strict=False
        )
        print(f"[{cls.__name__}]: create processor")
        model_preprocess = OpenCVImageProcessor()
        return model, model_preprocess
