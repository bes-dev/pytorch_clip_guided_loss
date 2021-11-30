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
import argparse
import cv2
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import kornia.augmentation as K
from vqvae import GumbelVQ, ste_quantize
from pytorch_clip_guided_loss import get_clip_guided_loss
from tqdm.autonotebook import tqdm


def main(args):
    # load model
    vqvae, vqvae_processor = GumbelVQ.from_pretrained(OmegaConf.load(args.cfg))
    clip_guided_loss = get_clip_guided_loss(args.clip_type, input_range = (0, 1))
    # model to inference device
    vqvae.to(args.device)
    clip_guided_loss.to(args.device)
    # initialize prompt
    clip_guided_loss.add_prompt(text=args.text)
    # initialize image
    n_toks = args.output_size // 2 ** (vqvae.decoder.num_resolutions - 1)
    z = vqvae.ids_to_embs(
        torch.randint(vqvae.quantize.n_embed, (1, n_toks * n_toks)).to(args.device)
    ).detach().requires_grad_(True)
    # initialize augmentations
    augs = nn.Sequential(
        K.RandomAffine(degrees=15, translate=0.1, shear=5, p=0.7, padding_mode='zeros', keepdim=True),
        K.RandomPerspective(distortion_scale=0.7, p=0.7),
        K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.7),
        K.RandomErasing(scale=(.1, .4), ratio=(.3, 1/.3), same_on_batch=True, p=0.7)
    )
    # initialize optimizer
    opt = torch.optim.AdamW([z], lr=args.lr)
    # initilize valid range for embeddings
    z_min = vqvae.quantize.embed.weight.min(dim=0).values[None, :, None, None]
    z_max = vqvae.quantize.embed.weight.max(dim=0).values[None, :, None, None]
    # start optimization
    iterator = tqdm(range(args.n_steps))
    for i in iterator:
        opt.zero_grad()
        x = ste_quantize(z.movedim(1, 3), vqvae.quantize.embed.weight).movedim(3, 1)
        x = vqvae.embs_to_img(x).add(1).div(2).clamp(0, 1)
        x = x.repeat_interleave(args.batch_size, dim=0)
        x = augs(x)
        loss = clip_guided_loss(image = x)["loss"]
        loss.backward()
        opt.step()
        with torch.inference_mode():
            z.copy_(z.maximum(z_min).minimum(z_max))
        iterator.set_description(f"loss: {loss.item()}")
    # save image
    z = ste_quantize(z.movedim(1, 3), vqvae.quantize.embed.weight).movedim(3, 1)
    cv2.imwrite(args.output_name, vqvae_processor.decode(vqvae.embs_to_img(z), rgb2bgr=True)[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="inference device.")
    parser.add_argument("-t", "--text", type=str, default=None, help="Text prompt.")
    parser.add_argument("--cfg", type=str, default="configs/vqvae.yaml", help="Path to VQVAE config.")
    parser.add_argument("--clip-type", type=str, default="ruclip", help="Type of CLIP model [clip, ruclip].")
    parser.add_argument("--output-size", type=int, default=256, help="Size of the output image.")
    parser.add_argument("--output-name", type=str, default="output.png", help="Name of the output image.")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--n-steps", type=int, default=100, help="Number steps of optimization.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    args = parser.parse_args()
    main(args)
