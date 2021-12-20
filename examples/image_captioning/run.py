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
import typing
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_clip.processor.text_processor import TextProcessor
from pytorch_clip_guided_loss import get_clip_guided_loss
from tqdm.autonotebook import tqdm
# image transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class STEQuantize(torch.autograd.Function):
    """ Quantize embeddings to codebook with
    gradients in style of Straight-Through Estimators.
    """
    @staticmethod
    def forward(ctx, embs: torch.Tensor, codebook: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """ Forward path.
        Arguments:
            embs (torch.Tensor): input embeddings.
            codebook (torch.Tensor): codebook.
        Returns:
            embs_q (torch.Tensor): quantized embeddings
        """
        d = embs.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * embs @ codebook.T
        indices = d.argmin(-1)
        embs_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
        return embs_q, indices

    @staticmethod
    def backward(ctx, grad_in: torch.Tensor, grad_ids: torch.Tensor) -> typing.Tuple[torch.Tensor, None]:
        """ Backward path like Straight-Through Estimators.
        Arguments:
            grad_in (torch.Tensor): input gradients.
        Returns:
            grad_out (torch.Tensor): STE gradients.
        """
        return grad_in, None

def ste_quantize(x: torch.Tensor, codebook: torch.tensor) -> torch.Tensor:
    """ Quantize embeddings to codebook with
    gradients in style of Straight-Through Estimators.
    Arguments:
        embs (torch.Tensor): input embeddings.
        codebook (torch.Tensor): codebook.
    Returns:
        embs_q (torch.Tensor): quantized embeddings
    """
    return STEQuantize.apply(x, codebook)


class MaskedGrad(torch.autograd.Function):
    """ Apply masked gradients
    """
    @staticmethod
    def forward(ctx, var: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ Forward path.
        Arguments:
            var (torch.Tensor): input variable.
            mask (torch.Tensor): mask of the gradient.
        Returns:
            var (torch.Tensor): input variable.
        """
        ctx.save_for_backward(mask)
        return var

    @staticmethod
    def backward(ctx, grad_in: torch.Tensor) -> typing.Tuple[torch.Tensor, None]:
        """ Backward path returns masked gradient for variable.
        Arguments:
            grad_in (torch.Tensor): input gradients.
        Returns:
            grad_out (torch.Tensor): masked gradients.
        """
        mask, = ctx.saved_tensors
        grad_out = grad_in * mask
        return grad_out, None


def masked_grad(var: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """ Apply masked gradients.
    Arguments:
        var (torch.Tensor): input variable.
        mask (torch.Tensor): mask of the gradient.
    Returns:
        var (torch.Tensor): input variable.
    """
    return MaskedGrad.apply(var, mask)


def init_params(
        tokenizer: TextProcessor,
        length_min: int,
        length_max: int,
        dictionary: nn.Module,
        device: str
) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Initialize random embeddings.
    Arguments:
        tokenizer (TextProcessor): text tokenizer.
        length_min (int): minimum length of the target text.
        length_max (int): maximum length of the target text.
        dictionary (nn.Module): dictionary of the embeddings.
        device (str): target device.
    Returns:
        embeds (torch.Tensor): random embeddings.
        attention_mask (torch.Tensor): attention mask.
        gradient_mask (torch.Tensor): gradient mask.
    """
    length_min = min(length_min, tokenizer.get_max_length() - 2)
    length_max = min(length_max, tokenizer.get_max_length() - 2)
    vocab_size = tokenizer.vocab_size()
    bos_emb = dictionary(torch.LongTensor([[tokenizer.bos_id]]).to(device))
    eos_emb = dictionary(torch.LongTensor([[tokenizer.eos_id]]).to(device))
    pad_emb = dictionary(torch.LongTensor([[tokenizer.pad_id]]).to(device))

    embeds = []
    attention_masks = []
    grad_masks = []
    for l in range(length_min, length_max + 1):
        ids = torch.randint(0, vocab_size, (1, l), device=device)
        embed = dictionary(ids)
        embed = torch.cat([bos_emb, embed, eos_emb, pad_emb.repeat_interleave(tokenizer.get_max_length() - 2 - l, dim=1)], dim=1)
        embeds.append(embed)
        attention_masks.append(
            torch.LongTensor([1] * (l + 2) + [0] * (tokenizer.get_max_length() - 2 - l)).unsqueeze(0)
        )
        grad_masks.append(
            torch.LongTensor([0] + [1] * l + [0] * (tokenizer.get_max_length() - 1 - l)).unsqueeze(0)
        )
    embeds = torch.cat(embeds, dim=0).detach()
    embeds.requires_grad = True
    attention_masks = torch.cat(attention_masks, dim=0).to(device)
    grad_masks = torch.cat(grad_masks, dim=0).unsqueeze(-1).to(device)
    return embeds, attention_masks, grad_masks


def remove_repeats(strings: typing.List[str]) -> typing.List[str]:
    """ remove repeated words and sentences
    Arguments:
        strings (typing.List[str]): input strings.
    Returns:
        strings (typing.List[str]): output strings.
    """
    out = []
    for s in strings:
        words = s.split()
        if len(words):
            output_words = [words[0]]
            for i in range(1, len(words)):
                if output_words[-1] != words[i]:
                   output_words.append(words[i])
            s_out = " ".join(output_words)
            if not s_out in out:
                out.append(s_out)
    return out


def main(args):
    # load model
    clip_guided_loss = get_clip_guided_loss(args.clip_type, input_range = (0, 1))
    img_transforms = A.Compose([
        ToTensorV2()
    ])
    tokenizer = clip_guided_loss.text_processor
    dictionary = clip_guided_loss.model.get_text_dictionary()
    # model to inference device
    clip_guided_loss.to(args.device)
    # initialize prompt
    image = (img_transforms(image=cv2.imread(args.img_path))["image"].unsqueeze(0) / 255.0).to(args.device)
    clip_guided_loss.add_prompt(image=image)
    # initialize text
    embeds, attention_mask, grad_mask = init_params(
        tokenizer,
        args.length_min,
        args.length_max,
        dictionary,
        args.device
    )
    # initilize valid range for embeddings
    range_min = dictionary.weight.min(dim=0).values[None, None, :]
    range_max = dictionary.weight.max(dim=0).values[None, None, :]
    # initialize optimizer
    opt = torch.optim.Adam([embeds], lr=args.lr)
    # start optimization
    iterator = tqdm(range(args.n_steps))
    for i in iterator:
        opt.zero_grad()
        x = masked_grad(embeds, grad_mask)
        x, ids = ste_quantize(x, dictionary.weight)
        loss = clip_guided_loss.text_loss(
            input_ids=ids,
            attention_mask=attention_mask,
            embed=x
        )["loss"]
        loss.backward()
        opt.step()
        with torch.inference_mode():
            embeds.copy_(embeds.maximum(range_min).minimum(range_max))
        iterator.set_description(f"loss: {loss.item()}")
    # print outputs
    x, ids = ste_quantize(embeds, dictionary.weight)
    strings = remove_repeats(tokenizer.decode(ids))
    input_ids, attention_mask = [], []
    for s in strings:
        out = tokenizer.encode(s, return_mask=True)
        input_ids.append(out["input_ids"])
        attention_mask.append(out["attention_mask"])
    input_ids = torch.cat(input_ids, dim=0).to(args.device)
    attention_mask = torch.cat(attention_mask, dim=0).to(args.device)
    embeds = dictionary(input_ids)
    loss = clip_guided_loss.text_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        embed=embeds,
        reduce=None
    )["loss"]
    print(f"best caption: {strings[loss.argmin()]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0", help="inference device.")
    parser.add_argument("--clip-type", type=str, default="ruclip", help="Type of CLIP model [clip, ruclip].")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--n-steps", type=int, default=100, help="Number steps of optimization.")
    parser.add_argument("--length-min", type=int, default=10, help="Minimum sequence length")
    parser.add_argument("--length-max", type=int, default=32, help="Maximum sequence length")
    parser.add_argument("--img-path", type=str, default=None, help="Path to input image")
    args = parser.parse_args()
    main(args)
