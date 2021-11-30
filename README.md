# pytorch_clip_guided_loss: Pytorch implementation of the CLIP guided loss for Text-To-Image, Image-To-Image, or Image-To-Text generation.

A simple library that implements CLIP guided loss in PyTorch.

<p align="center">
  <img src="resources/preview.png"/>
</p>

## Install package

```bash
pip install pytorch_clip_guided_loss
```

## Install the latest version

```bash
pip install --upgrade git+https://github.com/bes-dev/pytorch_clip_guided_loss.git
```

## Features
- The library supports multiple prompts (images or texts) as targets for optimization.
- The library automatically detects the language of the input text, and multilingual translate it via google translate.
- The library supports the original CLIP model by OpenAI and ruCLIP model by SberAI.

## Usage

### Simple code

```python
import torch
from pytorch_clip_guided_loss import get_clip_guided_loss

loss_fn = get_clip_guided_loss(clip_type="ruclip", input_range = (-1, 1)).eval().requires_grad_(False)
# text prompt
loss_fn.add_prompt(text="text description of the what we would like to generate")
# image prompt
loss_fn.add_prompt(image=torch.randn(1, 3, 224, 224))

# variable
var = torch.randn(1, 3, 224, 224).requires_grad_(True)
loss = loss_fn(image=var)["loss"]
loss.backward()
print(var.grad)
```

### VQGAN-CLIP

We provide our tiny implementation of the VQGAN-CLIP pipeline for image generation as an example of the usage of our library.
To start using our implementation of the VQGAN-CLIP please follow by [documentation](examples/vqgan_clip).