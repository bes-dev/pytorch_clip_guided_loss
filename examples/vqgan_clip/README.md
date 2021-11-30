# VQGAN-CLIP Text-To-Image pipeline

Tiny implementation of the Text-To-Image pipeline using VQVAE by SberAI and pytorch_clip_guided_loss library.

## Usage

### Install requirements

```bash
$ pip install -r requirements.txt
```

### Generate image from text

```bash
$ python run.py --help

usage: run.py [-h] [--device DEVICE] [-t TEXT] [--cfg CFG] [--clip-type CLIP_TYPE] [--output-size OUTPUT_SIZE] [--output-name OUTPUT_NAME] [--lr LR] [--n-steps N_STEPS] [--batch-size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE       inference device.
  -t TEXT, --text TEXT  Text prompt.
  --cfg CFG             Path to VQVAE config.
  --clip-type CLIP_TYPE
                        Type of CLIP model [clip, ruclip].
  --output-size OUTPUT_SIZE
                        Size of the output image.
  --output-name OUTPUT_NAME
                        Name of the output image.
  --lr LR               Learning rate.
  --n-steps N_STEPS     Number steps of optimization.
  --batch-size BATCH_SIZE
                        Batch size.
```

```bash
$ python run.py --text "A painting in the style of Picasso" --output-name output.png
```