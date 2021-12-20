# Simple CLIP guided image captioning

Simple gradient based CLIP guided image captioning.

## Usage

### Install requirements

```bash
$ pip install -r requirements.txt
```

### Generate image caption

```bash
$ python run.py --help

usage: run.py [-h] [--device DEVICE] [--clip-type CLIP_TYPE] [--lr LR] [--n-steps N_STEPS] [--length-min LENGTH_MIN] [--length-max LENGTH_MAX] [--img-path IMG_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE       inference device.
  --clip-type CLIP_TYPE
                        Type of CLIP model [clip, ruclip].
  --lr LR               Learning rate.
  --n-steps N_STEPS     Number steps of optimization.
  --length-min LENGTH_MIN
                        Minimum sequence length
  --length-max LENGTH_MAX
                        Maximum sequence length
  --img-path IMG_PATH   Path to input image
```

```bash
$ python run.py --img-path <path_to_img> --length-min 1 --length-max 32 --clip-type ruclip
```