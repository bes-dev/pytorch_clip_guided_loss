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
from clip_rcnn import ClipRCNN


def main(args):
    # build detector
    detector = ClipRCNN(
        scale=args.scale,
        sigma=args.sigma,
        min_size=args.min_size,
        aspect_ratio=[float(r) for r in args.aspect_ratio.split(",")],
        clip_type=args.clip_type,
        batch_size=args.batch_size,
        top_k=args.top_k
    )
    # add prompts
    if args.text_prompt is not None:
        detector.add_prompt(text=args.text_prompt)
    if args.image_prompt is not None:
        image = cv2.cvtColor(cv2.imread(args.image_prompt), cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image = img / 255.0
        detector.add_prompt(image=image)
    image = cv2.imread(args.image)
    boxes = detector.detect(image)
    for box in boxes:
        x, y, w, h = box["rect"]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)
    if args.output_image is None:
        cv2.imshow("image", images)
        cv2.waitKey()
    else:
        cv2.imwrite(args.output_image, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="Input image.")
    parser.add_argument("--device", type=str, default="cuda:0", help="inference device.")
    parser.add_argument("--text-prompt", type=str, default=None, help="Text prompt.")
    parser.add_argument("--image-prompt", type=str, default=None, help="Image prompt.")
    parser.add_argument("--clip-type", type=str, default="ruclip", help="Type of CLIP model [clip, ruclip].")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--scale", type=int, default=500, help="Scale (selective search).")
    parser.add_argument("--sigma", type=float, default=0.9, help="Sigma (selective search).")
    parser.add_argument("--min-size", type=float, default=0.05, help="Minimum area of the region proposal (selective search).")
    parser.add_argument("--aspect-ratio", type=str, default="0.5,1.5", help="Aspect ratio (selective search).")
    parser.add_argument("--top-k", type=int, default=1, help="top k predictions will be return.")
    parser.add_argument("--output-image", type=str, default=None, help="Output image name.")
    args = parser.parse_args()
    main(args)
