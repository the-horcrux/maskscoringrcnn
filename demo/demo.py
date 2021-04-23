import argparse
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time
import requests
from io import BytesIO
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../e2e_ms_rcnn_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=224,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    def load(url):
        """
        Given an url of an image, downloads the image and
        returns a PIL image
        """
        response = requests.get(url)
        pil_image = Image.open(BytesIO(response.content)).convert("RGB")
        image = np.array(pil_image)[:, :, [2, 1, 0]]
        return image

    def imshow(img):
        plt.imshow(img[:, :, [2, 1, 0]])
        plt.axis("off")

    url = "https://i.ibb.co/HrCxLdt/0672ab32-9fce-41f3-ae69-e39c48a0a292-FREC-Scab-3347.jpg"
    image = cv2.imread("/content/maskscoringrcnn/leaf_dataset/test/Apple_Scab_22.jpg")
    imshow(image)
    predictions = coco_demo.run_on_opencv_image(image)
    imshow(predictions)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()