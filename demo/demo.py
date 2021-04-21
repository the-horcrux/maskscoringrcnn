import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import argparse
import requests
from io import BytesIO
from PIL import Image
import numpy as np

import cv2

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

parser = argparse.ArgumentParser(
        description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--test-image",
        metavar="FILE",
        help="path to test image file",
    )

    args = parser.parse_args()

# update the config options with the config file
cfg.merge_from_file(args.config_file)


coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
)

def load(url):
    """
    Given an url of an image, downloads the image and
    returns a PIL image
    """
    response = requests.get(url)
    pil_image = Image.open(BytesIO(response.content)).convert("RGB")
    # convert to BGR format
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    return image

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")

image = cv2.imread(args.test_image)
cv2.namedWindow('ori_img', cv2.WINDOW_NORMAL)
cv2.imshow('ori_img', image)

# compute predictions
predictions = coco_demo.run_on_opencv_image(image)
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.imshow('result', predictions)
cv2.imwrite('demo/result.png', predictions)

cv2.waitKey(0)