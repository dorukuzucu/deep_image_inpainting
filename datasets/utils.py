import os
import glob
import random
import numpy as np

from PIL import Image
from typing import Tuple
from typing import Union
from pathlib import Path


def _randint(min_value, max_value):
    if not (isinstance(min_value, int) or isinstance(max_value, int)):
        raise ValueError("Min and Max values should be integers.")
    return random.randint(min_value, max_value)


def generate_random_bbox(img_size: Union[int, Tuple[int, int]], max_bbox_size: Union[int, Tuple[int, int]] = None):
    width, height = img_size
    min_bbox_size = min(width, height) // 6

    if max_bbox_size is None:
        mask_size = min(width, height) // 4
        max_bbox_size = (mask_size, mask_size)
    elif isinstance(max_bbox_size, int):
        max_bbox_size = (max_bbox_size, max_bbox_size)

    max_width, max_height = max_bbox_size

    bbox_width = _randint(min_bbox_size, max_width)
    bbox_height = _randint(min_bbox_size, max_height)
    bbox_x = _randint(0, width - 1)
    bbox_y = _randint(0, height - 1)

    return bbox_x, bbox_y, bbox_width, bbox_height


def bbox_to_mask(bbox, img_width, img_height):
    x, y, box_width, box_height = bbox
    mask = np.zeros((1, img_height, img_width), np.uint8)
    mask[0, y:(y+box_height), x:(x+box_width)] = 1
    return mask


def list_directory(dir_path, content_type=None, add_path=True, recursive=False):
    glob_str = os.path.join(dir_path, f"*.{content_type}")
    glob_data = glob.glob(glob_str, recursive=recursive)
    data_list = [data.split(os.sep)[-1] if not add_path else data for data in glob_data]
    return data_list


def read_img(img_path):
    if not os.path.isfile(img_path):
        raise Exception(f"Image is not found. {img_path}")

    img = Image.open(img_path)
    img = img.convert("RGB")
    return img
