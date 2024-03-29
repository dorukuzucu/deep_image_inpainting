import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from typing import List
from typing import Tuple
from typing import Union

import datasets.utils as dataset_utils


class _BaseDataset(Dataset):
    items: List[str]

    def __init__(self, img_format="jpg",
                 size: Union[int, Tuple[int, int]] = None,
                 grayscale: bool = False):
        self.image_format = img_format
        self.size = size if isinstance(size, tuple) else (size, size)
        self.gray_scale = grayscale

        self.transforms = None

    def initialize(self):
        self._list_dataset()
        self._build_transforms()

    def _list_dataset(self):
        raise NotImplementedError()

    def _build_transforms(self):
        transform_list = []
        transform_list.append(transforms.ToTensor())
        if self.size is not None:
            transform_list.append(transforms.Resize(self.size))
        if self.gray_scale:
            transform_list.append(transforms.Grayscale())
        transform_list.append(transforms.ConvertImageDtype(torch.float))
        self.transforms = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.items)

    def _mask_input(self, img):
        channels, height, width = img.size()
        mask_bbox = dataset_utils.generate_random_bbox((width, height))
        mask = dataset_utils.bbox_to_mask(mask_bbox, width, height)
        img_clone = img.clone()
        return img_clone * (1 - mask)

    def __getitem__(self, index):
        img_path = self.items[index]
        img = dataset_utils.read_img(img_path)
        img = self.transforms(img)
        masked_img = self._mask_input(img)
        return masked_img, img


class BsdDataset(_BaseDataset):
    items: List[str]

    def __init__(self, dataset_path: str, img_format="jpg", size: Union[int, Tuple[int, int]] = None,
                 grayscale: bool = False):
        super().__init__(img_format, size, grayscale)

        self.dataset_path = dataset_path
        self.initialize()

    def _list_dataset(self):
        self.items = dataset_utils.list_directory(dir_path=self.dataset_path, content_type=self.image_format)


class CelebADataset(_BaseDataset):
    items: List[str]

    def __init__(self, dataset_path: str, partition_file_path: str, img_format="jpg", stage_flag: int = 0,
                 size: Union[int, Tuple[int, int]] = None, grayscale: bool = False):
        super().__init__(img_format, size, grayscale)
        self.dataset_path = dataset_path
        self.partition_file_path = partition_file_path
        self.stage_flag = stage_flag

        self.initialize()

    def _list_dataset(self):
        partition = dataset_utils.txt_to_line_list(self.partition_file_path)
        self.items = [os.path.join(self.dataset_path, item.split(" ")[0]) for item in partition if item.split(" ")[1] == str(self.stage_flag)]
