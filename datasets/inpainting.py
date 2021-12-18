from typing import List
from typing import Tuple
from typing import Union

import torch
from torchvision import transforms
from torch.utils.data import Dataset

import datasets.utils as dataset_utils


class BsdDataset(Dataset):
    items: List[str]

    def __init__(self, dataset_path: str, img_format="jpg",
                 size: Union[int, Tuple[int, int]] = None,
                 grayscale: bool = False):
        self.dataset_path = dataset_path
        self.image_format = img_format
        self.size = size
        self.gray_scale = grayscale

        self.transforms = None
        self.initialize()

    def initialize(self):
        self._list_dataset()
        self._build_transforms()

    def _list_dataset(self):
        self.items = dataset_utils.list_directory(dir_path=self.dataset_path, content_type=self.image_format)

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

    def __getitem__(self, index):
        img_path = self.items[index]
        img = dataset_utils.read_img(img_path)
        img = self.transforms(img)
        return img
