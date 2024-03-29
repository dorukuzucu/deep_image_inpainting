import torch.nn as nn

from abc import ABC
from abc import abstractmethod


class BaseLoss(ABC):
    @abstractmethod
    def calculate(self, prediction, labels):
        pass

    def __call__(self, prediction, labels):
        return self.calculate(prediction, labels)


class MseLoss(BaseLoss):
    def __init__(self):
        self.loss = nn.MSELoss()

    def calculate(self, prediction, labels):
        return self.loss(prediction, labels)
