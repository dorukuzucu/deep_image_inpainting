import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter


class MetricManager:
    _train_loss_str: str
    _train_accuracy_str: str
    _val_loss_str: str
    _val_accuracy_str: str
    _writer: SummaryWriter

    def __init__(self, log_path):
        self._train_loss_str = "loss/train"
        self._train_accuracy_str = "accuracy/train"
        self._val_loss_str = "loss/val"
        self._val_accuracy_str = "accuracy/val"

        self._writer = SummaryWriter(log_dir=log_path)

    def add_train_loss(self, loss_val, epoch):
        self._writer.add_scalar(self._train_loss_str, loss_val, epoch)

    def add_val_loss(self, loss_val, epoch):
        self._writer.add_scalar(self._val_loss_str, loss_val, epoch)

    def add_train_accuracy(self, accuracy_val, epoch):
        self._writer.add_scalar(self._train_accuracy_str, accuracy_val, epoch)

    def add_val_accuracy(self, accuracy_val, epoch):
        self._writer.add_scalar(self._val_accuracy_str, accuracy_val, epoch)

