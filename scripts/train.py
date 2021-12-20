import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from tqdm import tqdm

from config.training import TrainConfig
from config.training import BsdDatasetConfig
from config.training import OptimizerConfig
from datasets.inpainting import BsdDataset
from datasets.inpainting import CelebADataset
from losses.inpainting import MseLoss
from models.context_encoder import RedNet
from scripts.data_savers import MetricManager
from scripts.data_savers import ModelFileManager


def build_optimizer(optimizer_config: OptimizerConfig, model: nn.Module):
    if optimizer_config.name.lower() not in ["sgd", "adam"]:
        raise NotImplementedError("Only SGD and Adam optimizers are supported")
    if optimizer_config.name == "SGD":
        optimizer_selection = SGD(params=model.parameters(), lr=optimizer_config.lr,
                                  weight_decay=optimizer_config.weight_decay,
                                  momentum=optimizer_config.momentum)
    else:
        optimizer_selection = Adam(params=model.parameters(), lr=optimizer_config.lr,
                                   weight_decay=optimizer_config.weight_decay)
    return optimizer_selection


def build_datasets(dataset_config: BsdDatasetConfig):
    if dataset_config.name.lower()=="bsd":
        train_dataset = BsdDataset(dataset_path=dataset_config.train_path, size=dataset_config.size)
        val_dataset = BsdDataset(dataset_path=dataset_config.val_path, size=dataset_config.size)
    elif dataset_config.name.lower()=="celeba":
        train_dataset = CelebADataset(dataset_path=dataset_config.train_path,
                                      partition_file_path=dataset_config.partition_file_path,
                                      size=dataset_config.size, stage_flag=0)
        val_dataset = CelebADataset(dataset_path=dataset_config.train_path,
                                    partition_file_path=dataset_config.partition_file_path,
                                    size=dataset_config.size, stage_flag=1)
    else:
        raise NotImplementedError(f"Dataset is not supported")

    return train_dataset, val_dataset


def build_loss(loss_name: str):
    if loss_name.lower() not in ["mse", "mseloss"]:
        raise NotImplementedError("Only MSE Loss is supported")
    if loss_name.lower() == "mse":
        return MseLoss()


def build_model():
    return RedNet(num_layers=30, input_channels=3)


def load_model(model_path):
    state_dict = torch.load(model_path)
    model = build_model()
    return model.load_state_dict(state_dict=state_dict)


class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, criteria, train_config: TrainConfig):
        """
        :param model: A deep learning model extends to nn.Module
        :param optimizer: optimizer for current test
        :param criteria: A loss function
        :param train_loader: Data loader for training data set
        :param val_loader: Data loader for validation data set
        :param train_config: TrainConfig object for training parameters
        """
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criteria = criteria
        self.train_config = train_config
        self.gpu_flag = self._gpu_flag()
        if self.gpu_flag:
            self.model.to("cuda:0")

        self.model_file_manager = ModelFileManager(model_save_path=train_config.output_path)
        self.metric_manager = MetricManager(log_path=train_config.output_path)

    def _gpu_flag(self):
        return "cuda" in self.train_config.device and torch.cuda.is_available()

    def train(self):
        for epoch in range(self.train_config.start_epoch, self.train_config.epochs):
            train_loss = self.train_single_epoch()
            self.metric_manager.add_train_loss(train_loss, epoch)
            print("Epoch:{} Loss:{}".format(epoch, train_loss))

            if (epoch + 1) % self.train_config.val_every == 0:
                self.optimizer.zero_grad(set_to_none=True)
                val_loss = self.evaluate()
                self.metric_manager.add_val_loss(val_loss, epoch)

                print(f"Validation: Loss:{val_loss}")
                self.save_model(epoch)

    def _forward(self, data, label):
        if self.gpu_flag:
            data = data.to(torch.device('cuda:0'))
            label = label.to(torch.device('cuda:0'))

        out = self.model(data)
        loss = self.criteria(out, label)
        return loss

    def _backward(self, loss):
        loss.backward()
        self.optimizer.step()

    def train_single_epoch(self):
        iter_loss = float(0)
        self.model.train()

        with tqdm(total=len(self.train_loader)) as bar:
            for data, label in self.train_loader:
                self.optimizer.zero_grad()
                self.model.zero_grad()

                loss = self._forward(data, label)
                self._backward(loss)

                iter_loss += float(loss.item())
                bar.update(1)

        return iter_loss / len(self.train_loader)

    def evaluate(self):
        val_loss = float(0.0)
        self.model.eval()

        with torch.no_grad():
            for data, label in self.val_loader:
                loss = self._forward(data, label)
                val_loss += float(loss.item())

        return val_loss / len(self.val_loader)

    def save_model(self, epoch):
        self.model_file_manager.save_model(model=self.model, epoch=epoch)
