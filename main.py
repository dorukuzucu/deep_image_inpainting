import argparse
from torch.utils.data import DataLoader

from config import utils
from config.training import TrainConfig
from config.training import DatasetConfig
from config.training import OptimizerConfig

from scripts.train import build_loss
from scripts.train import build_model
from scripts.train import build_datasets
from scripts.train import build_optimizer
from scripts.train import load_model
from scripts.train import Trainer


parser = argparse.ArgumentParser(description='Train a Image Restoration Network')
parser.add_argument("-config", type=str, required=False, default="train_config.json", help="Path to training config file")
parser.add_argument("-weights", type=str, required=False, default=None, help="Path to last weight file")


def _parse_configs(config_path):
    json_config = utils.read_json(config_path)
    try:
        train_config = TrainConfig.from_json(json_config["training"])
        dataset_config = DatasetConfig.from_json(json_config["dataset"])
        optimizer_config = OptimizerConfig.from_json(json_config["optimizer"])
        return train_config, dataset_config, optimizer_config
    except KeyError as e:
        raise KeyError(f"Missing configuration in config file {e}")


def main(train_config: TrainConfig, dataset_config: DatasetConfig, optimizer_config: OptimizerConfig, weight_path=None):
    model = build_model() if weight_path is None else load_model(weight_path)
    optimizer = build_optimizer(optimizer_config, model)
    # TODO refactor
    train_dataset, val_dataset = build_datasets(dataset_config)
    train_loader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config.batch_size, shuffle=True)
    loss = build_loss(train_config.loss)
    trainer = Trainer(model=model, optimizer=optimizer, train_loader=train_loader,
                      val_loader=val_loader, criteria=loss, train_config=train_config)
    trainer.train()


if __name__ == '__main__':
    args = parser.parse_args()
    train_cfg, dataset_cfg, optimizer_cfg = _parse_configs(args.config)
    main(train_cfg, dataset_cfg, optimizer_cfg, args.weights)
