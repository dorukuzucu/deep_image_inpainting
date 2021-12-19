from dataclasses import dataclass


class InvalidConfigError(Exception):
    def __init__(self, cls_name, params):
        self.cls_name = cls_name
        self.params = params

    def __str__(self):
        return f"Please validate {self.cls_name}. Required parameters are: {list(self.params)}"


class BaseConfig:
    @classmethod
    def from_json(cls, json_data: dict):
        params = list(cls.__annotations__.keys())
        if not set(params).issubset(set(json_data.keys())):
            raise InvalidConfigError(cls.__name__, cls.__annotations__.keys())
        return cls(**json_data)


@dataclass(init=True, repr=True)
class TrainConfig(BaseConfig):
    epochs: int
    val_every: int
    batch_size: int
    device: str
    output_path: str
    loss: str


@dataclass(init=True, repr=True)
class DatasetConfig(BaseConfig):
    train_path: str
    val_path: str
    name: str
    mask_type: str


@dataclass(init=True, repr=True)
class OptimizerConfig(BaseConfig):
    name: str
    lr: float
    weight_decay: float
    momentum: float
