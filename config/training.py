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
        try:
            config_class = cls(**json_data)
            return config_class
        except TypeError as e:
            raise Exception(str(e)[11:])


@dataclass(init=True, repr=True)
class TrainConfig(BaseConfig):
    epochs: int
    val_every: int
    batch_size: int
    device: str
    output_path: str
    loss: str
    start_epoch: int


class BaseDatasetConfig(BaseConfig):
    name: str
    mask_type: str
    size: int

    def __init__(self, name: str, mask_type: str, size: int):
        self.name = name
        self.mask_type = mask_type
        self.size = size


class BsdDatasetConfig(BaseDatasetConfig):
    train_path: str
    val_path: str

    def __init__(self, name: str, train_path: str, val_path: str, mask_type: str, size: int):
        super().__init__(name, mask_type, size)
        self.train_path = train_path
        self.val_path = val_path


class CelebADatasetConfig(BaseDatasetConfig):
    dataset_path: str
    partition_file_path: str

    def __init__(self, name: str, dataset_path: str, partition_file_path: str, mask_type: str, size: int):
        super().__init__(name, mask_type, size)
        self.dataset_path = dataset_path
        self.partition_file_path = partition_file_path


@dataclass(init=True, repr=True)
class OptimizerConfig(BaseConfig):
    name: str
    lr: float
    weight_decay: float
    momentum: float


class DatasetConfigClassFactory:
    @staticmethod
    def from_json(dataset_name):
        if dataset_name == "bsd":
            return BsdDatasetConfig
        elif dataset_name == "celeba":
            return CelebADatasetConfig
        else:
            raise NotImplementedError()
