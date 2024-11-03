import abc

from src.lib.models.config import ModelConfigBase
from src.lib.datasets.interfaces import Data, Dataset, Targets


class ModelInterface(abc.ABC):
    def __init__(self, name: str, config: ModelConfigBase):
        self.name = name
        self.config = config

    @abc.abstractmethod
    def fit(self, train_dataset: Dataset) -> "ModelInterface":
        ...

    @abc.abstractmethod
    def predict(self, data: Data) -> Targets:
        ...
