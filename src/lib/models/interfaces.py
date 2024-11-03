import abc
from dataclasses import dataclass

from src.lib.datasets.interfaces import Data, Dataset, Targets


@dataclass
class ModelConfigInterface(abc.ABC):
    ...


class ModelInterface(abc.ABC):
    NAME: str | None = None

    def __init__(self, config: ModelConfigInterface):
        self.config = config

    @abc.abstractmethod
    def fit(self, train_dataset: Dataset) -> "ModelInterface":
        ...

    @abc.abstractmethod
    def predict(self, data: Data) -> Targets:
        ...
