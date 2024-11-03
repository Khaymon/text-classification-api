import abc
from dataclasses import dataclass
import typing as T

from src.lib.datasets.interfaces import Data, Dataset, Targets
from src.lib.preprocessors.compose import ComposePreprocessor, ComposePrerpocessorConfig


@dataclass
class ModelConfigInterface(abc.ABC):
    preprocessor: ComposePrerpocessorConfig
    model_config: dict[str, T.Any] | None = None


class ModelInterface(abc.ABC):
    NAME: str | None = None

    def __init__(self, config: ModelConfigInterface):
        self.config = config
        self.preprocessor = ComposePreprocessor.from_config(config.preprocessor)

    @abc.abstractmethod
    def fit(self, train_dataset: Dataset) -> "ModelInterface":
        ...

    @abc.abstractmethod
    def predict(self, data: Data) -> Targets:
        ...
