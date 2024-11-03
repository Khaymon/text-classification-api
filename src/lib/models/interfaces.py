import abc
from dataclasses import asdict, dataclass
from pathlib import Path
import typing as T

from src.common.utils import JsonHelper, PickleHelper
from src.lib.datasets.interfaces import Data, Dataset, Targets
from src.lib.preprocessors.compose import ComposePreprocessor, ComposePrerpocessorConfig
from pydantic import BaseModel


class ModelConfig(BaseModel):
    preprocessor: ComposePrerpocessorConfig
    model_configuration: dict[str, T.Any] | None = None


class ModelInterface(abc.ABC):
    NAME: str | None = None

    def __init__(self, config: ModelConfig):
        self.config = config
        self.preprocessor = ComposePreprocessor.from_config(config.preprocessor)

    @abc.abstractmethod
    def fit(self, train_dataset: Dataset) -> "ModelInterface":
        ...

    @abc.abstractmethod
    def predict(self, data: Data) -> Targets:
        ...

    def save(self, path: Path) -> None:
        PickleHelper.save(self.preprocessor, path / "preprocessor.pkl")
        JsonHelper.save(asdict(self.config), path / "config.json")

        self._save(path)

    def load(self, path: Path) -> T.Self:
        self.preprocessor = PickleHelper.load(path / "preprocessor.pkl")
        self.config = ModelConfig.from_dict(JsonHelper.load(path / "config.json"))

        return self._load(path)
    
    @abc.abstractmethod
    def _save(self, path: Path) -> None:
        ...

    @abc.abstractmethod
    def _load(self, path: Path) -> T.Self:
        ...
