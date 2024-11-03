import abc
from dataclasses import asdict
from pathlib import Path
import typing as T

import src.common.utils as utils
from src.lib.datasets.interfaces import Data, Dataset, Targets
from src.lib.preprocessors.compose import ComposePreprocessor, ComposePrerpocessorConfig
from pydantic import BaseModel


LOGGER = utils.initialize_logging(__name__)


class ModelConfig(BaseModel):
    preprocessor: ComposePrerpocessorConfig
    model_configuration: dict[str, T.Any] | None = None


class ModelInterface(abc.ABC):
    NAME: str | None = None

    def __init__(self, config: ModelConfig):
        LOGGER.info(f"Create model {type(self).__name__} with config {config}")

        self.config = config
        self.preprocessor = ComposePreprocessor.from_config(config.preprocessor)

    @abc.abstractmethod
    def fit(self, train_dataset: Dataset) -> "ModelInterface":
        ...

    @abc.abstractmethod
    def predict(self, data: Data) -> Targets:
        ...

    def save(self, path: Path) -> None:
        LOGGER.info(f"Save model {type(self).__name__} to {path}")

        path.mkdir(parents=True, exist_ok=False)

        utils.PickleHelper.save(self.preprocessor, path / "preprocessor.pkl")
        utils.JsonHelper.save(self.config.model_dump(), path / "config.json")

        self._save(path)

    def load(self, path: Path) -> T.Self:
        LOGGER.info(f"Load model {type(self).__name__} from {path}")

        self.preprocessor = utils.PickleHelper.load(path / "preprocessor.pkl")
        self.config = ModelConfig(**utils.JsonHelper.load(path / "config.json"))

        return self._load(path)

    @classmethod
    def load(cls, path: Path) -> T.Self:
        config = ModelConfig(**utils.JsonHelper.load(path / "config.json"))
        model = cls(config)
        model.preprocessor = utils.PickleHelper.load(path / "preprocessor.pkl")
        return model._load(path)
    
    @classmethod
    @abc.abstractmethod
    def _save(cls, path: Path) -> None:
        ...

    @abc.abstractmethod
    def _load(self, path: Path) -> T.Self:
        ...
