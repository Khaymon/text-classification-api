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
        """
        Initialize the ModelInterface with the given configuration.

        Args:
            config (ModelConfig): The configuration for the model.
        """
        LOGGER.info(f"Create model {type(self).__name__} with config {config}")

        self.config = config
        self.preprocessor = ComposePreprocessor.from_config(config.preprocessor)

    @abc.abstractmethod
    def fit(self, train_dataset: Dataset) -> "ModelInterface":
        """
        Fit the model on the provided training dataset.

        Args:
            train_dataset (Dataset): The dataset to train the model on.
        
        Returns:
            ModelInterface: The fitted model instance.
        """
        ...

    @abc.abstractmethod
    def predict(self, data: Data) -> Targets:
        """
        Make predictions on the provided data.

        Args:
            data (Data): The data to make predictions on.
        
        Returns:
            Targets: The prediction results.
        """
        ...

    def save(self, path: Path) -> None:
        """
        Save the model and its configuration to the specified path.

        Args:
            path (Path): The directory path to save the model.
        """
        LOGGER.info(f"Save model {type(self).__name__} to {path}")

        path.mkdir(parents=True, exist_ok=False)

        utils.PickleHelper.save(self.preprocessor, path / "preprocessor.pkl")
        utils.JsonHelper.save(self.config.model_dump(), path / "config.json")

        self._save(path)

    @classmethod
    def load(cls, path: Path) -> T.Self:
        """
        Load a model instance from the specified path.

        Args:
            path (Path): The directory path from which to load the model.
        
        Returns:
            Self: The loaded model instance.
        """
        config = ModelConfig(**utils.JsonHelper.load(path / "config.json"))
        model = cls(config)
        model.preprocessor = utils.PickleHelper.load(path / "preprocessor.pkl")
        return model._load(path)
    
    @abc.abstractmethod
    def _save(self, path: Path) -> None:
        """
        Abstract method to handle model-specific saving logic.

        Args:
            path (Path): The directory path to save the model specifics.
        """
        ...

    @abc.abstractmethod
    def _load(self, path: Path) -> T.Self:
        """
        Abstract method to handle model-specific loading logic.

        Args:
            path (Path): The directory path from which to load the model specifics.
        
        Returns:
            Self: The loaded model instance.
        """
        ...
