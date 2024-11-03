from pathlib import Path
import typing as T

from sklearn.linear_model import LogisticRegression

from src.common.utils import PickleHelper
from src.lib.models.interfaces import ModelConfig, ModelInterface
from src.lib.datasets.interfaces import Data, Dataset, Targets


class LogisticRegressionModel(ModelInterface):
    NAME = "logistic_regression"

    def __init__(self, config: ModelConfig):
        """
        Initialize the LogisticRegressionModel with the given configuration.

        Args:
            config (ModelConfig): The configuration for the logistic regression model.
        """
        super().__init__(config)

        self._model = LogisticRegression(**(config.model_configuration or {}))

    def fit(self, train_dataset: Dataset) -> T.Self:
        """
        Fit the logistic regression model on the training dataset.

        Args:
            train_dataset (Dataset): The dataset to train the model on.
        
        Returns:
            Self: The fitted model instance.
        """
        X = train_dataset.data.to_pandas()
        y = train_dataset.targets.to_pandas()

        self._model.fit(self.preprocessor.fit_transform(X), y)

        return self
        
    def predict(self, data: Data) -> Targets:
        """
        Make predictions using the trained logistic regression model.

        Args:
            data (Data): The input data for prediction.
        
        Returns:
            Targets: The prediction results.
        """
        return Targets(
            self._model.predict(self.preprocessor.transform(data.to_pandas())).tolist()
        )

    def _save(self, path: Path) -> None:
        """
        Save the logistic regression model to the specified path.

        Args:
            path (Path): The directory path to save the model.
        """
        PickleHelper.save(self._model, path / "model.pkl")

    def _load(self, path: Path) -> T.Self:
        """
        Load the logistic regression model from the specified path.

        Args:
            path (Path): The directory path from which to load the model.
        
        Returns:
            Self: The loaded model instance.
        """
        self._model = PickleHelper.load(path / "model.pkl")
        return self
