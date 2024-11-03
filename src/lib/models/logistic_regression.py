import typing as T

from sklearn.linear_model import LogisticRegression

from src.lib.models.interfaces import ModelConfig, ModelInterface
from src.lib.datasets.interfaces import Data, Dataset, Targets


class LogisticRegressionModel(ModelInterface):
    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self._model = LogisticRegression(**(config.model_config or {}))

    def fit(self, train_dataset: Dataset) -> T.Self:
        X = train_dataset.data.to_pandas()
        y = train_dataset.targets.to_pandas()

        self._model.fit(self.preprocessor.fit_transform(X), y)

        return self
        
    def predict(self, data: Data) -> Targets:
        return Targets(
            self._model.predict(self.preprocessor.transform(data.to_pandas())).tolist()
        )