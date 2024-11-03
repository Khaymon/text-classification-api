from sklearn.linear_model import LogisticRegression

from src.lib.models.interfaces import ModelConfigInterface, ModelInterface
from src.lib.datasets.interfaces import Data, Dataset, Targets


class LogisticRegressionConfig(ModelConfigInterface):
    ...


class LogisticRegressionModel(ModelInterface):
    def __init__(self, config: LogisticRegressionConfig):
        super().__init__(config)

        self._model = LogisticRegression(**(config.model_config or {}))

    def fit(self, train_dataset: Dataset) -> "ModelInterface":
        preprocessed_data = self.config.
        self._model.fit()
        

    def predict(self, data: Data) -> Targets:
        ...
