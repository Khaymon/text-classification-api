from pathlib import Path
import typing as T

from catboost import CatBoostClassifier

from src.lib.datasets.data_models import Data, Dataset
from src.lib.models.data_models import ModelConfig
from src.lib.models.interface import ModelInterface


class CatBoostModel(ModelInterface):
    NAME = "catboost"

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self._model = CatBoostClassifier(**(config.model_configuration or {}))

    def fit(self, train_dataset: Dataset) -> T.Self:
        data = train_dataset.to_pandas()
        X = data.drop(["target"], axis=1)
        y = data["target"]

        self._model.fit(self.preprocessor.fit_transform(X), y, verbose=False)

        return self

    def predict(self, data: Data) -> list[int]:
        return self._model.predict(
            self.preprocessor.transform(data.to_pandas())
        ).tolist()

    def _save(self, path: Path) -> None:
        self._model.save_model(path / "model.cbm")

    def _load(self, path: Path) -> T.Self:
        self._model.load_model(path / "model.cbm")

        return self
