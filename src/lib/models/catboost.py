from pathlib import Path
import typing as T

from catboost import CatBoostClassifier

from src.lib.datasets.interfaces import Data, Dataset, Targets
from src.lib.models.interfaces import ModelConfig, ModelInterface


class CatBoostModel(ModelInterface):
    NAME = "catboost"

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self._model = CatBoostClassifier(**(config.model_configuration or {}))

    def fit(self, train_dataset: Dataset) -> T.Self:
        X = train_dataset.data.to_pandas()
        y = train_dataset.targets.to_pandas()

        self._model.fit(self.preprocessor.fit_transform(X), y, verbose=False)

        return self

    def predict(self, data: Data) -> Targets:
        return Targets(
            self._model.predict(self.preprocessor.transform(data.to_pandas()))
        )

    def _save(self, path: Path) -> None:
        self._model.save_model(path / "model.cbm")

    def _load(self, path: Path) -> T.Self:
        self._model.load_model(path / "model.cbm")

        return self
