from strenum import StrEnum

from .catboost import CatBoostModel
from .logistic_regression import LogisticRegressionModel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interface import ModelInterface
    from .data_models import ModelConfig


class ModelType(StrEnum):
    catboost = "catboost"
    logistic_regression = "logistic_regression"


class ModelsFactory:
    @staticmethod
    def get_model_class(model_type: ModelType) -> type["ModelInterface"]:
        if model_type == ModelType.catboost:
            return CatBoostModel
        elif model_type == ModelType.logistic_regression:
            return LogisticRegressionModel

        raise ValueError(f"Unknown model type: {model}")

    @staticmethod
    def create(model_type: ModelType, model_config: "ModelConfig") -> "ModelInterface":
        return ModelsFactory.get_model_class(model_type)(model_config)
