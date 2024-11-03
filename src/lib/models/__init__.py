from frozendict import frozendict

from .catboost import CatBoostModel
from .logistic_regression import LogisticRegressionModel


MODELS_MAP = frozendict({
    LogisticRegressionModel.NAME: LogisticRegressionModel,
    CatBoostModel.NAME: CatBoostModel,
})
