from frozendict import frozendict

from .logistic_regression import LogisticRegressionModel


MODELS_MAP = frozendict({
    LogisticRegressionModel.NAME: LogisticRegressionModel
})
