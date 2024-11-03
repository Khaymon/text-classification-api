from copy import deepcopy
from pydantic import BaseModel
import typing as T

import pandas as pd

import src.common.utils as utils
from src.lib.preprocessors.drop import DropPreprocessor
from src.lib.preprocessors.interfaces import DataPreprocessorConfig, DataPreprocessorInterface
from src.lib.preprocessors.tf_idf import TfIdfPreprocessor


LOGGER = utils.initialize_logging(__name__)


class ComposePrerpocessorConfig(BaseModel):
    preprocessors: list[DataPreprocessorConfig]


class ComposePreprocessor(DataPreprocessorInterface):
    NAME = "compose"

    def __init__(self, preprocessors: list[DataPreprocessorInterface]):
        self._preprocessors = preprocessors

    def fit(self, data: pd.DataFrame | pd.Series) -> T.Self:
        for preprocessor in self._preprocessors:
            preprocessor.fit(data)

        return self

    def transform(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        result_data = deepcopy(data)
        for preprocessor in self._preprocessors:
            result_data = preprocessor.transform(result_data)

        LOGGER.info(f"Obtained features {list(result_data.columns)} after preprocessing")

        return result_data

    @classmethod
    def from_config(cls, config: ComposePrerpocessorConfig) -> T.Self:
        preprocessors = []

        for preprocessor_config in config.preprocessors:
            if preprocessor_config.name == DropPreprocessor.NAME:
                preprocessors.append(DropPreprocessor.from_config(preprocessor_config))
            elif preprocessor_config.name == TfIdfPreprocessor.NAME:
                preprocessors.append(TfIdfPreprocessor.from_config(preprocessor_config))
            else:
                raise ValueError(f"Unknown preprocessor {preprocessor_config.name}")

        return cls(preprocessors)
