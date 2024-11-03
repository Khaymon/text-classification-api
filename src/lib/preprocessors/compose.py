from copy import deepcopy
from pydantic import BaseModel
import typing as T

import pandas as pd

from src.lib.preprocessors.drop import DropPreprocessor
from src.lib.preprocessors.interfaces import DataPreprocessorConfig, DataPreprocessorInterface
from src.lib.preprocessors.tf_idf import TfIdfPreprocessor


class ComposePrerpocessorConfig(BaseModel):
    preprocessors: list[DataPreprocessorConfig]

    @classmethod
    def from_dict(cls, values: dict) -> "ComposePrerpocessorConfig":
        preprocessors = [DataPreprocessorConfig(**config) for config in values["preprocessors"]]

        return cls(name=values["name"], preprocessors=preprocessors)


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
