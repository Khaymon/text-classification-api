from copy import deepcopy
import typing as T

import pandas as pd

from src.lib.preprocessors.interfaces import DataPreprocessorInterface


class ComposePreprocessor(DataPreprocessorInterface):
    NAME = "compose"

    def __init__(self, preprocessors: list[DataPreprocessorInterface]):
        self._preprocessors = preprocessors

    def fit(self, data: pd.DataFrame) -> T.Self:
        for preprocessor in self._preprocessors:
            preprocessor.fit(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        result_data = deepcopy(data)
        for preprocessor in self._preprocessors:
            result_data = preprocessor.transform(result_data)

        return result_data
