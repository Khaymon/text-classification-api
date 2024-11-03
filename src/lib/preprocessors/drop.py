import typing as T

import pandas as pd

from src.lib.preprocessors.interfaces import DataPreprocessorInterface


class DropPreprocessor(DataPreprocessorInterface):
    def __init__(self, columns: T.Iterable[str]):
        self._columns = columns

    def fit(self, data: pd.DataFrame) -> T.Self:
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.drop(columns=self._columns).copy()
