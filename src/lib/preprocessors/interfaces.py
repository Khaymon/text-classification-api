import abc
import typing as T

import pandas as pd


class DataPreprocessorInterface(abc.ABC):
    NAME: str | None = None

    @abc.abstractmethod
    def fit(self, data: pd.DataFrame) -> T.Self:
        ...

    @abc.abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        ...

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)

        return self.transform(data)
