import abc
from pydantic import BaseModel
import typing as T

import pandas as pd


class DataPreprocessorConfig(BaseModel):
    name: str
    params: dict[str, T.Any] | None = None


class DataPreprocessorInterface(abc.ABC):
    NAME: str | None = None

    @abc.abstractmethod
    def fit(self, data: pd.DataFrame | pd.Series) -> T.Self:
        ...

    @abc.abstractmethod
    def transform(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        ...

    def fit_transform(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        self.fit(data)

        return self.transform(data)

    @classmethod
    def from_config(cls, config: DataPreprocessorConfig) -> T.Self:
        return cls(**(config.params or {}))
