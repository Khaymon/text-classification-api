import abc
from dataclasses import dataclass
import typing as T

import pandas as pd

from src.lib.preprocessors.interfaces import DataPreprocessorConfig


@dataclass(kw_only=True)
class DataPreprocessorConfig:
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
