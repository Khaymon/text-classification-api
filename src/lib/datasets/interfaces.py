import abc
from copy import deepcopy
import typing as T

import pandas as pd


class Data:
    def __init__(self, texts: T.Sequence[str]):
        self._texts = list(texts)

    def to_list(self) -> T.List[str]:
        return deepcopy(self._texts)

    def to_pandas(self) -> pd.Series:
        return pd.Series(self._texts, name="text")


class Targets:
    def __init__(self, targets: T.Sequence[int]):
        self._targets = targets

    def to_list(self) -> T.List[str]:
        return deepcopy(self._targets)
    
    def to_pandas(self) -> pd.Series:
        return pd.Series(self._targets, name="target")


class Dataset:
    def __init__(self, data: Data, targets: Targets):
        self._data = data
        self._targets = targets

    @property
    def data(self):
        return deepcopy(self._data)

    @property
    def targets(self):
        return deepcopy(self._targets)

    def to_pandas(self) -> pd.DataFrame:
        return pd.concat([self._data.to_pandas(), self._targets.to_pandas()], axis=1)

    @abc.abstractmethod
    @classmethod
    def load(cls, *args, **kwargs) -> "Dataset":
        """
            Download the dataset from some source, preprocess it and create Dataset
        """
        ...
