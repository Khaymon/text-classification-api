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
        """
        Fit the preprocessor on the provided data.

        Args:
            data (pd.DataFrame | pd.Series): The data to fit the preprocessor on.
        
        Returns:
            Self: The fitted preprocessor instance.
        """
        ...

    @abc.abstractmethod
    def transform(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """
        Transform the data using the fitted preprocessor.

        Args:
            data (pd.DataFrame | pd.Series): The data to transform.
        
        Returns:
            pd.DataFrame | pd.Series: The transformed data.
        """
        ...

    def fit_transform(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """
        Fit the preprocessor on the data and then transform it.

        Args:
            data (pd.DataFrame | pd.Series): The data to fit and transform.
        
        Returns:
            pd.DataFrame | pd.Series: The transformed data.
        """
        self.fit(data)

        return self.transform(data)

    @classmethod
    def from_config(cls, config: DataPreprocessorConfig) -> T.Self:
        """
        Create a preprocessor instance from a configuration.

        Args:
            config (DataPreprocessorConfig): The configuration for the preprocessor.
        
        Returns:
            Self: The configured preprocessor instance.
        """
        return cls(**(config.params or {}))
