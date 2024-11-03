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
        """
        Initialize the ComposePreprocessor with a list of preprocessors.

        Args:
            preprocessors (list[DataPreprocessorInterface]): A list of data preprocessors to apply.
        """
        self._preprocessors = preprocessors

    def fit(self, data: pd.DataFrame | pd.Series) -> T.Self:
        """
        Fit all preprocessors on the provided data.

        Args:
            data (pd.DataFrame | pd.Series): The data to fit the preprocessors on.
        
        Returns:
            Self: The fitted ComposePreprocessor instance.
        """
        for preprocessor in self._preprocessors:
            preprocessor.fit(data)

        return self

    def transform(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """
        Transform the data using all preprocessors in sequence.

        Args:
            data (pd.DataFrame | pd.Series): The data to transform.
        
        Returns:
            pd.DataFrame | pd.Series: The transformed data.
        """
        result_data = deepcopy(data)
        for preprocessor in self._preprocessors:
            result_data = preprocessor.transform(result_data)

        LOGGER.info(f"Obtained features {list(result_data.columns)} after preprocessing")

        return result_data

    @classmethod
    def from_config(cls, config: ComposePrerpocessorConfig) -> T.Self:
        """
        Create a ComposePreprocessor instance from a configuration.

        Args:
            config (ComposePrerpocessorConfig): The configuration for the ComposePreprocessor.
        
        Returns:
            Self: A configured ComposePreprocessor instance.
        """
        preprocessors = []

        for preprocessor_config in config.preprocessors:
            if preprocessor_config.name == DropPreprocessor.NAME:
                preprocessors.append(DropPreprocessor.from_config(preprocessor_config))
            elif preprocessor_config.name == TfIdfPreprocessor.NAME:
                preprocessors.append(TfIdfPreprocessor.from_config(preprocessor_config))
            else:
                raise ValueError(f"Unknown preprocessor {preprocessor_config.name}")

        return cls(preprocessors)
