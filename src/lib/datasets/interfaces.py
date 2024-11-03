import abc
from copy import deepcopy
import typing as T
import pandas as pd


class Data:
    def __init__(self, texts: T.Sequence[str]):
        """
        Initialize the Data object with a list of texts.

        Args:
            texts (T.Sequence[str]): A sequence of text strings.
        """
        self._texts = list(texts)

    def to_list(self) -> T.List[str]:
        """
        Convert the data to a list of strings.

        Returns:
            T.List[str]: The list of text strings.
        """
        return deepcopy(self._texts)

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the data to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the text data.
        """
        return pd.DataFrame({"text": self._texts})


class Targets:
    def __init__(self, targets: T.Sequence[int]):
        """
        Initialize the Targets object with a list of target values.

        Args:
            targets (T.Sequence[int]): A sequence of target integers.
        """
        self._targets = list(targets)

    def to_list(self) -> T.List[int]:
        """
        Convert the targets to a list of integers.

        Returns:
            T.List[int]: The list of target integers.
        """
        return deepcopy(self._targets)
    
    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the targets to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the target data.
        """
        return pd.DataFrame({"target": self._targets})


class Dataset:
    NAME: str | None = None

    def __init__(self, data: Data, targets: Targets):
        """
        Initialize the Dataset with data and targets.

        Args:
            data (Data): The input data.
            targets (Targets): The corresponding targets.
        """
        self._data = data
        self._targets = targets

    @property
    def data(self):
        """
        Get a copy of the input data.

        Returns:
            Data: The input data.
        """
        return deepcopy(self._data)

    @property
    def targets(self):
        """
        Get a copy of the target data.

        Returns:
            Targets: The target data.
        """
        return deepcopy(self._targets)

    def to_pandas(self) -> pd.DataFrame:
        """
        Combine data and targets into a single pandas DataFrame.

        Returns:
            pd.DataFrame: The combined DataFrame.
        """
        return pd.concat([self._data.to_pandas(), self._targets.to_pandas()], axis=1)

    @classmethod
    @abc.abstractmethod
    def load(cls, *args, split: T.Literal["train", "test"], **kwargs) -> "Dataset":
        """
        Abstract method to load the dataset.

        Args:
            *args: Variable length argument list.
            split (T.Literal["train", "test"]): The dataset split to load.
            **kwargs: Arbitrary keyword arguments.
        
        Returns:
            Dataset: The loaded dataset instance.
        
        Raises:
            NotImplementedError: If the method is not implemented.
        """
        ...
