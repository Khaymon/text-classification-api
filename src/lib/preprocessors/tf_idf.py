import typing as T

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.lib.preprocessors.interfaces import DataPreprocessorInterface


class TfIdfPreprocessor(DataPreprocessorInterface):
    def __init__(self, text_column: str = "text", columns_suffix: str = "_tfidf"):
        self._text_column = text_column
        self._columns_suffix = columns_suffix
    
        self._transformer = TfidfVectorizer()

    def fit(self, data: pd.DataFrame) -> T.Self:
        self._transformer.fit(data[self._text_column])

        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        tf_idf = pd.DataFrame.sparse.from_spmatrix(self._transformer.transform(data[self._text_column]))
        tf_idf.columns = [f"{col}{self._columns_suffix}" for col in tf_idf.columns]

        return pd.concat([data, tf_idf], axis=1)
