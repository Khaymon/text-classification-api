import pytest
import pandas as pd

from src.lib.models.logistic_regression import LogisticRegressionModel
from src.lib.models.interfaces import ModelConfig
from src.lib.datasets.interfaces import Data, Dataset, Targets
from src.lib.preprocessors.compose import ComposePrerpocessorConfig, DataPreprocessorConfig
from src.lib.preprocessors.tf_idf import TfIdfPreprocessor

@pytest.fixture
def sample_dataset():
    data = Data([
        "Верблюдов-то за что? Дебилы, бл...",
        "Собаке - собачья смерть",
        "Страницу обнови, дебил",
        "тебя не убедил 6-страничный пдф"
    ])
    targets = Targets([1, 1, 1, 0])
    return Dataset(data, targets)

@pytest.fixture
def model_config():
    return ModelConfig(
        preprocessor=ComposePrerpocessorConfig(
            name="compose",
            preprocessors=[
                DataPreprocessorConfig(
                    name="tfidf",
                    params={
                        "text_column": "text"
                    }
                ),
                DataPreprocessorConfig(
                    name="drop",
                    params={
                        "columns": ["text"]
                    }
                )
            ]
        ),
        model_config={
            "random_state": 42
        }
    )

def test_model_initialization(model_config):
    """
    Test the initialization of the LogisticRegressionModel.

    Args:
        model_config (ModelConfig): The configuration for the model.
    
    {{ existing code }}

def test_model_fit(model_config, sample_dataset):
    """
    Test fitting the LogisticRegressionModel on a sample dataset.

    Args:
        model_config (ModelConfig): The configuration for the model.
        sample_dataset (Dataset): The sample dataset for training.
    
    {{ existing code }}

def test_model_predict(model_config, sample_dataset):
    """
    Test making predictions with the LogisticRegressionModel.

    Args:
        model_config (ModelConfig): The configuration for the model.
        sample_dataset (Dataset): The sample dataset for training.
    
    {{ existing code }}

def test_model_end_to_end(model_config, sample_dataset):
    """
    Test the end-to-end pipeline of training and predicting with the LogisticRegressionModel.

    Args:
        model_config (ModelConfig): The configuration for the model.
        sample_dataset (Dataset): The sample dataset for training.
    
    {{ existing code }}
