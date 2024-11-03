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
        "Хохлы, это отдушина затюканого россиянина",
        "Собаке - собачья смерть",
        "Страницу обнови, дебил",
        "тебя не убедил 6-страничный пдф"
    ])
    targets = Targets([1, 1, 1, 1, 0])
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
    model = LogisticRegressionModel(model_config)
    assert model is not None
    assert model.preprocessor is not None

def test_model_fit(model_config, sample_dataset):
    model = LogisticRegressionModel(model_config)
    fitted_model = model.fit(sample_dataset)
    assert fitted_model is model  # Should return self

def test_model_predict(model_config, sample_dataset):
    model = LogisticRegressionModel(model_config)
    model.fit(sample_dataset)
    
    test_data = Data([
        "Новый текст для проверки",
        "Еще один текст"
    ])
    
    predictions = model.predict(test_data)
    assert isinstance(predictions, Targets)
    assert len(predictions.to_list()) == 2
    print(predictions.to_list())
    assert all(isinstance(pred, (int, float)) for pred in predictions.to_list())

def test_model_end_to_end(model_config, sample_dataset):
    # Test the entire pipeline from fitting to prediction
    model = LogisticRegressionModel(model_config)
    model.fit(sample_dataset)
    
    # Predict on training data
    predictions = model.predict(sample_dataset.data)
    assert len(predictions.to_list()) == len(sample_dataset.data.to_list()) 