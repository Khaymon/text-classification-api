import os
import pytest
from pathlib import Path
from shutil import rmtree

from src.lib.storage.local_artifact_storage import LocalArtifactStorage
from src.lib.models.logistic_regression import LogisticRegressionModel
from src.lib.models.interfaces import ModelConfig, ModelInterface
from src.common.const import ARTIFACTS_DIR
from src.lib.preprocessors.compose import ComposePrerpocessorConfig, DataPreprocessorConfig

@pytest.fixture
def model_config():
    return ModelConfig(
        preprocessor=ComposePrerpocessorConfig(
            preprocessors=[
                DataPreprocessorConfig(name="tfidf", params={"text_column": "text"}),
                DataPreprocessorConfig(name="drop", params={"columns": ["text"]})
            ]
        ),
        model_configuration={"random_state": 42}
    )

@pytest.fixture
def logistic_regression_model(model_config):
    return LogisticRegressionModel(model_config)

@pytest.fixture
def local_storage():
    return LocalArtifactStorage()

@pytest.fixture(autouse=True)
def setup_and_teardown():
    # Setup: Ensure ARTIFACTS_DIR exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    yield
    # Teardown: Clean up ARTIFACTS_DIR
    rmtree(ARTIFACTS_DIR)

def test_save_artifact(local_storage, logistic_regression_model):
    dataset_name = "test_dataset"
    local_storage.save(logistic_regression_model, dataset_name)
    artifact_name = f"1__{logistic_regression_model.NAME}__{dataset_name}"
    assert (ARTIFACTS_DIR / artifact_name).exists()

def test_load_artifact(local_storage, logistic_regression_model):
    dataset_name = "test_dataset"
    artifact_name = f"1__{logistic_regression_model.NAME}__{dataset_name}"
    local_storage.save(logistic_regression_model, dataset_name)
    loaded_model = local_storage.load(artifact_name)
    assert isinstance(loaded_model, LogisticRegressionModel)

def test_list_artifacts(local_storage, logistic_regression_model):
    dataset_name = "test_dataset"
    artifact_name = f"1__{logistic_regression_model.NAME}__{dataset_name}"
    local_storage.save(logistic_regression_model, dataset_name)
    artifacts = local_storage.list()
    assert artifact_name in artifacts 
