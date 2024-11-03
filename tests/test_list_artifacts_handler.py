import pytest
from src.lib.web.handlers import train_handler
from src.lib.storage.local_artifact_storage import LocalArtifactStorage
from src.lib.web.interfaces import TrainRequest

@pytest.fixture
def train_request():
    return TrainRequest.model_validate({
        "dataset": {
            "name": "dvach"
        },
        "model": {
            "name": "logistic_regression",
            "configuration": {
                "preprocessor": {
                    "preprocessors": [
                        {
                            "name": "tfidf",
                            "params": {}
                        },
                        {
                            "name": "drop",
                            "params": {
                                "columns": ["text"]
                            }
                        }
                    ]
                },
                "model_configuration": {
                    "random_state": 42
                }
            }
        }
    })

def test_list_artifacts_handler(train_request):
    # Train a new model
    result = train_handler(train_request)
    artifact_name = result["artifact_name"]

    # List artifacts and check if the new artifact is present
    storage = LocalArtifactStorage()
    artifacts = storage.list()
    assert artifact_name in artifacts 