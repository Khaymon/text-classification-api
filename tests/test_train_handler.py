import pytest
from src.lib.web.handlers import train_handler
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

def test_train_handler(train_request):
    result = train_handler(train_request)
    assert "metrics" in result
    assert "accuracy" in result["metrics"]
    assert "f1" in result["metrics"]
    assert "precision" in result["metrics"]
    assert "recall" in result["metrics"] 
