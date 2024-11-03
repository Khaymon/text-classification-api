import pytest
from src.lib.web.handlers import train_handler, predict_handler
from src.lib.web.interfaces import TrainRequest, PredictRequest

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

@pytest.fixture
def trained_model_artifact(train_request):
    result = train_handler(train_request)
    return result["artifact_name"]

@pytest.fixture
def predict_request(trained_model_artifact):
    return PredictRequest.model_validate({
        "model_artifact_name": trained_model_artifact,
        "data": [
            "Новый текст для проверки",
            "Еще один текст"
        ]
    })

def test_predict_handler(predict_request):
    result = predict_handler(predict_request)
    assert "predictions" in result
    assert isinstance(result["predictions"], list)
    assert len(result["predictions"]) == 2
    assert all(isinstance(pred, (int, float)) for pred in result["predictions"]) 
