import sys
import unittest
from concurrent import futures

import grpc
from google.protobuf import empty_pb2
from google.protobuf.any_pb2 import Any

# TODO: ugly hack
sys.path.append(".")
import service_pb2
import service_pb2_grpc
from grpc_main import TextClassificationService
from src.lib.web.interfaces import TrainRequest, PredictRequest


class TestGRPCService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up a gRPC server for testing
        cls.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        service_pb2_grpc.add_TextClassificationServiceServicer_to_server(
            TextClassificationService(), cls.server
        )
        cls.port = cls.server.add_insecure_port("[::]:0")  # Bind to a free port
        cls.server.start()

        # Create a channel and a stub (client) for testing
        cls.channel = grpc.insecure_channel(f"localhost:{cls.port}")
        cls.stub = service_pb2_grpc.TextClassificationServiceStub(cls.channel)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop(None)
        cls.channel.close()

    def test_health_check(self):
        response = self.stub.HealthCheck(empty_pb2.Empty())
        self.assertEqual(response.status, "healthy")

    def test_get_datasets(self):
        response = self.stub.GetDatasets(empty_pb2.Empty())
        self.assertIsInstance(list(response.datasets), list)

    def test_get_models(self):
        response = self.stub.GetModels(empty_pb2.Empty())
        self.assertIsInstance(list(response.models), list)

    def test_list_model_artifacts(self):
        response = self.stub.ListModelArtifacts(empty_pb2.Empty())
        self.assertIsInstance(list(response.artifacts), list)

    def test_train_model(self):
        string_list = service_pb2.StringListValue(values=["text"])
        any_message = Any()
        any_message.Pack(string_list)

        train_request = service_pb2.TrainRequest(
            dataset=service_pb2.DatasetOptions(name="dvach"),
            model=service_pb2.ModelOptions(
                name="logistic_regression",
                configuration=service_pb2.ModelConfig(
                    preprocessor=service_pb2.PreprocessorConfig(
                        preprocessors=[
                            service_pb2.DataPreprocessorConfig(name="tfidf", params={}),
                            service_pb2.DataPreprocessorConfig(
                                name="drop", params={"columns": any_message}
                            ),
                        ]
                    ),
                    model_configuration={},
                ),
            ),
        )
        response = self.stub.TrainModel(train_request)
        self.assertIsInstance(response.artifact_name, str)
        self.assertIsInstance(response.metrics, service_pb2.Metrics)

    def test_predict(self):
        # Mock request for training
        string_list = service_pb2.StringListValue(values=["text"])
        any_message = Any()
        any_message.Pack(string_list)

        train_request = service_pb2.TrainRequest(
            dataset=service_pb2.DatasetOptions(name="dvach"),
            model=service_pb2.ModelOptions(
                name="logistic_regression",
                configuration=service_pb2.ModelConfig(
                    preprocessor=service_pb2.PreprocessorConfig(
                        preprocessors=[
                            service_pb2.DataPreprocessorConfig(name="tfidf", params={}),
                            service_pb2.DataPreprocessorConfig(
                                name="drop", params={"columns": any_message}
                            ),
                        ]
                    ),
                    model_configuration={},
                ),
            ),
        )
        response = self.stub.TrainModel(train_request)
        artifact_name = response.artifact_name
        # Mock request for prediction
        predict_request = service_pb2.PredictRequest(
            data=["sample text"], model_artifact_name=artifact_name
        )
        response = self.stub.Predict(predict_request)
        self.assertIsInstance(list(response.predictions), list)


if __name__ == "__main__":
    unittest.main()
