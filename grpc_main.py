from concurrent import futures
import grpc

import service_pb2
import service_pb2_grpc

from src.lib.web.handlers import (
    train_handler,
    predict_handler,
    list_model_artifacts_handler,
)
from src.lib.web.data_models import TrainRequest, PredictRequest
from src.lib.datasets.storage import DatasetsStorage
from src.lib.models import MODELS_MAP


class TextClassificationService(service_pb2_grpc.TextClassificationServiceServicer):
    def HealthCheck(self, request, context):
        return service_pb2.HealthStatus(status="healthy")

    def GetDatasets(self, request, context):
        datasets = DatasetsStorage().list()
        return service_pb2.DatasetsResponse(datasets=datasets)

    def GetModels(self, request, context):
        models = list(MODELS_MAP.keys())
        return service_pb2.ModelsResponse(models=models)

    def TrainModel(self, request, context):
        def unpack_message(value):
            if value.Is(service_pb2.StringValue.DESCRIPTOR):
                string_value = service_pb2.StringValue()
                value.Unpack(string_value)
                return string_value.value
            elif value.Is(service_pb2.FloatValue.DESCRIPTOR):
                float_value = service_pb2.FloatValue()
                value.Unpack(float_value)
                return float_value.value
            elif value.Is(service_pb2.StringListValue.DESCRIPTOR):
                string_list_value = service_pb2.StringListValue()
                value.Unpack(string_list_value)
                return list(string_list_value.values)
            else:
                raise ValueError(f"Unsupported message type: {value}")

        train_request = TrainRequest(
            dataset={"name": request.dataset_name},
            model={
                "name": request.model.name,
                "configuration": {
                    "preprocessor": {
                        "preprocessors": [
                            {
                                "name": preprocessor.name,
                                "params": {
                                    key: unpack_message(value)
                                    for key, value in preprocessor.params.items()
                                },
                            }
                            for preprocessor in request.model.configuration.preprocessor.preprocessors
                        ]
                    },
                    "model_configuration": {
                        key: unpack_message(value)
                        for key, value in request.model.configuration.model_configuration.items()
                    },
                },
            },
        )
        response = train_handler(train_request)
        return service_pb2.TrainResponse(
            artifact_name=response.artifact_name,
            metrics=service_pb2.Metrics(
                f1=response.metrics.f1,
                accuracy=response.metrics.accuracy,
                precision=response.metrics.precision,
                recall=response.metrics.recall,
            ),
        )

    def Predict(self, request, context):
        predict_request = PredictRequest(
            data=request.data, model_artifact_name=request.model_artifact_name
        )
        response = predict_handler(predict_request)
        return service_pb2.PredictResponse(predictions=response.predictions)

    def ListModelArtifacts(self, request, context):
        response = list_model_artifacts_handler()
        return service_pb2.ListModelArtifactsResponse(artifacts=response.artifacts)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_TextClassificationServiceServicer_to_server(
        TextClassificationService(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
