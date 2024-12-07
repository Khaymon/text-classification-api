from src.lib.models import ModelsFactory, ModelInterface, ModelType
from src.lib.datasets.data_models import Data, Dataset
from src.lib.datasets.storage import DatasetsStorage
import src.lib.web.data_models as data_models
from src.lib.storage.local_artifact_storage import LocalArtifactStorage

STORAGE = LocalArtifactStorage()


def train_handler(request: data_models.TrainRequest) -> data_models.TrainResponse:
    """
    Handle the training of a new model based on the provided TrainRequest.

    Args:
        request (TrainRequest): The request containing dataset and model configurations.

    Returns:
        TrainResponse: A response containing the artifact name and evaluation metrics.
    """
    datasets_storage = DatasetsStorage()

    train_dataset = datasets_storage.download(request.dataset_name)
    model = ModelsFactory.create(
        ModelType(request.model.name), request.model.configuration
    ).fit(train_dataset)

    return data_models.TrainResponse(
        artifact_name=STORAGE.save(model, request.dataset_name)
    )


def predict_handler(request: data_models.PredictRequest) -> data_models.PredictResponse:
    """
    Handle prediction requests using a specified model artifact.

    Args:
        request (PredictRequest): The request containing data and the model artifact name.

    Returns:
        PredictResponse: A response containing the predictions.
    """
    model: ModelInterface = STORAGE.load(request.model_artifact_name)
    predictions = model.predict(Data(texts=request.data))

    return data_models.PredictResponse(predictions=predictions)


def list_model_artifacts_handler() -> data_models.ListModelArtifactsResponse:
    """
    List all available model artifacts stored locally.

    Returns:
        ListModelArtifactsResponse: A response containing a list of artifact names.
    """
    return data_models.ListModelArtifactsResponse(artifacts=list(STORAGE.list()))


def upload_dataset_handler(
    request: data_models.UploadDatasetRequest,
) -> data_models.UploadDatasetResponse:
    dataset = Dataset(
        texts=[text for text, _ in request.data],
        targets=[int(target) for _, target in request.data],
    )
    DatasetsStorage().upload(dataset=dataset, name=request.name)

    return data_models.UploadDatasetResponse(message="success")
