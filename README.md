# Machine Learning Model Management API

## Overview

The **Machine Learning Model Management API** is a robust and scalable solution designed to facilitate the training, evaluation, storage, and deployment of machine learning models. Built with Python and FastAPI, the project leverages modern libraries and best practices to provide a seamless experience for data scientists and developers alike.

## Features

- **Model Training and Evaluation:** Train various machine learning models with customizable configurations and evaluate their performance using standard metrics.
- **Artifact Management:** Save, load, and list trained model artifacts efficiently using a local storage system.
- **Data Preprocessing:** Compose multiple data preprocessors to prepare datasets for training and prediction.
- **RESTful API Endpoints:** Interact with the system through well-defined API endpoints for health checks, dataset retrieval, model management, training, and prediction.
- **Comprehensive Testing:** Ensure code reliability and integrity with a suite of unit tests using pytest.


## Components

### 1. **Source Code (`src/`)**

- **`common/`**
  - **`const.py`:** Defines constant values used across the project, such as directory paths.
  - **`utils.py`:** Utility functions for JSON and pickle serialization, as well as logging initialization.

- **`lib/`**
  - **`datasets/`:** Handles dataset loading and interfaces.
    - **`dvach.py`:** Implements the `DvachDataset` class for loading the DVach dataset.
    - **`interfaces.py`:** Defines abstract classes and data structures for datasets, data, and targets.
  
  - **`models/`:** Manages machine learning models and their interfaces.
    - **`logistic_regression.py`:** Implements the `LogisticRegressionModel` class.
    - **`catboost.py`:** Implements the `CatBoostModel` class.
    - **`interfaces.py`:** Defines the `ModelInterface` abstract base class and `ModelConfig` data model.
  
  - **`preprocessors/`:** Composes and manages data preprocessors.
    - **`compose.py`:** Implements the `ComposePreprocessor` class to chain multiple preprocessors.
    - **`tf_idf.py`:** Implements the `TfIdfPreprocessor` class for TF-IDF vectorization.
    - **`drop.py`:** Implements the `DropPreprocessor` class to remove specified columns.
    - **`interfaces.py`:** Defines abstract classes and configurations for data preprocessors.
  
  - **`storage/`:** Manages artifact storage.
    - **`local_artifact_storage.py`:** Implements the `LocalArtifactStorage` class for saving and loading model artifacts.
  
  - **`trainer/`:** Handles the training and evaluation pipeline.
    - **`trainer.py`:** Implements the `Trainer` class for training models.
    - **`interfaces.py`:** Defines the `Metrics` data model.
  
  - **`web/`:** Contains web handlers and interfaces.
    - **`handlers.py`:** Implements API handlers for training, prediction, and artifact listing.
    - **`interfaces.py`:** Defines Pydantic models for API requests and validation.

- **`main.py`:** Initializes the FastAPI application and defines the API endpoints.

### 2. **Tests (`tests/`)**

- **`test_list_artifacts_handler.py`:** Tests the artifact listing functionality.
- **`test_logistic_regression.py`:** Tests the `LogisticRegressionModel` class.
- **`test_predict_handler.py`:** Tests the prediction handler.
- **`test_train_handler.py`:** Tests the training handler.
- **`test_local_artifact_storage.py`:** Tests the local artifact storage system.

### 3. **Configuration and Dependencies**

- **`.gitignore`:** Specifies files and directories to be ignored by Git.
- **`pytest.ini`:** Configures pytest settings.
- **`requirements.txt`:** Lists project dependencies, including `fastapi`, `uvicorn`, `dvc`, and `dvc-s3`.
- **`README.md`:** Provides an overview of the project (currently being updated).

## Use Cases

- **Training Models:** Users can train machine learning models by sending a `TrainRequest` to the `/models/train` endpoint with the desired dataset and model configurations.
  
- **Making Predictions:** Once trained, models can be used to make predictions by sending a `PredictRequest` to the `/models/predict` endpoint, specifying the model artifact and input data.
  
- **Managing Artifacts:** Users can list all saved model artifacts using the `/models/artifacts` endpoint, facilitating easy model versioning and management.

- **Health Monitoring:** The `/health` endpoint allows users to check the application's health status.

## Getting Started

### Prerequisites

- Python 3.8+
- Pip
- [DVC](https://dvc.org/) (for data versioning)
- [DVC S3](https://dvc.org/doc/user-guide/storage-s3) (if using S3 for storage)

### Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/ml-model-management-api.git
   cd ml-model-management-api
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**

   Ensure that the `PROJECT_ROOT` environment variable is set. By default, it uses the current working directory.

   ```bash
   export PROJECT_ROOT=/path/to/project  # On Windows: set PROJECT_ROOT=C:\path\to\project
   ```

5. **Initialize DVC:**

   ```bash
   dvc init
   ```

6. **Add Datasets:**

   ```bash
   dvc add data/dvach/train.csv
   dvc add data/dvach/test.csv
   ```

7. **Commit Changes:**

   ```bash
   git add .
   git commit -m "Initial commit with project structure and dependencies"
   ```

### Running the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```


Access the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

### Running Tests

Execute the test suite using pytest:

```bash
pytest
```


## Code References

- **Training a Model:**
  
  The `train_handler` function in `src/lib/web/handlers.py` handles training requests by loading the specified dataset, initializing the model from `MODELS_MAP`, and using the `Trainer` class to fit and evaluate the model.
  
  ```python
  def train_handler(request: TrainRequest) -> dict:
      train_dataset: Dataset = DATASETS_MAP[request.dataset.name].load(split="train")
      ...
      trainer = Trainer(model, train_dataset, test_dataset)
      model: ModelInterface = trainer.fit()
      metrics = trainer.evaluate()
      ...
  ```

- **Data Preprocessing:**
  
  The `ComposePreprocessor` class in `src/lib/preprocessors/compose.py` allows chaining multiple preprocessors like `TfIdfPreprocessor` and `DropPreprocessor`.

  ```python
  class ComposePreprocessor(DataPreprocessorInterface):
      def __init__(self, preprocessors: list[DataPreprocessorInterface]):
          self._preprocessors = preprocessors

      def fit(self, data: pd.DataFrame | pd.Series) -> T.Self:
          for preprocessor in self._preprocessors:
              preprocessor.fit(data)
          return self

      def transform(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
          result_data = deepcopy(data)
          for preprocessor in self._preprocessors:
              result_data = preprocessor.transform(result_data)
          return result_data
  ```

- **Artifact Storage:**
  
  The `LocalArtifactStorage` class in `src/lib/storage/local_artifact_storage.py` manages saving and loading model artifacts locally.

  ```python
  class LocalArtifactStorage(ArtifactStorageInterface):
      def save(self, artifact: ModelInterface, dataset_name: str) -> None:
          artifact_name = self._get_artifact_name(artifact, dataset_name)
          artifact.save(ARTIFACTS_DIR / artifact_name)
          return artifact_name

      def load(self, artifact_name: str) -> ModelInterface:
          model_name = self._get_model_name_from_artifact_name(artifact_name)
          return MODELS_MAP[model_name].load(ARTIFACTS_DIR / artifact_name)
  ```

- **API Endpoints:**
  
  Defined in `src/main.py`, endpoints like `/models/train`, `/models/predict`, and `/models/artifacts` interact with their respective handlers.

  ```python
  @app.post("/models/train")
  async def train(request: TrainRequest):
      return train_handler(request)
  ```

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

2. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Create a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) for the web framework.
- [DVC](https://dvc.org/) for data versioning.
- [Pydantic](https://pydantic-docs.helpmanual.io/) for data validation.
- [Scikit-learn](https://scikit-learn.org/) for machine learning tools.
