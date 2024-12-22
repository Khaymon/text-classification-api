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
    - **`storage.py`:** Implements the `DatasetStorage` class for loading and saving datasets.
    - **`interfaces.py`:** Defines the `DatasetInterface` abstract base class and `DatasetConfig` data model.
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

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) for the web framework.
- [DVC](https://dvc.org/) for data versioning.
- [Pydantic](https://pydantic-docs.helpmanual.io/) for data validation.
- [Scikit-learn](https://scikit-learn.org/) for machine learning tools.

## Running with Docker

To run the application using Docker, follow these steps:

1. **Build the Docker Images:**

   Navigate to the root directory of the project and build the Docker images using the following command:

   ```bash
   docker-compose build
   ```

2. **Start the Services:**

   Use Docker Compose to start all the services defined in the `docker-compose.yml` file:

   ```bash
   docker-compose up
   ```

   This command will start the FastAPI server, gRPC service, Streamlit app, and MLflow server.

3. **Access the Services:**

   - **FastAPI API Documentation:** Access the FastAPI API documentation at [http://localhost:8005/docs](http://localhost:8005/docs).
   - **Streamlit App:** Access the Streamlit app at [http://localhost:8501](http://localhost:8501).
   - **MLflow Tracking Server:** Access the MLflow tracking server at [http://localhost:5000](http://localhost:5000).
   - **Minio:** Access the Minio server at [http://localhost:9005](http://localhost:9005).
   - **gRPC:** Access the gRPC server at [http://localhost:50051](http://localhost:50051).

4. **Stop the Services:**

   To stop the services, press `Ctrl+C` in the terminal where the services are running, or use the following command:

   ```bash
   docker-compose down
   ```

   This will stop and remove the containers, but the data in the `mlruns` volume will be preserved.
