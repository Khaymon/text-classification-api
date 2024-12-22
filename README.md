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


## Running with Docker

To run the application using Docker, follow these steps:

1. **Build the Docker Images:**

   Navigate to the root directory of the project and build the Docker images using the following command:

   ```bash
   MINIO_ROOT_USER=your_minio_username MINIO_ROOT_PASSWORD=your_minio_password docker-compose build
   ```

2. **Start the Services:**

   Use Docker Compose to start all the services defined in the `docker-compose.yml` file:

   ```bash
   docker-compose up
   ```

   This command will start the FastAPI server, gRPC service, Streamlit app, Minio server, and MLflow server.

3. **Access the Services:**

   - **FastAPI API Documentation:** Access the FastAPI API documentation at [http://localhost:8005/docs](http://localhost:8005/docs).
   - **Streamlit App:** Access the Streamlit app at [http://localhost:8501](http://localhost:8501).
   - **MLflow Tracking Server:** Access the MLflow tracking server at [http://localhost:5000](http://localhost:5000).
   - **Minio:** Access the Minio server at [http://localhost:9005](http://localhost:9005).
   - **gRPC:** Access the gRPC server at [http://localhost:50051](http://localhost:50051).

4. ***Use the APP via streamlit.***

- Prepare a train.csv dataset with the following columns: `text` and `target`.
- Upload the dataset to the Minio server via streamlit (for example) -- go to `datasets` page and upload the dataset.
- Use `Train` page to train a model.
- Use `Predict` page to make predictions with the trained model.
- You can also access the training logs in the MLflow server.

5. **Stop the Services:**

   To stop the services, press `Ctrl+C` in the terminal where the services are running, or use the following command:

   ```bash
   docker-compose down
   ```

   This will stop and remove the containers, but the data in the `mlruns` volume will be preserved.
