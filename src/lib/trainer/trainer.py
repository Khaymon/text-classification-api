from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import src.common.utils as utils
from src.lib.models.interfaces import ModelInterface
from src.lib.datasets.interfaces import Dataset, Data, Targets
from src.lib.trainer.interfaces import Metrics


LOGGER = utils.initialize_logging(__name__)


class Trainer:
    def __init__(self, model: ModelInterface, train_dataset: Dataset, test_dataset: Dataset | None = None):
        """
        Initialize the Trainer with a model and datasets.

        Args:
            model (ModelInterface): The machine learning model to train.
            train_dataset (Dataset): The dataset used for training the model.
            test_dataset (Dataset | None, optional): The dataset used for testing the model. Defaults to None.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def fit(self) -> ModelInterface:
        """
        Fit the model using the training dataset.

        Returns:
            ModelInterface: The fitted model instance.
        """
        LOGGER.info(f"Fit model {self.model.NAME} with dataset {self.train_dataset.NAME}")
        self.model = self.model.fit(self.train_dataset)
        return self.model

    def predict(self, data: Data | None = None) -> Targets:
        """
        Make predictions using the model on the provided data or test dataset.

        Args:
            data (Data | None, optional): The data to make predictions on. If None, uses the test dataset. Defaults to None.
        
        Returns:
            Targets: The prediction results.
        """
        data = data or self.test_dataset.data if self.test_dataset is not None else None
        if data is None:
            raise ValueError("You should provide data for prediction if there is no test dataset")

        LOGGER.info(f"Predict with model {self.model.NAME}")

        return self.model.predict(data)
    
    def evaluate(self, dataset: Dataset | None = None) -> Metrics | None:
        """
        Evaluate the model's performance on the provided dataset.

        Args:
            dataset (Dataset | None, optional): The dataset to evaluate the model on. If None, uses the test dataset. Defaults to None.
        
        Returns:
            Metrics | None: The evaluation metrics.
        """
        if dataset is None:
            dataset = self.test_dataset

        predictions = self.model.predict(dataset.data)
        return self.compute_metrics(predictions, dataset.targets)

    def compute_metrics(self, predicted: Targets, true: Targets) -> Metrics:
        """
        Compute evaluation metrics based on predictions and true targets.

        Args:
            predicted (Targets): The predicted targets.
            true (Targets): The true targets.
        
        Returns:
            Metrics: The computed evaluation metrics.
        """
        true_list = true.to_list()
        predicted_list = predicted.to_list()
        return Metrics(
            accuracy=float(accuracy_score(true_list, predicted_list)),
            f1=float(f1_score(true_list, predicted_list)),
            precision=float(precision_score(true_list, predicted_list)),
            recall=float(recall_score(true_list, predicted_list)),
        )
