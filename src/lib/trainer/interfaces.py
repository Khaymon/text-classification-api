from pydantic import BaseModel


class Metrics(BaseModel):
    """
    Model to store evaluation metrics for a machine learning model.

    Attributes:
        f1 (float): The F1 score.
        accuracy (float): The accuracy score.
        precision (float): The precision score.
        recall (float): The recall score.
    """
    f1: float
    accuracy: float
    precision: float
    recall: float
