from pydantic import BaseModel


class Metrics(BaseModel):
    f1: float
    accuracy: float
    precision: float
    recall: float
