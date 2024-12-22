import pandas as pd
from pydantic import BaseModel


class Data(BaseModel):
    texts: list[str]

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame({"text": self.texts})


class Dataset(BaseModel):
    texts: list[str]
    targets: list[int]

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame({"text": self.texts, "target": self.targets})

    def to_csv(self) -> str:
        return self.to_pandas().to_csv(index=False)

    @classmethod
    def from_csv(cls, csv: str) -> "Dataset":
        data = pd.read_csv(csv)
        return cls(texts=data["text"].tolist(), targets=data["target"].tolist())
