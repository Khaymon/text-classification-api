import pandas as pd
from typing import Literal

from .interfaces import Dataset, Data, Targets


class DvachDataset(Dataset):
    NAME = "dvach"

    @classmethod
    def load(cls, *args, split: Literal["train", "test"], **kwargs) -> "Dataset":
        dvach_split_df = pd.read_csv(f"./data/dvach/{split}.csv")
        dvach_split_data = Data(dvach_split_df["comment"].tolist())
        dvach_split_targets = Targets(dvach_split_df["toxic"].tolist())
        return cls(dvach_split_data, dvach_split_targets)
