import pandas as pd
from typing import Literal

from .interfaces import Dataset, Data, Targets
from ...common.const import DATA_DIR


class DvachDataset(Dataset):
    NAME = "dvach"

    @classmethod
    def load(cls, *args, split: Literal["train", "test"], **kwargs) -> "Dataset":
        dvach_split_df = pd.read_csv(DATA_DIR / f"dvach/{split}.csv")
        dvach_split_data = Data(dvach_split_df["comment"].tolist())
        dvach_split_targets = Targets(dvach_split_df["toxic"].tolist())
        return cls(dvach_split_data, dvach_split_targets)
