from frozendict import frozendict

from .dvach import DvachDataset

DATASETS_MAP = frozendict({
    DvachDataset.NAME: DvachDataset,
})
