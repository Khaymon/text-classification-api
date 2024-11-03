import unittest
from src.lib.datasets.dvach import DvachDataset
from src.lib.datasets.interfaces import Data, Targets

class TestDvachDataset(unittest.TestCase):
    def test_load_train_data(self):
        dataset = DvachDataset.load(split="train")
        self.assertIsInstance(dataset, DvachDataset)
        self.assertIsInstance(dataset.data, Data)
        self.assertIsInstance(dataset.targets, Targets)
        self.assertGreater(len(dataset.data.to_list()), 0, "Data should not be empty")
        self.assertGreater(len(dataset.targets.to_list()), 0, "Targets should not be empty")

    def test_load_test_data(self):
        dataset = DvachDataset.load(split="test")
        self.assertIsInstance(dataset, DvachDataset)
        self.assertIsInstance(dataset.data, Data)
        self.assertIsInstance(dataset.targets, Targets)
        self.assertGreater(len(dataset.data.to_list()), 0, "Data should not be empty")
        self.assertGreater(len(dataset.targets.to_list()), 0, "Targets should not be empty")

if __name__ == "__main__":
    unittest.main()