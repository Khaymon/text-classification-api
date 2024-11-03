from pathlib import Path
import pickle


class PickleHelper:
    @staticmethod
    def save(obj, path: Path):
        with open(path, "wb") as fout:
            pickle.dump(obj, fout)

    @staticmethod
    def load(path: Path):
        with open(path, "rb") as fin:
            return pickle.load(fin)
