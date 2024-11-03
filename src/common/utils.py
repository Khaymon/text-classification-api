import json
from pathlib import Path
import pickle


class JsonHelper:
    @staticmethod
    def save(values: dict, path: Path):
        with open(path, "w") as fout:
            json.dump(values, fout)

    @staticmethod
    def load(path: Path) -> dict:
        with open(path, "r") as fin:
            return json.load(fin)


class PickleHelper:
    @staticmethod
    def save(obj, path: Path):
        with open(path, "wb") as fout:
            pickle.dump(obj, fout)

    @staticmethod
    def load(path: Path):
        with open(path, "rb") as fin:
            return pickle.load(fin)
