import json
import logging
from pathlib import Path
import pickle


DEFAULT_LOGGER_HANDLER_NAME = 'default'
DEFAULT_LOGGER_FORMAT = u'%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s'


class JsonHelper:
    @staticmethod
    def save(values: dict, path: Path):
        with open(path, "w") as fout:
            json.dump(values, fout, indent=4)

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


def initialize_logging(name: str | None = None) -> logging.Logger:
    """
    Initialize and configure the logging system.

    Args:
        name (str | None, optional): The name of the logger. Defaults to None.
    
    Returns:
        logging.Logger: The configured logger instance.
    """
    """Initialize logging with reasonable format."""

    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    _stream_handler = logging.StreamHandler()
    _stream_handler.set_name(DEFAULT_LOGGER_HANDLER_NAME)
    _stream_handler.setLevel(logging.DEBUG)
    _stream_handler.setFormatter(logging.Formatter(DEFAULT_LOGGER_FORMAT))
    log.addHandler(_stream_handler)
    log.propagate = False

    return log
