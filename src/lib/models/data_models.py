import typing as T

from pydantic import BaseModel

from src.lib.preprocessors.compose import ComposePrerpocessorConfig


class ModelConfig(BaseModel):
    preprocessor: ComposePrerpocessorConfig
    model_configuration: dict[str, T.Any] | None = None
