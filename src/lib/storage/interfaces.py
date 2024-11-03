from src.lib.models.interfaces import ModelInterface


class ArtifactStorageInterface:
    def save(self, artifact: ModelInterface) -> None:
        raise NotImplementedError

    def load(self, artifact_name: str) -> ModelInterface:
        raise NotImplementedError
    
    def list(self) -> list[str]:
        raise NotImplementedError
