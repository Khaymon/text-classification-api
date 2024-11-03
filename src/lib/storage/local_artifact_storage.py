import os

from src.lib.storage.interfaces import ArtifactStorageInterface
from src.lib.models.interfaces import ModelInterface
from src.common.const import ARTIFACTS_DIR
from src.lib.models import MODELS_MAP


class LocalArtifactStorage(ArtifactStorageInterface):
    def _get_artifact_name(self, artifact: ModelInterface, dataset_name: str) -> str:
        existing_artifacts = self.list()
        index = len(existing_artifacts) + 1
        return f"{index}__{artifact.NAME}__{dataset_name}"
    
    def _get_model_name_from_artifact_name(self, artifact_name: str) -> str:
        parts = artifact_name.split("__")
        return parts[1]

    def save(self, artifact: ModelInterface, dataset_name: str) -> None:
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        artifact_name = self._get_artifact_name(artifact, dataset_name)
        artifact.save(ARTIFACTS_DIR / artifact_name)
        return artifact_name

    def load(self, artifact_name: str) -> ModelInterface:
        assert artifact_name in self.list(), f"Artifact {artifact_name} not found"
        model_name = self._get_model_name_from_artifact_name(artifact_name)
        return MODELS_MAP[model_name].load(ARTIFACTS_DIR / artifact_name)
    
    def list(self) -> list[str]:
        return [f.name for f in os.scandir(ARTIFACTS_DIR) if f.is_dir()]
