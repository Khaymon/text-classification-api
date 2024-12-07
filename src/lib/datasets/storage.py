import tempfile

import minio

import src.common.utils as utils

from .data_models import Dataset


logger = utils.initialize_logging(__name__)


class DatasetsStorage:
    ENDPOINT = "localhost:9000"
    client = minio.Minio(
        ENDPOINT, access_key="minioadmin", secret_key="minioadmin", secure=False
    )

    def upload(self, dataset: Dataset, name: str):
        if self.client.bucket_exists(name):
            raise ValueError(f"Bucket {name} already exists")

        self.client.make_bucket(name)
        with tempfile.NamedTemporaryFile("r+b") as f:
            f.write(dataset.to_csv().encode("utf-8"))
            self.client.fput_object(name, "train.csv", f.name)

    def download(self, name: str) -> Dataset:
        if not self.client.bucket_exists(name):
            raise ValueError(f"Bucket {name} does not exist")

        with tempfile.NamedTemporaryFile("r+b") as f:
            f.write(self.client.get_object(name, "train.csv").read().decode())

            return Dataset.from_csv(f.name)

    def list(self) -> list[str]:
        logger.info(f"request list of datasets")
        return [b.name for b in self.client.list_buckets()]
