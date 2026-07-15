from google.cloud import storage


class StorageManager:
    """
    Generic storage manager backed by Google Cloud Storage.
    """

    def __init__(self, bucket_name: str):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)

    def upload_file(self, source: str, destination: str) -> None:
        blob = self.bucket.blob(destination)
        blob.upload_from_filename(source)


# Backward compatibility
GCSStorage = StorageManager
