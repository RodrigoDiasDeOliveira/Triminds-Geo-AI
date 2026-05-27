from google.cloud import storage


class GCSStorage:

    def __init__(self, bucket_name):

        self.client = storage.Client()

        self.bucket = self.client.bucket(bucket_name)

    def upload_file(self, source, destination):

        blob = self.bucket.blob(destination)

        blob.upload_from_filename(source)