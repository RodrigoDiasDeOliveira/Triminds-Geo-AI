from google.cloud import bigquery


class BigQueryClient:
    def __init__(self, project_id):

        self.client = bigquery.Client(project=project_id)

    def run_query(self, query):

        query_job = self.client.query(query)

        return query_job.result()

    def create_dataset(self, dataset_id):

        dataset = bigquery.Dataset(dataset_id)
        dataset.location = "US"

        return self.client.create_dataset(dataset, exists_ok=True)
