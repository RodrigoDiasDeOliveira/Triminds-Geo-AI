from google.cloud import dataproc_v1 as dataproc


class DataprocJob:
    def __init__(self, project_id, region, cluster_name):

        self.project_id = project_id
        self.region = region
        self.cluster_name = cluster_name

        self.client = dataproc.JobControllerClient(
            client_options={"api_endpoint": f"{region}-dataproc.googleapis.com:443"}
        )

    def submit_pyspark_job(self, main_python_file_uri, args=None):

        job = {
            "placement": {"cluster_name": self.cluster_name},
            "pyspark_job": {"main_python_file_uri": main_python_file_uri, "args": args or []},
        }

        operation = self.client.submit_job_as_operation(
            request={"project_id": self.project_id, "region": self.region, "job": job}
        )

        return operation.result()
