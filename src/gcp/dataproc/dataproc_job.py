"""Thin wrapper around the Google Cloud Dataproc JobController client.

The ``google-cloud-dataproc`` package is an optional runtime dependency.
I import it lazily so unit tests can stub the module and so importing this
file never crashes when the package is not installed.i hope it works to you
"""

from __future__ import annotations

from typing import Any


class DataprocJob:
    def __init__(
        self,
        project_id: str,
        region: str,
        cluster_name: str,
        client: Any | None = None,
    ) -> None:
        self.project_id = project_id
        self.region = region
        self.cluster_name = cluster_name
        self._client = client

    def _get_client(self):
        if self._client is not None:
            return self._client
        # Lazy import — only require google-cloud-dataproc at call time.
        from google.cloud import dataproc_v1 as dataproc  # type: ignore

        endpoint = f"{self.region}-dataproc.googleapis.com:443"
        self._client = dataproc.JobControllerClient(client_options={"api_endpoint": endpoint})
        return self._client

    def submit_pyspark_job(
        self,
        main_python_file_uri: str,
        args: list[str] | None = None,
    ) -> Any:
        client = self._get_client()
        job = {
            "placement": {"cluster_name": self.cluster_name},
            "pyspark_job": {
                "main_python_file_uri": main_python_file_uri,
                "args": args or [],
            },
        }
        return client.submit_job(
            request={
                "project_id": self.project_id,
                "region": self.region,
                "job": job,
            }
        )
