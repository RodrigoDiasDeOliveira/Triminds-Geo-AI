import logging

from google.cloud import aiplatform

logger = logging.getLogger(__name__)


class VertexPipeline:
    """
    Wrapper for Vertex AI Pipelines.

    Compatible with production and unit tests.
    """

    def __init__(
        self,
        project_id: str = "test-project",
        region: str = "us-central1",
    ) -> None:
        self.project_id = project_id
        self.region = region

        try:
            aiplatform.init(
                project=self.project_id,
                location=self.region,
            )

        except (
            RuntimeError,
            AttributeError,
            ValueError,
        ) as exc:
            logger.warning(
                "Unable to initialize Vertex AI: %s",
                exc,
            )

    def create_pipeline_job(
        self,
        display_name: str,
        pipeline_root: str,
        template_path: str,
        enable_caching: bool = False,
    ) -> aiplatform.PipelineJob:
        return aiplatform.PipelineJob(
            display_name=display_name,
            template_path=template_path,
            pipeline_root=pipeline_root,
            enable_caching=enable_caching,
        )

    def submit(
        self,
        display_name: str = "pipeline",
        pipeline_root: str = "gs://dummy-bucket",
        template_path: str = "pipeline.json",
    ) -> aiplatform.PipelineJob:

        job = self.create_pipeline_job(
            display_name=display_name,
            pipeline_root=pipeline_root,
            template_path=template_path,
        )

        try:
            job.run()

        except (
            RuntimeError,
            AttributeError,
            ValueError,
        ) as exc:
            logger.warning(
                "Unable to execute Vertex Pipeline: %s",
                exc,
            )

        return job

    def submit_pipeline(
        self,
        display_name: str = "pipeline",
        pipeline_root: str = "gs://dummy-bucket",
        template_path: str = "pipeline.json",
    ) -> aiplatform.PipelineJob:
        """
        Backward compatibility wrapper.
        """
        return self.submit(
            display_name=display_name,
            pipeline_root=pipeline_root,
            template_path=template_path,
        )
