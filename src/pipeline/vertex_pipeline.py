from google.cloud import aiplatform


class VertexPipeline:

    def __init__(self, project_id, region):

        self.project_id = project_id
        self.region = region

        aiplatform.init(
            project=project_id,
            location=region
        )

    def create_pipeline_job(self, display_name, pipeline_root, template_path):

        job = aiplatform.PipelineJob(
            display_name=display_name,
            template_path=template_path,
            pipeline_root=pipeline_root,
            enable_caching=False
        )

        job.run()

        return job