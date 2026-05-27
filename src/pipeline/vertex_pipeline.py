from google.cloud import aiplatform

from src.utils.config_loader import load_yaml


class VertexAIPipeline:

    def __init__(self, config_path):

        self.config = load_yaml(config_path)

        aiplatform.init(
            project=self.config["project_id"],
            location=self.config["region"]
        )

    def submit_training_job(
        self,
        display_name,
        container_uri,
        model_serving_container_image_uri
    ):

        job = aiplatform.CustomContainerTrainingJob(
            display_name=display_name,
            container_uri=container_uri,
            model_serving_container_image_uri=(
                model_serving_container_image_uri
            )
        )

        model = job.run(
            replica_count=1,
            machine_type="n1-standard-4"
        )

        return model