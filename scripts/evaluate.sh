from google.cloud import aiplatform


class VertexModelDeployer:

    def __init__(self, project_id, region):

        aiplatform.init(
            project=project_id,
            location=region
        )

    def deploy_model(
        self,
        model_path,
        display_name,
        machine_type="n1-standard-4"
    ):

        model = aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=model_path,
            serving_container_image_uri=(
                "us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest"
            )
        )

        endpoint = model.deploy(
            machine_type=machine_type
        )

        return endpoint