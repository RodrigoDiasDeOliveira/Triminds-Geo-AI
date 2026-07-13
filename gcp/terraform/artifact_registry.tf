resource "google_artifact_registry_repository" "docker" {

  repository_id = var.artifact_repository

  location = var.region

  format = "DOCKER"

  description = "Docker images for Satellite AI"
}
