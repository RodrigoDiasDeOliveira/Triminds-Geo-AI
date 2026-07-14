output "bucket_name" {

  value = google_storage_bucket.datasets.name
}

output "service_account" {

  value = google_service_account.vertex.email
}

output "artifact_registry" {

  value = google_artifact_registry_repository.docker.repository_id
}

output "pubsub_topic" {

  value = google_pubsub_topic.images.name
}

output "cloud_run_url" {

  value = google_cloud_run_service.inference.status[0].url
}
