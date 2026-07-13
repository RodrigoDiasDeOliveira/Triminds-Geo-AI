resource "google_logging_project_sink" "training_logs" {

  name = "satellite-training-logs"

  destination = "storage.googleapis.com/${google_storage_bucket.datasets.name}"

  filter = "resource.type=gce_instance"
}
