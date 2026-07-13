resource "google_cloud_scheduler_job" "training" {

  name = var.scheduler_name

  region = var.region

  schedule = "0 3 * * *"

  time_zone = "UTC"

  pubsub_target {

    topic_name = google_pubsub_topic.images.id

    data = base64encode("start-training")
  }
}
