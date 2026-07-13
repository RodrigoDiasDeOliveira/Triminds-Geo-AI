resource "google_pubsub_topic" "images" {

  name = var.topic_name
}

resource "google_pubsub_subscription" "processor" {

  name = "satellite-image-processor"

  topic = google_pubsub_topic.images.name
}
