resource "google_cloud_run_service" "inference" {

  name = "satellite-inference"

  location = var.region

  template {

    spec {

      service_account_name = google_service_account.vertex.email

      containers {

        image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_repository}/satellite-api:latest"
      }
    }
  }

  traffic {

    percent = 100

    latest_revision = true
  }
}
