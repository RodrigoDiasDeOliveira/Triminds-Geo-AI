resource "google_project_iam_member" "vertex_admin" {

  project = var.project_id

  role = "roles/aiplatform.user"

  member = "serviceAccount:${google_service_account.vertex.email}"
}

resource "google_project_iam_member" "storage_admin" {

  project = var.project_id

  role = "roles/storage.admin"

  member = "serviceAccount:${google_service_account.vertex.email}"
}
