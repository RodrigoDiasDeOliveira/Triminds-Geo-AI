resource "google_service_account" "vertex" {

  account_id = var.service_account_name

  display_name = "Satellite AI Service Account"
}
