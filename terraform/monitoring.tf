resource "google_monitoring_notification_channel" "email" {

  display_name = "Admin"

  type = "email"

  labels = {

    email_address = "admin@example.com"
  }
}
