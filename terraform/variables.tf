variable "project_id" {}

variable "region" {
  default = "us-central1"
}

variable "bucket_name" {}

variable "artifact_repository" {
  default = "satellite-ai"
}

variable "service_account_name" {
  default = "satellite-ai-sa"
}

variable "topic_name" {
  default = "satellite-images"
}

variable "scheduler_name" {
  default = "daily-training"
}
