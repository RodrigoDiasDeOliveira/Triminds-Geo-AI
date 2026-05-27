class TerraformGenerator:

    def generate_storage_bucket(self, bucket_name):

        return f"""
resource "google_storage_bucket" "{bucket_name}" {{
  name          = "{bucket_name}"
  location      = "US"
  force_destroy = true
}}
"""

    def generate_dataproc_cluster(self, cluster_name):

        return f"""
resource "google_dataproc_cluster" "{cluster_name}" {{
  name   = "{cluster_name}"
  region = "us-central1"

  cluster_config {{
    master_config {{
      num_instances = 1
      machine_type  = "n1-standard-4"
    }}

    worker_config {{
      num_instances = 2
      machine_type  = "n1-standard-4"
    }}
  }}
}}
"""