variable "project_id" {
  type = string
}

variable "region" {
  type = string
  default = "us-central1"
}

variable "zone" {
  type = string
  default = "us-central1-c"
}
variable "machine_type" {
  type = string
  default = "e2-standard-16"
}