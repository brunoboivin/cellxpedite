# Project Settings
# Ensure input and output buckets are distinct to avoid downloading input data to instance twice
variable "dataset" {
  type = "list"
  default = [
    "myplate1"
  ]
}
variable "num_instances"        { default = 1 }
variable "s3_input_bucket"      { default = "cellxpedite/input/myplate1" }
variable "s3_output_bucket"     { default = "cellxpedite/output/myplate1" }
variable "github_project"       { default = "https://github.com/brunoboivin/cellxpedite.git" }

# Analysis Settings
# analysis_type should be one of:
# standard: analyze all wells and compute threshold
# 39b: used 39b well mapping and pre-computed threshold
# rb9d: used rb9d well mapping and pre-computed threshold
# scna8a: used scn8a well mapping and pre-computed threshold
variable "analysis_type"        { default = "standard" }
variable "script_file"          { default = "analyze_plate.sh" }
variable "illum_pipeline"       { default = "illumination_correction.cppipe" }
variable "segment_pipeline"     { default = "cell_segmentation.cppipe" }

# AWS Credentials
variable "key_name"             { default = "bruno-bch" }
variable "private_key"          { default = "~/.aws/bruno-bch.pem" }
variable "iam_instance_profile" { default = "woolflab" }

# AWS Instance Details
variable "instance_type"        { default = "m4.large" }
variable "root_volume_size"     { default = 100 }
variable "aws_instance_name"    { default = "cellxpedite" }
variable "ami"                  { default = "ami-0cf03c0452cd88a51" }
variable "aws_region"           { default = "us-east-1" }
variable "availability_zone"    { default = "us-east-1a" }
variable "vpc"                  { default = "vpc-987fa6e1" }
variable "subnet"               { default = "subnet-63222606" }
variable "security_group"       { default = "sg-25d1c25b" }

# Parallelism Options
variable "num_jobs_parallel"    { default = 4 }