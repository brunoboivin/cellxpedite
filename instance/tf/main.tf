provider "aws" {
  region = "${var.aws_region}"
}

resource "aws_instance" "cellxpedite" {
  count = "${var.num_instances}"
  ami = "${var.ami}"
  associate_public_ip_address = true
  availability_zone = "${var.availability_zone}"
  iam_instance_profile = "${var.iam_instance_profile}"
  instance_type = "${var.instance_type}"
  key_name = "${var.key_name}"
  subnet_id = "${var.subnet}"
  vpc_security_group_ids = [
    "${var.security_group}"]

  root_block_device {
    volume_size = "${var.root_volume_size}"
  }
}

resource "null_resource" "configure-ips" {
  count = "${var.num_instances}"

  connection {
    user        = "ubuntu"
    private_key = "${file("${var.private_key}")}"
    host        = "${element(aws_instance.cellxpedite.*.public_ip, count.index)}"
  }

  provisioner "file" {
    source = "../../gitkeys/id_rsa"
    destination = "~/.ssh/id_rsa"
  }
  provisioner "file" {
    source = "../../gitkeys/id_rsa.pub"
    destination = "~/.ssh/id_rsa.pub"
  }
  provisioner "file" {
    source = "../scripts/${var.script_file}"
    destination = "${var.script_file}"
  }

  provisioner "remote-exec" {
    inline = [
      "chmod +x ${var.script_file}",
      "tmux new -d -s cellxpedite \"/bin/sh -c './${var.script_file} ${element(var.dataset, count.index)} ${var.s3_input_bucket} ${var.s3_output_bucket} ${var.github_project} ${var.num_jobs_parallel} ${var.analysis_type} ${var.illum_pipeline} ${var.segment_pipeline} >> ~/log.txt; exec bash'\""
    ]
  }
}
