#!/bin/bash
# ensure instance is ready
while [ ! -f /var/lib/cloud/instance/boot-finished ]
do
  echo "Waiting for AWS cloud-init..." >> waiting_for_AWS_cloud-init.log
  sleep 1
done
sleep 30

# create directory to stores AWS credentials
mkdir /home/ubuntu/.aws

# update & install linux packages
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y build-essential libfuse-dev libcurl4-openssl-dev libtool libssl-dev gsl-bin libgsl0-dev
sudo apt-get install -y pkg-config default-jdk git unzip parallel tmux awscli

# install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
bash /home/ubuntu/Miniconda3-py39_4.10.3-Linux-x86_64.sh -b -p $HOME/miniconda3
echo PATH="$HOME/miniconda3/bin:$PATH" >> /home/ubuntu/.profile
source /home/ubuntu/.profile
rm Miniconda3-py39_4.10.3-Linux-x86_64.sh
conda update -y conda
conda config --set always_yes True

# conda environment for using CELLXPEDITE
tee /home/ubuntu/cxp_env.yml <<EOL
name: cxp
channels:
  - conda-forge
  - anaconda
  - bioconda
dependencies:
  - click=8.0.1
  - csvkit=1.0.6
  - ipython=7.16.1
  - javabridge=1.0.19
  - java-jdk=8.0.112
  - jq=1.6
  - pip=21.3
  - python=3.6
  - pyyaml=5.4.1
  - pip:
    - cython==0.29.24
    - machine-learning-py==3.5.8
    - matplotlib==3.3.4
    - numpy==1.19.3
    - pandas==1.1.5
    - pillow==8.3.2
    - python-bioformats==4.0.3
    - scikit-image==0.17.2
    - scipy==1.5.3
EOL
conda env create -f cxp_env.yml

# docker
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

sudo groupadd docker
sudo usermod -aG docker $USER
sudo systemctl enable docker
sudo service docker start

# cellprofiler
sudo docker pull woolflabhms/cellprofiler:2.2.1
sudo docker pull woolflabhms/cellprofiler:3.0.0

# R and required packages (need at least 2GB of RAM to compile 'readr' pkg)
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo add-apt-repository 'deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu xenial/'
sudo apt-get update
sudo apt-get install -y r-base
sudo apt-get update
sudo Rscript -e 'install.packages(c("readr","dplyr","stringr","dtw","foreach","doParallel","ggplot2","magrittr","devtools"), repos="https://cran.rstudio.com")'
sudo Rscript -e 'devtools::install_github("shntnu/cytominer", ref="handle-na-hack", dependencies = TRUE, build_vignettes = TRUE)'
