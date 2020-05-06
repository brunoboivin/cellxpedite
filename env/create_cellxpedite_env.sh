#!/bin/bash
# ensure instance is ready
while [ ! -f /var/lib/cloud/instance/boot-finished ]
do
  echo "Waiting for AWS cloud-init..." >> waiting_for_AWS_cloud-init.log
  sleep 1
done
sleep 30
#-----------------------------
# Ubuntu & Java
#-----------------------------
# update Ubuntu and the apt package manager
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y build-essential git libfuse-dev libcurl4-openssl-dev libxml2-dev mime-support automake libtool
sudo apt-get install -y pkg-config libssl-dev unzip
sudo apt-get install -y default-jdk
#-----------------------------
# miniconda
# http://conda.pydata.org/docs/help/silent.html
#-----------------------------
curl -o /home/ubuntu/miniconda3.sh "https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
bash /home/ubuntu/miniconda3.sh -b -p $HOME/miniconda3
tee /home/ubuntu/.profile <<EOL
PATH="$HOME/miniconda3/bin:$PATH"
EOL
source /home/ubuntu/.profile
rm miniconda3.sh
# configure miniconda
conda update -y conda
conda config --set always_yes True
#-----------------------------
# awscli
#-----------------------------
# create the directory that stores AWS credentials
mkdir /home/ubuntu/.aws
#-----------------------------
# cxp python environment
#-----------------------------
# configure a conda environment for using CELLXPEDITE
tee /home/ubuntu/conda_env_cxp.yml <<EOL
name: cxp
channels:
  - conda-forge
  - anaconda
  - plotly
  - goodman # mysql-python for mac
  - bioconda
  - cyclus # java-jdk for windows
dependencies:
  - click
  - fabric
  - javabridge
  - java-jdk
  - jupyter
  - matplotlib
  - numpy
  - pandas
  - pillow
  - pip
  - plotly
  - pytest
  - python=2
  - scikit-image
  - scipy
  - seaborn
  - pip:
    - awscli
    - python-bioformats
    - seaborn
    - Cython
    - xlrd
    - xlwt
EOL
conda env create -f conda_env_cxp.yml
#-----------------------------
# Cell Painting
#-----------------------------
# configure a conda environment for using Cell Painting scripts with python 3.5
tee /home/ubuntu/conda_env_cellpainting_python_3.yml <<EOL
# run: conda env create -f conda_env_cellpainting_python_3.yml
# run: conda env update -f conda_env_cellpainting_python_3.yml
# run: conda env remove -n cellpainting_python_3
name: cellpainting_python_3
# in order of priority: lowest (top) to highest (bottom)
channels:
  - conda-forge
  - anaconda
dependencies:
  - python=3
  - csvkit
  - ipython
  - jq
  - pyyaml
EOL
conda env create -f conda_env_cellpainting_python_3.yml
#-----------------------------
# linux tools
#-----------------------------
sudo apt-get install -y parallel
sudo apt-get install -y tmux
#-----------------------------
# docker
#-----------------------------
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
# provide necessary permissions
sudo groupadd docker
sudo usermod -aG docker $USER
sudo systemctl enable docker
sudo service docker start
# include cellprofiler images
sudo docker pull woolflabhms/cellprofiler:2.2.1
sudo docker pull woolflabhms/cellprofiler:3.0.0
#-----------------------------
# ffmpeg
#-----------------------------
sudo apt-get install -y ffmpeg
#-----------------------------
# R
# * need at least 2GB of RAM to compile 'readr' pkg
#-----------------------------
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo add-apt-repository 'deb [arch=amd64,i386] https://cran.rstudio.com/bin/linux/ubuntu xenial/'
sudo apt-get update
sudo apt-get install -y r-base
sudo apt-get update
sudo Rscript -e 'install.packages(c("readr","dplyr","stringr","dtw","foreach","doParallel","ggplot2","magrittr","devtools"), repos="https://cran.rstudio.com")'
sudo Rscript -e 'devtools::install_github("shntnu/cytominer", ref="handle-na-hack", dependencies = TRUE, build_vignettes = TRUE)'
#-----------------------------
# command line tools
#-----------------------------
sudo apt-get install -y bc
#-----------------------------
# gsl and mlpy
#-----------------------------
sudo apt-get install -y gsl-bin libgsl0-dev
wget https://sourceforge.net/projects/mlpy/files/mlpy%203.5.0/mlpy-3.5.0.tar.gz
tar xvzf mlpy-3.5.0.tar.gz
source activate cxp
cd mlpy-3.5.0/
python setup.py install
