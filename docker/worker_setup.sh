# This script is working only for a particular spec of VM : 
# Unbuntu 18.04 x86_64


# Pre-requesites 
sudo apt install -y gcc make pkg-config
sudo apt install -y dkms build-essential linux-headers-generic

# Get Nvidia cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin

sudo mv cuda-ubuntu1804.pin  /etc/apt/preferences.d/cuda-repository-pin-600

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"

sudo apt-get update
sudo apt-get -y install cuda

export PATH=/usr/local/cuda-11.0/bin${PATH:+:${PATH}}$
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

#Check installation
source .bashrc
sudo apt install nvidia-cuda-toolkit
nvcc -V

# Get docker 
curl https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

# Get nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit

sudo systemctl restart docker
sudo apt-get install docker-ce nvidia-docker2
sudo systemctl restart docker

# Setting .env var for the CodaLab worker queue
echo "BROKER_URL=pyamqp://d99f8b9c-f061-44e5-a634-79de4a3e8133:fbc69cc5-f231-45ad-be37-4d20c2cd7937@competitions.codalab.org:5671/dc7b64c9-b08b-414b-a73e-7964d3d6df93" >> .env
echo "BROKER_USE_SSL=True" >> .env


# Create and run new CodaLab container
sudo mkdir -p /tmp/codalab && sudo nvidia-docker run \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /var/lib/nvidia-docker/nvidia-docker.sock:/var/lib/nvidia-docker/nvidia-docker.sock \
    -v /tmp/codalab:/tmp/codalab \
    -d \
    --name compute_worker \
    --env-file .env \
    --restart unless-stopped \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    codalab/competitions-v1-nvidia-worker:v1.5-compat



