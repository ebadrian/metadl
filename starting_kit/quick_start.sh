# Global env variables
ENV_NAME=metadl

# Create conda env
conda create -n $ENV_NAME python=3.7
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

cd ../../
# Clone and install meta-dataset package (in metadl's ROOT)
git clone https://github.com/ebadrian/meta_dataset
#git clone https://github.com/google-research/meta-dataset.git

wget -c https://competitions.codalab.org/my/datasets/download/57327142-2155-4b37-9ee7-74820f56c812 -O omniglot.zip
# Unzip data 
unzip omniglot.zip

# Install packages in metadl repo
cd metadl/
pip install -e .
pip install -r requirements.txt

# Installing meta-dataset package
cd ../meta-dataset
#cp ../metadl/misc/setup.py .
pip install -e .

# Install tensorflow addons 
pip install tensorflow_addons

# Install notebook
conda install -c conda-forge jupyterlab
pip install jupyterlab

# tutorial.ipynb ready to use



