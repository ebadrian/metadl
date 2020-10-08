FROM tensorflow/tensorflow:2.3.0-gpu

# List of common Python packages installed in the Docker image used for this
# challenge. Participants are welcome to suggest other popular Python packages
# to be installed. If necessary, we'll update the Docker image to satisfy the
# need for most participants.
# In the case where you want to use less common packages, you can simply put
# all these packages in the the same folder of your submission (together with
# `model.py`) and the CodaLab platform should be able to find them.
RUN pip install matplotlib
RUN pip install jupyter
RUN pip install scikit-learn

# Packages to be activated: Following packages are demanded by one of the
# participants. If another participant asks to install one of these packages,
# we'll uncomment corresponding line and rebuild this image.
# RUN pip install tensorflow_hub==0.7.0
# RUN pip install fastai
RUN pip install tensorflow_addons
RUN pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /app/codalab
ADD VERSION .

# Install 
# MetaDL package: https://github.com/ebadrian/metadl
RUN mkdir /app/packages
COPY .meta-dataset/ /app/packages/meta-dataset/
COPY meta_dataset_setup.py /app/packages/meta-dataset/setup.py
RUN cd /app/packages/meta-dataset; pip install -e .;
# and meta-dataset package: https://github.com/google-research/meta-dataset 
COPY .metadl/ /app/packages/metadl/
RUN cd /app/packages/metadl; pip install -e .;