#!/bin/bash
set -ex

USERNAME=ebadrian
IMAGE=metadl
VERSION=`cat VERSION`

rm -rf .metadl
rm -rf .meta-dataset

# git clone https://github.com/ebadrian/metadl.git
mkdir .metadl
cp -r ../metadl .metadl/
cp ../setup.py .metadl/

# WARNING: you should make following clone in parallel with this repo (metadl)
# git clone https://github.com/google-research/meta-dataset.git
cp -r ../../meta-dataset .meta-dataset

docker build -t $USERNAME/$IMAGE:gpu-$VERSION .
docker tag $USERNAME/$IMAGE:gpu-$VERSION $USERNAME/$IMAGE:gpu-latest

docker build -t $USERNAME/$IMAGE:cpu-$VERSION -f Dockerfile.cpu .
docker tag $USERNAME/$IMAGE:cpu-$VERSION $USERNAME/$IMAGE:cpu-latest

