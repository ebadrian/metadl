#!/bin/bash
set -ex

USERNAME=ebadrian
IMAGE=metadl
VERSION=`cat VERSION`

docker push $USERNAME/$IMAGE:gpu-$VERSION
docker push $USERNAME/$IMAGE:gpu-latest
#docker push $USERNAME/$IMAGE:cpu-$VERSION
#docker push $USERNAME/$IMAGE:cpu-latest

