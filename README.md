# paperspace-amazon-review
Repo for paperspace DL model training


# Docker Image for Training

Base image is tensorflow/tensorflow (2.0)

## Build docker image for training

docker image build --tag vtluk/paperspace-tf:1.0 .

## Push docker image to docker hub

docker image push

## Running Docker Image locally (interactive mode)

docker run -it --name tf vtluk/paperspace-tf:1.0
