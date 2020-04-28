# paperspace-amazon-review
Repo for paperspace DL model training


# Create Docker Image for Training

Base image is tensorflow/tensorflow with GPU (2.0)
Additional python packages are installed via pip using docker-requirements.txt file

## Build and push docker image for training

./docker.sh

## Running Docker Image locally (interactive mode)

docker run -it --name tf vtluk/paperspace-tf:1.0 /bin/bash


# Train Model

Use the following script to traing models on gradient paperspace.

P4000 GPU instance will be created for model training

```bash
Usage: train.sh <sample size>
    example: ./train.sh test # debug mode
    example: ./train.sh 1m
```

## Artifacts

Model training will generate the following artifacts on paperspace

```log
├── reports - csv report, missing words csv, network history
├── models - model binary, model json, model weights
```



  
