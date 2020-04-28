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
train.sh: [-b batch_size] [-c lstm_cells] [-d dropout_rate] [-e epochs] [-l log_level] [-m machine_type]
               [-p patience] [-r recurrent_dropout_rate] <sample size>
Parameter(s):
  sample_size:                size of data set to train. available values: test, 50k, 100k, 200k, 500k, 1m, 2m, 4m, all
Options:
  -b batch size:              batch size for training. Default 32
  -c lstm_cells:              number of LSTM cells used for training. Default 128
  -d dropout_rate:            dropout rate for LSTM network. Default 0
  -e epochs:                  max number of epochs for training. Default 20
  -l log_level:               log level for logging. Default INFO
  -m machine_type:            Gradient machine type. Options C3 (CPU) or P4000 (GPU). Default P4000
  -p patience:                patience for early stopping. Default 4
  -r recurrent_dropout_rate:  recurrent dropout rate for LSTM cells. Default 0
Example:
  ./train.sh test # DEBUG MODE
  ./train.sh 1m
  ./train.sh -e 40 -d 0.2 1m
```

## Artifacts

Model training will generate the following artifacts on paperspace

```log
├── reports - csv report, missing words csv, network history
├── models - model binary, model json, model weights
```



  
