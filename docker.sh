#!/bin/bash
# build and push docker image to docker hub

docker image build --tag vtluk/paperspace-tf-gpu:1.0 .
docker image push vtluk/paperspace-tf-gpu:1.0
