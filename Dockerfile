# latest stable release of tensorflow image
# https://www.tensorflow.org/install/docker
#
# Build image: docker build -t vtluk/paperspace-tf-gpu:1.0 .
# Push image: docker push vtluk/paperspace-tf-gpu:1.0
#

FROM tensorflow/tensorflow:2.2.0rc2-gpu-py3


COPY docker-requirements.txt requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt

# uncomment this when you want to debug so image stays up
#CMD bash


