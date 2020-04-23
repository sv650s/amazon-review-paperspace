# latest stable release of tensorflow image
# https://www.tensorflow.org/install/docker
FROM tensorflow/tensorflow:2.2.0rc2-gpu-py3


COPY docker-requirements.txt requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt

CMD bash


