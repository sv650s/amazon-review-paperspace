#!/bin/bash
# Training Reference: https://docs.paperspace.com/gradient/tutorials/train-a-model-with-the-cli
# Machine Types: https://support.paperspace.com/hc/en-us/articles/234711428-Machine-Pricing

gradient experiments run singlenode \
    --name train-test \
    --projectId pr1cl53bg \
    --container vtluk/paperspace-tf-gpu:1.0 \
    --machineType C3 \
    --command 'python train/train-test.py -i /storage -o /artifact' \
