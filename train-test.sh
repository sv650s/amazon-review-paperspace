#!/bin/bash
# Training Reference: https://docs.paperspace.com/gradient/tutorials/train-a-model-with-the-cli
# Machine Types: https://support.paperspace.com/hc/en-us/articles/234711428-Machine-Pricing
# Free instance: https://docs.paperspace.com/gradient/instances/free-instances?_ga=2.254671808.999355169.1587737794-211442023.1587536380
#       C3 or GPU+
#    --container vtluk/paperspace-tf-gpu:1.0 \

gradient experiments run singlenode \
    --name train-test \
    --projectId pr1cl53bg \
    --container janakiramm/python:3 \
    --machineType C3 \
    --command 'python train/train-test.py -i /storage -o /artifact' \
    --workspace https://github.com/sv650s/amazon-review-paperspace.git
