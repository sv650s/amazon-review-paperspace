#!/bin/bash
# Training Reference: https://docs.paperspace.com/gradient/tutorials/train-a-model-with-the-cli
# Machine Types: https://support.paperspace.com/hc/en-us/articles/234711428-Machine-Pricing
# Free instance: https://docs.paperspace.com/gradient/instances/free-instances?_ga=2.254671808.999355169.1587737794-211442023.1587536380
#       C3 (CPU) or P4000 (GPU)
#    --container vtluk/paperspace-tf-gpu:1.0 \

gradient experiments run singlenode \
    --name train-test \
    --projectId pr1cl53bg \
    --machineType P4000 \
    --container vtluk/paperspace-tf-gpu:1.0 \
    --command 'python train/test-env.py -i /storage -o /artifacts' \
    --workspace .


