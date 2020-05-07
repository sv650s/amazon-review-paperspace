#!/bin/bash
# Training Reference: https://docs.paperspace.com/gradient/tutorials/train-a-model-with-the-cli
# Machine Types: https://support.paperspace.com/hc/en-us/articles/234711428-Machine-Pricing
# Free instance: https://docs.paperspace.com/gradient/instances/free-instances?_ga=2.254671808.999355169.1587737794-211442023.1587536380
#       C3 (CPU) or P4000 (GPU)
#    --container vtluk/paperspace-tf-gpu:1.0 \

usage() {
    echo "`basename $0`: [-b batch_size] [-c lstm_cells] [-d dropout_rate] [-e epochs] [-l log_level] [-m machine_type]"
    echo "               [-p patience] [-r recurrent_dropout_rate] <sample size>"
    echo "Parameter(s):"
    echo "  sample_size:                size of data set to train. available values: test, 50k, 100k, 200k, 500k, 1m, 2m, 4m, all"
    echo "Options:"
    echo "  -b batch size:              batch size for training. Default 32"
    echo "  -c lstm_cells:              number of LSTM cells used for training. Default 128"
    echo "  -d dropout_rate:            dropout rate for LSTM network. Default 0"
    echo "  -e epochs:                  max number of epochs for training. Default 20"
    echo "  -l log_level:               log level for logging. Default INFO"
    echo "  -m machine_type:            Gradient machine type. Options C3 (CPU) or P4000 (GPU). Default P4000"
    echo "  -n enable_bidirectional:    Enable bidirectional network. Default False"
    echo "  -p patience:                patience for early stopping. Default 4"
    echo "  -r recurrent_dropout_rate:  recurrent dropout rate for LSTM cells. Default 0"
    echo "Example:"
    echo "  ./train.sh 1m"
    echo "  ./train.sh -e 40 -d 0.2 1m"
}

if [ $# -lt 1 ]; then
    echo -e "ERROR: missing required parameter: sample_size"
    usage
    exit 1
fi


MACHINE_TYPE="P4000"
BIDIRECTIONAL_OPT=
while getopts :b:c:d:e:l:m:np:r: o
   do
     case $o in
        b) BATCH_SIZE="$OPTARG" ;;
        c) LSTM_CELLS="$OPTARG" ;;
        d) DROPOUT_RATE="$OPTARG" ;;
        e) EPOCHS="$OPTARG" ;;
        l) LOG_LEVEL="$OPTARG" ;;
        m) MACHINE_TYPE="$OPTARG" ;;
        n) BIDIRECTIONAL_OPT=" -n ";;
        p) PATIENCE="$OPTARG" ;;
        r) RECURRENT_DROPOUT_RATE="$OPTARG" ;;
        *) usage && exit 0 ;;                     # display usage and exit
     esac
   done

shift $((OPTIND-1))
sample_size=$1
echo "Sample size: ${sample_size}"

if [ "x${BATCH_SIZE}" == "x" ]; then
    BATCH_SIZE_OPT=""
else
    BATCH_SIZE_OPT="-b ${BATCH_SIZE}"
fi
if [ "x${LSTM_CELLS}" == "x" ]; then
    LSTM_CELLS_OPT=""
else
    LSTM_CELLS_OPT="-c ${LSTM_CELLS}"
fi
if [ "x${DROPOUT_RATE}" == "x" ]; then
    DROPOUT_RATE_OPT=""
else
    DROPOUT_RATE_OPT="-d ${DROPOUT_RATE}"
fi
if [ "x${EPOCHS}" == "x" ]; then
    EPOCHS_OPT=""
else
    EPOCHS_OPT="-e ${EPOCHS}"
fi
if [ "x${LOG_LEVEL}" == "x" ]; then
    LOG_LEVEL_OPT=""
else
    LOG_LEVEL_OPT="-l ${LOG_LEVEL}"
fi
if [ "x${MACHINE_TYPE}" == "x" ]; then
    MACHINE_TYPE_OPT=""
else
    MACHINE_TYPE_OPT="-m ${MACHINE_TYPE}"
fi
if [ "x${PATIENCE}" == "x" ]; then
    PATIENCE_OPT=""
else
    PATIENCE_OPT="-p ${PATIENCE}"
fi
if [ "x${RECURRENT_DROPOUT_RATE}" == "x" ]; then
    RECURRENT_DROPOUT_RATE_OPT=""
else
    RECURRENT_DROPOUT_RATE_OPT="-r ${RECURRENT_DROPOUT_RATE}"
fi


echo "Running python with following command"
echo "python train/train.py -i /storage -o /artifacts ${BATCH_SIZE_OPT} ${BIDIRECTIONAL_OPT} ${LSTM_CELLS_OPT} ${DROPOUT_RATE_OPT} ${EPOCHS_OPT} ${LOG_LEVEL_OPT} ${PATIENCE_OPT} ${RECURRENT_DROPOUT_RATE_OPT} ${sample_size}" \

gradient experiments run singlenode \
    --name with_stop_nonlemmatized-${sample_size} \
    --projectId pr1cl53bg \
    --machineType ${MACHINE_TYPE} \
    --container vtluk/paperspace-tf-gpu:1.0 \
    --command "python train/train.py -i /storage -o /artifacts ${BATCH_SIZE_OPT} ${BIDIRECTIONAL_OPT} ${LSTM_CELLS_OPT} ${DROPOUT_RATE_OPT} ${EPOCHS_OPT} ${LOG_LEVEL_OPT} ${PATIENCE_OPT} ${RECURRENT_DROPOUT_RATE_OPT} ${sample_size}" \
    --workspace .


