#!/usr/bin/env bash
# This script copies util directory from capstone github repository into this repository
# files from util are ignored by git so we don't check them in in both places

UTIL_ORIG="../capstone/util"
UTIL_DEST="util"

# check to see if original capstone repo exists
if [ ! -d ${UTIL_ORIG} ]; then
    echo "ERROR: missing original capstone repo ${UTIL_ORIG}"
    exit 1
fi


if [ ! -d ${UTIL_DEST} ]; then
    echo "${UTIL_DEST} missing. Creating directory"
    mkdir ${UTIL_DEST}
fi

rsync -rauv --delete --exclude="__pycache__" ${UTIL_ORIG}/*.py ${UTIL_DEST}/