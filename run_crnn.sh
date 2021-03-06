#!/bin/sh

export CUDA_VISIBLE_DEVICES="0"

BATCH_SIZE=2
WORKER_SIZE=4
MAX_EPOCHS=10

python ./run_crnn.py --batch_size $BATCH_SIZE --workers $WORKER_SIZE --max_epochs $MAX_EPOCHS
