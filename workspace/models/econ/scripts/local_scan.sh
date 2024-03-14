#!/bin/bash

batch_sizes=(16 32 64 128 256 512 1024)

# learning_rates=(0.1 0.05 0.025 0.0125 0.00625 0.003125 0.0015625 )
learning_rates=(0.0001 0.00001 0.000001 0.0000001)

batch_sizes=(512)

learning_rates=(0.0015625)

for bs in ${batch_sizes[*]}
do
    for lr in ${learning_rates[*]}
    do
        echo "********************************"
        # /data/tbaldi/work/checkpoint/bs16_lr0.05/RN08_2b/accuracy_pixelate.txt
        # Check if the file exists
        ./scripts/test.sh \
                                    --batch_size $bs \
                                    --learning_rate $lr \
                                    --metric plot \
                                    --prune 1 \
                                    --num_batches 2000 \
                                    --prune_percentage 0.1 \
                                    --num_workers 1

    done
done

# --augmentation 1 \
# --aug_percentage 0.5 \
# --regularization 1 \
# --j_reg 0.1 \
