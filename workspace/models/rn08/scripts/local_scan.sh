#!/bin/bash

batch_sizes=(16 32 64 128 256 512 1024)
learning_rates=(0.1 0.05 0.025 0.0125 0.00625 0.003125 0.0015625)
precision=(2 3 4 5 6 7 8 9 10 11)

for bs in ${batch_sizes[*]}
do
    for lr in ${learning_rates[*]}
    do
        echo "********************************"
        ./scripts/test.sh \
                                        --batch_size $bs \
                                        --learning_rate $lr \
                                        --metric CKA \
                                        --num_batches 1000 \
                                        --num_workers 0
    done
done

