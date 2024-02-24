#!/bin/bash

batch_sizes=(16 32 64 128 256 512 1024)
learning_rates=(0.1 0.05 0.025 0.0125 0.00625 0.003125 0.0015625)
precision=(2 3 4 5 6 7 8 9 10 11)

for bs in ${batch_sizes[*]}
do
    for lr in ${learning_rates[*]}
    do
        echo "********************************"
        # /data/tbaldi/work/checkpoint/bs16_lr0.05/RN08_2b/accuracy_pixelate.txt
        file_path="/data/tbaldi/work/checkpoint/"$bs"_lr"$lr"/RN08_"$p"b/accuracy_pixelate.txt"
        # Check if the file exists
        if [ -f "$file_path" ]; then
            echo "File exists: $file_path"
        else
            ./scripts/test.sh \
                                        --batch_size $bs \
                                        --learning_rate $lr \
                                        --metric CKA \
                                        --num_batches 10 \
                                        --num_workers 0
        fi
    done
done

