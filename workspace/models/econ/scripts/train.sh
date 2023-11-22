#!/bin/bash

# Constants
ADD_PRECISION=3
SAVING_FOLDER="/loss_landscape/checkpoint/different_knobs_subset_10"    # /loss_landscape -> shared volume
DATA_DIR="../../../data/ECON/Elegun"
DATA_FILE="$DATA_DIR/nELinks5.npy"

# Default variable values
num_workers=4
max_epochs=25
no_train=false
size="baseline"
top_models=3
num_test=3
accelerator="auto"
batch_size=8
learning_rate=0.0015625

# ranges of the scan 
# batch_sizes=(16 32 64 128 256 512 1024)
# learning_rates=(0.1 0.05 0.025 0.0125 0.00625 0.003125 0.0015625)
precisions=(2 3 4 5 6 7 8 9 10 11)

# Function to display script usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo " -h, --help          Display this help message"
    echo "--num_workers        Number of workers"
    echo "--bs                 Batch size"
    echo "--lr                 Learning rate"
    echo "--max_epochs         Max number of epochs"
    echo "--size               Model size [baseline, small, large]"
    echo "--top_models         Number of top models to store"
    echo "--num_test           Number of time we repeat the computation"
    echo "--accelerator        Accelerator to use during training [auto, cpu, gpu, tpu]"
    echo "--no_train            Flag which specify if the model need to be train"
}

has_argument() {
    [[ ("$1" == *=* && -n ${1#*=}) || ( ! -z "$2" && "$2" != -*)  ]];
}

extract_argument() {
    echo "${2:-${1#*=}}"
}

# Function to handle options and arguments
handle_options() {
    while [ $# -gt 0 ]; do
        case $1 in
            -h | --help)
                usage
                return
                ;;
            --no_train)
                no_train=true
                echo "The model will not be trained"
                ;;
            --max_epochs)
                if has_argument $@; then
                    max_epochs=$(extract_argument $@)
                    echo "Max number of epochs: $max_epochs"
                    shift
                fi
                ;;
            --num_workers)
                if has_argument $@; then
                    num_workers=$(extract_argument $@)
                    echo "Number of workers: $num_workers"
                    shift
                fi
                ;;
            --accelerator)
                if has_argument $@; then
                    accelerator=$(extract_argument $@)
                    echo "Training accelerator: $accelerator"
                    shift
                fi
                ;;
            --size)
                if has_argument $@; then
                    size=$(extract_argument $@)
                    echo "Size of the model: $size"
                    shift
                fi
                ;;
            --top_models)
                if has_argument $@; then
                    top_models=$(extract_argument $@)
                    echo "Model to be stores: $top_models"
                    shift
                fi
                ;;
            --num_test)
                if has_argument $@; then
                    num_test=$(extract_argument $@)
                    echo "Number of test per model: $num_test"
                    shift
                fi
                ;;
            --bs)
                if has_argument $@; then
                    batch_size=$(extract_argument $@)
                    echo "Number of test per model: $batch_size"
                    shift
                fi
                ;;
            --lr)
                if has_argument $@; then
                    learning_rate=$(extract_argument $@)
                    echo "Number of test per model: $learning_rate"
                    shift
                fi
                ;;
            *)
                echo "Invalid option: $1" >&2
                usage
                exit 1
                ;;
        esac
        shift
    done
}

run_train() {
    saving_folder="$SAVING_FOLDER/bs$batch_size"_lr$learning_rate/ECON_"$precision"b/
    pids=()
    for i in $(eval echo "{1..$num_test}")
    do
        echo ""
        echo " BATCH SIZE $batch_size - LEARNING_RATE $learning_rate - PRECISION $precision - test $i "
        echo ""

        test_file="$saving_folder$size/$size"_emd_"$i.txt"
        echo $test_file
        # check if the model has been already computed 
        if [ -e "$test_file" ]; then
            echo "Already computed!"
        else 
            # create directories to retrieve informations of the training process
            # log_folder="$saving_folder"log
            # log_file=$log_folder"/log_"$size"_"$i".txt"

            # error_folder="$saving_folder"error
            # error_file=$error_folder"/err_"$size"_"$i".txt"

            # mkdir -p $log_folder
            # touch $log_file
            # mkdir -p $error_folder
            # touch $error_file

            # training of the model
            python code/train.py \
                --saving_folder "$saving_folder" \
                --data_dir "$DATA_DIR" \
                --data_file "$DATA_FILE" \
                --batch_size $batch_size \
                --num_workers $num_workers \
                --accelerator $accelerator \
                `if [ $no_train == true ]; then echo "--no_train"; fi` \
                --weight_precision $precision \
                --bias_precision $precision \
                --act_precision $(($precision + $ADD_PRECISION))   \
                --lr $learning_rate \
                --size $size \
                --top_models $top_models \
                --experiment $i \
                --max_epochs $max_epochs \
                >/dev/null &

            pids+=($!)
        fi
        echo ""
        echo "-----------------------------------------------------------"
    done

    # Wait for all background processes to finish
    for pid in "${pids[@]}"; do
        wait $pid
        echo "PID $pid finished!"
    done
}

# Main script execution
handle_options "$@"
# creating the directories if required
mkdir -p $DATA_DIR

#iterate over the precision
for precision in ${precisions[*]}
do
    # trainig with various batch sizes
    run_train
done

exit 0

# DEBUG
# mkdir -p "$SAVING_FOLDER/bs_16/ECON_2b/log"
# mkdir -p "$SAVING_FOLDER/bs_16/ECON_2b/error"   
# echo "********************** test **********************"
# python code/train.py \
#         --saving_folder "$SAVING_FOLDER/bs_1024/ECON_8"b/ \
#         --data_dir "$DATA_DIR" \
#         --data_file "$DATA_FILE" \
#         --batch_size 1024 \
#         --num_workers 1 \
#         --accelerator "auto" \
#         --no_train \
#         --weight_precision 8 \
#         --bias_precision 8 \
#         --act_precision 11   \
#         --lr 0.0015625 \
#         --size "small" \
#         --top_models 1 \
#         --experiment 1 \
#         --max_epochs 1 \
#         > >(tee -a "$SAVING_FOLDER/bs_16/ECON_2b/log/log.txt") \
#         2> >(tee -a "$SAVING_FOLDER/bs_16/ECON_2b/error/error.txt"    >&2)

# return
# END DEBUG

# . scripts/train.sh --num_workers 8 --bs 1024 --lr 0.0125 --max_epochs 25 --size small --top_models 3 --num_test 3 