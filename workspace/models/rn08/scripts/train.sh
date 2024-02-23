#!/bin/bash

# Constants
ADD_PRECISION=3
SAVING_FOLDER="/home/jovyan/checkpoint/different_knobs_subset_10"    
DATA_DIR="../../../data/RN08"


# Default variable values
num_workers=4
max_epochs=100
top_models=3
num_test=2
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
    echo "--top_models         Number of top models to store"
    echo "--num_test           Number of time we repeat the computation"
    echo "--accelerator        Accelerator to use during training [auto, cpu, gpu, tpu]"
    echo "--augmentation       Flag to activate data augmentation with noise dataset"
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
            --augmentation)
                if has_argument $@; then
                    augmentation=$(extract_argument $@)
                    echo "Flag to activate data augmentation: $augmentation"
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
    if [ "$augmentation" -eq 1 ]; then
        saving_folder="$SAVING_FOLDER/bs$batch_size"_lr$learning_rate/RN08_AUG_"$precision"b/
    else
        saving_folder="$SAVING_FOLDER/bs$batch_size"_lr$learning_rate/RN08_"$precision"b/
    fi
    pids=()
    for i in $(eval echo "{1..$num_test}")
    do
        echo ""
        echo " BATCH SIZE $batch_size - LEARNING_RATE $learning_rate - PRECISION $precision - test $i "
        echo ""

        test_file="$saving_folder"accuracy_$i.txt
        echo $test_file
        # check if the model has been already computed 
        if [ -e "$test_file" ]; then
            echo "Already computed!"
        else 
            # training of the model
            python code/train.py \
                --saving_folder "$saving_folder" \
                --data_dir "$DATA_DIR" \
                --batch_size $batch_size \
                --num_workers $num_workers \
                --accelerator $accelerator \
                --weight_precision $precision \
                --bias_precision $precision \
                --act_precision $(($precision + $ADD_PRECISION)) \
                --lr $learning_rate \
                --top_models $top_models \
                --experiment $i \
                --max_epochs $max_epochs \
                --augmentation $augmentation \
                >/$HOME/log_RN08_$i.txt 2>&1 &

            pids+=($!)
        fi
        echo ""
        echo "-----------------------------------------------------------"
    done

    # Wait for all background processes to finish
    for pid in "${pids[@]}"; do
        wait $pid
        current_date_time=$(date '+%Y-%m-%d %H:%M:%S')
        echo "$current_date_time: Process with PID $pid finished"
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
# archive everything and move it in the sahred folder
tar -czvf /loss_landscape/RN08_bs$batch_size"_lr$learning_rate".tar.gz $SAVING_FOLDER/ 

exit 0

# . scripts/train.sh --num_workers 0 --bs 16 --lr 0.003125 --max_epochs 100 --top_models 3 --num_test 1 --accelerator auto --augmentation 1

# python code/train.py \
#                 --saving_folder "/loss_landscape/checkpoint/different_knobs_subset_10" \
#                 --data_dir "../../../data/RN08" \
#                 --batch_size 1024 \
#                 --num_workers 0 \
#                 --accelerator auto \
#                 --weight_precision 2 \
#                 --bias_precision 2 \
#                 --act_precision 5   \
#                 --lr 0.003125 \
#                 --top_models 3 \
#                 --experiment 1 \
#                 --max_epochs 1