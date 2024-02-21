#!/bin/bash

# Constants
ADD_PRECISION=3
SAVING_FOLDER="/home/jovyan/checkpoint/different_knobs_subset_10"    # /loss_landscape -> shared volume
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
    echo "--augmentation        Flag to indicate to add noisy dataset into the training"

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
                    echo "batch size: $batch_size"
                    shift
                fi
                ;;
            --lr)
                if has_argument $@; then
                    learning_rate=$(extract_argument $@)
                    echo "learning rate: $learning_rate"
                    shift
                fi
                ;;
            --augmentation)
                if has_argument $@; then
                    augmentation=$(extract_argument $@)
                    echo "Add noise dataset to the training: $augmentation"
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
        saving_folder="$SAVING_FOLDER/bs$batch_size"_lr$learning_rate/ECON_AUG_"$precision"b/
    else
        saving_folder="$SAVING_FOLDER/bs$batch_size"_lr$learning_rate/ECON_"$precision"b/
    fi
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
            # training of the model
            python code/train.py \
                --saving_folder "$saving_folder" \
                --data_dir "$DATA_DIR" \
                --data_file "$DATA_FILE" \
                --batch_size $batch_size \
                --num_workers $num_workers \
                --accelerator $accelerator \
                --weight_precision $precision \
                --bias_precision $precision \
                --act_precision $(($precision + $ADD_PRECISION))   \
                --lr $learning_rate \
                --size $size \
                --top_models $top_models \
                --experiment $i \
                --max_epochs $max_epochs \
                --augmentation $augmentation \
                >/$HOME/log_ECON_$i.txt 2>&1 &

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
tar -czvf /loss_landscape/$size"_"bs$batch_size"_lr$learning_rate".tar.gz $SAVING_FOLDER/ 

exit 0

# . scripts/train.sh --num_workers 8 --bs 1024 --lr 0.0001 --max_epochs 100 --size small --top_models 1 --num_test 1 --augmentation 1

# python code/train.py \
#                 --saving_folder "/loss_landscape/checkpoint/different_knobs_subset_10" \
#                 --data_dir "../../../data/ECON/Elegun" \
#                 --data_file "../../../data/ECON/Elegun/nELinks5.npy" \
#                 --batch_size 1024 \
#                 --num_workers 8 \
#                 --accelerator auto \
#                 --weight_precision 2 \
#                 --bias_precision 2 \
#                 --act_precision 5   \
#                 --lr 0.025 \
#                 --size small \
#                 --top_models 3 \
#                 --experiment 2 \
#                 --max_epochs 25 