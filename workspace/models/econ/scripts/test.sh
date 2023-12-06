#!/bin/bash


SAVING_FOLDER="/data/tbaldi/checkpoint/ "    # /loss_landscape -> shared volume
DATA_DIR="../../../data/ECON/Elegun"
DATA_FILE="$DATA_DIR/nELinks5.npy"

# Default variable values
num_workers=4
size="baseline"
num_batches=1000
noise_type="gaussian"



# ranges of the scan 
# batch_sizes=(16 32 64 128 256 512 1024)
# learning_rates=(0.1 0.05 0.025 0.0125 0.00625 0.003125 0.0015625)
batch_sizes=(16 32 64 128 256 512 1024)
learning_rates=(0.025 0.0125 0.00625 0.003125 0.0015625)
precisions=(2 3 4 5 6 7 8 9 10 11)
percentages=(5 25 50 75 100)
bit_flip=0

# Function to display script usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo " -h, --help          Display this help message"
    echo "--num_workers        Number of workers"
    echo "--num_batches        Max number of batches to test"
    echo "--size               Model size [baseline, small, large]"
    echo "--noise_type         Type of noise [gaussian, random, salt_pepper]"
    echo "--bit_flip           percentage of bit to flip"
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
            --num_batches)
                if has_argument $@; then
                    num_batches=$(extract_argument $@)
                    echo "Max number of batches: $num_batches"
                    shift
                fi
                ;;
            --bit_flip)
                if has_argument $@; then
                    bit_flip=$(extract_argument $@)
                    echo "Max number of batches: $bit_flip"
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
            --size)
                if has_argument $@; then
                    size=$(extract_argument $@)
                    echo "Size of the model: $size"
                    shift
                fi
                ;;
            --noise_type)
                if has_argument $@; then
                    noise_type=$(extract_argument $@)
                    echo "Number of test per model: $noise_type"
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

run_test() {

    for per in ${percentages[*]}
    do
        echo ""
        echo " BATCH SIZE $bs - LEARNING_RATE $lr - PRECISION $p - noise $noise_type - precison $per% "
        echo ""

        
        # training of the model
        python code/test_encoder.py --saving_folder $SAVING_FOLDER \
                            --data_dir $DATA_DIR \
                            --data_file $DATA_FILE \
                            --batch_size $bs \
                            --num_workers $num_workers \
                            --lr $lr \
                            --size $size \
                            --percentage $per \
                            --precision $p \
                            --noise_type $noise_type \
                            --num_batches $num_batches \
                            --bit_flip $bit_flip 


        echo ""
        echo "-----------------------------------------------------------"
    done
}

# Main script execution
handle_options "$@"
# creating the directories if required
mkdir -p $DATA_DIR

#iterate over the precision
for bs in ${batch_sizes[*]}
do
    for lr in ${learning_rates[*]}
    do
        for p in ${precisions[*]}
        do
            # trainig with various batch sizes
            run_test
        done
    done
done

exit 0

# nohup bash scripts/test.sh --num_workers 4 --num_batches 5000 --size small --noise_type gaussian & 

# python code/test_encoder.py --saving_folder "/data/tbaldi/checkpoint/" \
#                         --data_dir "../../../data/ECON/Elegun" \
#                         --data_file "../../../data/ECON/Elegun/nELinks5.npy" \
#                         --batch_size 16 \
#                         --num_workers 8 \
#                         --lr 0.00625 \
#                         --size "baseline" \
#                         --percentage 5 \
#                         --precision 8 \
#                         --noise_type "gaussian" \
#                         --num_batches 1000 