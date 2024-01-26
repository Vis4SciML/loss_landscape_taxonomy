#!/bin/bash


SAVING_FOLDER="/home/jovyan/checkpoint/"    # /loss_landscape -> shared volume
DATA_DIR="../../../data/ECON/Elegun"
DATA_FILE="$DATA_DIR/nELinks5.npy"

# Default variable values
num_workers=12
batch_size=1024
learning_rate=0.1
size="baseline"
metric="noise"
num_batches=1



# ranges of the scan 
# batch_sizes=(16 32 64 128 256 512 1024)
# learning_rates=(0.1 0.05 0.025 0.0125 0.00625 0.003125 0.0015625)

precisions=(2 3 4 5 6 7 8 9 10 11)


# Function to display script usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo " -h, --help          Display this help message"
    echo "--metric             Name of the test"
    echo "--num_workers        Number of workers"
    echo "--size               Model size [baseline, small, large]"
    echo "--num_batches        Number to batches to test"
    echo "--batch_size         Select the model by batch size"
    echo "--learning_rate      Select the model by learning rate"
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
            --metric)
                if has_argument $@; then
                    metric=$(extract_argument $@)
                    echo "Max number of batches: $metric"
                    shift
                fi
                ;;
            --num_batches)
                if has_argument $@; then
                    num_batches=$(extract_argument $@)
                    echo "Max number of batches: $num_batches"
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
            --batch_size)
                if has_argument $@; then
                    batch_size=$(extract_argument $@)
                    echo "Number of test per model: $batch_size"
                    shift
                fi
                ;;
            --learning_rate)
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


# Main script execution
handle_options "$@"
# creating the directories if required
mkdir -p $DATA_DIR
for p in ${precisions[*]}
do
    echo precision: $p

    case $metric in
        noise)
            pids=()
            noise_type="gaussian"
            percentages=(5 10 15 20)
            for i in ${percentages[*]}
            do
                python code/test_encoder.py --saving_folder $SAVING_FOLDER \
                            --metric noise \
                            --data_dir $DATA_DIR \
                            --data_file $DATA_FILE \
                            --num_workers $num_workers \
                            --batch_size $batch_size \
                            --learning_rate $learning_rate \
                            --size $size \
                            --precision $p \
                            --noise_type $noise_type \
                            --percentage $i \
                            >/$HOME/log_$i.txt 2>&1 &
                pids+=($!)
            done
            for pid in "${pids[@]}"; do
                wait $pid
                current_date_time=$(date '+%Y-%m-%d %H:%M:%S')
                echo "$current_date_time: Process with PID $pid finished"
            done
            ;;
        bitflip)
            pids=()
            num_bits=(1 5 10 15)
            for b in ${num_bits[*]}
            do
                python code/test_encoder.py --saving_folder $SAVING_FOLDER \
                            --metric bitflip \
                            --data_dir $DATA_DIR \
                            --data_file $DATA_FILE \
                            --num_workers $num_workers \
                            --batch_size $batch_size \
                            --learning_rate $learning_rate \
                            --size $size \
                            --precision $p \
                            --bit_flip $b \
                            >/$HOME/log_$b.txt 2>&1 &
                pids+=($!)
            done
            for pid in "${pids[@]}"; do
                wait $pid
                current_date_time=$(date '+%Y-%m-%d %H:%M:%S')
                echo "$current_date_time: Process with PID $pid finished"
            done
            ;;
        CKA)
            python code/test_encoder.py --saving_folder $SAVING_FOLDER \
                            --metric CKA \
                            --data_dir $DATA_DIR \
                            --data_file $DATA_FILE \
                            --num_workers $num_workers \
                            --batch_size $batch_size \
                            --learning_rate $learning_rate \
                            --size $size \
                            --precision $p \
                            --num_batches $num_batches \
                            >/$HOME/log_$b.txt 2>&1 &
            ;;
        neural_efficiency)
            python code/test_encoder.py --saving_folder $SAVING_FOLDER \
                            --metric neural_efficiency \
                            --data_dir $DATA_DIR \
                            --data_file $DATA_FILE \
                            --num_workers $num_workers \
                            --batch_size $batch_size \
                            --learning_rate $learning_rate \
                            --size $size \
                            --precision $p \
                            --num_batches $num_batches \
                            >/$HOME/log_$b.txt 2>&1 &
            ;;
        fisher)
            python code/test_encoder.py --saving_folder $SAVING_FOLDER \
                            --metric fisher \
                            --data_dir $DATA_DIR \
                            --data_file $DATA_FILE \
                            --num_workers $num_workers \
                            --batch_size $batch_size \
                            --learning_rate $learning_rate \
                            --size $size \
                            --precision $p \
                            --num_batches $num_batches #\
                            # >/$HOME/log_$b.txt 2>&1 &
            ;;
        # ADD THE NEW METRIC HERE
        *)
            echo $metric not implemented yet!
            exit
            ;;
    esac
done

# archive everything and move it in the sahred folder
tar -C /home/jovyan/checkpoint/bs$batch_size"_lr"$learning_rate/ -czvf /loss_landscape/ECON_$size"_$metric"_bs$batch_size"_lr"$learning_rate.tar.gz ./

exit 0


# . scripts/test.sh \
#                                         --batch_size 1024 \
#                                         --learning_rate 0.1 \
#                                         --size baseline \
#                                         --metric fisher \
#                                         --num_batches 100000 \
#                                         --num_workers 12
