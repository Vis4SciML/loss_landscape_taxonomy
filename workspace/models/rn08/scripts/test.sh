#!/bin/bash


SAVING_FOLDER="/home/jovyan/checkpoint/"    # /loss_landscape -> shared volume
DATA_DIR="../../../data/RN08"

# Default variable values
num_workers=0
batch_size=1024
learning_rate=0.1
metric="noise"
num_batches=16

# default values
noise_type="pixelate" 

precisions=(2 3 4 5 6 7 8 9 10 11)


# Function to display script usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo " -h, --help          Display this help message"
    echo "--metric             Name of the test"
    echo "--num_workers        Number of workers"
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
                    echo "Metric: $metric"
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
            --batch_size)
                if has_argument $@; then
                    batch_size=$(extract_argument $@)
                    echo "Batch size: $batch_size"
                    shift
                fi
                ;;
            --learning_rate)
                if has_argument $@; then
                    learning_rate=$(extract_argument $@)
                    echo "Learning rate: $learning_rate"
                    shift
                fi
                ;;
            *)
                echo "Invalid option: $1" >&2
                usage
                return
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
            python code/test.py --saving_folder $SAVING_FOLDER \
                        --metric noise \
                        --data_dir $DATA_DIR \
                        --num_workers $num_workers \
                        --batch_size $batch_size \
                        --learning_rate $learning_rate \
                        --precision $p \
                        --num_batches $num_batches \
                        --noise_type $noise_type #\
                        #>/$HOME/log_$i.txt 2>&1 &
            ;;
        CKA)
            python code/test.py --saving_folder $SAVING_FOLDER \
                            --metric CKA \
                            --data_dir $DATA_DIR \
                            --num_workers $num_workers \
                            --batch_size $batch_size \
                            --learning_rate $learning_rate \
                            --precision $p \
                            --num_batches $num_batches #\
                            #>/$HOME/log_$metric.txt 
            ;;
        neural_efficiency)
            python code/test.py --saving_folder $SAVING_FOLDER \
                            --metric neural_efficiency \
                            --data_dir $DATA_DIR \
                            --num_workers $num_workers \
                            --batch_size $batch_size \
                            --learning_rate $learning_rate \
                            --precision $p \
                            --num_batches $num_batches #\
                            #>/$HOME/log_$metric.txt 
            ;;
        fisher)
            python code/test.py --saving_folder $SAVING_FOLDER \
                            --metric fisher \
                            --data_dir $DATA_DIR \
                            --num_workers $num_workers \
                            --batch_size $batch_size \
                            --learning_rate $learning_rate \
                            --precision $p \
                            --num_batches $num_batches #\
                            #>/$HOME/log_$metric.txt
            ;;
        plot)
            python code/test.py --saving_folder $SAVING_FOLDER \
                            --metric plot \
                            --data_dir $DATA_DIR \
                            --num_workers $num_workers \
                            --batch_size $batch_size \
                            --learning_rate $learning_rate \
                            --precision $p \
                            --num_batches $num_batches \
                            --steps 100 \
                            --distance 80 \
                            --normalization filter #\
                            #>/$HOME/log_$metric.txt
            ;;
        # ADD THE NEW METRIC HERE
        *)
            echo $metric not implemented yet!
            return
            ;;
    esac
done
# archive everything and move it in the sahred folder
tar -C /home/jovyan/checkpoint/bs$batch_size"_lr"$learning_rate/ -czvf /loss_landscape/RN08_$metric"_bs"$batch_size"_lr"$learning_rate.tar.gz ./

exit 0


# . scripts/test.sh \
#                                         --batch_size 32 \
#                                         --learning_rate 0.003125 \
#                                         --metric plot \
#                                         --num_batches 10000 \
#                                         --num_workers 0
