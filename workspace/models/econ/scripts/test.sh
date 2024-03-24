#!/bin/bash


SAVING_FOLDER="/home/jovyan/checkpoint/"    # /loss_landscape -> shared volume
# SAVING_FOLDER="/data/tbaldi/work/checkpoint/"
DATA_DIR="../../../data/ECON/Elegun"
DATA_FILE="$DATA_DIR/nELinks5.npy"

# Default variable values
num_workers=12
batch_size=1024
learning_rate=0.1
size="baseline"
metric="noise"
num_batches=1
augmentation=0
regularization=0
aug_percentage=0
j_reg=0
prune=0
prune_percentage=0


# ranges of the scan 
# batch_sizes=(16 32 64 128 256 512 1024)
# learning_rates=(0.1 0.05 0.025 0.0125 0.00625 0.003125 0.0015625)

precisions=(2 3 4 5 6 7 8 9 10 11)
#precisions=(7 8 9 10 11)

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
            --augmentation)
                if has_argument $@; then
                    augmentation=$(extract_argument $@)
                    echo "Learning rate: $augmentation"
                    shift
                fi
                ;;
            --regularization)
                if has_argument $@; then
                    regularization=$(extract_argument $@)
                    echo "Learning rate: $regularization"
                    shift
                fi
                ;;
            --aug_percentage)
                if has_argument $@; then
                    aug_percentage=$(extract_argument $@)
                    echo "Percentage of noise injected: $aug_percentage"
                    shift
                fi
                ;;
            --j_reg)
                if has_argument $@; then
                    j_reg=$(extract_argument $@)
                    echo "Weight of the jacobian regularization: $j_reg"
                    shift
                fi
                ;;
            --prune)
                if has_argument $@; then
                    prune=$(extract_argument $@)
                    echo "Flag for pruning: $prune"
                    shift
                fi
                ;;
            --prune_percentage)
                if has_argument $@; then
                    prune_percentage=$(extract_argument $@)
                    echo "Pruning percentage: $prune_percentage"
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
            percentages=(5)
            for j in ${percentages[*]}
            do
                echo "Valdation with $j%"
                pids=()
                noise_type=("gaussian" "random" "salt_pepper")
                for i in ${noise_type[*]}
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
                                --percentage $j \
                                --noise_type $i \
                                --num_batches $num_batches \
                                --aug_percentage $aug_percentage \
                                --j_reg $j_reg \
                                --prune $prune_percentage \
                                >/$HOME/log_$i.txt 2>&1 &
                    pids+=($!)
                done
                for pid in "${pids[@]}"; 
                do
                    wait $pid
                    current_date_time=$(date '+%Y-%m-%d %H:%M:%S')
                    echo "$current_date_time: Process with PID $pid finished"
                done
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
                            --aug_percentage $aug_percentage \
                            --j_reg $j_reg \
                            --prune $prune_percentage \
                            >/$HOME/log_$metric.txt 
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
                            --aug_percentage $aug_percentage \
                            --j_reg $j_reg \
                            --prune $prune_percentage \
                            >/$HOME/log_$metric.txt 
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
                            --num_batches $num_batches \
                            --aug_percentage $aug_percentage \
                            --j_reg $j_reg \
                            --prune $prune_percentage \
                            >/$HOME/log_$metric.txt
            ;;
        plot)
            python code/test_encoder.py --saving_folder $SAVING_FOLDER \
                            --metric plot \
                            --data_dir $DATA_DIR \
                            --data_file $DATA_FILE \
                            --num_workers $num_workers \
                            --batch_size $batch_size \
                            --learning_rate $learning_rate \
                            --size $size \
                            --precision $p \
                            --num_batches $num_batches \
                            --aug_percentage $aug_percentage \
                            --j_reg $j_reg \
                            --steps 300 \
                            --distance 200 \
                            --normalization filter \
                            --prune $prune_percentage \
                            >/$HOME/log_$metric.txt
            ;;
        hessian)
            # pids=()
            trial=(1 2 3)
            # for i in ${trial[*]}
            # do
            i=1
            python code/test_encoder.py --saving_folder $SAVING_FOLDER \
                    --metric hessian \
                    --data_dir $DATA_DIR \
                    --data_file $DATA_FILE \
                    --num_workers $num_workers \
                    --batch_size $batch_size \
                    --learning_rate $learning_rate \
                    --size $size \
                    --precision $p \
                    --aug_percentage $aug_percentage \
                    --j_reg $j_reg \
                    --trial $i \
                    --num_batches $num_batches \
                    --prune $prune_percentage #\
                    #>/$HOME/log_$i"_"$metric.txt 2>&1 &
            #     pids+=($!)
            # done
            # for pid in "${pids[@]}"; do
            #     wait $pid
            #     current_date_time=$(date '+%Y-%m-%d %H:%M:%S')
            #     echo "$current_date_time: Process with PID $pid finished"
            # done
            ;;
        # ADD THE NEW METRIC HERE
        *)
            echo $metric not implemented yet!
            return
            ;;
    esac
done
# archive everything and move it in the sahred folder
if [ "$augmentation" -eq 1 ]; then
    tar -C /home/jovyan/checkpoint/bs$batch_size"_lr"$learning_rate/ -czvf /loss_landscape/ECON_AUG_"$aug_percentage"_$size"_$metric"_bs$batch_size"_lr"$learning_rate.tar.gz ./
else
    if [ "$regularization" -eq 1 ] && [ "$prune" -eq 1 ]; then
            tar -C /home/jovyan/checkpoint/bs$batch_size"_lr"$learning_rate/ -czvf /loss_landscape/ECON_JREG_"$j_reg"_PRUNE_"$prune_percentage"_$size"_"bs$batch_size"_lr$learning_rate".tar.gz ./ 
    else
        if [ "$regularization" -eq 1 ]; then
            tar -C /home/jovyan/checkpoint/bs$batch_size"_lr"$learning_rate/ -czvf /loss_landscape/ECON_JREG_"$j_reg"_$size"_$metric"_bs$batch_size"_lr"$learning_rate.tar.gz ./
        else
            if [ "$prune" -eq 1 ]; then
                tar -C /home/jovyan/checkpoint/bs$batch_size"_lr"$learning_rate/ -czvf /loss_landscape/ECON_PRUNE_"$prune_percentage"_$size"_$metric"_bs$batch_size"_lr"$learning_rate.tar.gz ./
            else
                tar -C /home/jovyan/checkpoint/bs$batch_size"_lr"$learning_rate/ -czvf /loss_landscape/ECON_$size"_$metric"_bs$batch_size"_lr"$learning_rate.tar.gz ./
            fi
        fi
    fi
fi
exit 0


# . scripts/test.sh \
#                                         --batch_size 32 \
#                                         --learning_rate 0.003125 \
#                                         --size baseline \
#                                         --metric noise \
#                                         --num_batches 100 \
#                                         --num_workers 1
