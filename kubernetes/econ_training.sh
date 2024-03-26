#!/bin/sh

# Constants
ADD_PRECISION=3
SAVING_FOLDER="/home/jovyan/checkpoint/different_knobs_subset_10"
DATA_DIR="/home/jovyan/loss_landscape_taxonomy/data/ECON/Elegun"
DATA_FILE="$DATA_DIR/nELinks5.npy"

zero=0

# Default variable values
num_workers=4
max_epochs=25
size="baseline"
top_models=3
num_test=3
accelerator="auto"
augmentation=0
aug_percentage=0
regularization=0
j_reg=0
prune=0

# # ranges of the scan 
batch_sizes=(1024 512 256 128 64 32 16)
#batch_sizes=(256)

# learning_rates=(0.1 0.05 0.025 0.0125 0.00625 0.003125 0.0015625)
learning_rates=(0.0015625)
# learning_rates=(0.0001 0.000001)
# precisions=(2 3 4 5 6 7 8 9 10 11)

# Function to display script usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo " -h, --help          Display this help message"
    echo "--num_workers        Number of workers"
    echo "--max_epochs         Max number of epochs"
    echo "--size               Model size [baseline, small, large]"
    echo "--top_models         Number of top models to store"
    echo "--num_test           Number of time we repeat the computation"
    echo "--accelerator        Accelerator to use during training [auto, cpu, gpu, tpu]"
    echo "--augmentation       Flag to add noise dataset to the dataset"
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
                exit 0
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
            --augmentation)
                if has_argument $@; then
                    augmentation=$(extract_argument $@)
                    echo "Flag augmentation: $augmentation"
                    shift
                fi
                ;;
            --aug_percentage)
                if has_argument $@; then
                    aug_percentage=$(extract_argument $@)
                    echo "percentage of noise injected: $aug_percentage"
                    shift
                fi
                ;;
            --regularization)
                if has_argument $@; then
                    regularization=$(extract_argument $@)
                    echo "Flag to use Jacobian regularization: $regularization"
                    shift
                fi
                ;;
            --j_reg)
                if has_argument $@; then
                    j_reg=$(extract_argument $@)
                    echo "Weight of the Jacobian Regularization: $j_reg"
                    shift
                fi
                ;;
            --prune)
                if has_argument $@; then
                    prune=$(extract_argument $@)
                    echo "Flag to activate the pruning scanning: $prune"
                    shift
                fi
                ;;
        esac
        shift
    done
}

# Function to generate a Kubernetes Job YAML file
generate_job_yaml() {

    cat <<EOF >$job_name".yaml"
apiVersion: batch/v1
kind: Job
metadata:
    name: $(echo "$job_name" | sed 's/_/-/g')
spec:
    template:
        spec:
            restartPolicy: Never
            containers:
              - name: gpu-container
                image: gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/scipy
                command: ["/bin/bash","-c"]
                args: ["git clone https://github.com/balditommaso/loss_landscape_taxonomy.git;
                        cd /home/jovyan/loss_landscape_taxonomy;
                        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118;
                        pip3 install tensorboard==2.11.1 torchmetrics torchinfo pytorchcv pytorch_lightning==1.9.0 pyemd pandas pot;
                        pip3 install git+https://github.com/balditommaso/HAWQ.git@setup-pip;
                        cp -r /loss_landscape/ECON /home/jovyan/loss_landscape_taxonomy/data/;
                        cd /home/jovyan/loss_landscape_taxonomy/workspace/models/econ/;
                        . scripts/train.sh \
                                        --bs $bs \
                                        --lr $lr \
                                        --max_epochs $max_epochs \
                                        --size $size \
                                        --top_models $top_models \
                                        --num_test $num_test \
                                        --num_workers $num_workers \
                                        --augmentation $augmentation \
                                        --aug_percentage $aug_percentage \
                                        --regularization $regularization \
                                        --j_reg $j_reg \
                                        --prune $prune \
                                        --accelerator $accelerator;"]
                volumeMounts:
                  - mountPath: /loss_landscape
                    name: loss-landscape-volume
                resources:
                    limits:
                        nvidia.com/gpu: "1"
                        memory: "20G"
                        cpu: "12"
                    requests:
                        nvidia.com/gpu: "1"
                        memory: "16G"
                        cpu: "6"
            restartPolicy: Never
            volumes:
                  - name: loss-landscape-volume
                    persistentVolumeClaim:
                        claimName: loss-landscape-volume
EOF
    echo $job_name.yaml
}

# Function to start a Kubernetes Job
start_kubernetes_job() {
    kubectl apply -f $job_name".yaml"
}


# MAIN
handle_options "$@"
# iterate over all the possibilities
for bs in ${batch_sizes[*]}
do
    for lr in ${learning_rates[*]}
    do
        # archive everything and move it in the sahred folder
        if [ "$augmentation" -eq 1 ]; then
            job_name=$(echo "econ_aug_"$aug_percentage"_"$size"_bs"$bs"_lr$lr" | sed 's/\./_/g')
        else
            if [ "$regularization" -eq 1 ] && [ "$prune" -eq 1 ]; then
                job_name=$(echo "econ_jprune_"$j_reg"_"$size"_bs"$bs"_lr$lr" | sed 's/\./_/g')
            else
                if [ "$regularization" -gt 0 ]; then
                    job_name=$(echo "econ_jreg_"$j_reg"_"$size"_bs"$bs"_lr$lr" | sed 's/\./_/g')
                else
                    if [ "$prune" -gt 0 ]; then
                        job_name=$(echo "econ_prune_"$size"_bs"$bs"_lr$lr" | sed 's/\./_/g')
                    else
                        job_name=$(echo "econ_"$size"_bs"$bs"_lr$lr" | sed 's/\./_/g')
                    fi
                fi
            fi
        fi

        generate_job_yaml $job_name
        start_kubernetes_job
    done    
done

echo Jobs started
exit 0

# END MAIN

# SMALL
# bash econ_training.sh --num_workers 8 --max_epochs 25 --size small --top_models 3 --num_test 3

# BASELINE AUG
# bash econ_training.sh --num_workers 2 --max_epochs 25 --size baseline --top_models 3 --num_test 3 --augmentation 1 --aug_percentage 0.8
# BASELINE JREG
# bash econ_training.sh --num_workers 2 --max_epochs 25 --size baseline --top_models 3 --num_test 3 --regularization 1 --j_reg 0.1
# BASELINE PRUNE
# bash econ_training.sh --num_workers 2 --max_epochs 200 --size baseline --top_models 3 --num_test 3 --prune 1 --regularization 1 --j_reg 0.1

# LARGE
# bash econ_training.sh --num_workers 8 --max_epochs 25 --size large --top_models 3 --num_test 3
