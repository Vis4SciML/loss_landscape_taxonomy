#!/bin/sh

# Constants
ADD_PRECISION=3
SAVING_FOLDER="/home/jovyan/checkpoint/different_knobs_subset_10"
DATA_DIR="/home/jovyan/loss_landscape_taxonomy/data/ECON/Elegun"
DATA_FILE="$DATA_DIR/nELinks5.npy"

# Default variable values
num_workers=4
max_epochs=25
size="baseline"
top_models=3
num_test=3
accelerator="auto"

# # ranges of the scan 
# batch_sizes=(16 32 64 128 256 512 1024)
batch_sizes=(16)

# learning_rates=(0.1 0.05 0.025 0.0125 0.00625 0.003125 0.0015625)
learning_rates=(0.0001 0.000001)
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
        esac
        shift
    done
}

# TOD generate the right Job
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
                        . /home/jovyan/loss_landscape_taxonomy/workspace/models/econ/scripts/get_econ_data.sh;
                        cp -r /loss_landscape/ECON /home/jovyan/loss_landscape_taxonomy/data/
                        cd /home/jovyan/loss_landscape_taxonomy/workspace/models/econ/;
                        . scripts/train.sh \
                                        --bs $bs \
                                        --lr $lr \
                                        --max_epochs $max_epochs \
                                        --size $size \
                                        --top_models $top_models \
                                        --num_test $num_test \
                                        --num_workers $num_workers \
                                        --accelerator $accelerator;"]
                volumeMounts:
                  - mountPath: /loss_landscape
                    name: loss-landscape-volume
                resources:
                    limits:
                        nvidia.com/gpu: "1"
                        memory: "8G"
                        cpu: "2"
                    requests:
                        nvidia.com/gpu: "1"
                        memory: "8G"
                        cpu: "2"
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
        job_name=$(echo "econ_"$size"_bs"$bs"_lr$lr" | sed 's/\./_/g')
        generate_job_yaml $job_name
        start_kubernetes_job
    done    
done

echo Jobs started
exit 0

# END MAIN

# SMALL
# bash econ_training.sh --num_workers 8 --max_epochs 25 --size small --top_models 3 --num_test 3
# BASELINE
# bash econ_training.sh --num_workers 8 --max_epochs 25 --size baseline --top_models 3 --num_test 3
# LARGE
# bash econ_training.sh --num_workers 8 --max_epochs 25 --size large --top_models 3 --num_test 3
