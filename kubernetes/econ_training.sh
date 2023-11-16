#!/bin/sh

# Constants
ADD_PRECISION=3
SAVING_FOLDER="/home/jovyan/loss_landscape_taxonomy/workspace/checkpoint/different_knobs_subset_10"
DATA_DIR="/home/jovyan/loss_landscape_taxonomy/data/ECON/Elegun"
DATA_FILE="$DATA_DIR/nELinks5.npy"

# Default variable values
num_workers=4
max_epochs=25
size="baseline"
top_models=3
num_test=3
accelerator="auto"

# ranges of the scan 
batch_sizes=(16 32 64 128 256 512 1024)
learning_rates=(0.1 0.05 0.025 0.0125 0.00625 0.003125 0.0015625)
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

    cat <<EOF >$job_name
apiVersion: batch/v1
kind: Job
metadata:
  name: train-bs-job
spec:
  template:
    spec:
      containers:
      - name: gpu-container
        image: jupyter/scipy-notebook
        command: ["/bin/bash","-c"]
        args: ["echo Generated"]
        volumeMounts:
        - mountPath: /loss_landscape
          name: loss-landscape-volume
        resources:
          limits:
            nvidia.com/gpu: "1"
            memory: "32G"
            cpu: "4"
          requests:
            nvidia.com/gpu: "1"
            memory: "32G"
            cpu: "4"
      restartPolicy: Never
      volumes:
        - name: loss-landscape-volume
          persistentVolumeClaim:
            claimName: loss-landscape-volume
EOF
    echo "$job_name"
}

# Function to start a Kubernetes Job
start_kubernetes_job() {
    kubectl apply -f job.yaml
}


# MAIN
# iterate over all the possibilities
for bs in ${batch_sizes[*]}
do
    for lr in ${learning_rates[*]}
    do
        job_name=ECON_bs"$bs"_lr$lr.yaml
        generate_job_yaml "$job_name"
        # start_kubernetes_job
    done
done

echo Jobs started.

# END MAIN