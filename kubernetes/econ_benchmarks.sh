#!/bin/sh

# Constants
SAVING_FOLDER="/home/jovyan/checkpoint/different_knobs_subset_10"
DATA_DIR="/home/jovyan/loss_landscape_taxonomy/data/ECON/Elegun"
DATA_FILE="$DATA_DIR/nELinks5.npy"

# Default variable values
num_workers=14


# # ranges of the scan 
# batch_sizes=(16 32 64 128 256 512 1024)
# learning_rates=(0.025 0.0125 0.00625 0.003125 0.0015625)
batch_sizes=(1024)
learning_rates=(1 0.05)
bit_flip=0
noise_type="gaussian"

# precisions=(2 3 4 5 6 7 8 9 10 11)

# Function to display script usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo " -h, --help          Display this help message"
    echo "--num_workers        Number of workers"
    echo "--noise_type         Type of noise [gaussian, random, salt_pepper]"
    echo "--size               Model size [baseline, small, large]"
    echo "--bit_flip           Flag to simulate the radiation environment"
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
            --bit_flip)
                if has_argument $@; then
                    bit_flip=$(extract_argument $@)
                    echo "Max number of epochs: $bit_flip"
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
            --noise_type)
                if has_argument $@; then
                    noise_type=$(extract_argument $@)
                    echo "Training accelerator: $noise_type"
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
                args: ["cp /loss_landscape/checkpoint.tar.gz /home/jovyan/;
                        cd /home/jovyan/;
                        tar -xf checkpoint.tar.gz;
                        git clone https://github.com/balditommaso/loss_landscape_taxonomy.git;
                        cd /home/jovyan/loss_landscape_taxonomy;
                        conda env create -f environment.yml;
                        . /home/jovyan/loss_landscape_taxonomy/workspace/models/econ/scripts/get_econ_data.sh;
                        source activate loss_landscape;
                        cd /home/jovyan/loss_landscape_taxonomy/workspace/models/econ/;
                        . scripts/test.sh \
                                        --batch_size $bs \
                                        --learning_rate $lr \
                                        --size $size \
                                        --bit_flip $bit_flip \
                                        --noise_type $noise_type \
                                        --num_workers $num_workers"]
                volumeMounts:
                  - mountPath: /loss_landscape
                    name: loss-landscape-volume
                resources:
                    limits:
                        nvidia.com/gpu: "1"
                        memory: "128G"
                        cpu: "32"
                    requests:
                        nvidia.com/gpu: "1"
                        memory: "128G"
                        cpu: "32"
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
        job_name=$(echo "econ_benchmark_"$size"_bs"$bs"_lr$lr" | sed 's/\./_/g')
        generate_job_yaml $job_name
        start_kubernetes_job
    done    
done

echo Jobs started
exit 0

# END MAIN

# SMALL
# bash econ_benchmarks.sh --num_workers 12 --noise_type gaussian --size small --bit_flip 0
# BASELINE
# bash econ_benchmarks.sh --num_workers 12 --noise_type gaussian --size baseline --bit_flip 0
# LARGE
# bash econ_benchmarks.sh --num_workers 12 --noise_type gaussian --size large --bit_flip 0
