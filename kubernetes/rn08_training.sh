#!/bin/sh

# Default variable values
num_workers=8
max_epochs=50
top_models=3
num_test=5
accelerator="auto"
augmentation=0

# # ranges of the scan 
batch_sizes=(16 32 64 128 256 512 1024)
# batch_sizes=(512)
learning_rates=(0.1 0.05 0.025 0.0125 0.00625 0.003125 0.0015625)

# learning_rates=(0.0015625)
# precisions=(2 3 4 5 6 7 8 9 10 11)


# Function to display script usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo " -h, --help          Display this help message"
    echo "--num_workers        Number of workers"
    echo "--max_epochs         Max number of epochs"
    echo "--top_models         Number of top models to store"
    echo "--num_test           Number of time we repeat the computation"
    echo "--accelerator        Accelerator to use during training [auto, cpu, gpu, tpu]"
    echo "--augmentation       Flag to add noisy data in the training"
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
                    echo "Number of test per model: $augmentation"
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
                        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118;
                        pip3 install tensorboard==2.11.1 torchmetrics torchinfo pytorchcv pytorch_lightning==1.9.0;
                        pip3 install git+https://github.com/balditommaso/HAWQ.git@setup-pip;
                        cp -r /loss_landscape/RN08 /home/jovyan/loss_landscape_taxonomy/data/;
                        cd /home/jovyan/loss_landscape_taxonomy/workspace/models/rn08/;
                        . scripts/train.sh \
                                        --bs $bs \
                                        --lr $lr \
                                        --max_epochs $max_epochs \
                                        --top_models $top_models \
                                        --num_test $num_test \
                                        --num_workers $num_workers \
                                        --augmentation $augmentation \
                                        --accelerator $accelerator;"]
                volumeMounts:
                  - mountPath: /loss_landscape
                    name: loss-landscape-volume
                resources:
                    limits:
                        nvidia.com/gpu: "1"
                        memory: "4G"
                        cpu: "4"
                    requests:
                        nvidia.com/gpu: "1"
                        memory: "2G"
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
        job_name=$(echo "rn08_bs"$bs"_lr$lr" | sed 's/\./_/g')
        generate_job_yaml $job_name
        start_kubernetes_job
    done    
done

echo Jobs started
exit 0

# END MAIN

# bash rn08_training.sh --num_workers 0 --max_epochs 100 --top_models 3 --num_test 3 --augmentation 0

