#!/bin/sh

# Default variable values
num_workers=14
metric='noise'
num_batches=50000
size="baseline"
augmentation=0
regularization=0
aug_percentage=0
j_reg=0
prune=0
prune_percentage=0

# # ranges of the scan 
batch_sizes=(32 64 128 256 512 1024)
learning_rates=(0.1 0.05 0.025 0.0125 0.00625 0.003125 0.0015625)
batch_sizes=(512)
learning_rates=(0.0015625)


# Function to display script usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo " -h, --help          Display this help message"
    echo "--num_workers        Number of workers"
    echo "--metric             Metric of the analysis"
    echo "--num_batches        Number of batches to test"
    echo "--size               Size of the model"
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
            --metric)
                if has_argument $@; then
                    metric=$(extract_argument $@)
                    echo "Max number of epochs: $metric"
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
            --num_batches)
                if has_argument $@; then
                    num_batches=$(extract_argument $@)
                    echo "Training accelerator: $num_batches"
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
            --augmentation)
                if has_argument $@; then
                    augmentation=$(extract_argument $@)
                    echo "Percentage of noise injected: $augmentation"
                    shift
                fi
                ;;
            --regularization)
                if has_argument $@; then
                    regularization=$(extract_argument $@)
                    echo "Percentage of noise injected: $regularization"
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
                    echo "Percentage of noise injected: $prune"
                    shift
                fi
                ;;
            --prune_percentage)
                if has_argument $@; then
                    prune_percentage=$(extract_argument $@)
                    echo "Weight of the jacobian regularization: $prune_percentage"
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
                args: ["cp /loss_landscape/checkpoint.tar.gz /home/jovyan/;
                        cd /home/jovyan/;
                        tar -xf checkpoint.tar.gz;
                        git clone https://github.com/balditommaso/loss_landscape_taxonomy.git;
                        cd /home/jovyan/loss_landscape_taxonomy;
                        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118;
                        pip3 install tensorboard==2.11.1 torchmetrics torchinfo pytorchcv pytorch_lightning==1.9.0 pyemd pandas pot;
                        pip3 install scipy loss_landscapes;
                        pip3 install git+https://github.com/balditommaso/HAWQ.git@setup-pip;
                        pip3 install git+https://github.com/balditommaso/robustbench.git;
                        pip3 install git+https://github.com/jicampos/PyHessian.git@hawq;
                        cp -r /loss_landscape/ECON /home/jovyan/loss_landscape_taxonomy/data/;
                        cd /home/jovyan/loss_landscape_taxonomy/workspace/models/econ/;
                        . scripts/test.sh \
                                        --batch_size $bs \
                                        --learning_rate $lr \
                                        --size $size \
                                        --metric $metric \
                                        --num_batches $num_batches \
                                        --augmentation $augmentation \
                                        --regularization $regularization \
                                        --j_reg $j_reg \
                                        --aug_percentage $aug_percentage \
                                        --prune $prune \
                                        --prune_percentage $prune_percentage \
                                        --num_workers $num_workers"]
                volumeMounts:
                  - mountPath: /loss_landscape
                    name: loss-landscape-volume
                resources:
                    limits:
                        nvidia.com/gpu: "1"
                        memory: "14G"
                        cpu: "8"
                    requests:
                        nvidia.com/gpu: "1"
                        memory: "6G"
                        cpu: "5"
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
        #job_name=$(echo "econ_"$metric"_"$size"_bs"$bs"_lr$lr" | sed 's/\./_/g' | tr '[:upper:]' '[:lower:]')
        # archive everything and move it in the sahred folder
        if [ "$augmentation" -gt 0 ]; then
            job_name=$(echo "econ_aug_"$aug_percentage"_"$metric"_"$size"_bs"$bs"_lr$lr" | sed 's/\./_/g' | tr '[:upper:]' '[:lower:]')
        else
            if [ "$regularization" -gt 0 ]; then
                job_name=$(echo "econ_jreg_"$j_reg"_"$metric"_"$size"_bs"$bs"_lr$lr" | sed 's/\./_/g' | tr '[:upper:]' '[:lower:]')
            else
                if [ "$prune" -gt 0 ]; then
                job_name=$(echo "econ_prune_"$prune_percentage"_"$metric"_"$size"_bs"$bs"_lr$lr" | sed 's/\./_/g' | tr '[:upper:]' '[:lower:]')
                else
                    job_name=$(echo "econ_"$metric"_"$size"_bs"$bs"_lr$lr" | sed 's/\./_/g' | tr '[:upper:]' '[:lower:]')
                fi
            fi
        fi
        generate_job_yaml $job_name
        start_kubernetes_job
    done    
done

rm econ_$metric"*"

echo Jobs started
exit 0

# END MAIN

# SMALL
    # NoISE
    # bash econ_benchmarks.sh --size small --num_workers 12 --metric noise --num_batches 1000000
    # BIT FLIP
    # bash econ_benchmarks.sh --size small --num_workers 12 --metric bitflip --num_batches 1000000
    # CKA
    # bash econ_benchmarks.sh --size small --num_workers 12 --metric CKA --num_batches 100000
    # NE
    # bash econ_benchmarks.sh --size small --num_workers 12 --metric neural_efficiency --num_batches 100000

# BASELINE
    # NoISE
    # bash econ_hessian.sh --size baseline --num_workers 2 --metric noise --num_batches 1000000
    # BIT FLIP
    # bash econ_hessian.sh --size baseline --num_workers 2 --metric bitflip --num_batches 1000000
    # CKA
    # bash econ_hessian.sh --size baseline --num_workers 2 --metric CKA --num_batches 100000
    # NE
    # bash econ_hessian.sh --size baseline --num_workers 2 --metric neural_efficiency --num_batches 100000
    # fisher
    # bash econ_hessian.sh --size baseline --num_workers 2 --metric fisher --num_batches 100000
    # plot
    # bash econ_hessian.sh --size baseline --num_workers 2 --metric plot --num_batches 100000
    # hessian
    # bash econ_hessian.sh --size baseline --num_workers 2 --metric hessian --num_batches 10000
    # bash econ_hessian.sh --size baseline --num_workers 4 --metric hessian --num_batches 10000 --augmentation 1 --aug_percentage 0.8
    # bash econ_hessian.sh --size baseline --num_workers 4 --metric hessian --num_batches 10000 --regularization 1 --j_reg 0.1
    # bash econ_hessian.sh --size baseline --num_workers 4 --metric hessian --num_batches 10000 --prune 1 --prune_percentage 0.9


# LARGE
    # NoISE
    # bash econ_benchmarks.sh --size large --num_workers 12 --metric noise --num_batches 1000000
    # BIT FLIP
    # bash econ_benchmarks.sh --size large --num_workers 12 --metric bitflip --num_batches 1000000
    # CKA
    # bash econ_benchmarks.sh --size large --num_workers 12 --metric CKA --num_batches 100000
    # NE
    # bash econ_benchmarks.sh --size large --num_workers 12 --metric neural_efficiency --num_batches 100000

