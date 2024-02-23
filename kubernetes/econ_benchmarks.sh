#!/bin/sh

# Default variable values
num_workers=14
metric='noise'
num_batches=50000
size="baseline"

# # ranges of the scan 
batch_sizes=(16 32 64 128 256 512 1024)
learning_rates=(0.1 0.05 0.025 0.0125 0.00625 0.003125 0.0015625)
# batch_sizes=(128)
# learning_rates=(0.1 0.05)


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
                        conda env create -f environment.yml;
                        . /home/jovyan/loss_landscape_taxonomy/workspace/models/econ/scripts/get_econ_data.sh;
                        source activate loss_landscape;
                        cd /home/jovyan/loss_landscape_taxonomy/workspace/models/econ/;
                        . scripts/test.sh \
                                        --batch_size $bs \
                                        --learning_rate $lr \
                                        --size $size \
                                        --metric $metric \
                                        --num_batches $num_batches \
                                        --num_workers $num_workers"]
                volumeMounts:
                  - mountPath: /loss_landscape
                    name: loss-landscape-volume
                resources:
                    limits:
                        memory: "3G"
                        cpu: "2"
                    requests:
                        memory: "3G"
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
        job_name=$(echo "econ_"$metric"_"$size"_bs"$bs"_lr$lr" | sed 's/\./_/g' | tr '[:upper:]' '[:lower:]')
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
    # bash econ_benchmarks.sh --size baseline --num_workers 2 --metric noise --num_batches 1000000
    # BIT FLIP
    # bash econ_benchmarks.sh --size baseline --num_workers 2 --metric bitflip --num_batches 1000000
    # CKA
    # bash econ_benchmarks.sh --size baseline --num_workers 2 --metric CKA --num_batches 100000
    # NE
    # bash econ_benchmarks.sh --size baseline --num_workers 2 --metric neural_efficiency --num_batches 100000
    # fisher
    # bash econ_benchmarks.sh --size baseline --num_workers 2 --metric fisher --num_batches 100000
    # plot
    # bash econ_benchmarks.sh --size baseline --num_workers 2 --metric plot --num_batches 100000
    # hessian
    # bash econ_benchmarks.sh --size baseline --num_workers 2 --metric hessian --num_batches 10000


# LARGE
    # NoISE
    # bash econ_benchmarks.sh --size large --num_workers 12 --metric noise --num_batches 1000000
    # BIT FLIP
    # bash econ_benchmarks.sh --size large --num_workers 12 --metric bitflip --num_batches 1000000
    # CKA
    # bash econ_benchmarks.sh --size large --num_workers 12 --metric CKA --num_batches 100000
    # NE
    # bash econ_benchmarks.sh --size large --num_workers 12 --metric neural_efficiency --num_batches 100000

