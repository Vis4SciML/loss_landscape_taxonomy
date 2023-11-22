#!/bin/bash 

DATA_DIR="data/jets"
mkdir -p $DATA_DIR

# download train and val dataset
wget https://zenodo.org/record/3602254/files/hls4ml_LHCjet_100p_train.tar.gz -P $DATA_DIR
tar -zxf "$DATA_DIR/hls4ml_LHCjet_100p_train.tar.gz" -C $DATA_DIR

wget https://zenodo.org/record/3602254/files/hls4ml_LHCjet_100p_val.tar.gz -P $DATA_DIR
tar -zxf "$DATA_DIR/hls4ml_LHCjet_100p_val.tar.gz" -C $DATA_DIR

# cleanup
rm "$DATA_DIR/hls4ml_LHCjet_100p_train.tar.gz"
rm "$DATA_DIR/hls4ml_LHCjet_100p_val.tar.gz"
