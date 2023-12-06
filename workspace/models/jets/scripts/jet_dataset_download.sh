#! /bin/bash

DATA_DIR=$HOME/loss_landscape_taxonomy/data/JTAG

mkdir -p $DATA_DIR

wget https://zenodo.org/record/3602254/files/hls4ml_LHCjet_100p_train.tar.gz -P $DATA_DIR
wget https://zenodo.org/record/3602254/files/hls4ml_LHCjet_100p_val.tar.gz -P $DATA_DIR

tar -zxf $DATA_DIR/hls4ml_LHCjet_100p_train.tar.gz -C $DATA_DIR
tar -zxf $DATA_DIR/hls4ml_LHCjet_100p_val.tar.gz -C $DATA_DIR


