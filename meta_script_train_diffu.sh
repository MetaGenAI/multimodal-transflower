#!/bin/bash

#./script_train_diffu.sh transfusion_base4_lessN2 --base_filenames_file base_filenames_train_filtered_smol2.txt --max_epochs 500
#./script_train_diffu.sh transfusion_base4_lessN3 --base_filenames_file base_filenames_train_filtered2_shuf.txt --max_epochs 200 --continue_train
#./script_train_diffu.sh transfusion_base4_lessN4 --base_filenames_file base_filenames_train_filtered2_shuf.txt --max_epochs 10 $@
#./script_train_diffu.sh transfusion_base4_lessN5 --base_filenames_file base_filenames_train_filtered2_shuf.txt --max_epochs 10 $@
./script_train_diffu.sh transfusion_base4_lessN5 --base_filenames_file base_filenames_train_filtered2_shuf.txt --max_epochs 20 $@