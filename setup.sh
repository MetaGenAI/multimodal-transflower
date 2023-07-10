#!/bin/bash

gcloud init
./data/scripts/copy_from_gs.sh
pip install -r requirements_dev.txt
#./meta_script_train_diffu.sh $1
