#!/bin/bash

# export PYTHONPATH="/home_directory/.local/lib/python3.5/site-packages/"

#py=python3
py=python
dataset=multimodal
model=transformer
#exp=aistpp_big
exp=aistpp_testing

$py training/train.py --data_dir=./data/scaled_features --dataset_name=$dataset --model=$model --batch_size=80 --num_windows=1 --nepoch=5000 --nepoch_decay=5000 \
    --print_freq=1 --experiment_name=$exp --save_latest_freq=5000\
    --fix_lengths \
    --weight_decay=0 \
    --learning_rate=1e-4 \
    --dins="219,103" \
    --douts="219" \
    --input_modalities="joint_angles_scaled,mel_ddcpca_scaled" \
    --output_modalities="joint_angles_scaled" \
    --input_lengths="120,240" \
    --output_lengths="20" \
    --output_time_offset="121" \
    --predicted_inputs="0,0" \
    --nlayers=12 \
    --nhead=10 \
    --dhid=400 \
    --val_epoch_freq=0 \
    --dropout=0 \
    --workers=4 \
    --gpu_ids=0 \
    #--continue_train \
