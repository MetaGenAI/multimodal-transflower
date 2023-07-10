#!/bin/bash

#if using XLA
export XRT_WORKERS="localservice:0;grpc://localhost:40934"
export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0|GPU:0;/job:localservice/replica:0/task:0/device:XLA_GPU:0"

py=python3

exp=$1
#seq_id=1687209000059
#seq_id=1687208181080
#seq_id=1687210716516
seq_id=1687208181080
echo $exp $seq_id

mkdir -p inference/generated/${exp}/predicted_mods
mkdir -p inference/generated/${exp}/videos
fps=30
data_dir=data/edf_extracted_data_rel

# if we don't pass seq_id it will choose a random one from the test set
$py inference/generate.py --data_dir=$data_dir --output_folder=inference/generated --experiment_name=$exp \
    --seq_id $seq_id \
    ${@:2}
    #--generate_video \
