#!/bin/bash

#if using XLA
export XRT_WORKERS="localservice:0;grpc://localhost:40934"
export XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0|GPU:0;/job:localservice/replica:0/task:0/device:XLA_GPU:0"

#py=python3
py=python3
#exp=aistpp_2
#exp=aistpp_flower_1
#exp=aistpp_flower_gpu_nodrop
#exp=aistpp_flower_gpu_mfix
#exp=aistpp_flower_test3b
#exp=aistpp_flower_test3

#exp=aistpp_moglow_test2
#exp=aistpp_flower_test5
exp=aistpp_moglow_test_filter
#exp=aaaaaa
#exp=aistpp_transglower
#exp=aistpp_flower_expmap
#exp=aistpp_moglow_expmap
#exp=aistpp_test
#exp=aistpp_residual


#seq_id="gLH_sBM_cAll_d16_mLH1_ch04"
#seq_id="gWA_sBM_cAll_d26_mWA1_ch10"
#seq_id="gWA_sFM_cAll_d27_mWA2_ch17"
#seq_id="gLO_sBM_cAll_d14_mLO4_ch05"
#seq_id="gHO_sFM_cAll_d19_mHO1_ch02"
#seq_id="gKR_sBM_cAll_d28_mKR0_ch09"
#seq_id="gKR_sBM_cAll_d30_mKR5_ch06"
#seq_id="gWA_sFM_cAll_d25_mWA3_ch04"
#seq_id="gWA_sBM_cAll_d26_mWA1_ch05"
#seq_id="gJB_sBM_cAll_d09_mJB5_ch10"
#seq_id="gLH_sFM_cAll_d16_mLH4_ch05"
#seq_id="gKR_sBM_cAll_d28_mKR1_ch07"
seq_id="gMH_sBM_cAll_d24_mMH2_ch08"
#seq_id="gJS_sBM_cAll_d03_mJS3_ch02"
#seq_id="gPO_sBM_cAll_d10_mPO0_ch10"
#seq_id="gBR_sBM_cAll_d04_mBR2_ch01"
#seq_id="gWA_sBM_cAll_d26_mWA1_ch05"
#seq_id="gLO_sBM_cAll_d13_mLO3_ch06"
#seq_id="mambo"

mkdir inference/generated/${exp}
mkdir inference/generated/${exp}/predicted_mods
mkdir inference/generated/${exp}/videos

#in_mod=expmap_scaled
in_mod=joint_angles_scaled

$py inference/generate.py --data_dir=test_data --experiment_name=$exp \
    --seq_id $seq_id \
    --input_modalities=${in_mod}",mel_ddcpca_scaled" \
    --output_modalities=${in_mod} \
    --scalers="pkl_joint_angles_mats_scaler"
#    --scalers="bvh_expmap_scaler"

$py analysis/aistplusplus_api/generate_video_from_mats.py --pred_mats_file inference/generated/${exp}/predicted_mods/${seq_id}.${in_mod}.generated.npy \
    --output_folder inference/generated/${exp}/videos/ \
    --audio_file test_data/${seq_id}.mp3 \
    --trim_audio 0.66
#    --trim_audio 2


