import sys
import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
sys.path.append(ROOT_DIR)
import numpy as np; import scipy.linalg
# LUL
w_shape = [219,219]
w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
np_p, np_l, np_u = scipy.linalg.lu(w_init)

from training.datasets import create_dataset, create_dataloader

from models import create_model
from training.options.train_options import TrainOptions
import torch
import pytorch_lightning as pl
import numpy as np
import pickle, json, yaml
import sklearn
import argparse
import os, glob
from pathlib import Path

from analysis.visualization.generate_video_from_mats import generate_video_from_mats
from analysis.visualization.generate_video_from_expmaps import generate_video_from_expmaps
from analysis.visualization.generate_video_from_moglow_pos import generate_video_from_moglow_loc

from inference.utils import load_model_from_logs_path

if __name__ == '__main__':
    print("Hi")
    parser = argparse.ArgumentParser(description='Generate with model')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--seeds', type=str, help='sequences to use as seeds for each modality. in the format: mod,seq_id;mod,seq_id')
    parser.add_argument('--zero_seeds', type=str, help='modalities to seed with zeros, in the format: mod,mod')
    parser.add_argument('--zero_pads', type=str, help='modalities whose seeds to pad with zeros, in the format: mod,mod')
    parser.add_argument('--unpad_amount', type=int, default=0, help='amount of padding to remove from the beggining')
    parser.add_argument('--seeds_file', type=str, help='file from which to choose a random seed')
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--audio_format', type=str, default="wav")
    parser.add_argument('--sequence_length', type=int, default=-1)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--seq_id', type=str)
    parser.add_argument('--starting_index', type=int, default=0)
    parser.add_argument('--max_length', type=int, default=-1)
    parser.add_argument('--no-use_scalers', dest='use_scalers', action='store_false')
    parser.add_argument('--generate_video', action='store_true')
    parser.add_argument('--generate_bvh', action='store_true')
    parser.add_argument('--generate_ground_truth', action='store_true')
    parser.add_argument('--teacher_forcing', action='store_true')
    parser.add_argument('--use_temperature', action='store_true')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--save_jit', action='store_true')
    parser.add_argument('--no_concat_autoreg_mods', action='store_true')
    parser.add_argument('--version_index', type=int, default=-1, help="index of the checkpoint to get for the model weights. -1 means the latest one")
    #parser.add_argument('--save_jit_path', type=string, default="")
    #parser.add_argument('--nostrict', action='store_true')
    parser.add_argument('--fps', type=int, default=20)
    args, unknown_args = parser.parse_known_args()
    data_dir = args.data_dir
    audio_format = args.audio_format
    fps = args.fps
    output_folder = args.output_folder
    seq_id = args.seq_id
    if args.seeds is not None:
        seeds = {mod:seq for mod,seq in [tuple(x.split(",")) for x in args.seeds.split(";")]}
    else:
        seeds = {}

    if args.zero_seeds is not None:
        zero_seeds = args.zero_seeds.split(",")
    else:
        zero_seeds = []

    if args.zero_pads is not None:
        zero_pads = args.zero_pads.split(",")
    else:
        zero_pads = []

    if seq_id is None:
        temp_base_filenames = [x[:-1] for x in open(data_dir + "/base_filenames_test.txt", "r").readlines()]
        seq_id = np.random.choice(temp_base_filenames)
    if args.seeds_file is not None:
        print("choosing random seed from "+args.seeds_file)
        temp_base_filenames = [x[:-1] for x in open(args.seeds_file, "r").readlines()]
        seq_id = np.random.choice(temp_base_filenames)

    sequence_length=args.sequence_length
    if sequence_length==-1: sequence_length=None


    print(seq_id)

    #load hparams file
    default_save_path = "training/experiments/"+args.experiment_name
    logs_path = default_save_path
    model, opt = load_model_from_logs_path(logs_path, version_index=args.version_index, args=unknown_args)
    #model, opt = load_model_from_logs_path(logs_path, version_index=args.version_index, args=unknown_args, no_grad=False)
    print("Device: "+str(model.device))

    input_mods = opt.input_modalities.split(",")
    output_mods = opt.output_modalities.split(",")
    output_time_offsets = [int(x) for x in str(opt.output_time_offsets).split(",")]
    if args.use_scalers:
        print("USING SCALERS")
        scalers = [x+"_scaler.pkl" for x in output_mods]
    else:
        print("NOT USING SCALERS")
        scalers = []

    # Load input features (sequences must have been processed previously into features)
    features = {}
    for i,mod in enumerate(input_mods):
        if mod in seeds:
            print("loading", data_dir+"/"+seeds[mod]+"."+mod+".npy")
            feature = np.load(data_dir+"/"+seeds[mod]+"."+mod+".npy")
        elif mod in zero_seeds:
            feature = np.zeros((model.input_lengths[i],model.dins[i]))
        else:
            print(data_dir+"/"+seq_id+"."+mod+".npy")
            feature = np.load(data_dir+"/"+seq_id+"."+mod+".npy")
            feature = feature[args.starting_index:]
        if mod in zero_pads:
            feature = np.concatenate([np.zeros((model.input_lengths[i],model.dins[i])), feature], axis=0)
        if args.max_length != -1:
            feature = feature[:args.max_length]
        if model.input_proc_types[i] == "single":
            features["in_"+mod] = np.expand_dims(np.expand_dims(feature,1),1)
        else:
            features["in_"+mod] = np.expand_dims(feature,1)

        print("Feature mean: "+str(feature.mean()))
        print("Feature std: "+str(feature.std()))

    # Generate prediction
    # import pdb;pdb.set_trace()

    save_jit_path = output_folder+"/"+args.experiment_name+"/compiled_jit.pth"
    predicted_mods = model.generate(features, teacher_forcing=args.teacher_forcing, ground_truth=args.generate_ground_truth, sequence_length=sequence_length, use_temperature=args.use_temperature, temperature=args.temperature, save_jit=args.save_jit, save_jit_path=save_jit_path, concat_autoreg_mods=not args.no_concat_autoreg_mods)
    #print("--- %s seconds ---" % (time.time() - start_time))
    if len(predicted_mods) == 0:
        print("Sequence too short!")
    else:
        # import pdb;pdb.set_trace()
        for i, mod in enumerate(output_mods):
            predicted_mod = predicted_mods[i].cpu().numpy()
            if len(scalers)>0:
                print(scalers[i])
                transform = pickle.load(open(data_dir+"/"+scalers[i], "rb"))
                print(transform)
                print(predicted_mod.shape)
                predicted_mod = transform.inverse_transform(predicted_mod)
                #predicted_mod = transform.inverse_transform(feature)
            print(feature.shape)
            print(predicted_mod.shape)
            predicted_features_file = output_folder+"/"+args.experiment_name+"/predicted_mods/"+seq_id+"."+mod+".generated"
            predicted_mod = predicted_mod[:,:,args.unpad_amount:]
            np.save(predicted_features_file,predicted_mod)
            predicted_features_file += ".npy"

            if args.generate_video:
                if args.no_concat_autoreg_mods:
                    trim_audio = output_time_offsets[i] / fps #converting trim_audio from being in frames (which is more convenient as thats how we specify the output_shift in the models), to seconds
                else:
                    trim_audio = 0
                print("trim_audio: ",trim_audio)

                audio_file = data_dir + "/" + seq_id + "."+audio_format

                output_folder = output_folder+"/"+args.experiment_name+"/videos/"

                if mod == "joint_angles_scaled":
                    generate_video_from_mats(predicted_features_file,output_folder,audio_file,trim_audio,fps,plot_mats)
                elif mod[:13] == "expmap_scaled" or mod[:16] == "expmap_cr_scaled":
                    pipeline_file = f'{data_dir}/motion_{mod}_data_pipe.sav'
                    generate_video_from_expmaps(predicted_features_file,pipeline_file,output_folder,audio_file,trim_audio,args.generate_bvh)
                elif mod == "position_scaled":
                    control_file = f'{data_dir}/{seq_id}.moglow_control_scaled.npy'
                    data = np.load(predicted_features_file)[:,0,:]
                    control = np.load(control_file)
                    if args.use_scalers:
                        transform = pickle.load(open(data_dir+"/moglow_control_scaled_scaler.pkl", "rb"))
                        control = transform.inverse_transform(control)
                    control = control[int(opt.output_time_offsets.split(",")[0]):]
                    generate_video_from_moglow_loc(data,control,output_folder,seq_id,audio_file,fps,trim_audio)
                else:
                    print("Warning: mod "+mod+" not supported")
                    # raise NotImplementedError(f'Feature type {feature_type} not implemented')
                    pass
