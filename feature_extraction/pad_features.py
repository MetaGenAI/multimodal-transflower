import numpy as np
# import librosa
from pathlib import Path
import json
import os.path
import sys
import argparse
import pickle
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
sys.path.append(ROOT_DIR)
from utils import distribute_tasks

from analysis.pymo.parsers import BVHParser
from analysis.pymo.data import Joint, MocapData
from analysis.pymo.preprocessing import *
from sklearn.pipeline import Pipeline
import json

parser = argparse.ArgumentParser(description="Extract features from filenames")

parser.add_argument("data_path", type=str, help="Directory contining Beat Saber level folders")
parser.add_argument("--feature_name", type=str, help="file extension (the stuff after the base filename) to match")
parser.add_argument("--new_feature_name", type=str, help="new file extension (the stuff after the base filename)")
parser.add_argument("--length", type=int, help="total length to pad to")
parser.add_argument("--pad_along_feature_dim", action="store_true", help="whether to padd along feature dimension rather than time dimension")
parser.add_argument("--padding_const", type=float, default=0.0, help="the constant to fill the padding with")
parser.add_argument("--replace_existing", action="store_true")
parser.add_argument("--keep_feature_name", action="store_true")

args = parser.parse_args()

# makes arugments into global variables of the same name, used later in the code
globals().update(vars(args))
data_path = Path(data_path)


## distributing tasks accross nodes ##
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)

files = sorted(data_path.glob('**/*.'+feature_name+'.npy'), key=lambda path: path.parent.__str__())
tasks = distribute_tasks(files,rank,size)

for i in tasks:
    path = files[i]
    feature_file = path.__str__()
    if new_feature_name is None:
        if keep_feature_name:
            new_feature_name = feature_name
        else:
            new_feature_name = feature_name+"_padded"
    base_filename = feature_file[:-(len(feature_name)+4)]
    new_feature_file = base_filename+new_feature_name+".npy"
    # base_filename = data_path.joinpath(path).__str__()
    # new_feature_file = base_filename+"."+new_feature_name+".npy"
    features = np.load(feature_file)
    if (not pad_along_feature_dim and args.length > features.shape[0]) or (pad_along_feature_dim and args.length > features.shape[1]):
        if pad_along_feature_dim:
            assert len(features.shape) == 2
        if len(features.shape) == 1:
            features = np.concatenate([features, args.padding_const*np.ones((args.length-features.shape[0]))])
        elif len(features.shape) == 2:
            features_dim = features.shape[1]
            time_dim = features.shape[0]
            if pad_along_feature_dim:
                features = np.concatenate([args.padding_const*np.ones((time_dim,args.length-features_dim)), features], axis=1)
            else:
                features = np.concatenate([features, args.padding_const*np.ones((args.length-time_dim,features_dim))], axis=0)
        np.save(new_feature_file, features)
    else:
        print(features.shape[0])
