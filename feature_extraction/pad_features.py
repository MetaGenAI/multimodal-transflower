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
parser.add_argument("--files_extension", type=str, help="file extension (the stuff after the base filename) to match")
parser.add_argument("--length", type=int, help="total length to pad to")
parser.add_argument("--padding_const", type=float, default=0.0, help="the constant to fill the padding with")
parser.add_argument("--replace_existing", action="store_true")

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

files = sorted(data_path.glob('**/*.'+files_extension), key=lambda path: path.parent.__str__())
tasks = distribute_tasks(files,rank,size)

for i in tasks:
    path = files[i]
    # base_filename = data_path.joinpath(path).__str__()
    # new_feature_file = base_filename+"."+new_feature_name+".npy"
    features = np.load(path)
    if args.length > features.shape[0]:
        if len(features.shape) == 1:
            features = np.concatenate([features, args.padding_const*np.ones((args.length-features.shape[0]))])
        elif len(features.shape) == 2:
            features_dim = features.shape[1]
            features = np.concatenate([features, args.padding_const*np.ones((args.length-features.shape[0],features_dim))])
        np.save(path, features)
    else:
        print(features.shape[0])
