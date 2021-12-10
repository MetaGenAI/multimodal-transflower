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
parser.add_argument("--feature_name", metavar='', type=str, default=None, help="feature name to extract delta for")
parser.add_argument("--split_index", metavar='', type=int, default=None, help="index at which to split the features")
parser.add_argument("--replace_existing", action="store_true")
parser.add_argument("--new_feature_names", metavar='', type=str, default=None, help="coma separated list of the new features produced after splitting")

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

assert size == 1 # this should be done with one process

candidate_files = sorted(data_path.glob('**/*'+feature_name+'.npy'), key=lambda path: path.parent.__str__())
tasks = distribute_tasks(candidate_files,rank,size)

for i in tasks:
    path = candidate_files[i]
    print(path)
    feature_file = path.__str__()
    base_filename = feature_file[:-(len(feature_name)+4)]
    features = np.load(path)
    new_features = np.split(features,[args.split_index], axis=1)
    left_feats = new_features[0]
    right_feats = new_features[1]
    left_feat_name, right_feat_name = new_feature_names.split(",")
    new_feature_left_file = base_filename+left_feat_name
    new_feature_right_file = base_filename+right_feat_name
    np.save(new_feature_left_file, left_feats)
    np.save(new_feature_right_file, right_feats)

