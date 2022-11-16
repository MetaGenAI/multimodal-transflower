import numpy as np
# import librosa
from pathlib import Path
import json
import os.path
import os, glob
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
parser.add_argument("--feature_name", metavar='', type=str, default=None, help="feature name to trim")
parser.add_argument("--max_length", metavar='', type=int, default=None, help="number of frames to trim from the beginning")
parser.add_argument("--min_length", metavar='', type=int, default=None, help="number of frames to trim from the end")

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

candidate_files = sorted(data_path.glob('**/*'+feature_name+'.npy'), key=lambda path: path.parent.__str__())
tasks = distribute_tasks(candidate_files,rank,size)

for i in tasks:
    path = candidate_files[i]
    print(path)
    feature_file = path.__str__()
    base_filename = feature_file[:-(len(feature_name)+4)]
    features = np.load(path)
    length = features.shape[0]
    if min_length is not None and length < min_length:
        for f in glob.glob(base_filename+"*"):
            os.remove(f)
    elif max_length is not None and length > max_length:
        for f in glob.glob(base_filename+"*"):
            os.remove(f)


