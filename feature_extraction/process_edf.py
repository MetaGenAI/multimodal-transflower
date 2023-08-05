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
import joblib as jl

parser = argparse.ArgumentParser(description="Preprocess motion data")

parser.add_argument("data_path", type=str, help="Directory contining EDF records")
parser.add_argument("--no_replace_existing", action="store_true")
# parser.add_argument("--param", type=str, default="expmap", help="expmap, position")
# parser.add_argument("--do_mirror", action="store_true", help="whether to augment the data with mirrored motion")
# parser.add_argument("--fps", type=int, default=60)

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

candidate_edf_folders = sorted(data_path.glob('*'), key=lambda path: path.parent.__str__())
#candidate_motion_files = candidate_motion_files[:32]
tasks = distribute_tasks(candidate_edf_folders,rank,size)

from utils.edf_motion_utils import extract_features # only support motion for now

for task in tasks:
    folder = task
    features = extract_features(folder)
    features_file = folder + "/motion_features.npy"
    if not no_replace_existing or not os.path.isfile(features_file):
        np.save(features_file, features)
        print("Saved features to {}".format(features_file))
