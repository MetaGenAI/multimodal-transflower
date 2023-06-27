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

#import scipy.ndimage.filters as filters
import scipy.ndimage

parser = argparse.ArgumentParser(description="Extract features from filenames")

parser.add_argument("data_path", type=str, help="Directory contining Beat Saber level folders")
parser.add_argument("--feature_name", type=str, help="file extension (the stuff after the base filename) to match")
parser.add_argument("--new_feature_name", metavar='', type=str, default=None)
parser.add_argument("--filter_width", type=float, default=3.0)
parser.add_argument("--keep_feature_name", action="store_true")
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

files = sorted(data_path.glob('**/*.'+feature_name+'.npy'), key=lambda path: path.parent.__str__())
tasks = distribute_tasks(files,rank,size)

#TODO: make a general structure for these data processing files
for i in tasks:
    path = files[i]
    feature_file = path.__str__()
    if new_feature_name is None:
        if keep_feature_name:
            new_feature_name = feature_name
        else:
            new_feature_name = feature_name+"_smoothed"
    base_filename = feature_file[:-(len(feature_name)+4)]
    new_feature_file = base_filename+new_feature_name+".npy"
    if replace_existing or not os.path.isfile(new_feature_file):
        features = np.load(feature_file)
        features = scipy.ndimage.gaussian_filter1d(features, filter_width, axis=0, mode='nearest')
        np.save(new_feature_file, features)
