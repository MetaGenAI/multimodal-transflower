import numpy as np
from pathlib import Path
import json
import os.path
import sys
import argparse
import pickle

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(os.path.join(THIS_DIR, os.pardir), os.pardir))
sys.path.append(ROOT_DIR)
from audio_feature_utils import extract_features_hybrid, extract_features_mel, extract_features_multi_mel
from utils import distribute_tasks

parser = argparse.ArgumentParser(description="Fix lengths for time synnced modalities")
parser.add_argument("data_path", type=str, help="features path")
parser.add_argument("base_filenames_file", type=str, help="File listing the base names for the files for which to combine features")
parser.add_argument('--fix_length_types', default=None, help='Comma-separated list of approaches to fix length: end for cut end, beg for cut beginning, single for single-element sequence (e.g. sequence-level label). E.g. single,end,end. Assumes cut end if not specified')
parser.add_argument('--modalities', default='mp3_mel_100')
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

candidate_files = [x[:-1] for x in open(base_filenames_file,"r").readlines()]
print(candidate_files)
sys.stdout.flush()
tasks = distribute_tasks(candidate_files,rank,size)

modalities = modalities.split(",")
features = {mod:{} for mod in modalities}
features_filenames = {mod:{} for mod in modalities}

if fix_length_types is None:
    fix_length_types = ["end" for mod in modalities]
else:
    fix_length_types = fix_length_types.split(",")

assert len(fix_length_types) == len(modalities)
fix_length_types_dict = {mod:fix_length_types[i] for i,mod in enumerate(modalities)}

for task in tasks:
    base_filename = candidate_files[task]
    # print(base_filename)
    # sys.stdout.flush()
    for mod in modalities:
        feature_file = data_path.joinpath(base_filename+"."+mod+".npy")
        features[mod][base_filename] = np.load(feature_file)
        features_filenames[mod][base_filename] = feature_file

    #fix lengths for time-syncced modalities
    shortest_length = 99999999999
    first_match = True
    #find shortest length (among time-syncced modalities)
    for mod in modalities:
        if fix_length_types_dict[mod] == "none": continue
        length = features[mod][base_filename].shape[0]
        if length < shortest_length:
            #print(np.abs(length-shortest_length))
            if first_match:
                first_match = False
            else:
                if np.abs(length-shortest_length) > 2:
                    print("sequence length difference")
                    print(np.abs(length-shortest_length))
                    print(base_filename)
                    sys.stdout.flush()
                #assert np.abs(length-shortest_length) <= 2
            shortest_length = length

    for i,mod in enumerate(modalities):
        if fix_length_types[i] == "end":
            feats = features[mod][base_filename][:shortest_length]
        elif fix_length_types[i] == "beg":
            feats = features[mod][base_filename][shortest_length:]
        elif fix_length_types[i] == "none":
            feats = features[mod][base_filename]
        else:
            raise NotImplementedError("Don't have an implementation of fix_length_type "+fix_length_type[i])
        np.save(features_filenames[mod][base_filename],feats)
