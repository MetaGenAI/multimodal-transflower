import json
# import schema
import asyncio
from websockets.server import serve
import avro.schema
from avro.datafile import DataFileReader, DataFileWriter
from avro.io import DatumReader, DatumWriter, BinaryEncoder, BinaryDecoder
from io import BytesIO
from avro.utils import randbytes
from avro.datafile import MAGIC, META_SCHEMA
import time
from pymo.rotation_tools import Rotation, euler2expmap, euler2expmap2, expmap2euler, euler_reorder, unroll
from pymo.Quaternions import Quaternions
import argparse

# NODE_TYPES_LIST = ["Root", "Head", "LeftHand", "RightHand"]
NODE_TYPES_LIST = ["Head", "LeftHand", "RightHand"]
from constants import ROT_ORDER

import numpy as np


def read_data(data_file):
    reader = DataFileReader(open(data_file, "rb"), DatumReader())

    # Encode schema as binary and send it as the first message
    schema_buffer = BytesIO()
    data_buffer = BytesIO()
    encoder = BinaryEncoder(data_buffer)
    schema = avro.schema.parse(reader.schema)

    yield schema

    writer = DatumWriter(schema)

    # Send binary-encoded data
    # t0 = time.time()
    t0b = time.time()
    initial_time = None
    for i, datum in enumerate(reader):
        yield datum
        # print("sent data")
    reader.close()

def extract_features(path):
    print(path)
    fast_data_file = path+"/fast_data.avro"
    slow_data_file = path+"/slow_data.avro"
    fast_data = read_data(fast_data_file)
    slow_data = read_data(slow_data_file)

    fast_schema = next(fast_data)
    slow_schema = next(slow_data)

    slow_datum = next(slow_data)

    types_in_data = [n["nodeType"] for n in slow_datum["selfBio"][0]["motion"]["nodes"]]

    node_indices = [types_in_data.index(t) for t in NODE_TYPES_LIST]

    fast_data_list = list(read_data(fast_data_file))[1:]
    positions = np.array([[x["position"] for x in fast_datum["selfBio"][0]["motion"]["nodes"]] for fast_datum in fast_data_list])
    rotations = np.array([[x["rotation"] for x in fast_datum["selfBio"][0]["motion"]["nodes"]] for fast_datum in fast_data_list])

    q = Quaternions(rotations)
    eulers = q.euler(order=ROT_ORDER)

    exps = [unroll(np.array([euler2expmap(f, ROT_ORDER, use_deg=False) for f in eulers[:,i,:]])) for i in range(eulers.shape[1])]

    rots = np.stack([exps[i] for i in node_indices], axis=1)

    rots_flat = rots.reshape(rots.shape[0], -1)

    positions = np.stack([positions[:,i,:] for i in node_indices], axis=1)
    poss_flat = positions.reshape(positions.shape[0], -1)

    motion_features = np.concatenate([rots_flat, poss_flat], axis=1)

    return motion_features

if __name__ == "__main__":

    #read path from arguments

    parser = argparse.ArgumentParser(description='Process motion EDFs.')
    parser.add_argument('path', metavar='path', type=str,
                        help='path to data')
    path = parser.parse_args().path

    features = extract_features(path)

    features_file = path+"/features.npy"

    np.save(features_file, features)
