import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
# Add the folder from where the script is called to the system path
current_working_directory = os.getcwd()
sys.path.append(current_working_directory)

from analysis.pymo.parsers import BVHParser
from analysis.pymo.data import Joint, MocapData
from analysis.pymo.preprocessing import *
from analysis.pymo.viz_tools import *
from analysis.pymo.writers import *
from sklearn.pipeline import Pipeline
import joblib as jl
from analysis.visualization.utils import generate_video_from_images, join_video_and_audio
import argparse


import matplotlib
matplotlib.use("Agg")

def generate_video_from_expmaps(features_file, pipeline_file, output_folder, audio_file, trim_audio=0, generate_bvh=False):
    data = np.load(features_file)
    # pipeline = jl.load("data/scaled_features/motion_data_pipe.sav")
    # containing_path = os.path.dirname(features_file)
    # pipeline_file = containing_path + "/" + "motion_expmap_data_pipe.sav"
    pipeline = jl.load(pipeline_file)

    filename = os.path.basename(features_file)
    seq_id = filename.split(".")[0]

    if len(data.shape) == 2:
        bvh_data=pipeline.inverse_transform([data])
    else:
        bvh_data=pipeline.inverse_transform([data[:,0,:]])
    # import pdb; pdb.set_trace()
    if generate_bvh:
        writer = BVHWriter()
        with open(output_folder+"/"+seq_id+".bvh",'w') as f:
            writer.write(bvh_data[0], f)

    bvh2pos = MocapParameterizer('position')
    pos_data = bvh2pos.fit_transform(bvh_data)
    video_file = f'{output_folder}/{seq_id}.mp4'
    #render_mp4(pos_data[0], video_file, axis_scale=100, elev=45, azim=45)

    render_mp4(pos_data[0], video_file, axis_scale=300, elev=45, azim=45)
    if audio_file is not None:
        join_video_and_audio(video_file, audio_file, trim_audio)
    # draw_stickfigure3d(pos_data[0], 10)
    # sketch_move(pos_data[0], data=None, ax=None, figsize=(16,8)):



def main():
    parser = argparse.ArgumentParser(description="Generate a video from expmaps.")
    parser.add_argument("--features_file", required=True, help="Path to the numpy file having the expmap data.")
    parser.add_argument("--pipeline_file", required=True, help="Path to the pipeline file.")
    parser.add_argument("--output_folder", required=False, help="Path to the output folder. Optional.")
    parser.add_argument("--audio_file", required=False, default=None, help="Path to the audio file. Optional.")
    parser.add_argument("--trim_audio", required=False, type=int, default=0, help="Trim audio. Optional.")
    parser.add_argument("--generate_bvh", required=False, action='store_true', help="Generate BVH file. Optional.")

    args = parser.parse_args()

    # If output_folder is not provided, set it to the folder containing the features_file
    if args.output_folder is None:
        args.output_folder = os.path.dirname(args.features_file)

    generate_video_from_expmaps(
        features_file=args.features_file,
        pipeline_file=args.pipeline_file,
        output_folder=args.output_folder,
        audio_file=args.audio_file,
        trim_audio=args.trim_audio,
        generate_bvh=args.generate_bvh
    )

if __name__ == "__main__":
    main()
