#!/bin/bash

# Input video or video directory
video_path_or_dir=$1

# Variables
ext=".mp4"
out_dir="output"
shot_info_path="shots.json"

# check if the path is a video or a folder of videos
if [ -d "$video_path_or_dir" ]; then
    video_param="--video_dir $video_path_or_dir"
else
    video_param="--video $video_path_or_dir"
fi

# Run TransNet V2 for shot boundary detection 
python transnet/infer_on_dir.py $video_param --out_dir $out_dir

# Generate JSON for transitions
python cvt_to_clipshot_anno.py $video_param --shot_info_path $shot_info_path

# Create pseudo dataset
python create_pseudo_dataset.py --shot_json $out_dir/$shot_info_path
