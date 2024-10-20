import argparse
import glob
import json
import os
from decord import VideoReader
from tqdm import tqdm

parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--video_dir", type=str, help="path to video dir to process")
group.add_argument("--video", type=str, help="path to a single video")

parser.add_argument("--ext", default='*.mp4', help="Extension of video files")
parser.add_argument("--out_dir", default='output', help="Output directory")
parser.add_argument("--shot_info_path", default='shot.json', help="File with shot info of all videos")

args = parser.parse_args()

if args.video is None:
    t = tqdm(glob.glob(os.path.join(args.video_dir, args.ext)))
else:
    t = tqdm([args.video])

skipped = 0
info = {}
for video_path in t:
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # get all transitions
    pred_shot_path = f'{args.out_dir}/shot_boundaries/{video_name}.scenes.txt'
    if not os.path.exists(pred_shot_path):
        t.set_description(f'{pred_shot_path} not found, skipped, likely haven\'t run transnet?')
        skipped += 1
        continue

    t.set_description(f'Processing {video_path}')
    with open(pred_shot_path, 'r') as inf:
        shots = [line.strip().split(' ') for line in inf]
    
        transitions = [[prev_shot[1], next_shot[0]]
                       for prev_shot, next_shot in zip(shots[:-1], shots[1:])]

    vr = VideoReader(video_path)
    info[video_path] = {
        'frame_num': len(vr),
        'transitions': transitions
    }

print(f'{skipped}/{len(t)} skipped')
anno_path = os.path.join(args.out_dir, args.shot_info_path)
json.dump(info, open(anno_path, 'w'), indent=2)
