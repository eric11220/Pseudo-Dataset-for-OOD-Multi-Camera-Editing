import argparse
import glob
import numpy as np
import os
from tqdm import tqdm

from transnetv2 import TransNetV2, infer

parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--video_dir", type=str, help="path to video dir to process")
group.add_argument("--video", type=str, help="path to a single video")

parser.add_argument("--ext", default='*.mp4', help="Extension of video files")
parser.add_argument("--out_dir", default='../output', help="Output directory")
args = parser.parse_args()

shot_result_dir = os.path.join(args.out_dir, 'shot_boundaries')
os.makedirs(shot_result_dir, exist_ok=True)

model = TransNetV2()

failed_paths = []
if args.video is None:
    t = tqdm(glob.glob(os.path.join(args.video_dir, args.ext)))
else:
    t = tqdm([args.video])

for video_path in t:
    t.set_description(f'Processing {video_path}')
    success = infer(model, video_path, shot_result_dir, tqdm_obj=t)
    if not success:
        failed_paths.append(video_path)

status_path = os.path.join(shot_result_dir, 'failed_shot_pred_videos.npy')
np.save(status_path, failed_paths)
print(f'#Videos failed to be processed: {len(failed_paths)}/{len(t)}')
print(f'With filed video paths in {status_path}')
