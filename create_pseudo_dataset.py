import argparse
import json
import numpy as np
import os
import pickle as pk

from PIL import Image
from bisect import bisect, bisect_left
from copy import deepcopy
from decord import VideoReader, cpu
from time import time
from tqdm import tqdm

from shot_utils import (
    ShotVisualMatcher,
    ShotScaleCluster,
    get_seg_starts_ends
)

# pseudo dataset parameters
n_past_shot = 5
num_sample = 15

NONE_CAM = -1

gap_stats_path = 'tvmce_stats/past_frame_gap_stats.pk'

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--shot_json', default='output/shot.json', required=True)

    parser.add_argument('--cluster_shots', default=True, choices=[False, True], type=str2bool)
    parser.add_argument('--n_pseudo_cams', default=6, type=int)
    parser.add_argument('--gap', default=None, type=int)

    parser.add_argument('--json_name', default='pseudo_data.json')
    parser.add_argument('--out_dir', default='output')
    parser.add_argument('--pseudo_policy', default='most_similar', choices=['most_similar', 'random'])
    args = parser.parse_args()

    # shared directory for frames
    args.frame_dir = os.path.join(args.out_dir, 'frames')
    os.makedirs(args.out_dir, exist_ok=True)
    args.out_json_path = os.path.join(args.out_dir, args.json_name)
    return args

# Ad-hoc distribution
def setup_sample_distribution(last_gaps_dict, num_fit_point=1, eps=1e-6):
    def gauss(x, A, mu, sigma):
        return A * np.exp(-(x-mu)**2 / (2.*sigma**2))

    last_gaps = np.array(sorted(list(last_gaps_dict.keys())))
    last_gap_cnts = np.array([last_gaps_dict[last_gap] for last_gap in last_gaps])

    coeffs = {}
    for i in np.arange(len(last_gaps)):
        start = max(0, i-num_fit_point)
        end   = min(i+num_fit_point+1, len(last_gaps))

        next_ = last_gaps[i+1] if i+1 < len(last_gaps) else last_gaps[i-1]
        past_ = last_gaps[i-1] if i-1 >= 0             else last_gaps[i+1]
        mean_dist = (abs(next_ - last_gaps[i]) + abs(past_ - last_gaps[i]))/2

        coeff = (last_gap_cnts[i], last_gaps[i], mean_dist/6)
        coeffs[last_gaps[i]] = coeff

    x = np.arange(last_gaps[0], last_gaps[-1]+1)
    y = []
    anchors = sorted(list(coeffs.keys()))
    for i in x:
        if i not in anchors:
            right = bisect(anchors, i)
            left = right-1

            left_anchor, right_anchor = anchors[left], anchors[right]
            y_val = (gauss(i, *coeffs[left_anchor]) + gauss(i, *coeffs[right_anchor]))/2
        else:
            y_val = coeffs[i][0]
        y.append(y_val)
    y = np.array(y)
    y = y / np.sum(y)
    return x, y

def sample_gap(gaps, probs, last_frame):
    max_allowed_gap = last_frame // num_sample
    if max_allowed_gap < gaps[0]:
        return -1

    allowed_ind = gaps <= max_allowed_gap
    allowed_probs = probs[allowed_ind]
    allowed_probs = allowed_probs / np.sum(allowed_probs)
    gap = np.random.choice(gaps[allowed_ind], p=allowed_probs, replace=False)
    return gap

def create_entry(video_name, cur_frame, gap, shot_indices, shot_cams, seg_id, cand_indices,
                 frame_indices, frame2cam, select_cam, cam_list, boundary, prev_gap=5):

    last_prev_frame = cur_frame - gap
    ind = np.arange(last_prev_frame - prev_gap*(num_sample-1),
                    last_prev_frame+prev_gap,
                    prev_gap)

    all_frames_available = np.all([i in frame2cam for i in ind])
    if not all_frames_available:
        return None, frame_indices

    prev_inds = np.arange(max(0, seg_id-n_past_shot), seg_id)
    frame_indices = frame_indices.union(ind).union(cand_indices)

    # sub-sample cameras
    cand_indices = deepcopy(cand_indices)
    if len(cam_list) > args.n_pseudo_cams:
        cam2cand = {cam: cand_ind for cand_ind, cam in zip(cand_indices, cam_list)}
        cams = np.array(list(cam2cand.keys()))
        cams = cams[cams != select_cam]
        np.random.shuffle(cams)
        cam_list = cams[:arsg.n_pseudo_cams-1].tolist() + [select_cam]
        cand_indices = [cam2cand[cam] for cam in cam_list]

    entry = {
        "videoID": video_name,
        "sampleInterval": 5,
        "startFrame": ind[0],
        "outputList": ind.tolist(),
        "outputCam": [frame2cam[frame] for frame in ind.tolist()],

        "prevShotOutputList": [shot_indices[i] for i in prev_inds],
        "prevShotCamList": [shot_cams[i] for i in prev_inds],

        "candidates": cand_indices,
        "boundary": boundary,

        "selectCAM": select_cam,
        "CAMList": cam_list
    }
    return entry, frame_indices

def get_pseudo_candidates(segments, shot_similarity, shot_cam_labels, seg_id, cam_list,
                          frame_indices, policy='most_similar'):

    curr_cam, select_cam = shot_cam_labels[seg_id], shot_cam_labels[seg_id+1]

    indices = np.arange(len(segments))
    if policy == 'most_similar':
        desc_sim_ind = shot_similarity[seg_id].argsort()[::-1]

    keep_indices = np.ones_like(indices, dtype=bool)
    if cam_list[0] == NONE_CAM:
        if policy == 'most_similar':
            keep_indices[desc_sim_ind == seg_id+1] = False
            desc_sim_ind = desc_sim_ind[keep_indices]
            shot_indices = desc_sim_ind[:len(cam_list)-1]
        else:
            keep_indices[indices == seg_id+1] = False
            shot_indices = np.random.choice(
                indices[keep_indices], len(cam_list)-1, replace=False)

        shot_indices = np.append(shot_indices, seg_id+1)
        cand_indices = [segments[select_seg_id][1] for select_seg_id in shot_indices]
    else:
        cand_indices = []
        for cam in cam_list:
            if cam == select_cam:
                select_seg_id = seg_id+1
            else:
                same_cam_ind = shot_cam_labels == cam
                # same_cam_ind[seg_id] = False
                if policy == 'most_similar':
                    same_cam_desc_ind = np.where(same_cam_ind[desc_sim_ind])[0][0]
                    select_seg_id = desc_sim_ind[same_cam_desc_ind]
                elif policy == 'random':
                    select_seg_id = np.random.choice(np.where(same_cam_ind)[0])
            frame_ind = segments[select_seg_id][1]
            cand_indices.append(frame_ind)

    frame_indices = frame_indices.union(cand_indices)
    return cand_indices, curr_cam, select_cam

if __name__ == '__main__':
    args = parse_args()
    np.random.seed(41)

    # get gap distribution
    info = pk.load(open(gap_stats_path, 'rb'))
    last_gaps_dict = info['last_gap']
    gaps, probs = setup_sample_distribution(last_gaps_dict)

    # visual matcher and shot scale matcher
    visual_matcher = ShotVisualMatcher()
    scale_cluster  = ShotScaleCluster()

    out_json, meta = [], {}

    info = json.load(open(args.shot_json))

    video_skipped = 0
    t = tqdm(enumerate(info.items()), total=len(info))
    for video_id, (video_path, video_info) in t:
        t.set_description(f'Processing {video_path}')

        video_name = os.path.basename(video_path)

        transitions = np.array(video_info['transitions'], dtype=int)
        num_frame = video_info['frame_num']

        # #transition too small =>
        # likely not real camera switch, but some visual effect
        if len(transitions) < 10:
            video_skipped += 1
            continue

        vr = VideoReader(video_path, ctx=cpu(0))

        allowed_segs = video_info.get('same_scene_shots', None)
        segments = get_seg_starts_ends(transitions, len(vr), allowed_segs=allowed_segs)

        n_segments = len(segments)
        cam_list = list(range(1, min(args.n_pseudo_cams, n_segments)+1))

        shot_similarity = None
        if args.pseudo_policy == 'most_similar':
            shot_similarity = visual_matcher(vr, segments)

        if not args.cluster_shots:
            shot_cam_labels = np.ones(n_segments, dtype=int) * NONE_CAM
            cam_list = [NONE_CAM for _ in range(len(cam_list))]
        elif n_segments < args.n_pseudo_cams:
            shot_cam_labels = np.arange(1, n_segments+1)
        else:
            shot_cam_labels = scale_cluster(
                vr, segments,
                result_dir=None,
                n_pseudo_cams=args.n_pseudo_cams)

        frame2cam = {
            frame: cam
            for (_, start, end), cam in zip(segments.tolist(), shot_cam_labels.tolist())
            for frame in range(start, end+1)}

        frame_indices, shot_indices, shot_cams = set(), [], []
        for seg_id, (shot_id, start_prev_shot, end_prev_shot) in enumerate(segments[:-1]):
            start_next_shot = segments[seg_id+1][1]

            cand_indices, curr_cam, select_cam = get_pseudo_candidates(
                segments, shot_similarity, shot_cam_labels, seg_id, cam_list, frame_indices,
                policy=args.pseudo_policy
            )
            shot_cams.append(curr_cam)

            # boundary case
            gap = sample_gap(gaps, probs, end_prev_shot) if args.gap is None else args.gap
            if gap > 0:
                entry, frame_indices = create_entry(video_name,
                                                    start_next_shot,
                                                    gap,
                                                    shot_indices,
                                                    shot_cams,
                                                    seg_id,
                                                    cand_indices,
                                                    frame_indices,
                                                    frame2cam,
                                                    select_cam,
                                                    cam_list,
                                                    boundary=1)
                if entry is not None:
                    out_json.append(entry)

            # middle case
            # TODO: a better way to assign pseudo candidates
            # currently same as boundary case, using first frame
            mid_frame = np.random.randint(start_prev_shot, end_prev_shot+1)
            gap = sample_gap(gaps, probs, mid_frame) if args.gap is None else args.gap
            if gap > 0:
                if curr_cam == NONE_CAM:
                    cand_indices[-1] = mid_frame # put GT at the end
                else:
                    cand_indices[cam_list.index(curr_cam)] = mid_frame

                entry, frame_indices = create_entry(video_name,
                                                    mid_frame,
                                                    gap,
                                                    shot_indices,
                                                    shot_cams,
                                                    seg_id,
                                                    cand_indices,
                                                    frame_indices,
                                                    frame2cam,
                                                    curr_cam,
                                                    cam_list,
                                                    boundary=0)
                if entry is not None:
                    out_json.append(entry)
            shot_indices.append((start_prev_shot + end_prev_shot)//2)

        if len(frame_indices) == 0:
            continue
        frame_indices = sorted(list(frame_indices))

        # save frames to shared args.frame_dir
        shared_frame_dir = os.path.join(args.frame_dir, video_name)
        os.makedirs(shared_frame_dir, exist_ok=True)

        to_load_indices = []
        for frame_ind in frame_indices:
            frame_path = os.path.join(shared_frame_dir, f'{frame_ind}.jpg')
            if not os.path.exists(frame_path):
                to_load_indices.append(frame_ind)

        batch_load_size = 64
        for batch_start in range(0, len(to_load_indices), batch_load_size): # prevent from OOM
            batch_end = batch_start + batch_load_size
            batch_to_load_indices = to_load_indices[batch_start:batch_end]

            for frame_ind, frame in zip(
                batch_to_load_indices, vr.get_batch(batch_to_load_indices).asnumpy()
            ):
                img = Image.fromarray(frame)
                width, height = img.size
                short_side = min(img.size)
                if short_side > 256:
                    ratio = 256/width if width == short_side else 256/height
                    new_w, new_h = int(width * ratio), int(height * ratio)
                    img = img.resize((new_w, new_h))

                frame_path = os.path.join(shared_frame_dir, f'{frame_ind}.jpg')
                img.save(frame_path)

        # pre-save boundary information
        boundaries = segments[:, 1].tolist()
        avail_frame2cam = {int(frame): int(frame2cam[frame]) for frame in frame_indices}
        meta[video_name] = {
            'closest_frame_to_boundary': {
                # bisect_left finds self, hence -1
                boundary: frame_indices[bisect_left(frame_indices, boundary)]
                for boundary in boundaries[1:]
            },
            'frame2cam': avail_frame2cam,
            'segments': segments[:, 1:].tolist()
        }
    print(f'{video_skipped} videos skipped due to insufficient #transitions')

    json.dump({
        'data': out_json, 'meta': meta
    }, open(args.out_json_path, 'w'), cls=NpEncoder, indent=2)
