import numpy as np
from .shot_scale_cluster import ShotScaleCluster
from .shot_visual_matcher import ShotVisualMatcher

def get_seg_starts_ends(transitions, num_frames, allowed_segs=None):
    # discard transition at the first frame
    shot_ids = list(range(len(transitions)+1))
    if transitions[0][0] == 0:
        transitions = transitions[1:]
        shot_ids = shot_ids[1:]

    skip_shots = []
    for shot_id, transition in zip(shot_ids, transitions):
        if transition[1] - transition[0] != 1:
            skip_shots.append(shot_id)

    transitions = [transition
                   for shot_id, transition in enumerate(transitions)
                   if shot_id not in skip_shots]
    shot_ids = [shot_id
                for shot_id in shot_ids
                if shot_id not in skip_shots]

    starts = [0] + [transition[1] for transition in transitions]
    ends   = [transition[0] for transition in transitions] + [num_frames-1]

    segments = []
    for shot_id, start, end in zip(shot_ids, starts, ends):
        if allowed_segs is not None and str(shot_id) not in allowed_segs:
            continue
        segments.append((shot_id, start, end))
    segments = np.array(segments, dtype=int)
    return segments
