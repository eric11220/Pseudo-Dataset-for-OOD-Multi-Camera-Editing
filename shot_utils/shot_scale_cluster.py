import cv2
import numpy as np
import os

import torch
import torch.nn as nn
from torchvision import transforms

from decord import VideoReader, cpu
from sklearn.cluster import KMeans
from tsn import TSN

class ShotScaleCluster(nn.Module):
    def __init__(self,
                 tsn_ckpt_path = '/home/apple/research/others/tsn-pytorch/checkpoints/movienet_bn_inception__rgb_checkpoint.pth.tar',
                 num_crops=1, max_num_segments=25, bsize=192, device='cuda'):
        super(ShotScaleCluster, self).__init__()

        self.bsize = bsize
        self.device = device
        self.max_num_segments = max_num_segments
        self.num_crops = num_crops

        self.load_shot_scale_tsn(tsn_ckpt_path)
        self.init_shot_scale_transforms(num_crops)

    def load_shot_scale_tsn(self, ckpt_path):
        self.model = TSN(num_class=5,
                         num_segments=1,
                         base_model='BNInception',
                         modality='RGB')

        ckpt = torch.load(ckpt_path)
        print('Loading MovieShot checkpoint')
        print("model epoch {} best prec@1: {}".format(ckpt['epoch'], ckpt['best_prec1']))
        base_dict = {
            '.'.join(k.split('.')[1:]): v
            for k,v in list(ckpt['state_dict'].items())
        }
        self.model.load_state_dict(base_dict)
        self.model.eval()
        self.model.to(self.device)

    def init_shot_scale_transforms(self, num_crops):
        mean, std = (104, 117, 128), (1., 1., 1.)
        # if num_crops == 1:
        #     cropping = transforms.Compose([
        #         GroupScale(256),
        #         GroupCenterCrop(224),
        #     ])
        # elif num_crops == 10:
        #     cropping = transforms.Compose([
        #         GroupOverSample(224, 256)
        #     ])

        # self.transform = transforms.Compose([
        #     cropping,
        #     Stack(roll=True), # args.arch == 'BNInception'),
        #     ToTorchFormatTensor(div=False), # args.arch != 'BNInception'),
        #     GroupNormalize(mean, std),
        # ])

        self.transform = transforms.Compose([
            transforms.Resize(256, transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.Normalize(mean, std)
        ])

    def prepare_tsn_inputs(self, frames, num_segments):
        num_frames = len(frames)
        tick = num_frames / float(num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)], dtype=int)

        tensor = torch.from_numpy(frames[offsets]).float()
        tensor = tensor.permute(0, 3, 1, 2)
        tensor = tensor[:, [2,1,0]]
        inputs = self.transform(tensor)
        return inputs

    def write_result(self, result_dir, vr, cluster_info, mid_frames, cluster_labels):
        def get_shot_type(pred):
            return {
                0: 'Extreme_Close-up',
                1: 'Close-up',
                2: 'Medium',
                3: 'Full',
                4: 'Long'
            }[pred]

        os.makedirs(result_dir, exist_ok=True)
        writer = cv2.VideoWriter(os.path.join(result_dir, 'out.mp4'),
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 vr.get_avg_fps(),
                                 (vr[0].shape[1], vr[0].shape[0]))

        cluster_info = np.array(cluster_info, dtype=object)
        for cluster_id in np.unique(cluster_labels):
            ind = cluster_labels == cluster_id
            cluster_dir = os.path.join(result_dir, str(cluster_id))
            os.makedirs(cluster_dir, exist_ok=True)

            for (shot_id, _, pred), frame_id in zip(cluster_info[ind], mid_frames[ind]):
                shot_type = get_shot_type(int(pred))
                frame = vr[frame_id].asnumpy()
                frame = cv2.putText(frame, f'{shot_type}', (10,60), cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(0,255,0), thickness=1, lineType=cv2.LINE_AA)
                cv2.imwrite(os.path.join(cluster_dir, f'{shot_id}_{frame_id}.jpg'), frame)

        writer.release()

    def get_predictions(self, batch_inputs, shot_ids, shot_lens, cluster_info):
        batch_inputs = torch.cat(batch_inputs, dim=0)
        feats = self.model.forward_feature(batch_inputs)
        preds = self.model.forward_linear(feats)

        shot_lens = np.cumsum([0] + shot_lens)
        for shot_id, shot_start, shot_end in zip(shot_ids, shot_lens[:-1], shot_lens[1:]):
            num_segments = shot_end - shot_start
            shot_feats = feats[shot_start:shot_end]
            shot_preds = preds[shot_start:shot_end]

            avg_feat = shot_feats.view(self.num_crops, num_segments, -1) \
                            .mean(dim=0).mean(dim=0).cpu().numpy()
            avg_pred = shot_preds.view(self.num_crops, num_segments, -1) \
                            .mean(dim=0).mean(dim=0).argmax().cpu().numpy()

            cluster_info.append((shot_id, avg_feat, avg_pred))

    @torch.no_grad()
    def forward(self, vr, segments, n_pseudo_cams=6, result_dir=None):
        cur_batch_size = 0
        batch_inputs, shot_ids, shot_lens = [], [], []

        cluster_info, mid_frames = [], []
        for shot_id, start, end in segments:
            frames = vr.get_batch(range(start, end+1)).asnumpy()
            num_segments = min(len(frames), self.max_num_segments)
            inputs = self.prepare_tsn_inputs(frames, num_segments).to(self.device)

            mid_frames.append((end + start)//2)

            batch_inputs.append(inputs)
            cur_batch_size += len(inputs)
            shot_ids.append(shot_id)
            shot_lens.append(len(inputs))
            if cur_batch_size > self.bsize:
                self.get_predictions(batch_inputs, shot_ids, shot_lens, cluster_info)

                cur_batch_size = 0
                batch_inputs, shot_ids, shot_lens = [], [], []
        if cur_batch_size > 0:
            self.get_predictions(batch_inputs, shot_ids, shot_lens, cluster_info)

        cluster_shot_feats = np.stack([info[1] for info in cluster_info])
        kmeans = KMeans(n_clusters=n_pseudo_cams, random_state=0).fit(cluster_shot_feats)

        if result_dir is not None:
            mid_frames = np.array(mid_frames)
            self.write_result(result_dir, vr, cluster_info, mid_frames, kmeans.labels_)

        return kmeans.labels_+1
