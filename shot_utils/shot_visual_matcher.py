import cv2
import math
import numpy as np
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
from torchvision import transforms as T

from PIL import Image
from decord import VideoReader, cpu
from matplotlib import pyplot as plt

from .swin_transformer import SwinTransformer


class ShotVisualMatcher(nn.Module):
    def __init__(self, batch_size=64, device='cuda', n_ransac_iter=1000, tolerance=0.05, min_inlier=10):
        super(ShotVisualMatcher, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.n_ransac_iter = n_ransac_iter
        self.tolerance = tolerance
        self.min_inlier = min_inlier

        self.load_model_transform()

    def load_model_transform(self, ckpt='checkpoint/swin_tiny_patch4_window7_224.pth'):
        self.model = SwinTransformer(num_classes=1000,
                                     attn_drop_rate=0.,
                                     drop_rate=0.,
                                     drop_pat_rate=0.2).to(self.device)

        swint_state_dict = torch.load(ckpt)['model']
        msg = self.model.load_state_dict(swint_state_dict, strict=False)

        # self.transform = T.Compose([
        #     T.Resize(256),
        #     T.CenterCrop(224),
        #     T.ToTensor(),
        #     T.Normalize(mean=[0.485, 0.456, 0.406],
        #                 std=[0.229, 0.224, 0.225])
        # ])

        self.t_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        # model = models.resnet50(pretrained=True)
        # self.model = nn.Sequential(*list(model.children())[:-1]).to(self.device)

        # layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']
        # module_list = [getattr(model, l) for l in layers]
        # self.model = torch.nn.Sequential(*module_list).to(self.device)

    def forward_rn50(self, frames):
        frames = torch.from_numpy(frames)
        frames = frames.permute(0, 3, 1, 2)
        frames = frames / 255
        frames = self.t_transform(frames)

        feats = []
        for ind in range(0, len(frames), self.batch_size):
            inputs = frames[ind:ind+self.batch_size].to(self.device)
            feats.append(self.model.forward_features(inputs).cpu())
        feats = torch.cat(feats, dim=0)
        return feats


    def forward_multiscale(self, frames, ratios=[1]):
        feats = self.forward_rn50(frames)
        feats = feats.view(feats.shape[0], -1)
        feats = F.normalize(feats, dim=-1)
        return feats

    def match_shots(self, shot_end_feats, shot_start_feats, segments):
        sim_mat = torch.mm(shot_end_feats, shot_start_feats.permute(1,0))
        return sim_mat.numpy()

    def visualize_similarity(self, end_frames, start_frames, segments, sim_mat, out_root, img_per_row=10):
        os.makedirs(out_root, exist_ok=True)
        row, col = int(math.ceil(len(start_frames)/img_per_row)), img_per_row+1

        for shot_id, end_frame, sim in zip(segments[:,0], end_frames, sim_mat):
            fig, ax = plt.subplots(row, col, figsize=(16,12), gridspec_kw = {'wspace':0, 'hspace':0})
            for axis in ax.ravel():
                axis.set_axis_off()

            ax[0, 0].imshow(end_frame)
            ind = sim.argsort()[::-1]
            for rank, (next_frame, match_ratio) in enumerate(zip(start_frames[ind], sim[ind])):
                r, c = rank//img_per_row, rank%img_per_row
                next_frame = cv2.putText(
                    next_frame, f'{match_ratio*100:.2f}', (10,10), cv2.FONT_HERSHEY_SIMPLEX,  
                    fontScale=1, color=(0,255,0), thickness=1, lineType=cv2.LINE_AA
                ) 
                ax[r, 1+c].imshow(next_frame)

            plt.tight_layout()
            fig.savefig(os.path.join(out_root, f'{shot_id}.jpg'))
            plt.close(fig)

    @torch.no_grad()
    def forward(self, vr, segments):
        shot_end_frames = vr.get_batch(segments[:,2]).asnumpy()
        shot_start_frames = vr.get_batch(segments[:,1]).asnumpy()

        shot_end_feats = self.forward_multiscale(shot_end_frames)
        shot_start_feats = self.forward_multiscale(shot_start_frames)

        sim_mat = self.match_shots(shot_end_feats, shot_start_feats, segments)
        return sim_mat

        # print('Visualizing similarity results')
        # self.visualize_similarity(shot_end_frames,
        #                           shot_start_frames,
        #                           segments,
        #                           sim_mat,
        #                           out_root=os.path.join('plots', os.path.basename(video_path)))
