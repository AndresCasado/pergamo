import glob
import os
import pickle as pkl

import numpy as np
import torch
from tqdm import tqdm

from tools.animation_storage_tools import save_pc2


def predict_sequence(regressor, sequence, mean_pose, sd_pose, mean_shirt, max_offset, min_offset, epoch):
    regex_pkl = "*_enc.pkl"
    regex_pkl_filenames = glob.glob(os.path.join(sequence, regex_pkl))
    regex_pkl_filenames.sort()
    body_poses_pca_features = [pkl.load(open(x, "rb")) for x in regex_pkl_filenames]

    vertices = []
    for pose in tqdm(body_poses_pca_features):
        pose = torch.tensor(pose).float().cuda()
        pose = (pose - mean_pose) / sd_pose

        vertices_pred = regressor(pose.unsqueeze(0))
        vertices_pred = vertices_pred * (max_offset - min_offset) + min_offset + mean_shirt
        vertices.append(vertices_pred.cpu())

    vertices = [x.squeeze().detach().cpu().numpy() for x in vertices]
    vertices = np.array(vertices)
    save_pc2(sequence + "_" + str(epoch) + ".pc2", vertices)
