import glob
import os
import pickle as pkl

import numpy as np
import torch
import tqdm
from scipy.spatial.transform import Rotation as R

from encoder.pose_encoder_10D_torch import PoseEncoder10D


def encode_sequence(directory, sequence, pose_encoder):
    regex_pkl = ('[0-9]' * 4) + "_bp.pkl"
    regex_pkl_filenames = glob.glob(os.path.join(directory, sequence, regex_pkl))
    regex_pkl_filenames.sort()
    body_poses_as_matrix_smooth = None

    iterator = tqdm.tqdm(regex_pkl_filenames, desc="Encoding poses")
    for filename in iterator:
        body_poses_as_matrix = pkl.load(open(filename, "rb"))

        # This reproduces the smoothing performed during reconstruction
        if body_poses_as_matrix_smooth is None:
            body_poses_as_matrix_smooth = body_poses_as_matrix
        else:
            body_poses_as_matrix_smooth = body_poses_as_matrix_smooth * 0.5 + body_poses_as_matrix * 0.5

        body_pose_as_angleaxis = np.concatenate([R.from_matrix(x).as_rotvec() for x in body_poses_as_matrix_smooth])
        body_pose_as_angleaxis = body_pose_as_angleaxis.reshape([1, 23 * 3])
        body_pose_as_angleaxis = torch.from_numpy(body_pose_as_angleaxis).float()
        body_poses_encoded = pose_encoder.forward(body_pose_as_angleaxis)
        output_filename = filename.replace("_bp.pkl", "_enc.pth")
        torch.save(body_poses_encoded, output_filename)


def encode_sequences(directory, sequences):
    pose_encoder_path = "../pose_encoder_10D_converted.pth"
    pose_encoder = PoseEncoder10D()
    pose_encoder.load_state_dict(torch.load(pose_encoder_path))

    for sequence in sequences:
        encode_sequence(directory, sequence, pose_encoder)


def main_example():
    directory = "../data/train_sequence/poses/"
    sequences = ["dan-005"]
    encode_sequences(directory, sequences)

    directory = "../data/validation_sequence/poses/"
    sequences = ["dan-013"]
    encode_sequences(directory, sequences)


if __name__ == "__main__":
    main_example()
