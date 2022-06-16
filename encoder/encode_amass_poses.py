import glob
import os
import pickle as pkl

import torch

from encoder.pose_encoder_10D_torch import PoseEncoder10D


def encode_sequence(directory, sequence, pose_encoder):
    regex_pkl = ('[0-9]' * 4) + "_bp.pkl"
    regex_pkl_filenames = glob.glob(os.path.join(directory, sequence, regex_pkl))
    regex_pkl_filenames.sort()

    for filename in regex_pkl_filenames:
        body_pose_as_angleaxis = pkl.load(open(filename, "rb"))
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
    directory = "../data/test_sequence/"
    sequences = ["Subject_6_F_7_poses"]
    encode_sequences(directory, sequences)


if __name__ == "__main__":
    main_example()
