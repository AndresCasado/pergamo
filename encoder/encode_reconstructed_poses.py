import glob
import os
import pickle as pkl

import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation as R


def encode_sequence(dir, sequence, pose_encoder):
    regex_pkl = ('[0-9]' * 4) + "_bp.pkl"
    regex_pkl_filenames = glob.glob(os.path.join(dir, sequence, regex_pkl))
    regex_pkl_filenames.sort()
    body_poses_as_matrix_smooth = None

    for filename in regex_pkl_filenames:
        body_poses_as_matrix = pkl.load(open(filename, "rb"))

        # This reproduces the smoothing performed during reconstruction
        if body_poses_as_matrix_smooth is None:
            body_poses_as_matrix_smooth = body_poses_as_matrix
        else:
            body_poses_as_matrix_smooth = body_poses_as_matrix_smooth * 0.5 + body_poses_as_matrix * 0.5

        body_pose_as_angleaxis = np.concatenate([R.from_matrix(x).as_rotvec() for x in body_poses_as_matrix_smooth])
        body_pose_as_angleaxis = body_pose_as_angleaxis.reshape([1, 23 * 3])
        body_poses_encoded = pose_encoder.predict(body_pose_as_angleaxis)
        output_filename = filename.replace("_bp.pkl", "_enc.pkl")
        with open(output_filename, 'wb') as f:
            pkl.dump(body_poses_encoded, f)


def encode_sequences(dir, sequences):
    pose_encoder_path = "pose_encoder_10D"
    pose_encoder = tf.keras.models.load_model(pose_encoder_path, compile=False)

    for sequence in sequences:
        encode_sequence(dir, sequence, pose_encoder)


def main_example():
    dir = "../data/train_sequence/poses/"
    sequences = ["dan-005"]
    encode_sequences(dir, sequences)

    dir = "../data/validation_sequence/poses/"
    sequences = ["dan-013"]
    encode_sequences(dir, sequences)


if __name__ == "__main__":
    main_example()
