import glob
import os
import pickle as pkl

import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation as R


def batched_slerp(batched_origins, batched_ends, t=0.5):
    batched_dot = np.einsum('bi,bi->b', batched_origins, batched_ends)

    ori_norms = np.linalg.norm(batched_origins, axis=1)
    end_norms = np.linalg.norm(batched_ends, axis=1)

    cosines = batched_dot / (ori_norms * end_norms)
    omegas = np.arccos(cosines)

    sin_first = np.sin((1 - t) * omegas)
    sin_second = np.sin(t * omegas)
    sin_omega = np.sin(omegas)

    first_factor = np.einsum('b,bi->bi', sin_first, batched_origins)
    second_factor = np.einsum('b,bi->bi', sin_second, batched_ends)

    final_interpolated = np.einsum('b,bi->bi', 1 / sin_omega, first_factor + second_factor)

    return final_interpolated


def encode_sequence(dir, sequence, pose_encoder):
    regex_pkl = ('[0-9]' * 4) + "_bp.pkl"
    regex_pkl_filenames = glob.glob(os.path.join(dir, sequence, regex_pkl))
    regex_pkl_filenames.sort()
    body_poses_as_angleaxis_smooth = None

    for filename in regex_pkl_filenames:
        body_poses_as_matrix = pkl.load(open(filename, "rb"))
        body_poses_as_angleaxis = R.from_matrix(body_poses_as_matrix).as_rotvec()

        # This reproduces the smoothing performed during reconstruction
        if body_poses_as_angleaxis_smooth is None:
            body_poses_as_angleaxis_smooth = body_poses_as_angleaxis
        else:
            body_poses_as_angleaxis_smooth = batched_slerp(
                body_poses_as_angleaxis_smooth,
                body_poses_as_angleaxis,
                t=0.5,
            )

        bp_angle_smooth_reshaped = body_poses_as_angleaxis_smooth.reshape(1, 23 * 3)
        body_poses_encoded = pose_encoder.predict(bp_angle_smooth_reshaped)
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
