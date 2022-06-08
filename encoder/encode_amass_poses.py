import glob
import os
import pickle as pkl

import tensorflow as tf


def encode_sequence(dir, sequence, pose_encoder):
    regex_pkl = ('[0-9]' * 4) + "_bp.pkl"
    regex_pkl_filenames = glob.glob(os.path.join(dir, sequence, regex_pkl))
    regex_pkl_filenames.sort()

    for filename in regex_pkl_filenames:
        body_pose_as_angleaxis = pkl.load(open(filename, "rb"))
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
    dir = "../data/test_sequence/"
    sequences = ["Subject_6_F_7_poses"]
    encode_sequences(dir, sequences)


if __name__ == "__main__":
    main_example()
