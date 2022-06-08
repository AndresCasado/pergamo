import glob
import os
import pickle as pkl


def encode_sequence(dir, sequence):
    regex_pkl = ('[0-9]' * 4) + ".pkl"
    regex_pkl_filenames = glob.glob(os.path.join(dir, sequence, regex_pkl))
    regex_pkl_filenames.sort()

    for filename in regex_pkl_filenames:
        smpl_model = pkl.load(open(filename, "rb"))
        smpl_body_pose = smpl_model["body_pose"].detach().cpu().squeeze().numpy()

        output_filename = filename.replace(".pkl", "_bp.pkl")
        with open(output_filename, 'wb') as f:
            pkl.dump(smpl_body_pose, f)


def encode_sequences(dir, sequences):
    for sequence in sequences:
        encode_sequence(dir, sequence)


def main_example():
    dir = "../data/train_sequence/poses/"
    sequences = ["dan-005"]
    encode_sequences(dir, sequences)

    dir = "../data/validation_sequence/poses/"
    sequences = ["dan-013"]
    encode_sequences(dir, sequences)


if __name__ == "__main__":
    main_example()
