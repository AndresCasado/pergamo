import glob
import os
import pickle as pkl


def process_sequence(directory, sequence):
    regex_pkl = ('[0-9]' * 4) + ".pkl"
    regex_pkl_filenames = glob.glob(os.path.join(directory, sequence, regex_pkl))
    regex_pkl_filenames.sort()

    for filename in regex_pkl_filenames:
        smpl_model = pkl.load(open(filename, "rb"))
        smpl_body_pose = smpl_model["body_pose"].detach().cpu().squeeze().numpy()

        output_filename = filename.replace(".pkl", "_bp.pkl")
        with open(output_filename, 'wb') as f:
            pkl.dump(smpl_body_pose, f)


def process_sequences(directory, sequences):
    for sequence in sequences:
        process_sequence(directory, sequence)


def main_example():
    directory = "../data/train_sequence/poses/"
    sequences = ["dan-005"]
    process_sequences(directory, sequences)

    directory = "../data/validation_sequence/poses/"
    sequences = ["dan-013"]
    process_sequences(directory, sequences)


if __name__ == "__main__":
    main_example()
