import os
import pickle as pkl

import numpy as np
import smplx
import torch
import tqdm
from scipy.spatial.transform import Rotation as R


def separate_arms(body_poses_as_matrix, angle=20, left_arm=16, right_arm=15):
    num_joints = body_poses_as_matrix.shape[1]

    poses = body_poses_as_matrix.reshape((-1, num_joints, 3, 3))
    rot = R.from_euler('z', -angle, degrees=True)
    poses[:, left_arm] = (rot * R.from_matrix(poses[:, left_arm])).as_matrix()
    rot = R.from_euler('z', angle, degrees=True)
    poses[:, right_arm] = (rot * R.from_matrix(poses[:, right_arm])).as_matrix()

    poses[:, 22] *= R.from_rotvec(R.from_matrix(poses[:, 22]).as_rotvec() * 0.1).as_matrix()
    poses[:, 21] *= R.from_rotvec(R.from_matrix(poses[:, 22]).as_rotvec() * 0.1).as_matrix()

    return poses


def process_sequence(directory, motion_path, betas, smpl_model):
    sequence = motion_path.split(".")[0]

    motion = np.load(os.path.join(directory, motion_path), mmap_mode='r')
    global_orient = motion["poses"][:, 0:3]
    motion_poses = motion["poses"][:, 3:72]

    iterator = tqdm.tqdm(
        iterable=enumerate(zip(motion_poses, global_orient)),
        desc="Processing poses",
        total=len(motion_poses),
    )
    for i, (pose, orient) in iterator:
        matrix_pose = np.expand_dims(R.from_rotvec(pose.reshape(23, 3)).as_matrix(), axis=0)
        matrix_pose = separate_arms(matrix_pose)

        pose = R.from_matrix(matrix_pose.squeeze()).as_rotvec()
        orient = torch.from_numpy(R.from_rotvec(orient).as_matrix()).unsqueeze(0).float()

        matrix_pose = torch.from_numpy(matrix_pose).float()

        fmodel = smpl_model(betas=betas,
                            global_orient=orient,
                            trans=torch.from_numpy(motion["trans"]),
                            body_pose=matrix_pose)

        output_dir = os.path.join(directory, sequence)
        os.makedirs(output_dir, exist_ok=True)
        pkl.dump(pose, open(os.path.join(output_dir, f"{i:04}_bp.pkl"), "wb"))
        pkl.dump(fmodel, open(os.path.join(output_dir, f"{i:04}.pkl"), "wb"))


def process_sequences(directory, motion_paths):
    smpl_model_path = '../data/smpl/smpl_neutral.pkl'
    smpl_model = smplx.build_layer(smpl_model_path)

    betas = torch.from_numpy(pkl.load(open("../data/mean_betas.pkl", "rb"))).unsqueeze(0)

    for motion_path in motion_paths:
        process_sequence(directory, motion_path, betas, smpl_model)


def main_example():
    directory = "../data/test_sequence/"
    motion_paths = ["Subject_6_F_7_poses.npz"]
    process_sequences(directory, motion_paths)


if __name__ == "__main__":
    main_example()
