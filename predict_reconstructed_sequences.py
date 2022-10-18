import glob
import os
import pickle as pkl

import igl
import numpy as np
import torch
import tqdm
from scipy.spatial.transform import Rotation as R
from encoder.encode_reconstructed_poses import batched_slerp
from regressor import Regressor
from tools.animation_storage_tools import save_pc2, save_kaolin_mesh
from tools.collision_tools import push_vertices
from tools.posing_tools import load_poser


def predict_reconstructed_sequence(
        directory,
        sequence,
        regressor,
        mean_pose,
        sd_pose,
        mean_shirt,
        max_offset,
        min_offset,
        betas,
        epoch
):
    regex_pkl = "*_enc.pkl"
    regex_pkl_filenames = glob.glob(os.path.join(directory, sequence, regex_pkl))
    regex_pkl_filenames.sort()

    regex_bp = ('[0-9]' * 4) + ".pkl"
    regex_bp_filenames = glob.glob(os.path.join(directory, sequence, regex_bp))
    regex_bp_filenames.sort()

    if not (regex_bp_filenames and regex_pkl_filenames):
        raise RuntimeError('I need files to process!')

    body_poses_pca_features = [torch.load(x) for x in regex_pkl_filenames]
    smpl_data_all = [pkl.load(open(x, "rb")) for x in regex_bp_filenames]

    global_orient_all = []
    body_poses_as_matrices_all = []
    for x in smpl_data_all:
        global_orient_all.append(
            x["global_orient"].cpu()
        )
        body_poses_as_matrices_all.append(
            x["body_pose"].detach().cpu()
        )

    for i in range(1, len(body_poses_as_matrices_all)):
        end_rotation_mat = body_poses_as_matrices_all[i][0]
        start_rotation_mat = body_poses_as_matrices_all[i - 1][0]

        end_rotation_vec = R.from_matrix(end_rotation_mat).as_rotvec()
        start_rotation_vec = R.from_matrix(start_rotation_mat).as_rotvec()

        test = batched_slerp(end_rotation_vec, start_rotation_vec)
        slerped_mat = R.from_rotvec(test).as_matrix()

        body_poses_as_matrices_all[i] = torch.from_numpy(slerped_mat).float().unsqueeze(0)

    poser, lbs_weights, smpl_layer = load_poser()

    # Read with IGL so that the subdivision later doesn't conflict
    template_verts, template_faces = igl.read_triangle_mesh("data/mean_shirt.obj")

    offsets_acc = None
    vertices = []
    vertices_body = []
    iterator = tqdm.tqdm(
        iterable=zip(body_poses_pca_features, body_poses_as_matrices_all, global_orient_all),
        total=len(body_poses_pca_features),
    )
    for pose, body_pose, orient in iterator:
        pose = pose.float().cuda()
        pose = (pose - mean_pose) / sd_pose

        offsets_pred = regressor(pose.unsqueeze(0))
        if offsets_acc is None:
            offsets_acc = offsets_pred
        else:
            offsets_acc = offsets_acc * 0.5 + offsets_pred * 0.5
            offsets_pred = offsets_acc

        vertices_pred = offsets_pred * (max_offset - min_offset) + min_offset + mean_shirt

        kwargs = {
            'betas': betas.squeeze().unsqueeze(0),
            'body_pose': body_pose,
            'global_orient': orient,
        }

        with torch.no_grad():
            posed_vertices = poser.pose(
                vertices=vertices_pred.cpu().squeeze().unsqueeze(0),
                betas=betas,
                smplx_kwargs=kwargs,
                lbs_weights=lbs_weights,
                unpose=False
            )[0]

            body = smpl_layer(**kwargs)

        body_vertices = body.vertices.squeeze().cpu().detach().numpy()

        posed_vertices = posed_vertices.squeeze().cpu().detach().numpy()

        pushed = push_vertices(
            posed_vertices,
            body_vertices,
            smpl_layer.faces.astype(np.int32),
            epsilon=1e-2,
        )

        vertices.append(pushed)
        vertices_body.append(body_vertices)

    meshes = [
        igl.loop(verts, template_faces, number_of_subdivs=1)
        for verts in vertices
    ]

    # Save one mesh, the animation can be loaded with the PC2 file
    first_mesh = meshes[0]
    save_kaolin_mesh(
        path=os.path.join(directory, f"{sequence}_{epoch}.obj"),
        verts=first_mesh[0],
        faces=first_mesh[1],
    )

    mesh_vertices = [mesh[0] for mesh in meshes]
    mesh_vertices = np.array(mesh_vertices)

    vertices_body = np.array(vertices_body)

    save_pc2(os.path.join(directory, f"{sequence}_{epoch}.pc2"), mesh_vertices)
    save_pc2(os.path.join(directory, f"{sequence}_{epoch}_body.pc2"), vertices_body)


def predict_reconstructed_sequences(directory, sequences):
    params = pkl.load(open("data/params_enc.pkl", "rb"))

    mean_shirt = torch.from_numpy(params["mean_shirt"]).cuda()
    max_offset = torch.from_numpy(params["max_offset"]).cuda()
    min_offset = torch.from_numpy(params["min_offset"]).cuda()
    mean_pose = torch.from_numpy(params["mean_pose"]).cuda()
    sd_pose = torch.from_numpy(params["sd_pose"]).cuda()

    betas = torch.from_numpy(pkl.load(open("data/mean_betas.pkl", "rb"))).unsqueeze(0)

    regressor = Regressor(in_channels=10, out_channels=4424)
    chp = torch.load("data/checkpoints/regressor-epoch-97-losses-0.075.pth")
    epoch = 97  # Same as the loaded weights
    regressor.load_state_dict(chp)
    regressor = regressor.cuda()

    for sequence in sequences:
        predict_reconstructed_sequence(
            directory,
            sequence,
            regressor,
            mean_pose,
            sd_pose,
            mean_shirt,
            max_offset,
            min_offset,
            betas,
            epoch)


def main_example():
    directory = "data/train_sequence/poses/"
    sequences = ["dan-005"]
    predict_reconstructed_sequences(directory, sequences)

    directory = "data/validation_sequence/poses/"
    sequences = ["dan-013"]
    predict_reconstructed_sequences(directory, sequences)


if __name__ == "__main__":
    main_example()
