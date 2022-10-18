import glob
import os
import pickle as pkl

import igl
import numpy as np
import torch
import trimesh
from tqdm import tqdm

from tools.colision_tools import push_vertices
from tools.posing_tools import load_poser
from tools.animation_storage_tools import save_pc2
from regressor import Regressor


def predict_amass_sequence(
        dir,
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
    regex_pkl_filenames = glob.glob(os.path.join(dir, sequence, regex_pkl))
    regex_pkl_filenames.sort()
    body_poses_pca_features = [pkl.load(open(x, "rb")) for x in regex_pkl_filenames]

    regex_bp = ('[0-9]' * 4) + ".pkl"
    regex_bp_filenames = glob.glob(os.path.join(dir, sequence, regex_bp))
    regex_bp_filenames.sort()
    smpl_models = [pkl.load(open(x, "rb")) for x in regex_bp_filenames]
    global_orient = [x["global_orient"] for x in smpl_models]
    body_poses_as_matrices = [x["body_pose"].detach().cpu() for x in smpl_models]

    poser, lbs_weights, smpl_layer = load_poser()

    offsets_acc = None
    vertices = []
    vertices_body = []
    for pose, body_pose, orient in tqdm(zip(body_poses_pca_features, body_poses_as_matrices, global_orient)):
        pose = torch.tensor(pose).float().cuda()
        pose = (pose - mean_pose) / sd_pose

        offsets_pred = regressor(pose.unsqueeze(0))
        if offsets_acc is None:
            offsets_acc = offsets_pred
        else:
            offsets_acc = offsets_acc * 0.6 + offsets_pred * 0.4
            offsets_pred = offsets_acc

        vertices_pred = offsets_pred * (max_offset - min_offset) + min_offset + mean_shirt

        kwargs = {
            'betas': betas,
            'body_pose': body_pose.squeeze().unsqueeze(0),
            'global_orient': orient.unsqueeze(0)
        }

        with torch.no_grad():
            posed_vertices = poser.pose(
                vertices=vertices_pred.cpu(),
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

    v, f = igl.read_triangle_mesh("data/mean_shirt.obj")
    meshes = [igl.loop(verts, f, number_of_subdivs=1) for verts in vertices]

    template = trimesh.Trimesh(meshes[0][0], meshes[0][1], process=False)
    template.export(os.path.join(dir, sequence + "_" + str(epoch) + ".obj"))

    mesh_vertices = [mesh[0] for mesh in meshes]
    mesh_vertices = np.array(mesh_vertices)

    vertices_body = np.array(vertices_body)

    save_pc2(os.path.join(dir, sequence + "_" + str(epoch) + ".pc2"), mesh_vertices)
    save_pc2(os.path.join(dir, sequence + "_" + str(epoch) + "_body.pc2"), vertices_body)


def predict_amass_sequences(dir, sequences):
    params = pkl.load(open("data/params_enc.pkl", "rb"))

    mean_shirt = torch.from_numpy(params["mean_shirt"]).cuda()
    max_offset = torch.from_numpy(params["max_offset"]).cuda()
    min_offset = torch.from_numpy(params["min_offset"]).cuda()
    mean_pose = torch.from_numpy(params["mean_pose"]).cuda()
    sd_pose = torch.from_numpy(params["sd_pose"]).cuda()

    betas = torch.from_numpy(pkl.load(open("data/mean_betas.pkl", "rb"))).unsqueeze(0)

    regressor = Regressor(in_channels=10, out_channels=4424)
    chp = torch.load("data/checkpoints/regressor-epoch-97-losses-0.075.pth")
    regressor.load_state_dict(chp)
    regressor = regressor.cuda()

    for sequence in sequences:
        predict_amass_sequence(
            dir,
            sequence,
            regressor,
            mean_pose,
            sd_pose,
            mean_shirt,
            max_offset,
            min_offset,
            betas,
            97
        )


def main_example():
    dir = "data/test_sequence/"
    sequences = ["Subject_6_F_7_poses"]
    predict_amass_sequences(dir, sequences)


if __name__ == "__main__":
    main_example()
