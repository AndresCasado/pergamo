import argparse
import hashlib
import os
import pickle
import re
import typing
import warnings
from collections import namedtuple

import kaolin.io.obj as kaobj
import matplotlib.pyplot as plt
import numpy as np
import smplx
import torch
import torch.nn.functional as F
import torchvision.utils as tvutils
import tqdm

from tools.animation_storage_tools import save_kaolin_mesh
from tools.collision_tools import push_vertices
from tools.custom_logging import ImageLogger, LossLogger, MeshLogger
from tools.image_gradient import compute_image_gradient
from tools.kaolin_rendering import KaolinExPoseRenderer
from tools.mesh_edge_lengths import compute_mesh_edge_lengths
from tools.posing_tools import compute_lbs_weights, TshirtPoser, TshirtOffsetter
from tools.vae import MeshOffsetVAE


def change_range(tensor, oldmin=None, oldmax=None, newmin=0.0, newmax=1.0):
    if oldmin is None:
        oldmin = tensor.min()
    if oldmax is None:
        oldmax = tensor.max()
    return (tensor - oldmin) / (oldmax - oldmin) * (newmax - newmin) + newmin


def read_converted_expose_path(filepath):
    if os.path.isfile(filepath):
        warnings.warn('Loading cached ExPose conversion')
        basename, extension = os.path.splitext(filepath)
        if extension == '.torch':
            smpl_params = torch.load(filepath)
        elif extension == '.pkl':
            smpl_params = pickle.load(open(filepath, 'rb'))
            ignore_keys = ['vertices', 'joints', 'full_pose', 'v_shaped', 'faces']
            for k in ignore_keys:
                del smpl_params[k]
        else:
            raise TypeError(f"I don't understand the extension {extension}")
    else:
        raise FileNotFoundError(f'The file with the conversion does not exist: {filepath}')
    return smpl_params


def read_silhouette_new(filepath):
    loaded = np.load(filepath)
    silhouette = torch.from_numpy(loaded).float()
    return silhouette


def read_pifu_normals(filepath):
    # Normals from PiFuHD
    from PIL import Image
    pifu_result = Image.open(filepath)
    pifu_result_normals = pifu_result.crop(
        # (x0, y0, x1, y1)
        (512, 0, 512 * 2, 512)
    )
    pifu_result_normals = torch.from_numpy(np.array(pifu_result_normals))
    pifu_result_normals = pifu_result_normals / 255.0

    return pifu_result_normals


def process_one(
        template_obj_path: str,
        smpl_model_path: str,
        smpl_params,
        autoencoder_state_dict_path: str,
        script_meaning: str,
        log_path: str,
        pifu_result_normals,
        silhouette,
        expose,
        parsing_resolution,
        vae_prev_param: torch.Tensor,
        offset_prev_param: torch.Tensor,
        show_normals: bool = False,
):
    DEVICE = torch.device('cuda')

    VAE_REG_WEIGHT = 0.01  # 0.04
    TMP_REG_WEIGHT = 1  # 15.0

    normals = torch.where(
        silhouette.unsqueeze(-1) > 0.5,  # Condition
        pifu_result_normals,  # Value if condition met
        torch.ones_like(pifu_result_normals) * 0.5,  # Value if not met
    )
    if show_normals:
        plt.imshow(normals)
        plt.show()

    # tvutils.save_image(normals.permute(2, 0, 1), 'normalitas.png')

    # SMPL-X
    smpl_model = smplx.build_layer(smpl_model_path)

    tshirt_obj = kaobj.import_mesh(template_obj_path)
    tshirt_vs = tshirt_obj.vertices[None].to(DEVICE)
    tshirt_fs = tshirt_obj.faces.to(DEVICE)

    edges = set()
    for face in tshirt_obj.faces.numpy():
        for fid in range(3):
            vi = face[fid]
            vj = face[(fid + 1) % 3]
            if (vj, vi) in edges:
                edges.remove((vj, vi))
            else:
                edges.add((vi, vj))

    boundary_vertices = list(set([v for pair in edges for v in pair]))

    is_not_boundary = np.ones_like(tshirt_obj.vertices)
    is_not_boundary[boundary_vertices] = [0, 0, 0]
    is_not_boundary = torch.from_numpy(is_not_boundary).to(DEVICE)

    lbs = compute_lbs_weights(
        tshirt_obj.vertices,
        smpl_model().vertices[0],
        smpl_model.lbs_weights,
    )
    tshirt_poser = TshirtPoser(
        smpl_model=smpl_model,
        lbs_weights=lbs,
        **smpl_params,
        device=torch.device('cuda'),
    )

    renderer = KaolinExPoseRenderer(
        parsing_resolution, parsing_resolution,
        expose_params=expose,
        smooth_normals=True,
    )

    autoencoder = MeshOffsetVAE(device=DEVICE)
    autoencoder_state_dict = torch.load(autoencoder_state_dict_path)['model']
    autoencoder = autoencoder.to(DEVICE)
    autoencoder.load_state_dict(autoencoder_state_dict)

    offsetter = TshirtOffsetter(vertices=tshirt_vs)

    im_logger = ImageLogger(log_path, every=10)
    loss_logger = LossLogger(script_meaning)
    mesh_logger = MeshLogger(log_path, every=30)

    loss_fun = torch.nn.MSELoss()
    gt_improb = silhouette[None].clone().to(DEVICE).float()
    gt_imnormal = normals.clone().to(DEVICE) * 2.0 - 1.0

    def loss_only_silhouette(input_x, _):
        posed_tshirt = tshirt_poser(offsetter(autoencoder.decode(input_x)))

        render_result = renderer(
            posed_tshirt,
            faces=tshirt_fs,
        )

        (imnormal,), improb, imdx = render_result
        improb = F.interpolate(improb[None], size=(512, 512))[0]
        imnormal = F.interpolate(imnormal[None], size=(512, 512, 3))[0]

        loss_improb = loss_fun(improb, gt_improb)

        loss_reg_temporal_vae = 0.0
        if vae_prev_param is not None:
            loss_reg_temporal_vae = (input_x - vae_prev_param).square().mean()

        loss_reg_vae = input_x.square().mean()

        loss = (loss_improb
                + loss_reg_vae * VAE_REG_WEIGHT
                + loss_reg_temporal_vae * TMP_REG_WEIGHT)

        with torch.no_grad():
            normal_diff = imnormal - gt_imnormal

        loss_components = {
            'loss_improb': loss_improb,
            'loss_reg_vae': loss_reg_vae,
            'loss_reg_temporal_vae': loss_reg_temporal_vae
        }
        log_images = {
            'normal': change_range(imnormal, -1., 1.),
            'diff': change_range(normal_diff, -2., 2.),
        }
        log_meshes = {
            'mesh': (posed_tshirt, tshirt_fs),
        }
        loss_logger.log_losses(loss_components)
        im_logger.save_images(log_images)
        mesh_logger.save_meshes(log_meshes)

        return loss

    last_improb_of_next_loss = None

    def loss_normal_and_silhouette_vpk(input_x, _):
        posed_tshirt = tshirt_poser(offsetter(autoencoder.decode(input_x)))

        render_result = renderer(
            posed_tshirt,
            faces=tshirt_fs,
        )

        (imnormal,), improb, imdx = render_result
        improb = F.interpolate(improb[None], size=(512, 512))[0]
        imnormal = F.interpolate(imnormal[None], size=(512, 512, 3))[0]

        nonlocal last_improb_of_next_loss
        last_improb_of_next_loss = improb.detach()

        loss_improb = loss_fun(improb, gt_improb)
        normal_diff = imnormal - gt_imnormal
        loss_imnormal = normal_diff.square().mean()

        loss_reg_temporal_vae = 0.0
        if vae_prev_param is not None:
            loss_reg_temporal_vae = (input_x - vae_prev_param).square().mean()

        loss_reg_vae = input_x.square().mean()

        loss = (loss_imnormal
                + loss_improb
                + loss_reg_vae * VAE_REG_WEIGHT
                + loss_reg_temporal_vae * TMP_REG_WEIGHT)

        loss_components = {
            'loss_imnormal': loss_imnormal,
            'loss_improb': loss_improb,
            'loss_reg_vae': loss_reg_vae,
            'loss_reg_temporal_vae': loss_reg_temporal_vae
        }
        log_images = {
            'normal': change_range(imnormal, -1., 1.),
            'diff': change_range(normal_diff, -2., 2.)
        }
        log_meshes = {
            'mesh': (posed_tshirt, tshirt_fs),
        }
        loss_logger.log_losses(loss_components)
        im_logger.save_images(log_images)
        mesh_logger.save_meshes(log_meshes)

        return loss

    def loss_vpk_freevertex_lengths(input_x, free_vertex):
        t_pose_tshirt = offsetter(autoencoder.decode(input_x))
        free_vertex = free_vertex * is_not_boundary
        free_vertex = torch.clamp(free_vertex, -0.02, 0.02)
        free_vertex[..., [0, 1]] = torch.clamp(free_vertex[..., [0, 1]], -0.004, 0.004)
        free_moved_tshirt = t_pose_tshirt + free_vertex
        posed_tshirt = tshirt_poser(free_moved_tshirt)

        t_posed_lengths = compute_mesh_edge_lengths(t_pose_tshirt[0], tshirt_fs)
        freemoved_lengths = compute_mesh_edge_lengths(free_moved_tshirt[0], tshirt_fs)

        render_result = renderer(
            posed_tshirt,
            faces=tshirt_fs,
        )

        (imnormal,), _, _ = render_result
        imnormal = F.interpolate(imnormal[None], size=(512, 512, 3))[0]

        horizontal_gradient_diff = compute_image_gradient(imnormal) - compute_image_gradient(gt_imnormal[None])
        horizontal_gradient_diff *= last_improb_of_next_loss.unsqueeze(-1)
        vertical_gradient_diff = compute_image_gradient(imnormal, True) - compute_image_gradient(gt_imnormal[None],
                                                                                                 True)
        vertical_gradient_diff *= last_improb_of_next_loss.unsqueeze(-1)
        loss_imnormal = horizontal_gradient_diff.square().mean() + vertical_gradient_diff.square().mean()

        loss_reg_lengths = (t_posed_lengths - freemoved_lengths).square().mean()

        loss_reg_temporal_offset = 0.0

        if offset_prev_param is not None:
            loss_reg_temporal_offset = (free_vertex - offset_prev_param).square().mean()

        loss_reg_offset_z = torch.norm(free_vertex[..., [2]])

        loss = (
                loss_imnormal * 15.0 / 0.0023
                + loss_reg_lengths * 1e4
                + loss_reg_offset_z * 1e-1
                + loss_reg_temporal_offset * 1e10
        )

        loss_components = {
            'loss_imnormal': loss_imnormal,
            'loss_reg_lengths': loss_reg_lengths,
            'loss_reg_offset_z': loss_reg_offset_z,
            'loss_reg_temporal_offset': loss_reg_temporal_offset
        }
        log_images = {
            'normal': change_range(imnormal, -1., 1.),
            'normal-gt': change_range(gt_imnormal[None], -1., 1.),
            'diff': change_range(horizontal_gradient_diff, -2., 2.)
        }
        log_meshes = {
            'mesh': (posed_tshirt, tshirt_fs),
            'tposed': (free_moved_tshirt, tshirt_fs),
        }
        loss_logger.log_losses(loss_components)
        im_logger.save_images(log_images)
        mesh_logger.save_meshes(log_meshes)

        return loss

    it_vae = 200
    input_x = torch.zeros(1, 25, requires_grad=True, device=DEVICE)
    if vae_prev_param is not None:
        it_vae = 15
        input_x = torch.clone(vae_prev_param).requires_grad_(True).to(DEVICE)
        print(f'Vae norm: {vae_prev_param.square().mean()}')

    it_vertex = 200
    free_vertex = torch.zeros_like(tshirt_vs, requires_grad=True, device=DEVICE)
    if offset_prev_param is not None:
        it_vertex = 31
        print(f'Vert norm: {offset_prev_param.square().mean()}')
        print(f'Vert max: {offset_prev_param.max()}')
        offset_prev_param = torch.clamp(offset_prev_param, -0.019, 0.019)
        offset_prev_param[..., [0, 1]] = torch.clamp(offset_prev_param[..., [0, 1]], -0.0039, 0.0039)
        free_vertex = torch.clone(offset_prev_param).requires_grad_(True).to(DEVICE)

    parameters = [input_x, free_vertex]

    adams = [
        torch.optim.Adam(parameters, lr=1e-2),
        torch.optim.Adam(parameters, lr=1e-3),
        torch.optim.Adam([free_vertex], lr=1e-4)
    ]

    OptimizationStep = namedtuple('OptimizationStep', ['loss', 'iterations'])

    optimization_steps = [
        OptimizationStep(loss_only_silhouette, it_vae),
        OptimizationStep(loss_normal_and_silhouette_vpk, it_vae),
        OptimizationStep(loss_vpk_freevertex_lengths, it_vertex)
    ]

    for adam, step in zip(adams, optimization_steps, ):
        step_loss = step.loss
        step_iterations = step.iterations

        iterator = tqdm.tqdm(range(step_iterations))
        for i in iterator:
            adam.zero_grad()
            loss = step_loss(*parameters)
            loss.backward()

            adam.step()
            loss_item = loss.detach().item()
            iterator.set_postfix({
                'loss': loss_item,
            })

    # Postprocess

    input_x, free_vertex = parameters
    input_x = input_x.requires_grad_(False)
    free_vertex = free_vertex.requires_grad_(False)
    free_vertex = free_vertex * is_not_boundary

    real_body = smpl_model.to(DEVICE)(**smpl_params)
    posed_tshirt = tshirt_poser(offsetter(autoencoder.decode(input_x)) + free_vertex)
    pushed_vertices = push_vertices(
        posed_tshirt[0].cpu().detach().numpy(),
        real_body.vertices.cpu().detach().numpy()[0],
        smpl_model.faces.astype(np.int32),
        0.010000
    )

    render_result = renderer(
        torch.from_numpy(pushed_vertices).unsqueeze(0).cuda(),
        faces=tshirt_fs,
    )

    rest, name = os.path.split(log_path)
    (imnormal,), _, _ = render_result

    tvutils.save_image(
        tensor=(imnormal * 0.5 + 0.5)[0].permute((2, 0, 1)),
        fp=os.path.join(rest, 'pushed_' + name + '.png'),
    )

    save_kaolin_mesh(
        os.path.join(rest, 'pushed_' + name + '.obj'),
        pushed_vertices,
        tshirt_fs,
    )

    save_kaolin_mesh(
        os.path.join(rest, 'body_' + name + '.obj'),
        real_body.vertices.cpu().detach()[0],
        smpl_model.faces.astype(np.int32),
    )
    return [input_x, free_vertex]


def run_sequence_loading_data(
        info_dicts: typing.Iterable[typing.Dict[str, str]]
):
    PreviousFrame = namedtuple(
        'PreviousFrame',
        [
            'smpl_params',
            'vae_param',
            'offset',
        ]
    )

    prev_frame = PreviousFrame(None, None, None)

    for info_dict in info_dicts:
        smpl_params_path = info_dict['smpl_params']

        smpl_params = read_converted_expose_path(smpl_params_path)
        info_dict['smpl_params'] = smpl_params

        silhouette = read_silhouette_new(info_dict['silhouette'])
        silhouette = F.interpolate(silhouette[None, None], size=(512, 512))[0, 0]
        info_dict['silhouette'] = silhouette

        expose_path = info_dict['expose']
        expose_np_loaded = np.load(expose_path, allow_pickle=True)
        expose = {x: expose_np_loaded[x] for x in expose_np_loaded}
        info_dict['expose'] = expose

        pifu_result_normals = read_pifu_normals(info_dict['pifu_result_normals'])
        info_dict['pifu_result_normals'] = pifu_result_normals

        if prev_frame.smpl_params is not None:
            for k in smpl_params:
                if smpl_params[k] is not None:
                    smpl_params[k] = smpl_params[k] * 0.5 + prev_frame.smpl_params[k] * 0.5  # TODO Interpolation
                    smpl_params[k] = smpl_params[k].detach()
        optimized_parameters = process_one(
            **info_dict,
            vae_prev_param=prev_frame.vae_param,
            offset_prev_param=prev_frame.offset,
        )

        vae_prev_param, offset_prev_param = optimized_parameters  # type: torch.Tensor, torch.Tensor
        prev_frame = PreviousFrame(

            smpl_params,
            vae_prev_param,
            offset_prev_param,
        )


def process_sequence(
        sequence_name,
        base_directory,
        tshirt_template_path,
        smpl_model_path,
        vae_state_dict_path,
        output_path,
) -> typing.Iterable[typing.Dict[str, str]]:
    # region ExPose
    expose_folder = os.path.join(base_directory, sequence_name, sequence_name + '_expose')
    expose_subfolders = list(sorted(os.listdir(expose_folder)))
    expose_pattern = re.compile(r'.*?(\d+)\..*_\d+')
    expose_params = {}
    for curr_subfolder in tqdm.tqdm(expose_subfolders, desc="Saving ExPose poses paths"):
        match = expose_pattern.match(curr_subfolder)
        frame = int(match[1])

        npz_file_path = os.path.join(expose_folder, curr_subfolder, curr_subfolder + '_params.npz')
        expose_params[frame] = npz_file_path

    # endregion

    # region SMPL-converted poses
    smpl_params_folder = os.path.join(base_directory, sequence_name, sequence_name + '_smpl')
    smpl_pattern = re.compile(r'body_.*?(\d+).*\..*_\d+\.pkl')
    smpl_params = {}
    for file in tqdm.tqdm(os.listdir(smpl_params_folder), desc="Saving SMPL-converted poses paths"):
        match = smpl_pattern.match(file)
        if match:
            frame = int(match[1])
            smpl_path_to_load = os.path.join(smpl_params_folder, file)
            smpl_params[frame] = smpl_path_to_load
    # endregion

    # region Silhouettes
    checked_resolution = False
    parsing_resolution = None

    silhouettes_folder = os.path.join(base_directory, sequence_name, sequence_name + '_parsing')
    silhouette_pattern = re.compile(r'.*?(\d+).*\.npy')
    silhouettes = {}
    for file in tqdm.tqdm(os.listdir(silhouettes_folder), desc="Saving silhouettes paths"):
        match = silhouette_pattern.match(file)
        if match:
            frame = int(match[1])
            sil_path = os.path.join(silhouettes_folder, file)
            silhouettes[frame] = sil_path

            if not checked_resolution:
                loaded_matrix = np.load(sil_path)
                parsing_resolution, _ = loaded_matrix.shape
                checked_resolution = True
    # endregion

    # region Normals from PifuHD
    pifu_folder = os.path.join(base_directory, sequence_name, sequence_name + '_pifu')
    pifu_pattern = re.compile(r'.*?(\d+).*_512\.png')
    pifu_normals = {}
    for file in tqdm.tqdm(os.listdir(pifu_folder), desc="Saving Pifu normals paths"):
        match = pifu_pattern.match(file)
        if match:
            frame = int(match[1])
            pifu_normals[frame] = os.path.join(pifu_folder, file)
        else:
            raise ValueError(f"This shouldn't happen. Bad file? {file}")

    # endregion

    script_meaning = 'reconstruction'
    for frame in expose_params:
        log_path = os.path.join(output_path, f'{script_meaning}-{sequence_name}/frame{frame:04d}')
        this_dict = {
            'template_obj_path': tshirt_template_path,
            'smpl_model_path': smpl_model_path,
            'smpl_params': smpl_params[frame],
            'autoencoder_state_dict_path': vae_state_dict_path,
            'script_meaning': script_meaning,
            'log_path': log_path,
            'pifu_result_normals': pifu_normals[frame],
            'silhouette': silhouettes[frame],
            'expose': expose_params[frame],
            'parsing_resolution': parsing_resolution,
        }

        yield this_dict


def main():
    descr = 'Reconstruct meshes from a sequence of frames. ' \
            'A sequence with name <NAME> needs the following folders: ' \
            '<NAME_expose> ' \
            '<NAME_parsing> ' \
            '<NAME_pifu> ' \
            '<NAME_smpl> '

    parser = argparse.ArgumentParser(
        description=descr,
    )
    parser.add_argument(
        '--dir',
        type=str,
        help='Directory where the sequence folders are',
        required=True,
    )
    parser.add_argument(
        '--tshirt_template',
        type=str,
        default='./data/tshirt_4424verts.obj',
        help='Path of the template',
    )
    parser.add_argument(
        '--smpl_path',
        type=str,
        default='./data/smpl/smpl_neutral.pkl',
        help='Path to SMPL model',
    )
    parser.add_argument(
        '--vae_state_dict',
        type=str,
        default='./data/offset_vae_test.pth',
        help='Path to the weights of the VAE',
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='./output',
        help='Where to output results and logs'
    )
    res = parser.parse_args()

    base_directory = res.dir
    tshirt_template_path = res.tshirt_template
    smpl_path = res.smpl_path
    vae_state_dict_path = res.vae_state_dict
    output_path = res.output_path

    tshirt_exists = os.path.exists(tshirt_template_path)
    print(f'Tshirt template: {tshirt_template_path}. Exists? {tshirt_exists}')

    smpl_exists = os.path.exists(smpl_path)
    if not smpl_exists:
        raise RuntimeError(f'SMPL model not found in {smpl_path}')
    else:
        with open(smpl_path, 'rb') as file:
            read_hash = hashlib.md5(file.read()).hexdigest()
        if not read_hash == 'b3d3e3e236add66eb09bfc48a5ae87c4':
            # TODO Should this be an error or a warning?
            raise RuntimeError('The md5 of the loaded SMPL is not the same as the one we used!')

    vae_exists = os.path.exists(vae_state_dict_path)
    if not vae_exists:
        raise RuntimeError(f'VAE weights not found in {vae_state_dict_path}')

    os.makedirs(output_path, exist_ok=True)

    sequence_folders = os.listdir(base_directory)
    for seq_name in sequence_folders:
        for sub_type in ['smpl', 'expose', 'pifu', 'parsing']:
            full_subdir_path = os.path.join(base_directory, seq_name, f'{seq_name}_{sub_type}')
            exists = os.path.exists(full_subdir_path)
            if not exists:
                error_msg = (
                    f' Folder "{sub_type}" does not exist for sequence "{seq_name}". '
                    f'Full path tested: "{full_subdir_path}".'
                )
                raise RuntimeError(error_msg)
    else:
        print('All sequences are correct')

    for sequence_folder in sequence_folders:
        processed_sequence_information = process_sequence(
            sequence_name=sequence_folder,
            base_directory=base_directory,
            tshirt_template_path=tshirt_template_path,
            smpl_model_path=smpl_path,
            vae_state_dict_path=vae_state_dict_path,
            output_path=output_path,
        )

        run_sequence_loading_data(processed_sequence_information)


if __name__ == '__main__':
    main()
