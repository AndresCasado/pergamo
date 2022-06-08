import os
import pickle
import re
import typing
import warnings
from collections import namedtuple

import kaolin.io.obj as kaobj
import matplotlib.pyplot as plt
import numpy as np
import smplx as pysmplx
import torch
import torch.nn.functional as F
import tqdm
import cv2

import urjc_lib.kaohelpers as kaohelpers
import urjc_lib.math.igor_vertices as igoverts
import urjc_lib.optimization.loggers as myloggers
import urjc_lib.optimization.vae_pose_kaolin as vpk
import urjc_lib.smpl_fit_kaolin
from urjc_lib.math.custom_kaolin_laplacian import laplacian_cot_curvature_or_something
from urjc_lib.math.image_gradient import image_gradient_fun
from urjc_lib.math.kaolin_edges import lengths
from urjc_lib.models.meshoffsetautoencoder import MeshOffsetVAE

parsing_categories = [
    'Background',
    'Hat',
    'Hair',
    'Glove',
    'Sunglasses',
    'Upper-clothes',
    'Dress',
    'Coat',
    'Socks',
    'Pants',
    'Jumpsuits',
    'Scarf',
    'Skirt',
    'Face',
    'Left-arm',
    'Right-arm',
    'Left-leg',
    'Right-leg',
    'Left-shoe',
    'Right-shoe',
]


def warn_and_wait(message):
    warnings.warn(message)
    input('Continue?\n')


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
        raise FileNotFoundError('The file with the conversion does not exist.')
        smpl_params = urjc_lib.smpl_fit_kaolin.fit_smpl_to_expose(
            expose_npz_path,
            smplx_model_path,
            smpl_model_path,
            debug=True,
        )
        smpl_params = {
            k: smpl_params[k].detach()
            for k in smpl_params
        }
        torch.save(smpl_params, filepath)

    return smpl_params


def read_silhouette(filepath):
    loaded = np.load(filepath)
    up_cat = parsing_categories.index('Upper-clothes')
    silhouette_prob_argmax = loaded.argmax(axis=2)
    silhouette = np.where(silhouette_prob_argmax == 5, 1.0, 0.0)
    silhouette = torch.from_numpy(silhouette)

    return silhouette


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
        smplx_model_path: str,
        smpl_model_path: str,
        smpl_params,
        autoencoder_state_dict_path: str,
        script_meaning: str,
        log_path: str,
        pifu_result_normals,
        silhouette,
        expose,
        vae_prev_param: torch.Tensor,
        offset_prev_param: torch.Tensor,
        show_normals: bool = False,
):
    DEVICE = torch.device('cuda')

    VAE_REG_WEIGHT = 0.01 #0.04
    TMP_REG_WEIGHT = 1 # 15.0

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
    smpl_model = pysmplx.build_layer(smpl_model_path)

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

    lbs = igoverts.compute_skinning_weights(
        tshirt_obj.vertices,
        smpl_model().vertices[0],
        smpl_model.lbs_weights,
    )
    tshirt_poser = vpk.TshirtPoser(smpl_model=smpl_model, lbs_weights=lbs, **smpl_params)

    renderer = kaohelpers.KaolinExPoseRenderer(
        512, 512,
        # 600, 600,
        expose_params=expose,
        smooth_normals=True,
    )

    autoencoder = MeshOffsetVAE(device=DEVICE)
    autoencoder_state_dict = torch.load(autoencoder_state_dict_path)['model']
    autoencoder = autoencoder.to(DEVICE)
    autoencoder.load_state_dict(autoencoder_state_dict)

    offsetter = vpk.TshirtOffsetter(vertices=tshirt_vs)

    im_logger = myloggers.ImageLogger(log_path, every=10)
    loss_logger = myloggers.LossLogger(script_meaning)
    mesh_logger = myloggers.MeshLogger(log_path, every=30)

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
            'normal': vpk.change_range(imnormal, -1., 1.),
            'diff': vpk.change_range(normal_diff, -2., 2.),
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
            'normal': vpk.change_range(imnormal, -1., 1.),
            'diff': vpk.change_range(normal_diff, -2., 2.)
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

        t_posed_lengths = lengths(t_pose_tshirt[0], tshirt_fs)
        freemoved_lengths = lengths(free_moved_tshirt[0], tshirt_fs)

        render_result = renderer(
            posed_tshirt,
            faces=tshirt_fs,
        )

        (imnormal,), _, _ = render_result
        imnormal = F.interpolate(imnormal[None], size=(512, 512, 3))[0]

        horizontal_gradient_diff = image_gradient_fun(imnormal) - image_gradient_fun(gt_imnormal[None])
        horizontal_gradient_diff *= last_improb_of_next_loss.unsqueeze(-1)
        vertical_gradient_diff = image_gradient_fun(imnormal, True) - image_gradient_fun(gt_imnormal[None], True)
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
            'normal': vpk.change_range(imnormal, -1., 1.),
            'normal-gt': vpk.change_range(gt_imnormal[None], -1., 1.),
            'diff': vpk.change_range(horizontal_gradient_diff, -2., 2.)
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
        print('Vae norm:' + str(vae_prev_param.square().mean()))

    it_vertex = 200
    free_vertex = torch.zeros_like(tshirt_vs, requires_grad=True, device=DEVICE)
    if offset_prev_param is not None:
        it_vertex = 31
        print('Vert norm:' + str(offset_prev_param.square().mean()))
        print('Vert max:' + str(torch.max(offset_prev_param)))
        offset_prev_param = torch.clamp(offset_prev_param, -0.019, 0.019)
        offset_prev_param[..., [0, 1]] = torch.clamp(offset_prev_param[..., [0, 1]], -0.0039, 0.0039)
        free_vertex = torch.clone(offset_prev_param).requires_grad_(True).to(DEVICE)

    parameters = [input_x, free_vertex]

    adams = [
        torch.optim.Adam(parameters, lr=1e-2),
        torch.optim.Adam(parameters, lr=1e-3),
        torch.optim.Adam([free_vertex], lr=1e-4)
    ]

    optimization_steps = [
        vpk.OptimizationStep(loss_only_silhouette, it_vae),
        vpk.OptimizationStep(loss_normal_and_silhouette_vpk, it_vae),
        vpk.OptimizationStep(loss_vpk_freevertex_lengths, it_vertex)
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
    from urjc_lib.math.mesh_postprocess import push_vertices

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
    imnormal = imnormal.squeeze().cpu().detach().numpy()
    imnormal = (imnormal + 1) / 2.0 * 65535
    imnormal = imnormal.astype(np.uint16)
    imnormal = imnormal[..., [2, 1, 0]]
    cv2.imwrite(os.path.join(rest, 'pushed_' + name + '.png'), imnormal)

    kaohelpers.save_kaolin_mesh(
        os.path.join(rest, 'pushed_' + name + '.obj'),
        pushed_vertices,
        tshirt_fs,
    )

    kaohelpers.save_kaolin_mesh(
        os.path.join(rest, 'body_' + name + '.obj'),
        real_body.vertices.cpu().detach()[0],
        smpl_model.faces.astype(np.int32),
    )
    return [input_x, free_vertex]


def run():
    template_obj_path = '/home/mslab/Models/tshirt_4424verts.obj'
    out_folder = 'out_dan'
    filename = 'dan'
    smplx_model_path = 'models/smplx/SMPLX_NEUTRAL.npz'
    smpl_model_path = 'models/smpl/smpl_neutral.pkl'
    converted_expose_path = 'out_dan/smplx/dan.png_000.pkl'
    autoencoder_state_dict_path = 'offset_vae_test.torch'
    script_meaning = 'rd_edges_study'
    log_path = f'data/kaolin/{script_meaning}'

    process_one(
        template_obj_path,
        out_folder,
        filename,
        smplx_model_path,
        smpl_model_path,
        converted_expose_path,
        autoencoder_state_dict_path,
        script_meaning,
        log_path,
    )


buff_base = '/home/marc/DatosBuff'


def check_all_buff(seq):
    b = [
        seq + x
        for x in [
            '',
            '_smpl',
            '_pifu',
            '_expose',
            '_parsing'
        ]
    ]
    valid = True
    for s in b:
        exists = os.path.exists(os.path.join(buff_base, s))
        valid = valid and exists
    if valid:
        print(f'{seq} vale')
    else:
        print(f'{seq} no vale')

    return valid


def all_buff_marc_files():
    buff_seqs = [
        'mslab002-red-hips',
        'mslab002-red-legs',
        'mslab002-red-shoulders-backwards',
        'mslab002-red-twist',
        'mslab002-red-walk',
        'mslab002-white-hips',
        'mslab002-white-legs',
        'mslab002-white-shoulders-backwards',
        'mslab002-white-twist',
        'mslab002-white-walk',
    ]
    buff_seqs = [
        'clip-02600',
        'clip-06100',
        'clip-06300',
    ]
    buff_seqs = [
        'dan-101',
        'dan-102',
        'dan-103',
        'dan-104',
        'dan-105',
        'dan-106',
        'dan-107',
        'dan-108',
        'dan-109',
        'dan-110',
        'dan-111',
        'dan-112',
        'dan-201',
        'dan-202',
        'dan-203',
        'dan-204',
        'dan-205',
        'dan-206',
        'dan-207',
        'dan-208',
        'dan-209',
        'dan-210',
        'dan-211',
        'dan-212',
        'dan-213',
        'dan-214',
        'dan-301',
        'dan-302',
        'dan-303',
        'dan-304',
        'dan-305',
        'dan-306',
        'dan-307',
        'dan-308',
        'dan-309',
    ]
    buff_seqs = [
        'dan-001',
        'dan-002',
        'dan-003',
        'dan-004',
        'dan-005',
        'dan-006',
        'dan-007',
        'dan-008',
        'dan-009',
        'dan-010',
        'dan-011',
        'dan-012',
        'dan-013',
        'dan-014',
    ]
    buff_seqs = ["shortlong_hips_96"]
    script_meaning = 'SEQ_real_data_mslab002'
    suff = input(f'Add suffix? {script_meaning} ')
    if suff:
        suff = suff.split(' ')
        script_meaning = '_'.join([script_meaning, *suff])

    for seq in buff_seqs:
        valid = check_all_buff(seq)

        if not valid:
            warnings.warn(f'Me salto {seq}')
            continue

        template_obj_path = 'tshirt_4424verts.obj'
        smplx_model_path = 'models/smplx/SMPLX_NEUTRAL.npz'
        smpl_model_path = 'models/smpl/smpl_neutral.pkl'
        autoencoder_state_dict_path = 'offset_vae_test.torch'

        p = os.path.join(buff_base, seq)
        expose_ffff = os.path.join(buff_base, seq + '_expose')

        expose_files = list(sorted(os.listdir(expose_ffff)))
        expose_pattern = re.compile(r'(\d+)\.png_\d+')
        expose = {}
        print('Reading poses')
        for r in tqdm.tqdm(expose_files):
            match = expose_pattern.match(r)
            frame = int(match[1])
            for root_dir, directories, files in os.walk(os.path.join(expose_ffff, r)):
                npz_file = [file for file in files if file.endswith('.npz')][0]
                this_expose = np.load(os.path.join(expose_ffff, r, npz_file), allow_pickle=True)
                w = {x: this_expose[x] for x in this_expose}
                expose[frame] = w

        smpl_params = {}
        smpl_converted_directory = f'{buff_base}/{seq}_smpl'
        smpl_pattern = re.compile(r'(\d+)\.png_\d+\.pkl')
        print('Loading converted poses')
        for file in tqdm.tqdm(os.listdir(smpl_converted_directory)):
            match = smpl_pattern.match(file)
            if match:
                frame = int(match[1])
                smpl_loaded = read_converted_expose_path(os.path.join(smpl_converted_directory, file))
                smpl_params[frame] = smpl_loaded

        silhouettes = {}
        npy_pattern = re.compile(r'(\d+)_tshirt\.npy')
        print('Reading silhouettes')
        where_silhouettes = f'{buff_base}/{seq}_parsing'
        for file in tqdm.tqdm(os.listdir(where_silhouettes)):
            match = npy_pattern.match(file)
            if match:
                frame = int(match[1])
                sil_path = os.path.join(where_silhouettes, file)
                read_sil = read_silhouette_new(sil_path)
                read_sil = F.interpolate(read_sil[None, None], size=(512, 512))[0, 0]
                silhouettes[frame] = read_sil

        pifu_result_normals = {}
        pifu_pattern = re.compile(r'result_(\d+)_512\.png')
        pifu_directory = f'{buff_base}/{seq}_pifu'
        print('Reading Pifu normals')
        for file in tqdm.tqdm(os.listdir(pifu_directory)):
            match = pifu_pattern.match(file)
            if match:
                frame = int(match[1])
                pifu_result_normals[frame] = read_pifu_normals(os.path.join(pifu_directory, file))
            else:
                raise ValueError(f"This shouldn't happen. Bad file? {file}")

        os.makedirs('/tmp/retesting_pipeline', exist_ok=True)
        for i in expose:
            # log_path = f'/mnt/HDDTera/auto_results/{script_meaning}-{seq}_{subject}/frame{i:04d}'
            log_path = f'/tmp/mslab_seqs/{script_meaning}-{seq}/frame{i:04d}'
            os.makedirs(log_path, exist_ok=True)

            this_dict = {
                'template_obj_path': template_obj_path,
                'smplx_model_path': smplx_model_path,
                'smpl_model_path': smpl_model_path,
                'smpl_params': smpl_params[i],  # Converted SMPL-X to SMPL
                'autoencoder_state_dict_path': autoencoder_state_dict_path,
                'script_meaning': script_meaning,
                'log_path': log_path,
                'pifu_result_normals': pifu_result_normals[i],
                'silhouette': silhouettes[i],
                'expose': expose[i],  # SMPL-X from ExPose

            }
            yield this_dict


PreviousFrame = namedtuple(
    'PreviousFrame',
    [
        'smpl_params',
        'vae_param',
        'offset',
    ]
)


def run_sequence(
        info: typing.List[typing.Dict[str, typing.Any]]
):
    prev_frame = PreviousFrame(None, None, None)
    for i in info():
        smpl_params = i['smpl_params']
        if prev_frame.smpl_params is not None:
            for k in smpl_params:
                if smpl_params[k] is not None:
                    smpl_params[k] = smpl_params[k] * 0.5 + prev_frame.smpl_params[k] * 0.5
                    smpl_params[k] = smpl_params[k].detach()
        optimized_parameters = process_one(
            **i,
            vae_prev_param=prev_frame.vae_param,
            offset_prev_param=prev_frame.offset,
        )

        vae_prev_param, offset_prev_param = optimized_parameters  # type: torch.Tensor, torch.Tensor
        prev_frame = PreviousFrame(
            smpl_params,
            vae_prev_param,
            offset_prev_param,
        )


def main():
    info_dicts = all_buff_marc_files
    run_sequence(info_dicts)


if __name__ == '__main__':
    main()
