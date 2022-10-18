import typing

import kaolin.io.obj as kaobj
import smplx
import smplx.lbs as smplx_lbs
import torch
import torch.nn as nn
from smplx.lbs import batch_rodrigues


def custom_lbs(vertices, joint_transforms, skinning_weights, inverse=False):
    T = torch.einsum('nj, bjik->nik', skinning_weights, joint_transforms)
    if inverse:
        T = T.inverse()
    homogen_coord = torch.ones([*vertices.shape[0:2], 1], device=vertices.device)
    vertices_homogeneous = torch.cat([vertices, homogen_coord], dim=2)
    vertices_posed_homogeneous = torch.einsum('nij,bnj->bni', T, vertices_homogeneous)
    vertices_posed = vertices_posed_homogeneous[:, :, :3]

    return vertices_posed


def pairwise_distance(pointcloud_a, pointcloud_b):
    rA = torch.sum(torch.square(pointcloud_a), dim=1)
    rB = torch.sum(torch.square(pointcloud_b), dim=1)
    distances = - 2 * torch.matmul(pointcloud_a, pointcloud_b.T) + rA[:, None] + rB[None, :]
    return distances


def find_nearest_neighbour(pointcloud_a, pointcloud_b):
    nearest_neighbour = torch.argmin(pairwise_distance(pointcloud_a, pointcloud_b), dim=1)
    return nearest_neighbour


def compute_lbs_weights(
        pointcloud_a, pointcloud_b, lbs_weights
):
    nearest_neighbours = find_nearest_neighbour(pointcloud_a, pointcloud_b)
    skinning_weights = lbs_weights[nearest_neighbours]
    return skinning_weights


class Poser:
    def __init__(
            self,
            smplx_model: typing.Union[smplx.SMPLXLayer, smplx.SMPLLayer],
            device=None,
    ):
        if device is None:
            device = torch.device('cpu')
        self.device = device

        self.smplx_model = smplx_model

    def build_full_pose(
            self,
            body_pose: torch.Tensor = None,
            global_orient: torch.Tensor = None,
            **kwargs,
    ) -> torch.Tensor:

        list_tensors = [
            global_orient if global_orient is not None else torch.eye(3, device=self.device).view(1, 1, 3, 3),
            body_pose if body_pose is not None else torch.eye(3, device=self.device).expand(
                1, self.smplx_model.NUM_BODY_JOINTS, 3, 3
            ),
        ]

        if isinstance(self.smplx_model, (smplx.SMPLXLayer, smplx.SMPLX)):
            list_tensors.extend([
                torch.eye(3, device=self.device).view(1, 1, 3, 3),  # jaw
                torch.eye(3, device=self.device).view(1, 1, 3, 3),  # leye
                torch.eye(3, device=self.device).view(1, 1, 3, 3),  # reye
                torch.eye(3, device=self.device).expand(1, self.smplx_model.NUM_HAND_JOINTS, 3, 3),  # left_hand
                torch.eye(3, device=self.device).expand(1, self.smplx_model.NUM_HAND_JOINTS, 3, 3),  # right_hand
            ])
        t = torch.cat(
            list_tensors,
            dim=1,
        )
        return t

    def pose(
            self,
            vertices: torch.Tensor,
            betas: torch.Tensor,
            smplx_kwargs: typing.Dict[str, torch.Tensor],
            lbs_weights: torch.Tensor,
            unpose: bool = False,
    ) -> torch.Tensor:
        target_betas = betas
        target_thetas = self.build_full_pose(**smplx_kwargs)

        v_shaped = self.smplx_model.v_template + smplx_lbs.blend_shapes(target_betas, self.smplx_model.shapedirs)
        J = smplx_lbs.vertices2joints(self.smplx_model.J_regressor, v_shaped)  # if not unpose else body_smplx.vertices)

        _, joint_transforms = smplx_lbs.batch_rigid_transform(
            target_thetas, J, self.smplx_model.parents, dtype=torch.float
        )

        lbs_result = custom_lbs(
            vertices,
            joint_transforms=joint_transforms,
            skinning_weights=lbs_weights,
            inverse=unpose,
        )

        return lbs_result


class TshirtPoser(nn.Module):
    def __init__(
            self,
            smpl_model: typing.Union[smplx.SMPLXLayer, smplx.SMPLLayer],
            lbs_weights: torch.Tensor,
            betas: torch.Tensor = None,
            body_pose: torch.Tensor = None,
            global_orient: torch.Tensor = None,
            transl: torch.Tensor = None,
            reverse: bool = False,
            device=None,
    ):
        super().__init__()
        if device is None:
            device = torch.device('cpu')
        self.device = device

        self.smplx_model = smpl_model.to(device)
        self.lbs_weights = lbs_weights.clone().to(device)

        self.poser = Poser(smpl_model, device)

        if body_pose is None:
            self.body_pose = torch.eye(3, device=device).expand(1, smpl_model.NUM_BODY_JOINTS, 3, 3).clone()
        else:
            if body_pose.dim() < 4:
                body_pose = batch_rodrigues(body_pose[0])[None]
            self.body_pose = body_pose.clone().to(device)

        if global_orient is None:
            self.global_orient = torch.eye(3, device=device).view(1, 1, 3, 3)
        else:
            self.global_orient = global_orient.clone().to(device)

        if betas is None:
            self.betas = torch.zeros(1, 10, device=device)
        else:
            self.betas = betas.clone().to(device)

        if transl is None:
            self.transl = torch.zeros(1, 3, device=device)
        else:
            self.transl = transl.clone().to(device)

        self.reverse = reverse

    def forward(
            self,
            vertices,
            body_pose: torch.Tensor = None,
            betas: torch.Tensor = None,
            global_orient: torch.Tensor = None,
            transl: torch.Tensor = None,
            reverse: bool = None,
    ):
        smplx_kwargs = {
            'body_pose': body_pose if body_pose is not None else self.body_pose,
            'global_orient': global_orient if global_orient is not None else self.global_orient,
        }
        betas = betas if betas is not None else self.betas

        _reverse = reverse if reverse is not None else self.reverse
        posed = self.poser.pose(vertices, betas, smplx_kwargs, self.lbs_weights, _reverse)

        t = transl if transl is not None else self.transl

        posed = posed + t

        return posed


class TshirtOffsetter(nn.Module):
    def __init__(self, vertices: torch.FloatTensor):
        super().__init__()
        self.vertices = vertices

    def forward(self, offset):
        n, flat = offset.shape
        return self.vertices + offset.view(n, -1, 3)


def load_poser():
    smpl_model = smplx.create('data/smpl/smpl_neutral.pkl')
    smpl_layer = smplx.build_layer('data/smpl/smpl_neutral.pkl')
    template_verts = kaobj.import_mesh('data/mean_shirt.obj').vertices

    poser = Poser(smplx_model=smpl_model)

    lbs_weights = compute_lbs_weights(
        template_verts,
        smpl_model().vertices[0],
        smpl_model.lbs_weights,
    )

    return poser, lbs_weights, smpl_layer
