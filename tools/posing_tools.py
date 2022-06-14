import os
import pickle
import typing

import trimesh
import smplx
import smplx.lbs as smplx_lbs
import torch
from matplotlib import pyplot as plt


def plot_3d_points(points: torch.Tensor, *additional, labels: typing.List[str] = None, same_size=True):
    fig = plt.figure()  # type: plt.Figure
    ax = fig.add_subplot(111, projection='3d')  # type: plt.Axes

    maxs = []
    mins = []
    for array in [points, *additional]:
        if labels:
            next_label, *labels = labels
        else:
            next_label = None

        xs = array[:, 0]
        ys = array[:, 1]
        zs = array[:, 2]
        ax.scatter(xs=xs, ys=ys, zs=zs, label=next_label)

        # Add max
        m, _ = array.max(dim=0)
        maxs.append(m)
        # Add min
        m, _ = array.min(dim=0)
        mins.append(m)

    maxs_tensor = torch.stack(maxs, dim=0)
    mins_tensor = torch.stack(mins, dim=0)

    if same_size:
        ax.autoscale(enable=False, axis='both')  # you will need this line to change the Z-axis
        max, _ = maxs_tensor.max(dim=0)
        min, _ = mins_tensor.min(dim=0)
        center = (max + min) * 0.5
        center_diff = max - center
        center_diff = center_diff.max()

        ax.set_xlim(center[0] - center_diff, center[0] + center_diff)
        ax.set_ylim(center[1] - center_diff, center[1] + center_diff)
        ax.set_zlim(center[2] - center_diff, center[2] + center_diff)
    ax.legend()
    plt.show()


def scatter_labeled(labeled_points: typing.List[typing.Tuple[str, torch.Tensor]], same_size=True, ):
    labels, points = zip(*labeled_points)
    plot_3d_points(*points, labels=labels, same_size=same_size)


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


def load_poser():
    smpl_model = smplx.create('data/smpl/smpl_neutral.pkl')
    smpl_layer = smplx.build_layer('data/smpl/smpl_neutral.pkl')
    template_verts = torch.from_numpy(trimesh.load_mesh('data/mean_shirt.obj', process=False).vertices).float()

    poser = Poser(smplx_model=smpl_model)

    lbs_weights = compute_lbs_weights(
        template_verts,
        smpl_model().vertices[0],
        smpl_model.lbs_weights,
    )

    return poser, lbs_weights, smpl_layer

