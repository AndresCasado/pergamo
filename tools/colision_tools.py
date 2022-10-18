import os
import typing

import igl
import numpy as np
import torch
from matplotlib import pyplot as plt


def scatter_labeled(labeled_points: typing.List[typing.Tuple[str, torch.Tensor]], same_size=True, ):
    labels, points = zip(*labeled_points)
    plot_3d_points(*points, labels=labels, same_size=same_size)


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


def push_vertices(
        vs: torch.Tensor,
        base_vs: torch.Tensor,
        base_fs: torch.Tensor,
        epsilon: float,
):
    vs_np = vs
    base_vs_np = base_vs
    base_fs_np = base_fs

    s, i, c, n = igl.signed_distance(
        p=vs_np,
        v=base_vs_np,
        f=base_fs_np,
        return_normals=True,
    )
    d = (s - epsilon).clip(max=0.0)
    dn = np.einsum('n,nd->nd', d, n)
    moved = vs - dn

    return moved


def save_kaolin_mesh(
        path: typing.Union[str, bytes, os.PathLike],
        verts: torch.Tensor,
        faces: torch.Tensor,
        # normals: torch.Tensor,
):
    # TODO Save mesh normals or materials
    # It will be useful to use vn.unique(return_inverse=True), but rembember
    # that return_inverse is very inefficient
    lines = []
    for v in verts:
        x, y, z = v
        lines.append(f'v {x} {y} {z}\n')
    for f in faces + 1:
        x, y, z = f
        lines.append(f'f {x} {y} {z}\n')
    with open(path, 'w') as file:
        file.writelines(lines)
