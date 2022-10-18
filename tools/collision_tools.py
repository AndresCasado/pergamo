import igl
import numpy as np
import torch


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
