import torch
from torch import nn as nn


def _batch_rodrigues(
        rot_vecs: torch.Tensor,
        epsilon: float = 1e-8,
) -> torch.Tensor:
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    # TODO Copied from smplx, check if everything is correct and efficient

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
        dim=1,
    ).view(
        (batch_size, 3, 3)
    )

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


class PoseEncoder10D(nn.Module):
    def __init__(self):
        super().__init__()

        self.dense = nn.Linear(207, 500)
        self.relu = nn.ReLU()
        self.dense_1 = nn.Linear(500, 10)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 23, 3)
        x = _batch_rodrigues(x.view(-1, 3)).view(-1, 23, 3, 3)
        x = x - torch.eye(3)
        x = x.view(-1, 207)
        x = self.dense(x)
        x = self.relu(x)
        x = self.dense_1(x)
        return x
