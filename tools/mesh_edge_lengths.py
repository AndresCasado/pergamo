import torch


def compute_mesh_edge_lengths(vertices: torch.Tensor, faces: torch.Tensor):
    vertices_by_face = vertices[faces]

    xs = vertices_by_face[:, 0, :]
    ys = vertices_by_face[:, 1, :]
    zs = vertices_by_face[:, 2, :]

    edges_01 = ys - xs
    edges_12 = zs - ys
    edges_20 = xs - zs

    edges = torch.stack([edges_01, edges_12, edges_20], dim=1)

    _lengths = edges.norm(dim=2)

    return _lengths
