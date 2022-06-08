import kaolin.ops.mesh as kops_mesh
import torch


def compute_normals_per_vertex(
        vs: torch.Tensor,
        fs: torch.Tensor,
        face_normals: torch.Tensor = None):
    if face_normals is None:
        face_normals = kops_mesh.face_normals(kops_mesh.index_vertices_by_faces(vs, fs))

    device = vs.device

    # Get number of vertices and faces
    _, nvs, _ = vs.shape
    nfs, _ = fs.shape

    # For each vertex we want to know what faces it is in
    # We will build a tensor for this. Each row will be a vertex, and each column will be a face
    # The value will be 0. if the vertex is not in the face, and 1. if it is

    # First, we create the tensor with zeros
    vertex_in_face = torch.zeros(nvs, nfs, device=device)

    # Then we will need the indices where we will put ones
    # The following tensor has a sequence repeated 3 times
    # [[0,0,0], [1,1,1], [2,2,2], ...]
    w = torch.tensor(range(nfs), device=device)[None].repeat(3, 1).t()

    # We index the rows with fs, which contains the vertices, and pair it with w
    # Simple example:
    #
    # fs = [[0, 1, 2], [2, 3, 4]]
    # w = [[0, 0, 0], [1, 1, 1]]  <- in fs there are 2 faces, so 0,1 repeated
    #
    # That would make the following pairs:
    # 0,0   1,0   2,0   2,1   3,1   4,1
    # and then the following sentence will set ones where we wanted:
    # [[1, 0],  <- vertex 0 is in face 0, but not in face 1
    #  [1, 0],
    #  [1, 1],  <- vertex 2 is in both faces
    #  [0, 1],
    #  [0, 1]]  <- vertex 4 is in face 1, but not in face 0
    vertex_in_face[fs, w] = 1.0

    # Now, for each vertex, we want the sum of the normals from each of the faces it is part of
    # vertex_in_face x face_normals = (nvs x nfs) x (nfs x 3) = (nvs x 3)
    sum_normals = torch.einsum('ij,bjk->bik', vertex_in_face, face_normals)

    # The sum takes non-unit normals, which makes it able to take into account their importance. But at this point we
    # don't want the importance anymore, so we normalize them
    normalized_sum_normals = sum_normals / sum_normals.norm(dim=2, keepdim=True)

    return normalized_sum_normals
