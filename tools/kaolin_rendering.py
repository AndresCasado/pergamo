import typing

import numpy as np
import torch
import torch.nn as nn
from kaolin.ops.mesh import index_vertices_by_faces, face_normals as fn
from kaolin.render.mesh import dibr_rasterization

from tools.normal_tools import compute_normals_per_vertex


def append_ones(x):
    return torch.cat(
        [x, torch.ones(*x.shape[:-1], 1, device=x.device)],
        dim=-1,
    )


def batched_matrix_mm(first, second):
    return torch.einsum('ij,bnj->bni', first, second)


class BaseKaolinRenderer(nn.Module):
    def __init__(
            self,
            height: int,
            width: int,
            smooth_normals: bool = True,
            device=None,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.smooth_normals = smooth_normals

        if device is None:
            device = torch.device('cuda')
        self.device = device
        self.extra_dibr_params = {}

    def compute_vertices_camera(self, vertices, **kwargs):
        raise NotImplementedError('Method not implemented')

    def compute_vertices_image(self, vertices, **kwargs):
        raise NotImplementedError('Method not implemented')

    def forward(
            self,
            vertices: torch.FloatTensor,
            faces: torch.LongTensor,
            smooth_normals: bool = None,
            attribs: typing.List[torch.FloatTensor] = None,
            **kwargs,
    ) -> typing.Tuple[typing.Tuple[torch.Tensor, ...], torch.FloatTensor, torch.LongTensor]:
        if smooth_normals is None:
            smooth_normals = self.smooth_normals

        vs_camera = self.compute_vertices_camera(vertices, **kwargs)
        face_vs_camera = index_vertices_by_faces(vs_camera, faces)

        vs_image = self.compute_vertices_image(vs_camera, **kwargs)
        face_vs_image = index_vertices_by_faces(vs_image, faces)

        face_normals = fn(face_vs_camera)
        face_normals_unit = face_normals / face_normals.norm(dim=2, keepdim=True)
        face_normals_unit = face_normals_unit.unsqueeze(-2).repeat(1, 1, 3, 1)
        face_camera_normals_z = face_normals_unit[:, :, 1]

        face_attributes = []

        if smooth_normals:
            vertex_normals = compute_normals_per_vertex(vs_camera, faces, face_normals)
            face_vertex_normals = index_vertices_by_faces(vertex_normals, faces)
            face_attributes.append(face_vertex_normals)
        else:
            face_attributes.append(face_normals_unit)

        if attribs is not None:
            face_attributes.extend(attribs)

        result = dibr_rasterization(
            self.height,
            self.width,
            face_vs_camera[:, :, :, 2],
            face_vs_image,
            face_attributes,
            face_camera_normals_z,
            **self.extra_dibr_params,
        )

        if smooth_normals:
            (imnormal, *rendered_attribs,), improb, imdx = result
            imnormal = imnormal / (imnormal.norm(dim=3, keepdim=True) + 1e-10)
            result = (imnormal, *rendered_attribs), improb, imdx

        return result


class KaolinExPoseRenderer(BaseKaolinRenderer):
    def __init__(
            self,
            height: int,
            width: int,
            expose_params: typing.Dict[str, np.ndarray],
            smooth_normals: bool = False,
            device=None,
    ):
        if device is None:
            device = torch.device('cuda')

        super().__init__(
            height, width,
            smooth_normals=smooth_normals,
            device=device,
        )

        self.expose = {}
        for k in ['transl', 'focal_length_in_px', 'center']:
            self.expose[k] = torch.from_numpy(expose_params[k])

        self.extra_dibr_params = {
            'sigmainv': 333333,
            'boxlen': 0.02,
            'knum': 30,
            'multiplier': 1000,
        }

    def compute_vertices_camera(
            self,
            vertices,
            **kwargs,
    ):
        x, y, z = self.expose['transl']
        V = torch.tensor([
            [1.0, 0.0, 0.0, x],
            [0.0, -1.0, 0.0, -y],
            [0.0, 0.0, -1.0, -z],
            [0.0, 0.0, 0.0, 1.0],
        ], device=self.device, dtype=torch.float32)

        vertices = append_ones(vertices)
        vs_camera = batched_matrix_mm(V, vertices)
        vs_camera = vs_camera[..., :-1]

        return vs_camera

    def compute_vertices_image(
            self,
            vertices_camera,
            **kwargs,
    ):
        centerx, centery = self.expose['center']
        focal_length = self.expose['focal_length_in_px']

        cx = 1 - 2 * centerx / (self.width - 1)
        cy = 2 * centery / (self.height - 1) - 1
        fx = 2 * focal_length / self.width
        fy = 2 * focal_length / self.height

        P = torch.tensor([
            [fx, 0., cx, 0.],
            [0., fy, cy, 0.],
            [0., 0., -1.0010005, -0.10005003],
            [0., 0., -1., 0.],
        ], device=self.device, )

        vertices_camera = append_ones(vertices_camera)
        vs_image = batched_matrix_mm(P, vertices_camera)
        vs_image = vs_image[:, :, :3] / vs_image[:, :, 3, None]
        vs_image = vs_image[..., :-1]

        return vs_image
