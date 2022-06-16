import os
import struct
import typing

import numpy as np
import torch


def save_pc2(filename, data):
    if isinstance(data, list):
        data = np.array(data)

    data = data.astype(np.float32)
    num_frames = data.shape[0]
    num_vertices = data.shape[1]

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'wb') as f:
        # Write header
        header_format = '<12siiffi'
        header = struct.pack(header_format, b'POINTCACHE2\0', 1, num_vertices, 0, 1.0, num_frames)

        f.write(header)

        # Write animation
        for f_id in range(num_frames):
            for v_id in range(num_vertices):
                vert = struct.pack('<fff', float(data[f_id][v_id][0]),
                                   float(data[f_id][v_id][1]),
                                   float(data[f_id][v_id][2]))
                f.write(vert)

    print('Saved:', filename)


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
