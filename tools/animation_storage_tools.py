import os
import struct

import numpy as np


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
