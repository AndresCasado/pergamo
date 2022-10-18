import os
import time
import typing
import warnings

import torch
from torch.utils import tensorboard as tb
from torchvision import utils as tvutils

from tools.animation_storage_tools import save_kaolin_mesh


class BaseLogger:
    def __init__(self, path, every=1):
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.i = 0
        self.every = every

    def check_every(self):
        every = self.i % self.every == 0
        self.i += 1
        return every


class ImageLogger(BaseLogger):
    def save_images(self, named_images: typing.Dict[str, torch.Tensor]):
        if self.check_every():
            for name, image in named_images.items():
                image = image[0].cpu()
                if image.dim() == 3:
                    image = image.permute(2, 0, 1)
                path = os.path.join(self.path, name)
                os.makedirs(path, exist_ok=True)
                tvutils.save_image(image, fp=os.path.join(path, f'{name}_{self.i:09d}.png'))


class MeshLogger(BaseLogger):
    def save_meshes(self, named_meshes: typing.Dict[str, typing.Tuple[torch.Tensor, torch.Tensor]]):
        if self.check_every():
            for name, (vertices, faces) in named_meshes.items():
                if vertices.dim() == 3 and vertices.shape[0] > 1:
                    warnings.warn('MeshLogger only saves the first mesh of the batch')
                if vertices.dim() > 3:
                    raise ValueError('Vertices tensor has an incompatible shape')

                vertices = vertices[0].cpu()
                faces = faces.cpu()
                path = os.path.join(self.path, name)
                os.makedirs(path, exist_ok=True)
                save_kaolin_mesh(
                    path=os.path.join(path, f'{name}_{self.i:09d}.obj'),
                    verts=vertices,
                    faces=faces,
                )


class LossLogger:
    def __init__(self, name):
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        path = f'tb_data/{name}_{timestamp}'
        self.sum_writer = tb.SummaryWriter(path)
        self.registered_name = set()  # type: set
        self.i = 0

    def log_losses(self, losses: typing.Dict[str, typing.Union[float, torch.Tensor]]):
        for name, loss in losses.items():
            if name not in self.registered_name:
                self.registered_name.add(name)
                for j in range(self.i):
                    self.sum_writer.add_scalar(name, float('nan'), j)
            if isinstance(loss, torch.Tensor):
                t = loss.item()
            elif isinstance(loss, float):
                t = loss
            else:
                t = float('nan')
            self.sum_writer.add_scalar(name, t, self.i)
        self.i += 1
        self.sum_writer.flush()
