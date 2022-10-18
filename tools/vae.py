import typing

import torch
from torch import nn


def loss_weighter(
        values_dict: typing.Dict[str, typing.Tuple[torch.Tensor, float]]
) -> typing.Tuple[torch.Tensor, typing.Dict[str, torch.Tensor]]:
    final_value: torch.Tensor = 0.0  # Typing seems weird, but it works after the first addition of weighted values
    weighted_dict = {}

    for name in values_dict:
        value, weight = values_dict[name]
        if weight is None:
            # Values can be skipped by setting the weight to None
            continue

        weighted_value = value * weight

        final_value += weighted_value
        weighted_dict['w_' + name] = (weighted_value, 1.0)

    return final_value, weighted_dict


class MeshOffsetVAE(nn.Module):
    def __init__(
            self,
            template_vertices=None,
            template_faces=None,
            device=None,
    ):
        super().__init__()

        # TODO MeshOffsetVAE device
        device = torch.device('cuda')

        sizes = [13272, 2000, 2000, 2000]
        latent = 25

        # Seq before torch.cross problem: 13272, 4000, 2000, 1000, 500, 300, 150
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers.extend([
                nn.Linear(in_size, out_size),
                nn.LeakyReLU(),
            ])
        layers.pop()

        reversed_layers = []
        for in_size, out_size in reversed(list(zip(sizes[1:], sizes))):
            reversed_layers.extend([
                nn.Linear(in_size, out_size),
                nn.LeakyReLU(),
            ])
        reversed_layers.pop()

        self.encoder = nn.Sequential(
            *layers
        )
        self.fc_mu = nn.Linear(sizes[-1], latent)
        self.fc_var = nn.Linear(sizes[-1], latent)

        self.decoder_input = nn.Linear(latent, sizes[-1])

        self.decoder = nn.Sequential(
            *reversed_layers,
        )

        self.recons_loss_function = nn.L1Loss()
        self.kldiv_loss_function = nn.KLDivLoss(reduction='batchmean')

        if template_vertices is not None:
            self.template_vertices: torch.Tensor = template_vertices.to(device)
        if template_faces is not None:
            self.template_faces: torch.Tensor = template_faces.to(device)

    def encode(self, input_meshes: torch.Tensor):
        r = self.encoder(input_meshes)
        mu = self.fc_mu(r)
        var = self.fc_var(r)

        return mu, var

    def decode(self, encoded: torch.Tensor):
        r = self.decoder_input(encoded)
        r = self.decoder(r)
        return r

    def reparametrize(self, mu, var):
        std = torch.exp(0.5 * var)
        rnd = torch.randn_like(std)
        return rnd * std + mu

    def forward(self, input_offset: torch.Tensor):
        # Pass the input through the encoder and decoder
        mu, var = self.encode(input_offset)
        z = self.reparametrize(mu, var)
        decoded = self.decode(z)

        return decoded

    def loss(self, input_offset: torch.Tensor):
        mu, var = self.encode(input_offset)
        z = self.reparametrize(mu, var)
        decoded = self.decode(z)

        # Compute ground_truth normals
        input_offset_view = input_offset.view(-1, *self.template_vertices.shape)
        offsetted_template = self.template_vertices + input_offset_view
        offsetted_fvs = offsetted_template[:, self.template_faces]
        edges01 = offsetted_fvs[:, :, 1] - offsetted_fvs[:, :, 0]
        edges02 = offsetted_fvs[:, :, 2] - offsetted_fvs[:, :, 0]
        offsetted_normals = torch.cross(edges01, edges02, dim=2)

        # Compute normals from the decoded offsets
        decoded_template = self.template_vertices + decoded.view(-1, *self.template_vertices.shape)
        decoded_fvs = decoded_template[:, self.template_faces]
        edges01 = decoded_fvs[:, :, 1] - decoded_fvs[:, :, 0]
        edges02 = decoded_fvs[:, :, 2] - decoded_fvs[:, :, 0]
        decoded_normals = torch.cross(edges01, edges02, dim=2)

        # Compute losses
        recons_loss = self.recons_loss_function(input_offset, decoded)
        normals_loss = self.recons_loss_function(offsetted_normals, decoded_normals)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + var - mu.pow(2) - var.exp(), dim=1), dim=0)

        values_dict = {
            'recons_loss': (recons_loss, 1.0),
            'normals_loss': (normals_loss, 0.0),
            'kld_loss': (kld_loss, 5e-4),
        }
        final_value, weighted_dict = loss_weighter(values_dict)
        values_dict['final_value'] = (final_value, 1.0)

        return final_value, {**values_dict, **weighted_dict}

    def save_progress(self, path, i):
        sd = self.state_dict()
        torch.save({'model': sd}, 'offset_vae_smooth.torch')

    def sample(self, n, device):
        z = torch.randn(n, 15)
        samples = self.decode(z)

        return samples
