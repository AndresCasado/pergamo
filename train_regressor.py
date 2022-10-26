import os
import pickle as pkl
import random
import time

import kaolin.io.obj as kaobj
import numpy as np
import torch
import torch.nn as tnn
import torch.utils.data
import torchvision.datasets as tvd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from regressor import Regressor
from tools.normal_tools import compute_normals_per_vertex
from tools.prediction_tools import predict_sequence

os.makedirs('runs/', exist_ok=True)
# writer = SummaryWriter(log_dir="runs/")

if __name__ == "__main__":
    # To avoid changing the seeds if this file is ever imported
    torch.manual_seed(3023)
    torch.cuda.manual_seed(3023)
    random.seed(3023)
    np.random.seed(3023)

HYPER_PARAMETERS = {
    "input_size": 10,
    "output_size": 4424,
    "batch_size": 64,
    "epochs": [10, 40, 40, 10],
    "learning_rates": [5e-3, 1e-3, 1e-4, 1e-5]
}


class Dataset(tvd.DatasetFolder):
    def __getitem__(self, index):
        mesh_path, target = self.samples[index]
        pose_path = mesh_path.replace("meshes", "poses").replace(".obj", "_enc.pkl")

        mesh = self.loader(mesh_path)
        vertices = np.array(mesh.vertices)
        vertices = torch.from_numpy(vertices).float()

        pose = torch.load(pose_path).squeeze().reshape(HYPER_PARAMETERS["input_size"])

        return target, pose, vertices


class AverageMeter(object):
    def __init__(self):
        self.value, self.avg, self.sum, self.count = 0, 0, 0, 0
        self.reset()

    def reset(self):
        self.value, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


def compute_loss(
        normalized_offsets_pred,
        normalized_offsets,
        vertices_pred,
        vertices,
        template_fs,
        recon_criterion
):
    normals_pred = torch.nan_to_num(compute_normals_per_vertex(vertices_pred, template_fs), nan=0)
    normals_gt = torch.nan_to_num(compute_normals_per_vertex(vertices, template_fs), nan=0)

    recon_loss = recon_criterion(normalized_offsets, normalized_offsets_pred)
    normal_loss = recon_criterion(normals_gt, normals_pred)

    loss_value = recon_loss + normal_loss
    return loss_value, recon_loss, normal_loss


def validate(
        val_loader,
        mean_pose,
        sd_pose,
        mean_shirt,
        min_offset,
        max_offset,
        template_fs,
        regressor,
        optimizer,
        recon_criterion,
        epoch
):
    regressor.eval()

    # Prepare value counters and timers
    batch_time, data_time = AverageMeter(), AverageMeter()
    total_loss, vertex_loss, normal_loss = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for i, (_, pose, vertices) in enumerate(tqdm(val_loader)):
        # Use GPU if available
        pose, vertices = pose.cuda(), vertices.cuda()

        normalized_pose = (pose - mean_pose) / sd_pose
        normalized_offsets = (vertices - mean_shirt - min_offset) / (max_offset - min_offset)

        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Generator
        optimizer.zero_grad()

        normalized_offsets_pred = regressor(normalized_pose)

        vertices_pred = normalized_offsets_pred * (max_offset - min_offset) + min_offset + mean_shirt

        tloss, vloss, nloss = compute_loss(
            normalized_offsets_pred,
            normalized_offsets,
            vertices_pred,
            vertices,
            template_fs,
            recon_criterion,
        )
        optimizer.zero_grad()

        total_loss.update(tloss.item(), pose.size(0))
        vertex_loss.update(vloss.item(), pose.size(0))
        normal_loss.update(nloss.item(), pose.size(0))

        # Record time to do forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()

    # Print model accuracy
    print(
        f'Epoch: [{epoch}]\t'
        f'Time {batch_time.value:.3f} ({batch_time.avg:.3f})\t'
        f'Data {data_time.value:.3f} ({data_time.avg:.3f})\t'
        f'Loss {total_loss.value:.6f} ({total_loss.avg:.6f})\t'
    )

    print(f'Finished training epoch {epoch}')
    # writer.add_scalar("Validation/total_loss", total_loss.avg, epoch)
    # writer.add_scalar("Validation/vertex_loss", vertex_loss.avg, epoch)
    # writer.add_scalar("Validation/normal_loss", normal_loss.avg, epoch)
    return total_loss.avg


def train(
        train_loader,
        mean_pose,
        sd_pose,
        mean_shirt,
        min_offset,
        max_offset,
        template_fs,
        regressor,
        optimizer,
        recon_criterion,
        epoch
):
    print(f'Starting training epoch {epoch}')

    regressor.train()

    # Prepare value counters and timers
    batch_time, data_time = AverageMeter(), AverageMeter()
    total_loss, vertex_loss, normal_loss = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for i, (_, pose, vertices) in enumerate(tqdm(train_loader)):
        # Use GPU if available
        pose, vertices = pose.cuda(), vertices.cuda()

        normalized_pose = (pose - mean_pose) / sd_pose
        normalized_offsets = (vertices - mean_shirt - min_offset) / (max_offset - min_offset)

        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Generator
        optimizer.zero_grad()

        normalized_offsets_pred = regressor(normalized_pose)

        vertices_pred = normalized_offsets_pred * (max_offset - min_offset) + min_offset + mean_shirt

        tloss, vloss, nloss = compute_loss(
            normalized_offsets_pred,
            normalized_offsets,
            vertices_pred,
            vertices,
            template_fs,
            recon_criterion,
        )

        normalized_offsets_pred = regressor(
            normalized_pose + torch.randn(normalized_pose.shape).cuda() * torch.mean(sd_pose)
        )

        vertices_pred = normalized_offsets_pred * (max_offset - min_offset) + min_offset + mean_shirt

        n_tloss, n_vloss, n_nloss = compute_loss(
            normalized_offsets_pred,
            normalized_offsets,
            vertices_pred,
            vertices,
            template_fs,
            recon_criterion,
        )

        tloss = (tloss + n_tloss * 0.05) / 1.05
        vloss = (vloss + n_vloss * 0.05) / 1.05
        nloss = (nloss + n_nloss * 0.05) / 1.05

        tloss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss.update(tloss.item(), pose.size(0))
        vertex_loss.update(vloss.item(), pose.size(0))
        normal_loss.update(nloss.item(), pose.size(0))

        # Record time to do forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()

        # writer.add_scalar("Train/zcore", tloss.item(), epoch * len(train_loader) + i)

    # Print model accuracy
    print(
        f'Epoch: [{epoch}]\t'
        f'Time {batch_time.value:.3f} ({batch_time.avg:.3f})\t'
        f'Data {data_time.value:.3f} ({data_time.avg:.3f})\t'
        f'Loss {total_loss.value:.6f} ({total_loss.avg:.6f})\t')

    print(f'Finished training epoch {epoch}')
    # writer.add_scalar("Train/total_loss", total_loss.avg, epoch)
    # writer.add_scalar("Train/vertex_loss", vertex_loss.avg, epoch)
    # writer.add_scalar("Train/normal_loss", normal_loss.avg, epoch)
    return total_loss.avg


def weights_init(m):
    if isinstance(m, tnn.Linear):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)


def main():
    # Training
    train_imagefolder = Dataset(
        'data/train_sequence/meshes/',
        extensions=".obj",
        loader=kaobj.import_mesh,
    )
    train_loader = torch.utils.data.DataLoader(
        train_imagefolder,
        batch_size=HYPER_PARAMETERS["batch_size"],
        shuffle=True,
    )

    # Validation
    val_imagefolder = Dataset(
        'data/validation_sequence/meshes/',
        extensions=".obj",
        loader=kaobj.import_mesh,
    )
    val_loader = torch.utils.data.DataLoader(
        val_imagefolder,
        batch_size=HYPER_PARAMETERS["batch_size"],
        shuffle=True,
    )

    # The model
    regressor = Regressor(
        in_channels=HYPER_PARAMETERS["input_size"],
        out_channels=HYPER_PARAMETERS["output_size"],
    )
    regressor = regressor.apply(weights_init)
    regressor = regressor.cuda()

    # Loss
    recon_criterion = torch.nn.L1Loss()
    recon_criterion = recon_criterion.cuda()

    os.makedirs('checkpoints', exist_ok=True)

    best_losses = 1e10

    params = pkl.load(open("data/params_enc.pkl", "rb"))

    mean_shirt = torch.from_numpy(params["mean_shirt"]).cuda()
    max_offset = torch.from_numpy(params["max_offset"]).cuda()
    min_offset = torch.from_numpy(params["min_offset"]).cuda()
    mean_pose = torch.from_numpy(params["mean_pose"]).cuda()
    sd_pose = torch.from_numpy(params["sd_pose"]).cuda()

    template_fs = kaobj.import_mesh('data/mean_shirt.obj').faces.cuda()

    epoch_acc = 0
    for epochs, learning_rate in zip(HYPER_PARAMETERS["epochs"], HYPER_PARAMETERS["learning_rates"]):
        optimizer = torch.optim.Adam(list(regressor.parameters()), lr=learning_rate)
        for epoch in range(epochs):
            epoch = epoch + epoch_acc
            print(f"Starting epoch {epoch} with learning rate {learning_rate}")
            train_loss = train(
                train_loader,
                mean_pose,
                sd_pose,
                mean_shirt,
                min_offset,
                max_offset,
                template_fs,
                regressor,
                optimizer,
                recon_criterion,
                epoch,
            )

            # Save checkpoint and replace old best model if current model is better
            if train_loss < best_losses or epoch == (epochs - 1):
                best_losses = train_loss
                torch.save(
                    regressor.state_dict(),
                    f'checkpoints/regressor-epoch-{epoch + 1}-losses-{train_loss:.3f}.pth'
                )
            with torch.no_grad():
                validate(
                    val_loader,
                    mean_pose,
                    sd_pose,
                    mean_shirt,
                    min_offset,
                    max_offset,
                    template_fs,
                    regressor,
                    optimizer,
                    recon_criterion,
                    epoch,
                )

            folders = [
                "data/train_sequence/poses/dan-005",
                "data/validation_sequence/poses/dan-013",
            ]
            for folder in folders:
                predict_sequence(
                    regressor,
                    folder,
                    mean_pose,
                    sd_pose,
                    mean_shirt,
                    max_offset,
                    min_offset,
                    epoch,
                )

        epoch_acc += epochs


if __name__ == '__main__':
    main()
