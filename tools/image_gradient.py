import torch
import torch.nn.functional as F

__kernel = torch.tensor([
    [0.0, 0.0, 0.0, ],
    [0.0, 1.0, -1., ],
    [0.0, 0.0, 0.0, ],
])


def compute_image_gradient(image: torch.Tensor, transpose: bool = False):
    device = image.device
    b, h, w, c = image.shape

    kernel = __kernel
    if transpose:
        kernel = kernel.t()
    kernel = kernel.to(device)[None, None].repeat(3, 1, 1, 1, )
    permuted_image = image.permute(0, 3, 1, 2)
    convoluted = F.conv2d(permuted_image, kernel, padding='same', groups=c)

    permuted_convoluted = convoluted.permute(0, 2, 3, 1)

    return permuted_convoluted
