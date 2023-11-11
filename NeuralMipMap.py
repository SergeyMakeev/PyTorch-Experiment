import torch
import math

import torchvision.utils


def load_albedo(path):
    _albedo = torchvision.io.read_image(path)
    albedo = _albedo.permute(1, 2, 0).reshape(-1, 3).to(torch.float32)
    return albedo / 255.0


def load_normals(path):
    _normals = torchvision.io.read_image(path)
    normals = _normals.permute(1, 2, 0).reshape(-1, 3).to(torch.float32)
    normals = (normals / 255.0) * 2.0 - 1.0
    return normals / normals.norm(dim=1, keepdim=True)


def save_albedo(albedo, width, height, path):
    _albedo = albedo.reshape(height, width, 3).permute(2, 0, 1)
    torchvision.utils.save_image(_albedo, path)


def save_normals(normals, width, height, path):
    _normals = normals / normals.norm(dim=1, keepdim=True)
    _normals = _normals.reshape(height, width, 3).permute(2, 0, 1)
    _normals = (_normals + 1.0) * 0.5
    torchvision.utils.save_image(_normals, path)


def run():

    img_width = 256
    img_height = 256
    num_pixels = img_width * img_height
    num_samples = 16

    dtype = torch.float
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    torch.set_default_device(device)

    # generate random light directions
    torch.manual_seed(13)
    random_light_directions = torch.randn(num_samples, 3, dtype=dtype)
    x = random_light_directions / random_light_directions.norm(dim=1, keepdim=True)

    # load random normals (normalized) (num_pixels, 3)
    normal_map = load_normals("in_normal.png").to(device)

    # load albedo (num_pixels, 3)
    albedo = load_albedo("in_albedo.png").to(device)

    y_dp = torch.matmul(normal_map, x.t())          # (num_pixels, num_samples)
    y0 = y_dp.t().unsqueeze(-1)                     # (num_samples, num_pixels, 1)
    y_m = y0 * albedo                               # (num_samples, num_pixels, 3)
    y = y_m.flatten()                               # linearize (num_samples * num_pixels * 3)

    # our initial guess

    # initial normals/albedo
    _normal = torch.full((num_pixels, 3), 0.001, dtype=dtype, requires_grad=True)
    _albedo = torch.full((num_pixels, 3), 0.001, dtype=dtype, requires_grad=True)

    # use the original values for initialization (for test)
    # _normal = normal_map.clone().detach().requires_grad_(True)
    # _albedo = albedo.clone().detach().requires_grad_(True)

    learning_rate = 40.0
    for t in range(100000):
        # Forward pass: compute predicted y using operations on Tensors.
        _normal_norm = _normal / _normal.norm(dim=1, keepdim=True)
        _y_dp = torch.matmul(_normal_norm, x.t())  # (num_pixels, num_samples)
        _y0 = _y_dp.t().unsqueeze(-1)  # (num_samples, num_pixels, 1)
        _y_m = _y0 * _albedo  # (num_samples, num_pixels, 3)
        y_pred = _y_m.flatten()  # linearize (num_samples * num_pixels * 3)

        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the scalar value held in the loss.
        loss = (y_pred - y).pow(2).mean()

        if t % 1000 == 0:
            print(t, loss.item())
            save_albedo(_albedo, img_width, img_height, f"out/in_albedo_{t}.png")
            save_normals(_normal, img_width, img_height, f"out/in_normals_{t}.png")

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
        # the gradient of the loss with respect to a, b, c, d respectively.
        loss.backward()

        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.
        with torch.no_grad():
            _normal -= learning_rate * _normal.grad
            _albedo -= learning_rate * _albedo.grad

            # Manually zero the gradients after updating weights
            _normal.grad = None
            _albedo.grad = None

    print("-----------------")
    save_albedo(_albedo, img_width, img_height, f"out/in_albedo_fin.png")
    save_normals(_normal, img_width, img_height, f"out/in_normals_fin.png")


run()