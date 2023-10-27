import torch
import math

dtype = torch.float
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
torch.set_default_device(device)

# generate random values for x, y, and z components
random_light_directions = torch.randn(1000, 3, dtype=dtype)

x = random_light_directions / random_light_directions.norm(dim=1, keepdim=True)

# set some normal (we are going to try to predict that)
normal_ = torch.tensor([0.0, 1.0, 0.0])

# generate random normal
normal = torch.randn_like(normal_)

# normalize normal
normal = normal / normal.norm()

# albedo to guess
# albedo = torch.tensor([0.3, 0.7, 0.1])
albedo = torch.rand(3)

print("Random surface properties to guess")
print("-----------------")
print("r: " + str(albedo[0].item()))
print("g: " + str(albedo[1].item()))
print("b: " + str(albedo[2].item()))
print("nx: " + str(normal[0].item()))
print("ny: " + str(normal[1].item()))
print("nz: " + str(normal[2].item()))
print("-----------------")


y0 = torch.matmul(x, normal).clamp(0, 1)
y1 = y0.unsqueeze(1)        # (1000) -> (1000, 1)
y2 = albedo.unsqueeze(0)    # (3) -> (1,3)
y3 = y1 * y2                # multiply
y = y3.view(-1)             # linearize

# our initial guess
std = 1e-2
_normal = torch.tensor([0.01, 0.01, 0.01], dtype=dtype, requires_grad=True)
_albedo = torch.tensor([0.01, 0.01, 0.01], dtype=dtype, requires_grad=True)

learning_rate = 1e-3
for t in range(100000):
    # Forward pass: compute predicted y using operations on Tensors.
    _normal_norm = _normal / _normal.norm()

    _y0 = torch.matmul(x, _normal_norm).clamp(0, 1)
    _y1 = _y0.unsqueeze(1)        # (1000) -> (1000, 1)
    _y2 = _albedo.unsqueeze(0)    # (3) -> (1,3)
    _y3 = _y1 * _y2               # multiply
    y_pred = _y3.view(-1)         # linearize

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.
    loss = (y_pred - y).pow(2).mean()
    if t < 10 or (t < 100 and t % 10 == 0) or t % 1000 == 0:
        print(t, loss.item())

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

_normal_norm = _normal / _normal.norm()

dp = torch.dot(normal, _normal_norm)

print("Answer:")
print("dp:" + str(dp.item()))
print("-----------------")
print("r:" + str(_albedo[0].item()))
print("g:" + str(_albedo[1].item()))
print("b:" + str(_albedo[2].item()))
print("x:" + str(_normal_norm[0].item()))
print("y:" + str(_normal_norm[1].item()))
print("z:" + str(_normal_norm[2].item()))
print("-----------------")
print("The original values")
print("r: " + str(albedo[0].item()))
print("g: " + str(albedo[1].item()))
print("b: " + str(albedo[2].item()))
print("nx: " + str(normal[0].item()))
print("ny: " + str(normal[1].item()))
print("nz: " + str(normal[2].item()))
print("-----------------")


