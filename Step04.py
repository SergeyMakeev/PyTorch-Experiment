#
# guess a normal
#   gradient descent algo is trying to "guess" an original normal by looking to dot products only
#
import torch
import math
import matplotlib.pyplot as plt

dtype = torch.float
device = "cpu"
torch.set_default_device(device)

# generate random values for x, y, and z components
random_vectors = torch.randn(1000, 3, dtype=dtype)

x = random_vectors / random_vectors.norm(dim=1, keepdim=True)

# set some normal (we are going to try to predict that)
normal_ = torch.tensor([0.0, 1.0, 0.0])

# generate random normal
normal = torch.randn_like(normal_)

# normalize normal
normal = normal / normal.norm()

print("Random normal to guess")
print("-----------------")
print("x: " + str(normal[0].item()))
print("y: " + str(normal[1].item()))
print("z: " + str(normal[2].item()))
print("-----------------")


y = torch.matmul(x, normal).clamp(0, 1)

# our initial guess
std = 1e-2
_normal = torch.tensor([0.01, 0.01, 0.01], dtype=dtype, requires_grad=True)

learning_rate = 1e-4
for t in range(80000):
    # Forward pass: compute predicted y using operations on Tensors.
    _normal_norm = _normal / _normal.norm()
    y_pred = torch.matmul(x, _normal_norm).clamp(0, 1)

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

        # Manually zero the gradients after updating weights
        _normal.grad = None

_normal_norm = _normal / _normal.norm()


dp = torch.dot(normal, _normal_norm)


print("Answer:")
print("dp:" + str(dp.item()))
print("-----------------")
print("x:" + str(_normal_norm[0].item()))
print("y:" + str(_normal_norm[1].item()))
print("z:" + str(_normal_norm[2].item()))
print("-----------------")


