# -*- coding: utf-8 -*-
import torch
import math
import matplotlib.pyplot as plt


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# Create random input and output data
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

# Randomly initialize weights
a = torch.randn((), device=device, dtype=dtype)
b = torch.randn((), device=device, dtype=dtype)
c = torch.randn((), device=device, dtype=dtype)
d = torch.randn((), device=device, dtype=dtype)
e = torch.randn((), device=device, dtype=dtype)
f = torch.randn((), device=device, dtype=dtype)

learning_rate = 1e-12
for t in range(2000):
    # Forward pass: compute predicted y
    y_pred = a + (b * x) + (c * x ** 2) + (d * x ** 3) + (e * x ** 4) + (f * x ** 5)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    # Backprop to compute gradients of a, b, c, d with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()
    grad_e = (grad_y_pred * x ** 4).sum()
    grad_f = (grad_y_pred * x ** 5).sum()

    # Update weights using gradient descent
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d
    e -= learning_rate * grad_d
    f -= learning_rate * grad_d


print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3 + {e.item()} x^4 + {f.item()} x^5')


_y = []

for _x in x:
    res = math.sin(_x)
    approx_res = a + (b * _x) + (c * _x ** 2) + (d * _x ** 3) + (e * _x ** 4) + (f * _x ** 5)
    _y.append(approx_res)


plt.plot(x, y, color='r', label='sin')
plt.plot(x, torch.tensor(_y), color='g', label='approx')
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Sin Approx")
plt.show()


