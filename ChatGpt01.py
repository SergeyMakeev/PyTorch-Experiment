# generated by chat gpt
import math

import numpy as np
import matplotlib.pyplot as plt

# Generate data from the sine function
x = np.linspace(0, 2*np.pi, 100) # 100 sample points
y = np.sin(x)

# Construct the design matrix A
A = np.column_stack([x**i for i in range(6)])

# Compute the polynomial coefficients
coefficients = np.linalg.lstsq(A, y, rcond=None)[0]

# Evaluate the polynomial
y_poly = np.dot(A, coefficients)

print(coefficients)

a = coefficients[0]
b = coefficients[1]
c = coefficients[2]
d = coefficients[3]
e = coefficients[4]
f = coefficients[5]

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3 + {e} x^4 + {f} x^5')


_y = []

for _x in x:
    res = math.sin(_x)
    approx_res = a + (b * _x) + (c * _x ** 2) + (d * _x ** 3) + (e * _x ** 4) + (f * _x ** 5)
    _y.append(approx_res)


# Plot the results
plt.plot(x, y, label='sin(x)')
# plt.plot(x, y_poly, label='5th order polynomial fit')
plt.plot(x, _y, label='5th order polynomial fit')
plt.legend()
plt.show()
