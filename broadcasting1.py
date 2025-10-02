# dot product and matrix multiplication
import numpy as np
a = np.array([[1, 2],
              [3, 4]])

b = np.array([[5, 6],
              [7, 8]])

print("Dot product:\n", np.dot(a, b))
print("Matrix multiplication:\n", a @ b)  # Same as np.matmul(a, b)
