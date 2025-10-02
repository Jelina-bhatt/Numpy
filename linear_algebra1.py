#determinant and inverse
import numpy as np
from numpy.linalg import det, inv

# Define a square matrix
a = np.array([[1, 2],
              [3, 4]])

# Determinant
print("Determinant:", det(a))

# Inverse
print("Inverse:\n", inv(a))
