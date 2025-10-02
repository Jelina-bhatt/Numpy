# eigen value and eigen vectors
import numpy as np
from numpy.linalg import eig

# Define a square matrix
a = np.array([[1, 2],
              [3, 4]])

# Eigenvalues and Eigenvectors
vals, vecs = eig(a)

print("Eigenvalues:", vals)
print("Eigenvectors:\n", vecs)

