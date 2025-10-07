# broadcasting: 1d with 2d
import numpy as np
mat = np.array([[1, 2, 3],
                [4, 5, 6]])

vec = np.array([10, 20, 30])

print(mat + vec)
# [[11 22 33]
#  [14 25 36]]
