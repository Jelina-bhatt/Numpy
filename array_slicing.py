import numpy as np
# 1D slicing
arr = np.array([10, 20, 30, 40, 50])
print("Elements from index 1 to 3:", arr[1:4])  # [20 30 40]
print("Every second element:", arr[::2])        # [10 30 50]

# 2D slicing
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print("First two rows, last two cols:\n", arr2d[0:2, 1:3])
# [[2 3]
#  [5 6]]

print("Second row:", arr2d[1, :])  # [4 5 6]
print("First column:", arr2d[:, 0]) # [1 4 7]
