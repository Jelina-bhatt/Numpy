import numpy as np

# 1D array
arr = np.array([10, 20, 30, 40, 50])
print("First element:", arr[0])    # 10
print("Last element:", arr[-1])    # 50
print("Third element:", arr[2])    # 30

# 2D array
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print("Element at row 0, col 2:", arr2d[0, 2])  # 3
print("Element at row 2, col 1:", arr2d[2, 1])  # 8
