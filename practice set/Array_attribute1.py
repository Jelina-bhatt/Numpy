import numpy as np

arr = np.array([5, 10, 15, 20])

print("Array:", arr)
print("Shape:", arr.shape)     # (4,)
print("Dimensions:", arr.ndim) # 1D
print("Size:", arr.size)       # 4 elements
print("Data type:", arr.dtype) # int64 (may vary)
print("Item size:", arr.itemsize, "bytes")
