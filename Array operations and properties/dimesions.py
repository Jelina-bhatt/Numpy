#to count the number of dimensions
import numpy as np

arr1d = np.array([1,2,3])
arr2d = np.array([[1,2,3],[4,5,6]])
arr3d = np.array([[[1,2],[3,4],[5,6],[7,8]]])

print(arr1d.ndim)
print(arr2d.ndim)
print(arr3d.ndim)