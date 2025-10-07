"""
np.insert(array, index, value, axis=None)
if axis = 0 ; row-wise
else axis =1 ; column wise
"""
import numpy as np

arr = np.array([10,20,30,40,50,60])
print(arr)
new_arr = np.insert(arr, 2, 100, axis=0)
print(new_arr)