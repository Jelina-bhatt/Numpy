"""
np.delete(array, index, axis=None)
flatten array
"""

import numpy as np

arr = np.array([10,30,40,40,60,70,80])
print(arr)
new_arr = np.delete(arr, 3, axis=None)
print(new_arr)
