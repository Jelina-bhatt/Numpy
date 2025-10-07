# to convert multid array to 1d array
"""
.ravel() -> views #affects in original array
.flatten() ->copy #doesnt affect in original array

"""

import numpy as np

arr_2d = np.array (([1,2,3],[4,5,6]))
print(arr_2d.flatten())
print(arr_2d.ravel())