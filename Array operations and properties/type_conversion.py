#converts data type with the help of astype()

import numpy as np

arr = np.array ([1.2,2.5,3.8])
int_arr = arr.astype(int)
print(int_arr)
print(int_arr.dtype)
print(arr.dtype)