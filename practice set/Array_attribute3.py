# changing D type
import numpy as np
arr = np.array([1.5, 2.3, 3.7])

print("Original dtype:", arr.dtype)  # float64
arr_int = arr.astype(int)            # convert to int
print("Converted dtype:", arr_int.dtype)
print("Converted array:", arr_int)   # [1 2 3]
