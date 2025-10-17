
import numpy as np

arr = np.array([1,2,np.inf,4,-np.inf,6])

print(np.isinf(arr))

cleaned_arr= np.nan_to_num(arr, posinf=100, neginf=500)
print(cleaned_arr)