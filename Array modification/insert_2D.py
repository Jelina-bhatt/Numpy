import numpy as np
 
arr_2d = np.array([[1,2],[3,4]])

#insert a new row at index 1
new_arr = np.insert(arr_2d, 1, [5,6], axis=1)
print(new_arr)