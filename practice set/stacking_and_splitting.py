import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("Horizontal stack:", np.hstack((a, b))) # [1 2 3 4 5 6]
print("Vertical stack:\n", np.vstack((a, b))) 
# [[1 2 3]
#  [4 5 6]]
