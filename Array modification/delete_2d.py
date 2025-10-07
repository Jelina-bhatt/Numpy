import numpy as np

# Create 2D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("Original array:")
print(arr_2d)

# Delete first row
new_arr = np.delete(arr_2d, 0, axis=0)
print("\nAfter deleting first row:")
print(new_arr)

# Insert a new row at index 1
modified_arr = np.insert(new_arr, 1, [7, 8, 9], axis=0)
print("\nAfter inserting new row:")
print(modified_arr)
