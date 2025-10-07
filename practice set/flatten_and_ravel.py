import numpy as np

# Create an array
arr = np.arange(1, 9)  # [1 2 3 4 5 6 7 8]

# Reshape into 2D
reshaped = arr.reshape(2, 4)
print("Reshaped:\n", reshaped)

# Flatten makes a COPY
print("Flattened:", reshaped.flatten())

# Ravel makes a VIEW
print("Ravelled:", reshaped.ravel())
