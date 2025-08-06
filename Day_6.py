# Indexing and Slicing 

# print(arr3[1, 2])  # Second row, third column
# What happens?:
# -- Accesses the element at row index 1, column index 2 in arr3.
# -- Output: 6 (from the 3x4 array shown earlier).

# print(arr3[:, 1:3])  # All rows, columns 1-2
# What happens?:
# -- : this will selects all rows.
# -- 1:3 selects columns at indices 1 and 2 (Python slices are upper bound exclusive).

import numpy as np

# Creating arrays
arr1 = np.array([1, 2, 3, 4, 5])          # 1D array
arr2 = np.array([[1, 2, 3], [4, 5, 6]])   # 2D array

# Array operations (vectorization)
print(arr1 * 2)         # Multiply each element by 2
print(arr1 + arr1)      # Add arrays element-wise

# Reshaping
arr3 = np.arange(12).reshape(3, 4)        # Create 3x4 array from 0 to11
print(arr3)

# Matrix operations
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
print(np.dot(matrix1, matrix2))          # Matrix multiplication

# Statistical operations
data = np.random.normal(0, 1, 1000)       # 1000 random numbers from normal distribution
print(f"Mean: {data.mean()}, Std: {data.std()}")
