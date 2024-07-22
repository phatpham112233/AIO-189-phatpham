### a.Length of a Vector:
import numpy as np

def compute_vector_length(vector):
    len_of_vector = np.linalg.norm(vector)
    return len_of_vector


vector = np.array([3, 4])
print(compute_vector_length(vector))  

### b.Dot Product:

def compute_dot_product(vector1, vector2):
    result = np.dot(vector1, vector2)
    return result

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(compute_dot_product(v1, v2))  

###c.Multiplying a Vector by a Matrix
def matrix_multi_vector(matrix, vector):
    result = np.dot(matrix, vector)
    return result

matrix = np.array([[1, 2], [3, 4]])
vector = np.array([5, 6])
print(matrix_multi_vector(matrix, vector))  

### d.Multiplying a Matrix by a Matrix:
def matrix_multi_matrix(matrix1, matrix2):
    result = np.dot(matrix1, matrix2)
    return result

matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
print(matrix_multi_matrix(matrix1, matrix2))  

### c.Matrix Inverse:
def inverse_matrix(matrix):
    result = np.linalg.inv(matrix)
    return result

# Example usage
matrix = np.array([[1, 2], [3, 4]])
print(inverse_matrix(matrix))  # Output should be [[-2.0, 1.0], [1.5, -0.5]]
