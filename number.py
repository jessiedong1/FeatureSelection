#NumPy array
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6]])
print("x:\n{}".format(x))

from scipy import sparse

# Create a 2D Numpy array with a diagonal of ones, and zeros everywhere else
eye = np.eye(4)
print("NumPy array:\n{}".format(eye))

#Convert the NumPy array to a SciPy sparse matrix in CSR format
#Only the nonzero entries are stored

sparse_matrix = sparse.csr_matrix(eye)
print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))

#Create sparse representation directly
data = np.ones(5)
row_indices = np.arange(5)
col_indices = np.arange(5)
print("DATA\n{}".format(data))
eye_coo = sparse.coo_matrix((data,(row_indices,col_indices)))
print("COO representation:\n{}".format(eye_coo))




