import numpy as np

# this is not working
def power_iteration(A, max_iter=10000, tol=1e-12):
    v = 2*np.random.rand(A.shape[0]) - 1.0 # make the values in the guess vector [-1, 1]
    v = v / np.linalg.norm(v)
    for i in range(max_iter):
        v = A@v
        v = v/np.linalg.norm(v)
        lambda_ = v@A@v
        res = np.linalg.norm((lambda_*v - A@v)**2)
        if res < tol:
            return lambda_, v
        
def run_power_iteration(A):
    lambda_, v = power_iteration(A)
    return lambda_, v

def run_eigh_max(A):
    eigenValues, eigenVectors = np.linalg.eigh(A)
    # sort the eigenvalues for ease of comparison
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]  
    return eigenValues[0], eigenVectors[0]

def generate_symmetric_matrix():
    A = np.random.rand(10, 10)
    return (A + A.T)/2

matrix = generate_symmetric_matrix()
print(run_power_iteration(matrix))
print(run_eigh_max(matrix))
