import numpy as np

def generate_symmetric_matrix(n):
    """
    Generates a random symmetric matrix of size n x n.
    """
    A = np.random.randn(n, n)
    return np.dot(A, A.T)  # A*A^T to ensure the matrix is symmetric

def sorted_eigen(A, reverse=False):
    """
    Computes the eigenvalues and eigenvectors of matrix A.
    Returns eigenvalues and eigenvectors sorted in descending order of eigenvalues.
    """
    eigvals, eigvecs = np.linalg.eig(A)  # Using eigh as it's efficient for symmetric matrices
    # Sort eigenvalues and eigenvectors
    if reverse == False:
        sorted_indices = np.argsort(-eigvals)[::-1]  # Sort indices in descending order
    if reverse == True:
       sorted_indices = np.argsort(eigvals)[::-1] 
    sorted_eigvals = eigvals[sorted_indices]
    sorted_eigvecs = eigvecs[:, sorted_indices]
    return sorted_eigvals, sorted_eigvecs

def is_symmetric(A, tol=1e-6):
    """
    Checks if a matrix A is symmetric.
    Returns True if symmetric, otherwise False.
    """
    return np.allclose(A, A.T, atol=tol)

def normalize_eigenvectors(eigvecs):
    """
    Normalizes the eigenvectors (makes them unit vectors).
    """
    norms = np.linalg.norm(eigvecs, axis=0)
    return eigvecs / norms

def generate_positive_definite_matrix(n):
    """
    Generates a random positive-definite matrix of size n x n.
    """
    A = np.random.randn(n, n)
    return np.dot(A.T, A)  # A^T * A will be positive definite

def is_positive_definite(A):
    """
    Checks if a matrix is positive definite.
    Returns True if positive definite, otherwise False.
    """
    try:
        eigvals = np.linalg.eigvals(A)
        return np.all(eigvals > 0)
    except np.linalg.LinAlgError:
        return False

def random_unit_vector(dim):
    """
    Generates a random unit vector in a given dimension.
    
    Parameters:
    dim (int): The dimension of the unit vector.
    
    Returns:
    numpy.ndarray: A random unit vector of the specified dimension.
    """
    # Generate a random vector
    random_vector = np.random.randn(dim)
    
    # Normalize the vector to make it a unit vector
    unit_vector = random_vector / np.linalg.norm(random_vector)
    
    return unit_vector

def generate_dense_symmetric_normal_matrix(size, density, mean=0, std_dev=1, seed=None):
    """
    Generates a dense symmetric matrix with values sampled from a normal distribution.

    Parameters:
        size (int): Size of the matrix (size x size).
        density (float): Density of the non-zero entries (0 < density <= 1).
        mean (float): Mean of the normal distribution.
        std_dev (float): Standard deviation of the normal distribution.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: A dense symmetric matrix.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate a dense matrix with values sampled from a normal distribution
    matrix = np.random.normal(mean, std_dev, size=(size, size))
    
    # Create a mask to enforce the specified density
    mask = np.random.rand(size, size) < density
    
    # Apply the mask to zero out some elements
    matrix = np.multiply(matrix, mask)
    
    # Symmetrize the matrix
    symmetric_matrix = (matrix + matrix.T) / 2
    
    # Ensure the diagonal is non-zero by sampling new values for it
    np.fill_diagonal(symmetric_matrix, np.random.normal(mean, std_dev, size=size))
    
    return symmetric_matrix

def generate_diagonally_dominant_matrix(n, seed=None):
    """
    Generate a diagonally dominant random symmetric matrix of size n x n.

    Parameters:
    n (int): The size of the matrix (n x n).
    seed (int, optional): A seed for the random number generator for reproducibility.

    Returns:
    numpy.ndarray: A diagonally dominant random symmetric matrix.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate a random symmetric matrix
    A = np.random.rand(n, n)
    A = (A + A.T) / 2  # Symmetrize the matrix

    # Adjust diagonal to make it diagonally dominant
    for i in range(n):
        A[i, i] = sum(np.abs(A[i])) + np.random.rand()  # Add a random positive value for strict dominance

    return A
