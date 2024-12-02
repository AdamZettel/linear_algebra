import numpy as np
from scipy.sparse.linalg import LinearOperator
import matplotlib.pyplot as plt
import matrix_module as mm


# Davidson diagonalization
def davidson(A, k, max_iter=100, tol=1e-8):
    """
    Davidson algorithm for finding the smallest k eigenvalues and eigenvectors.
    
    Parameters:
        A : ndarray or LinearOperator
            Symmetric matrix (dense or sparse) to diagonalize.
        k : int
            Number of eigenvalues to compute.
        max_iter : int
            Maximum number of iterations.
        tol : float
            Convergence tolerance for residuals.
    
    Returns:
        eigvals : ndarray
            Array of the smallest k eigenvalues.
        eigvecs : ndarray
            Corresponding eigenvectors.
    """
    n = A.shape[0]
    V = np.random.rand(n, k)  # Initial subspace
    V, _ = np.linalg.qr(V)    # Orthogonalize the subspace
    eigvals, eigvecs = None, None
    
    for iteration in range(max_iter):
        # Project the matrix onto the subspace
        T = V.T @ A @ V
        evals, evecs = np.linalg.eigh(T)
        
        # Approximate eigenvectors
        eigvals = evals[:k]
        eigvecs = V @ evecs[:, :k]
        
        # Residuals
        residuals = A @ eigvecs - eigvecs @ np.diag(eigvals)
        residual_norms = np.linalg.norm(residuals, axis=0)
        
        # Convergence check
        if np.all(residual_norms < tol):
            print(f"Converged in {iteration + 1} iterations.")
            return eigvals, eigvecs
        
        # Expand the subspace
        for i in range(k):
            if residual_norms[i] > tol:
                r = residuals[:, i]
                q = r / (eigvals[i] - np.diag(A))  # Preconditioning
                q -= V @ (V.T @ q)  # Orthogonalize against V
                q /= np.linalg.norm(q)
                V = np.hstack((V, q.reshape(-1, 1)))
        
        # Orthogonalize V
        V, _ = np.linalg.qr(V)
    
    print("Maximum iterations reached.")
    return eigvals, eigvecs

# Example usage
size = 100  # Matrix size
num_eigenvalues = 2  # Number of eigenvalues to find
A = mm.generate_symmetric_matrix(size)

eigvals, eigvecs = davidson(A, num_eigenvalues)
print("Eigenvalues:", eigvals)

eigvals_exact, eigvecs_exact = mm.sorted_eigen(A)
print(eigvals_exact)
