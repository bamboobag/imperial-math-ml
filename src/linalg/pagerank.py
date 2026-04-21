import numpy as np
from typing import Tuple

def is_stochastic(M: np.ndarray) -> bool:
    r"""
    Validates if a matrix is column-stochastic.

    A matrix $M \in \mathbb{R}^{n \times n}$ is column-stochastic if:
    1. $M_{ij} \geq 0$ for all $i, j$
    2. $\sum_{i=1}^n M_{ij} = 1$ for all $j=1, \dots, n$

    Mathematical condition:
    $$ \mathbf{1}^T M = \mathbf{1}^T $$
    where $\mathbf{1}$ is a vector of ones.

    Args:
        M: Input matrix of shape (n, n).

    Returns:
        bool: True if M is column-stochastic within numerical tolerance.
    """
    assert isinstance(M, np.ndarray), "Input M must be a numpy array"
    assert M.ndim == 2, "Input M must be a 2D matrix"
    assert M.shape[0] == M.shape[1], "Input M must be a square matrix"

    # Non-negativity check
    if np.any(M < -1e-15):
        return False

    # Column sum check
    col_sums = np.sum(M, axis=0)
    return np.allclose(col_sums, 1.0, atol=1e-12)


def power_iteration(A: np.ndarray, 
                    num_iterations: int = 1000, 
                    tol: float = 1e-12) -> Tuple[float, np.ndarray]:
    r"""
    Finds the dominant eigenvalue and eigenvector of a matrix $A$ using Power Iteration.

    The algorithm starts with a random vector $b_0$ and performs the iteration:
    $$ b_{k+1} = \frac{A b_k}{\|A b_k\|} $$
    
    The eigenvalue $\lambda$ is estimated using the Rayleigh Quotient:
    $$ \lambda \approx \frac{b_k^T A b_k}{b_k^T b_k} $$

    Args:
        A: A square matrix of shape (n, n).
        num_iterations: Maximum number of iterations to perform.
        tol: Convergence tolerance (change in eigenvector norm).

    Returns:
        eigenvalue: The dominant eigenvalue (scalar).
        eigenvector: The corresponding normalized eigenvector.

    Raises:
        AssertionError: If A is not square or not a 2D array.
    """
    assert isinstance(A, np.ndarray), "Input A must be a numpy array"
    assert A.ndim == 2, "Input A must be a 2D matrix"
    n, m = A.shape
    assert n == m, "Input A must be a square matrix"

    # Initialize a random vector
    b_k = np.random.rand(n)
    b_k /= np.linalg.norm(b_k)

    for _ in range(num_iterations):
        # Calculate the matrix-vector product
        b_k1 = np.dot(A, b_k)

        # Re-normalize the vector
        b_k1_norm = np.linalg.norm(b_k1)
        if b_k1_norm < 1e-15:
            return 0.0, b_k
        
        b_k1 /= b_k1_norm

        # Check convergence
        if np.linalg.norm(b_k - b_k1) < tol:
            b_k = b_k1
            break
        
        b_k = b_k1

    # Rayleigh quotient for eigenvalue estimation
    eigenvalue = float(np.dot(b_k.T, np.dot(A, b_k)) / np.dot(b_k.T, b_k))
    
    return eigenvalue, b_k
