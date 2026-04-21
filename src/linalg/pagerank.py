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
                    tol: float = 1e-12,
                    return_history: bool = False) -> Tuple[float, np.ndarray, list]:
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
        return_history: If True, also returns the list of residuals (convergence history).

    Returns:
        eigenvalue: The dominant eigenvalue (scalar).
        eigenvector: The corresponding normalized eigenvector.
        history: List of residuals $\|b_{k+1} - b_k\|$ at each step.

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

    history = []
    for _ in range(num_iterations):
        # Calculate the matrix-vector product
        b_k1 = np.dot(A, b_k)

        # Re-normalize the vector
        b_k1_norm = np.linalg.norm(b_k1)
        if b_k1_norm < 1e-15:
            return 0.0, b_k, history
        
        b_k1 /= b_k1_norm

        # Check convergence
        residual = float(np.linalg.norm(b_k - b_k1))
        history.append(residual)
        
        if residual < tol:
            b_k = b_k1
            break
        
        b_k = b_k1

    # Rayleigh quotient for eigenvalue estimation
    eigenvalue = float(np.dot(b_k.T, np.dot(A, b_k)) / np.dot(b_k.T, b_k))
    
    return eigenvalue, b_k, history


def pagerank(M: np.ndarray, 
             d: float = 0.85, 
             num_iterations: int = 1000, 
             tol: float = 1e-12,
             return_history: bool = False) -> Tuple[np.ndarray, list]:
    r"""
    Computes the PageRank vector for a given stochastic matrix.

    The PageRank vector is the stationary distribution of the Google Matrix $G$:
    $$ G = d M + \frac{1-d}{n} J $$
    where $d$ is the damping factor, $M$ is the column-stochastic matrix, 
    and $J$ is the $n \times n$ matrix of all ones.

    Args:
        M: Column-stochastic matrix of shape (n, n).
        d: Damping factor (typically 0.85).
        num_iterations: Maximum number of iterations for power iteration.
        tol: Convergence tolerance.
        return_history: If True, also returns the list of residuals.

    Returns:
        pagerank_vector: The normalized PageRank vector.
        history: Convergence history from power iteration.
    """
    assert is_stochastic(M), "Input M must be a column-stochastic matrix"
    n = M.shape[0]
    
    # Construct Google Matrix G
    # J = np.ones((n, n))
    # G = d * M + (1 - d) / n * J
    
    # Efficient matrix-vector product implementation without explicitly forming J
    # Gv = d * Mv + (1-d)/n * Jv
    # Jv = sum(v) * ones
    
    # We can use power_iteration with a custom operator or just form G for simplicity here
    # since n is typically small in this educational repository.
    J = np.ones((n, n))
    G = d * M + (1 - d) / n * J
    
    _, v, history = power_iteration(G, num_iterations=num_iterations, tol=tol, return_history=True)
    
    # Ensure it's a probability distribution (sums to 1)
    v = v / np.sum(v)
    
    return v, history
