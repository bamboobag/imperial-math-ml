import numpy as np
from typing import Tuple

def modified_gram_schmidt(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Performs the Modified Gram-Schmidt (MGS) orthonormalization to compute the QR decomp.

    The MGS algorithm decomposes a matrix $A \in \mathbb{R}^{m \times n}$ into an orthogonal 
    matrix $Q \in \mathbb{R}^{m \times n}$ and an upper triangular matrix $R \in \mathbb{R}^{n \times n}$ 
    such that $A = QR$. 

    Compared to the Classical Gram-Schmidt, the Modified version is more numerically stable 
    against floating-point rounding errors.

    Mathematical Definition:
    For a set of vectors $\{a_1, \dots, a_n\}$, we compute $q_i$ as:
    $$ v_i^{(1)} = a_i $$
    $$ q_i = \frac{v_i^{(i)}}{\|v_i^{(i)}\|_2} $$
    $$ v_j^{(i+1)} = v_j^{(i)} - \langle q_i, v_j^{(i)} \rangle q_i \quad \text{for } j > i $$

    Args:
        A: A matrix of shape (m, n) where m >= n.

    Returns:
        Q: Orthonormal matrix of shape (m, n).
        R: Upper triangular matrix of shape (n, n).

    Raises:
        AssertionError: If input is not a 2D array or if dimensions are incompatible.
    """
    assert isinstance(A, np.ndarray), "Input A must be a numpy array"
    assert A.ndim == 2, "Input A must be a 2D matrix"
    
    m, n = A.shape
    assert m >= n, "Tall or square matrices only (m >= n); QR decomp requires linear independence"

    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)
    V = A.astype(np.float64).copy()

    for i in range(n):
        R[i, i] = np.linalg.norm(V[:, i])
        
        # Handle zero-norm vectors (singular matrix case)
        if R[i, i] > 1e-15:
            Q[:, i] = V[:, i] / R[i, i]
        else:
            Q[:, i] = 0.0
            
        # Vectorized update for remaining columns
        if i < n - 1:
            # Compute projections for all j > i simultaneously
            # R[i, i+1:] = Q[:, i].T @ V[:, i+1:]
            # V[:, i+1:] -= Q[:, i:i+1] @ R[i:i+1, i+1:]
            
            # Using broadcasting for vectorization
            projections = np.dot(Q[:, i], V[:, i+1:])
            R[i, i+1:] = projections
            V[:, i+1:] -= np.outer(Q[:, i], projections)

    return Q, R
