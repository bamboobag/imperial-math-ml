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
