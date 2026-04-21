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
    # TODO: Implement stochastic matrix validation
    pass


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
    # TODO: Implement power iteration algorithm
    pass
