import numpy as np

def hilbert_matrix(n: int) -> np.ndarray:
    r"""
    Generates a Hilbert matrix of order n.
    
    A Hilbert matrix $H$ has entries $H_{ij} = \frac{1}{i+j-1}$.
    It is a classic example of an ill-conditioned matrix.
    
    Args:
        n: The order of the matrix.
        
    Returns:
        A Hilbert matrix of shape (n, n).
    """
    assert isinstance(n, int) and n > 0, "n must be a positive integer"
    return np.fromfunction(lambda i, j: 1 / (i + j + 1), (n, n))
