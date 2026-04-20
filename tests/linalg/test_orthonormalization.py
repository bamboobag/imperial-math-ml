import numpy as np
import pytest
from src.linalg.orthonormalization import modified_gram_schmidt

def test_mgs_orthogonality():
    """
    Verify that Q^T Q = I (orthogonality condition).
    """
    # Create a random 5x3 matrix
    A = np.random.randn(5, 3)
    Q, R = modified_gram_schmidt(A)
    
    # Check dimensions
    assert Q.shape == (5, 3)
    assert R.shape == (3, 3)
    
    # Check orthogonality: Q^T Q should be Identity
    identity_approx = Q.T @ Q
    assert np.allclose(identity_approx, np.eye(3), atol=1e-12)

def test_mgs_reconstruction():
    """
    Verify that QR = A (reconstruction condition).
    """
    A = np.random.randn(10, 4)
    Q, R = modified_gram_schmidt(A)
    
    A_reconstructed = Q @ R
    assert np.allclose(A_reconstructed, A, atol=1e-12)

def test_mgs_upper_triangular():
    """
    Verify that R is upper triangular.
    """
    A = np.random.randn(6, 6)
    Q, R = modified_gram_schmidt(A)
    
    # Check if R is upper triangular
    assert np.allclose(R, np.triu(R), atol=1e-12)

def test_mgs_singular_matrix():
    """
    Test behavior with a singular matrix (linearly dependent columns).
    """
    # Create a matrix where the third column is the sum of the first two
    A = np.zeros((4, 3))
    A[:, 0] = [1, 0, 0, 0]
    A[:, 1] = [0, 1, 0, 0]
    A[:, 2] = [1, 1, 0, 0]
    
    Q, R = modified_gram_schmidt(A)
    
    # The third column of Q should still be handled gracefully (though it won't be part of an orthonormal basis for R^4)
    # Modified Gram-Schmidt should still produce Q and R such that QR = A
    assert np.allclose(Q @ R, A, atol=1e-12)
    # However, Q^T Q won't be Identity if the rank is less than n, 
    # but the columns that ARE produced should be orthogonal to each other.
    # In our implementation, zero columns are produced for singular components.
    
    # Check orthogonality of non-zero columns
    non_zero_cols = np.linalg.norm(Q, axis=0) > 1e-10
    Q_reduced = Q[:, non_zero_cols]
    if Q_reduced.shape[1] > 0:
        assert np.allclose(Q_reduced.T @ Q_reduced, np.eye(Q_reduced.shape[1]), atol=1e-12)

def test_mgs_identity():
    """
    Test with identity matrix.
    """
    A = np.eye(5)
    Q, R = modified_gram_schmidt(A)
    
    assert np.allclose(Q, np.eye(5), atol=1e-12)
    assert np.allclose(R, np.eye(5), atol=1e-12)

def test_mgs_invalid_input():
    """
    Test defensive programming assertions.
    """
    with pytest.raises(AssertionError):
        # 1D array
        modified_gram_schmidt(np.array([1, 2, 3]))
        
    with pytest.raises(AssertionError):
        # m < n
        modified_gram_schmidt(np.zeros((2, 3)))
