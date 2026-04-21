import numpy as np
import pytest
from src.linalg.pagerank import is_stochastic, power_iteration

def test_is_stochastic_valid():
    """Verify is_stochastic correctly identifies valid column-stochastic matrices."""
    M = np.array([
        [0.5, 0.3, 0.2],
        [0.2, 0.4, 0.4],
        [0.3, 0.3, 0.4]
    ])
    assert is_stochastic(M) is True

def test_is_stochastic_invalid_sum():
    """Verify is_stochastic rejects matrices where column sums != 1."""
    M = np.array([
        [0.5, 0.3, 0.2],
        [0.2, 0.4, 0.4],
        [0.3, 0.3, 0.5]
    ])
    assert is_stochastic(M) is False

def test_is_stochastic_invalid_negative():
    """Verify is_stochastic rejects matrices with negative entries."""
    M = np.array([
        [1.1, 0.5],
        [-0.1, 0.5]
    ])
    assert is_stochastic(M) is False

def test_power_iteration_basic():
    """Test power iteration on a simple 2x2 matrix with known dominant eigenvalue."""
    # A = [[2, 1], [1, 2]] -> eigenvalues are 3 and 1
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    eigenvalue, eigenvector = power_iteration(A)
    
    assert np.isclose(eigenvalue, 3.0, atol=1e-10)
    # Verify A*v = lambda*v
    assert np.allclose(A @ eigenvector, eigenvalue * eigenvector, atol=1e-10)

def test_power_iteration_stochastic():
    """Verify that power iteration finds eigenvalue 1 for a stochastic matrix."""
    M = np.array([
        [0.8, 0.3],
        [0.2, 0.7]
    ])
    eigenvalue, eigenvector = power_iteration(M)
    assert np.isclose(eigenvalue, 1.0, atol=1e-10)
    # Check stationary distribution property: M*v = v
    assert np.allclose(M @ eigenvector, eigenvector, atol=1e-10)

def test_power_iteration_convergence():
    """Check convergence on a larger random symmetric matrix."""
    np.random.seed(42)
    n = 10
    A = np.random.rand(n, n)
    A = A + A.T  # Symmetric matrices have real eigenvalues
    eigenvalue, eigenvector = power_iteration(A, num_iterations=2000, tol=1e-12)
    
    # Verify eigenvalue/eigenvector property
    result = A @ eigenvector
    expected = eigenvalue * eigenvector
    assert np.allclose(result, expected, atol=1e-10)

def test_pagerank_invalid_input():
    """Test defensive assertions in pagerank functions."""
    with pytest.raises(AssertionError):
        is_stochastic(np.array([1, 2, 3])) # 1D
    
    with pytest.raises(AssertionError):
        is_stochastic(np.zeros((2, 3))) # Not square
        
    with pytest.raises(AssertionError):
        power_iteration(np.zeros((2, 3))) # Not square
