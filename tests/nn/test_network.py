import numpy as np
import pytest
from src.nn.network import FeedForwardNN

def test_nn_initialization():
    layer_sizes = (10, 20, 15, 5)
    nn = FeedForwardNN(layer_sizes)
    
    assert nn.W1.shape == (10, 20)
    assert nn.b1.shape == (20,)
    assert nn.W2.shape == (20, 15)
    assert nn.b2.shape == (15,)
    assert nn.W3.shape == (15, 5)
    assert nn.b3.shape == (5,)

def test_nn_forward_pass_shapes():
    layer_sizes = (10, 20, 15, 3)
    nn = FeedForwardNN(layer_sizes)
    
    batch_size = 8
    x = np.random.randn(batch_size, 10)
    y_hat = nn.forward(x)
    
    assert y_hat.shape == (batch_size, 3)

def test_nn_softmax_output():
    layer_sizes = (5, 10, 10, 4)
    nn = FeedForwardNN(layer_sizes)
    
    x = np.random.randn(2, 5)
    y_hat = nn.forward(x)
    
    # Check that outputs are probabilities (sum to 1)
    np.testing.assert_allclose(np.sum(y_hat, axis=1), 1.0, atol=1e-7)
    assert np.all(y_hat >= 0)
    assert np.all(y_hat <= 1)

def test_nn_input_validation():
    layer_sizes = (5, 10, 10, 2)
    nn = FeedForwardNN(layer_sizes)
    
    x = np.random.randn(3, 10)  # Wrong input size
    with pytest.raises(AssertionError, match="Input size mismatch"):
        nn.forward(x)
