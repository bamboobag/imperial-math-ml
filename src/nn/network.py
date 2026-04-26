import numpy as np
from typing import Tuple
from src.nn.activations import relu, softmax

class FeedForwardNN:
    r"""
    A 2-hidden layer feed-forward neural network implementation using NumPy.
    
    Architecture:
    1. Input Layer: $x \in \mathbb{R}^{n_{in}}$
    2. Hidden Layer 1: $h_1 = \sigma(W_1 x + b_1)$
    3. Hidden Layer 2: $h_2 = \sigma(W_2 h_1 + b_2)$
    4. Output Layer: $y = \text{softmax}(W_3 h_2 + b_3)$
    
    Where $\sigma$ is the ReLU activation function.
    """
    
    def __init__(self, layer_sizes: Tuple[int, int, int, int], seed: int = 42):
        r"""
        Initializes the network weights and biases using He initialization.
        
        Args:
            layer_sizes: A tuple (input_size, hidden1_size, hidden2_size, output_size).
            seed: Random seed for reproducibility.
        """
        np.random.seed(seed)
        n_in, n_h1, n_h2, n_out = layer_sizes
        self.layer_sizes = layer_sizes

        # He initialization for ReLU layers: Var(W) = 2/n_in
        self.W1 = np.random.randn(n_in, n_h1) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_h1)

        self.W2 = np.random.randn(n_h1, n_h2) * np.sqrt(2.0 / n_h1)
        self.b2 = np.zeros(n_h2)

        # Glorot initialization for the output layer (Softmax)
        self.W3 = np.random.randn(n_h2, n_out) * np.sqrt(1.0 / n_h2)
        self.b3 = np.zeros(n_out)

    def forward(self, x: np.ndarray) -> np.ndarray:
        r"""
        Performs the forward pass of the network.
        
        Args:
            x: Input data of shape (batch_size, input_size).
            
        Returns:
            Output probabilities of shape (batch_size, output_size).
            
        Raises:
            AssertionError: If input shape does not match expected input size or dimension.
        """
        assert x.ndim == 2, f"Input must be 2D (batch_size, input_size), got {x.ndim}D"
        assert x.shape[1] == self.layer_sizes[0], \
            f"Input size mismatch. Expected {self.layer_sizes[0]}, got {x.shape[1]}"

        # Layer 1
        self.z1 = x @ self.W1 + self.b1
        self.a1 = relu(self.z1)

        # Layer 2
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)

        # Layer 3 (Output)
        self.z3 = self.a2 @ self.W3 + self.b3
        self.y_hat = softmax(self.z3)

        return self.y_hat
