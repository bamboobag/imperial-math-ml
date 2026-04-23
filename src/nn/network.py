import numpy as np
from typing import List, Tuple
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
        Initializes the network weights and biases.
        
        Args:
            layer_sizes: A tuple (input_size, hidden1_size, hidden2_size, output_size).
            seed: Random seed for reproducibility.
        """
        # TODO: Implement weight and bias initialization
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        r"""
        Performs the forward pass of the network.
        
        Args:
            x: Input data of shape (batch_size, input_size).
            
        Returns:
            Output probabilities of shape (batch_size, output_size).
            
        Raises:
            AssertionError: If input shape does not match expected input size.
        """
        # TODO: Implement forward pass
        pass
