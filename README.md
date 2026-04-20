# Mathematical Foundations for Machine Learning
---
*Implementation of core computational algorithms following Imperial ReCoDE standards*

## Project Objective
This library implements fundamental Machine Learning mathematics from first principles. The focus is on bridging the gap between theoretical coursework and production-grade numerical computing, specifically prepared for the Imperial MSc MLDS.

## Engineering Standards
*   **Numerical Stability**: Prioritizing algorithms that mitigate floating-point errors (e.g., Modified Gram-Schmidt).
*   **Vectorization**: Leveraging NumPy broadcasting to eliminate explicit loops and ensure performance.
*   **Defensive Programming**: Runtime validation of matrix dimensions and properties via strict type hinting and assertions.
*   **Reproducibility**: Full test coverage of mathematical edge cases via `pytest`.

## Library Modules

### 1. Numerical Linear Algebra
*   Orthonormalization: Modified Gram-Schmidt (MGS) for forming orthonormal bases.
*   Iterative Solvers: Power Iteration for Eigenvector discovery (PageRank).

### 2. Multivariate Calculus and Optimization
*   Automatic Differentiation: Manual derivation of Jacobians for neural architectures.
*   Optimization: Gradient descent implementations in raw NumPy.

### 3. Statistics and Dimensionality
*   Principal Component Analysis: Implementation via Eigendecomposition.
*   Data Normalization: Scalable feature centering and variance scaling.

## Setup and Validation
This project uses `uv` for reproducible dependency management.
```bash
# Sync environment
uv sync

# Run mathematical validation
uv run pytest