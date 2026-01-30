# Principal Component Analysis (PCA)

PCA is the most important dimensionality reduction technique. It finds the directions of maximum variance in data and projects onto them. Understanding PCA connects linear algebra (eigendecomposition, SVD) to statistics (variance, covariance) to optimization (maximizing a quadratic form).

---

## The Problem

Given data X ∈ ℝⁿˣᵈ (n samples, d features), find:
1. The directions of maximum variance
2. A lower-dimensional representation Z ∈ ℝⁿˣᵏ (k < d) that preserves as much information as possible

---

## Setup: Centered Data

Assume data is centered (mean subtracted):
```
X ← X - mean(X, axis=0)
```

The sample covariance matrix:
```
C = (1/n) XᵀX ∈ ℝᵈˣᵈ
```

Or with Bessel's correction: C = (1/(n-1)) XᵀX

C is symmetric positive semi-definite.

---

## Derivation 1: Maximum Variance

### Finding the First Principal Component

I want to find direction w ∈ ℝᵈ (unit vector, ||w|| = 1) such that the variance of the projection Xw is maximized.

The projected data is z = Xw ∈ ℝⁿ.

Variance of projection:
```
Var(z) = (1/n) Σᵢ (wᵀxᵢ)² = (1/n) ||Xw||² = (1/n) wᵀXᵀXw = wᵀCw
```

So I want to maximize wᵀCw subject to ||w|| = 1.

### Lagrangian

```
L(w, λ) = wᵀCw - λ(wᵀw - 1)
```

Setting ∂L/∂w = 0:
```
2Cw - 2λw = 0
Cw = λw
```

This is the eigenvalue equation. w must be an eigenvector of C.

### Which Eigenvector?

The variance along w is:
```
wᵀCw = wᵀ(λw) = λ wᵀw = λ
```

To maximize variance, choose the eigenvector with the **largest eigenvalue**.

### Subsequent Components

The k-th principal component maximizes variance among directions orthogonal to the first k-1 components.

Result: The principal components are the eigenvectors of C, ordered by decreasing eigenvalue.

---

## Derivation 2: Minimum Reconstruction Error

Alternative view: Find the k-dimensional subspace that minimizes reconstruction error.

Let W ∈ ℝᵈˣᵏ have orthonormal columns (the subspace basis).
Projection of xᵢ onto subspace: WWᵀxᵢ
Reconstruction error: ||xᵢ - WWᵀxᵢ||²

Total error:
```
E = Σᵢ ||xᵢ - WWᵀxᵢ||²
```

Minimizing this over W gives the same answer: columns of W are the top k eigenvectors of XᵀX.

### Why?

The two objectives are complementary:
- Variance explained = Σⱼ₌₁ᵏ λⱼ
- Reconstruction error = Σⱼ₌ₖ₊₁ᵈ λⱼ

Since total variance = Σⱼ λⱼ is constant, maximizing explained variance = minimizing reconstruction error.

---

## The PCA Procedure

### Via Eigendecomposition

1. Center data: X ← X - mean
2. Compute covariance: C = (1/n)XᵀX
3. Eigendecompose: C = VΛVᵀ (V has eigenvectors as columns)
4. Sort eigenvectors by decreasing eigenvalue
5. Take top k eigenvectors: Vₖ = first k columns of V
6. Project: Z = XVₖ ∈ ℝⁿˣᵏ

### Via SVD (Preferred)

1. Center data: X ← X - mean
2. Compute SVD: X = UΣVᵀ
3. Principal components are columns of V
4. Singular values: σᵢ, eigenvalues of XᵀX are σᵢ²
5. Project: Z = XVₖ = UₖΣₖ (first k columns)

SVD is preferred because it's more numerically stable than forming XᵀX.

---

## Understanding the Output

### Principal Components (V)

The columns of V are the principal component directions:
- v₁: direction of maximum variance
- v₂: direction of maximum variance orthogonal to v₁
- etc.

These form an orthonormal basis for ℝᵈ.

### Projected Data (Z = XV)

Each row zᵢ = Vᵀxᵢ gives the coordinates of xᵢ in the PC basis.

### Eigenvalues (Λ)

λₖ = variance along the k-th principal component.

Proportion of variance explained by first k components:
```
Explained ratio = (Σⱼ₌₁ᵏ λⱼ) / (Σⱼ₌₁ᵈ λⱼ)
```

---

## Choosing the Number of Components

### Variance Threshold

Keep k components such that explained variance ≥ threshold (e.g., 95%):
```
Find smallest k where (Σⱼ₌₁ᵏ λⱼ) / (Σⱼ λⱼ) ≥ 0.95
```

### Scree Plot

Plot eigenvalues vs component index. Look for "elbow" where eigenvalues drop off.

### Cross-Validation

For supervised tasks, choose k that gives best downstream performance.

---

## Implementation

```python
import numpy as np

class PCA:
    """Principal Component Analysis from scratch."""

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        """Fit PCA model.

        X: (n_samples, n_features)
        """
        n, d = X.shape

        # Center data
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # SVD
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Store results
        self.components_ = Vt                    # (d, d) or (k, d)
        self.singular_values_ = s                # (min(n,d),)
        self.explained_variance_ = s**2 / n     # eigenvalues of cov
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()

        # Determine number of components
        if self.n_components is None:
            self.n_components = min(n, d)

        return self

    def transform(self, X):
        """Project data onto principal components.

        X: (n_samples, n_features)
        Returns: (n_samples, n_components)
        """
        X_centered = X - self.mean_
        return X_centered @ self.components_[:self.n_components].T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z):
        """Reconstruct data from reduced representation.

        Z: (n_samples, n_components)
        Returns: (n_samples, n_features)
        """
        return Z @ self.components_[:self.n_components] + self.mean_

    def reconstruction_error(self, X):
        """Compute mean squared reconstruction error."""
        Z = self.transform(X)
        X_reconstructed = self.inverse_transform(Z)
        return np.mean((X - X_reconstructed)**2)
```

---

## Geometric Interpretation

### As Rotation

PCA rotates the coordinate system to align with the principal axes of the data ellipsoid.

Original coordinates → PC coordinates is a rotation (orthogonal transformation).

### As Projection

When we keep k < d components, we project onto a k-dimensional subspace that captures the most variance.

This is the "best" k-dimensional approximation in the least-squares sense.

### Visualizing in 2D

For 2D data, the first PC points along the major axis of the data ellipse, the second PC along the minor axis.

---

## Connection to SVD

For centered X:
```
X = UΣVᵀ

XᵀX = VΣ²Vᵀ (eigendecomposition of XᵀX)
XXᵀ = UΣ²Uᵀ (eigendecomposition of XXᵀ)
```

So:
- V contains right singular vectors = eigenvectors of XᵀX = principal components
- σᵢ² = eigenvalues of XᵀX = n × (variance along i-th PC)
- U contains left singular vectors = eigenvectors of XXᵀ

The projected data:
```
Z = XVₖ = UₖΣₖ
```

---

## When PCA Fails

### Non-Linear Structure

PCA finds linear subspaces. If data lies on a curved manifold, PCA may not capture it well.

Alternatives: kernel PCA, autoencoders, t-SNE, UMAP

### Large Variance ≠ Important

PCA maximizes variance, but high-variance directions may be noise, not signal.

For supervised tasks, consider supervised dimensionality reduction (LDA).

### Outliers

PCA is sensitive to outliers (they inflate variance).

Consider robust PCA or preprocessing to handle outliers.

---

## Connection to Modern ML

### Autoencoders

A linear autoencoder (encoder + decoder, both linear, no activation) learns the same subspace as PCA.

The encoder learns V, the decoder learns Vᵀ.

### Word Embeddings

Word2Vec and GloVe produce embeddings that can be analyzed with PCA to find semantic dimensions.

### Attention Visualization

PCA on attention patterns or hidden states reveals structure in transformer representations.

### Latent Spaces

The "latent space" in VAEs, GANs, etc. is analogous to the low-dimensional PC space.

---

## Exercises

### Conceptual
1. Why must data be centered before PCA?
2. If all eigenvalues are equal, what does this say about the data?
3. How does PCA relate to the SVD of the data matrix?
4. Why is SVD more stable than eigendecomposition of XᵀX?

### Implementations
1. [ ] Implement PCA via eigendecomposition
2. [ ] Implement PCA via SVD
3. [ ] Verify both give the same result
4. [ ] Implement variance explained ratio

### Experiments
1. [ ] Apply PCA to MNIST, visualize in 2D (first 2 PCs)
2. [ ] Plot variance explained vs number of components
3. [ ] Visualize principal component "images" for MNIST
4. [ ] Reconstruct images with different numbers of PCs, observe quality

---

## Key Formulas

| Item | Formula |
|------|---------|
| Covariance matrix | C = (1/n)XᵀX |
| First PC | v₁ = argmax_{||v||=1} vᵀCv = eigenvector for λ_max |
| Variance along v | vᵀCv = λ (if v is eigenvector) |
| Projection | Z = XVₖ |
| Reconstruction | X̂ = ZVₖᵀ + μ |
| Explained variance | Σₖ λₖ / Σ λₖ |
| Reconstruction error | Σⱼ>ₖ λⱼ |

---

## Key Takeaways

1. PCA finds directions of maximum variance
2. Principal components are eigenvectors of covariance matrix
3. Eigenvalues give variance along each component
4. Equivalent to minimum reconstruction error
5. SVD provides stable computation
6. Linear method - fails on non-linear structure
7. Fundamental building block for understanding representation learning

---

## Next Steps

- [ ] Kernel PCA (non-linear extension)
- [ ] Apply PCA as preprocessing before classification
- [ ] Study autoencoders (non-linear generalization)
- [ ] Explore t-SNE/UMAP for visualization
