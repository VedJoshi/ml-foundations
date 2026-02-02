# Principal Component Analysis (PCA)

Dimensionality reduction via projection onto directions of maximum variance.

---

## Problem

Given data X ∈ ℝⁿˣᵈ (n samples, d features):
1. Find directions of maximum variance
2. Project onto k-dimensional subspace (k < d) preserving maximum information

---

## Setup

Center the data:
```
X ← X - mean(X, axis=0)
```

Sample covariance matrix:
```
C = (1/n) XᵀX ∈ ℝᵈˣᵈ
```

C is symmetric positive semi-definite.

---

## Derivation 1: Maximum Variance

### First Principal Component

Find unit vector w ∈ ℝᵈ maximizing variance of projection Xw.

The projected data is z = Xw ∈ ℝⁿ.

Variance of projection:
```
Var(z) = (1/n) Σᵢ (wᵀxᵢ)² = (1/n) ||Xw||² = wᵀCw
```

Maximize wᵀCw subject to ||w|| = 1.

### Lagrangian

```
L(w, λ) = wᵀCw - λ(wᵀw - 1)
```

where λ is the Lagrange multiplier for the unit norm constraint.

Setting ∂L/∂w = 0:
```
2Cw - 2λw = 0
Cw = λw
```

w must be an eigenvector of C with eigenvalue λ.

### Which Eigenvector?

Variance along w:
```
wᵀCw = wᵀ(λw) = λ wᵀw = λ
```

Maximum variance ⟹ eigenvector with largest eigenvalue.

### Subsequent Components

The k-th principal component maximizes variance among directions orthogonal to the first k-1 components.

Result: Principal components are eigenvectors of C, ordered by decreasing eigenvalue.

---

## Derivation 2: Minimum Reconstruction Error

Let W ∈ ℝᵈˣᵏ have orthonormal columns (the subspace basis).
- Projection of xᵢ onto subspace: WWᵀxᵢ
- Reconstruction error: ||xᵢ - WWᵀxᵢ||²

Total reconstruction error:
```
E = Σᵢ ||xᵢ - WWᵀxᵢ||²
```

Minimizing E over W gives columns of W as top k eigenvectors of XᵀX.

### Equivalence

- Variance explained = Σⱼ₌₁ᵏ λⱼ
- Reconstruction error = Σⱼ₌ₖ₊₁ᵈ λⱼ

Total variance = Σⱼ λⱼ is constant, so maximizing explained variance = minimizing reconstruction error.

---

## PCA Procedure

### Via Eigendecomposition

1. Center data: X ← X - mean
2. Compute covariance: C = (1/n)XᵀX
3. Eigendecompose: C = VΛVᵀ where V has eigenvectors as columns
4. Sort eigenvectors by decreasing eigenvalue
5. Take top k eigenvectors: Vₖ = first k columns of V
6. Project: Z = XVₖ ∈ ℝⁿˣᵏ

### Via SVD (Preferred)

1. Center data: X ← X - mean
2. Compute SVD: X = UΣVᵀ where U ∈ ℝⁿˣⁿ, Σ diagonal, V ∈ ℝᵈˣᵈ
3. Principal components are columns of V
4. Singular values σᵢ relate to eigenvalues: λᵢ = σᵢ²/n
5. Project: Z = XVₖ = UₖΣₖ (first k columns)

SVD is more numerically stable than forming XᵀX explicitly.

---

## Output Interpretation

### Principal Components (V)
- v₁: direction of maximum variance
- v₂: direction of maximum variance orthogonal to v₁
- Form an orthonormal basis for ℝᵈ

### Projected Data (Z = XV)
Each row zᵢ = Vᵀxᵢ gives coordinates of xᵢ in the PC basis.

### Eigenvalues (Λ)
λₖ = variance along k-th principal component.

Explained variance ratio:
```
(Σⱼ₌₁ᵏ λⱼ) / (Σⱼ₌₁ᵈ λⱼ)
```

---

## Choosing Number of Components

### Variance Threshold
Keep smallest k such that explained variance ≥ threshold (e.g., 95%).

### Scree Plot
Plot eigenvalues vs component index. Look for "elbow" where eigenvalues drop off.

### Cross-Validation
For supervised tasks, choose k by downstream performance.

---

## Implementation

```python
import numpy as np

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        """
        X: (n_samples, n_features)
        """
        n, d = X.shape

        # Center data
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # SVD: X = UΣVᵀ
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Store results
        self.components_ = Vt                         # (d, d) principal components
        self.singular_values_ = s                     # singular values
        self.explained_variance_ = s**2 / n           # eigenvalues of covariance
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()

        if self.n_components is None:
            self.n_components = min(n, d)

        return self

    def transform(self, X):
        """
        X: (n_samples, n_features)
        Returns: (n_samples, n_components)
        """
        X_centered = X - self.mean_
        return X_centered @ self.components_[:self.n_components].T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z):
        """
        Z: (n_samples, n_components)
        Returns: (n_samples, n_features) reconstructed data
        """
        return Z @ self.components_[:self.n_components] + self.mean_

    def reconstruction_error(self, X):
        """Mean squared reconstruction error."""
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

For centered X = UΣVᵀ:
```
XᵀX = VΣ²Vᵀ (eigendecomposition of XᵀX)
XXᵀ = UΣ²Uᵀ (eigendecomposition of XXᵀ)
```

- V: right singular vectors = eigenvectors of XᵀX = principal components
- σᵢ² = eigenvalues of XᵀX = n × (variance along i-th PC)
- U: left singular vectors = eigenvectors of XXᵀ
- Projected data: Z = XVₖ = UₖΣₖ

---

## Geometric Interpretation

- PCA rotates coordinates to align with principal axes of data ellipsoid
- Original → PC coordinates is an orthogonal transformation (rotation)
- Keeping k < d components projects onto best k-dimensional subspace in least-squares sense
- For 2D data: first PC = major axis of ellipse, second PC = minor axis

---

## Limitations

- **Non-linear structure:** PCA finds linear subspaces. Alternatives: kernel PCA, autoencoders, t-SNE, UMAP.
- **Variance ≠ importance:** High-variance directions may be noise. For classification, consider supervised methods (LDA).
- **Outliers:** PCA is sensitive to outliers (they inflate variance). Consider robust PCA.

---

## Connection to Modern ML

- **Linear autoencoders:** A linear autoencoder learns the same subspace as PCA (encoder learns V, decoder learns Vᵀ).
- **Word embeddings:** PCA on Word2Vec/GloVe embeddings reveals semantic dimensions.
- **Attention visualization:** PCA on attention patterns or hidden states reveals structure in transformer representations.
- **Latent spaces:** VAE/GAN latent spaces are analogous to the low-dimensional PC space.

---

## Key Formulas

| Item | Formula |
|------|---------|
| Covariance matrix | C = (1/n)XᵀX |
| First PC | v₁ = argmax_{‖v‖=1} vᵀCv = eigenvector for λ_max |
| Variance along v | vᵀCv = λ (if v is eigenvector) |
| Projection | Z = XVₖ |
| Reconstruction | X̂ = ZVₖᵀ + μ |
| Explained variance | Σₖ λₖ / Σ λₖ |
| Reconstruction error | Σⱼ>ₖ λⱼ |

---

## Exercises

### Conceptual
1. Why must data be centered before PCA?
2. If all eigenvalues are equal, what does this imply about the data?
3. Why is SVD more stable than eigendecomposition of XᵀX?

### Implementations
1. [ ] Implement PCA via eigendecomposition
2. [ ] Implement PCA via SVD
3. [ ] Verify both give the same result
4. [ ] Implement variance explained ratio

### Experiments
1. [ ] Apply PCA to MNIST, visualize in 2D
2. [ ] Plot variance explained vs number of components
3. [ ] Visualize principal component "images" for MNIST
4. [ ] Reconstruct images with different numbers of PCs
