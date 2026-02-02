# Principal Component Analysis (PCA)

Dimensionality reduction via projection onto directions of maximum variance.

---

## Problem

Given data $X \in \mathbb{R}^{n \times d}$ ($n$ samples, $d$ features):
1. Find directions of maximum variance
2. Project onto $k$-dimensional subspace ($k < d$) preserving maximum information

---

## Setup

Center the data:

$$X \leftarrow X - \text{mean}(X, \text{axis}=0)$$

Sample covariance matrix:

$$C = \frac{1}{n} X^\top X \in \mathbb{R}^{d \times d}$$

$C$ is symmetric positive semi-definite.

---

## Derivation 1: Maximum Variance

### First Principal Component

Find unit vector $w \in \mathbb{R}^d$ maximizing variance of projection $Xw$.

The projected data is $z = Xw \in \mathbb{R}^n$.

Variance of projection:

$$\text{Var}(z) = \frac{1}{n} \sum_i (w^\top x_i)^2 = \frac{1}{n} \|Xw\|^2 = w^\top Cw$$

Maximize $w^\top Cw$ subject to $\|w\| = 1$.

### Lagrangian

$$L(w, \lambda) = w^\top Cw - \lambda(w^\top w - 1)$$

where $\lambda$ is the Lagrange multiplier for the unit norm constraint.

Setting $\partial L/\partial w = 0$:

$$2Cw - 2\lambda w = 0$$
$$Cw = \lambda w$$

$w$ must be an eigenvector of $C$ with eigenvalue $\lambda$.

### Which Eigenvector?

Variance along $w$:

$$w^\top Cw = w^\top(\lambda w) = \lambda w^\top w = \lambda$$

Maximum variance ⟹ eigenvector with largest eigenvalue.

### Subsequent Components

The $k$-th principal component maximizes variance among directions orthogonal to the first $k-1$ components.

Result: Principal components are eigenvectors of $C$, ordered by decreasing eigenvalue.

---

## Derivation 2: Minimum Reconstruction Error

Let $W \in \mathbb{R}^{d \times k}$ have orthonormal columns (the subspace basis).
- Projection of $x_i$ onto subspace: $WW^\top x_i$
- Reconstruction error: $\|x_i - WW^\top x_i\|^2$

Total reconstruction error:

$$E = \sum_i \|x_i - WW^\top x_i\|^2$$

Minimizing $E$ over $W$ gives columns of $W$ as top $k$ eigenvectors of $X^\top X$.

### Equivalence

- Variance explained = $\sum_{j=1}^{k} \lambda_j$
- Reconstruction error = $\sum_{j=k+1}^{d} \lambda_j$

Total variance = $\sum_j \lambda_j$ is constant, so maximizing explained variance = minimizing reconstruction error.

---

## PCA Procedure

### Via Eigendecomposition

1. Center data: $X \leftarrow X - \text{mean}$
2. Compute covariance: $C = \frac{1}{n}X^\top X$
3. Eigendecompose: $C = V\Lambda V^\top$ where $V$ has eigenvectors as columns
4. Sort eigenvectors by decreasing eigenvalue
5. Take top $k$ eigenvectors: $V_k$ = first $k$ columns of $V$
6. Project: $Z = XV_k \in \mathbb{R}^{n \times k}$

### Via SVD (Preferred)

1. Center data: $X \leftarrow X - \text{mean}$
2. Compute SVD: $X = U\Sigma V^\top$ where $U \in \mathbb{R}^{n \times n}$, $\Sigma$ diagonal, $V \in \mathbb{R}^{d \times d}$
3. Principal components are columns of $V$
4. Singular values $\sigma_i$ relate to eigenvalues: $\lambda_i = \sigma_i^2/n$
5. Project: $Z = XV_k = U_k\Sigma_k$ (first $k$ columns)

SVD is more numerically stable than forming $X^\top X$ explicitly.

---

## Output Interpretation

### Principal Components ($V$)
- $v_1$: direction of maximum variance
- $v_2$: direction of maximum variance orthogonal to $v_1$
- Form an orthonormal basis for $\mathbb{R}^d$

### Projected Data ($Z = XV$)
Each row $z_i = V^\top x_i$ gives coordinates of $x_i$ in the PC basis.

### Eigenvalues ($\Lambda$)
$\lambda_k$ = variance along $k$-th principal component.

Explained variance ratio:

$$\frac{\sum_{j=1}^{k} \lambda_j}{\sum_{j=1}^{d} \lambda_j}$$

---

## Choosing Number of Components

### Variance Threshold
Keep smallest $k$ such that explained variance $\geq$ threshold (e.g., 95%).

### Scree Plot
Plot eigenvalues vs component index. Look for "elbow" where eigenvalues drop off.

### Cross-Validation
For supervised tasks, choose $k$ by downstream performance.

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

        # SVD: X = UΣV^T
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

When we keep $k < d$ components, we project onto a $k$-dimensional subspace that captures the most variance.

This is the "best" $k$-dimensional approximation in the least-squares sense.

### Visualizing in 2D

For 2D data, the first PC points along the major axis of the data ellipse, the second PC along the minor axis.

---

## Connection to SVD

For centered $X = U\Sigma V^\top$:

$$X^\top X = V\Sigma^2 V^\top \quad \text{(eigendecomposition of } X^\top X\text{)}$$
$$XX^\top = U\Sigma^2 U^\top \quad \text{(eigendecomposition of } XX^\top\text{)}$$

- $V$: right singular vectors = eigenvectors of $X^\top X$ = principal components
- $\sigma_i^2$ = eigenvalues of $X^\top X$ = $n \times$ (variance along $i$-th PC)
- $U$: left singular vectors = eigenvectors of $XX^\top$
- Projected data: $Z = XV_k = U_k\Sigma_k$

---

## Limitations

- **Non-linear structure:** PCA finds linear subspaces. Alternatives: kernel PCA, autoencoders, t-SNE, UMAP.
- **Variance ≠ importance:** High-variance directions may be noise. For classification, consider supervised methods (LDA).
- **Outliers:** PCA is sensitive to outliers (they inflate variance). Consider robust PCA.

---

## Connection to Modern ML

- **Linear autoencoders:** A linear autoencoder learns the same subspace as PCA (encoder learns $V$, decoder learns $V^\top$).
- **Word embeddings:** PCA on Word2Vec/GloVe embeddings reveals semantic dimensions.
- **Attention visualization:** PCA on attention patterns or hidden states reveals structure in transformer representations.
- **Latent spaces:** VAE/GAN latent spaces are analogous to the low-dimensional PC space.

---

## Key Formulas

| Item | Formula |
|------|---------|
| Covariance matrix | $C = \frac{1}{n}X^\top X$ |
| First PC | $v_1 = \arg\max_{\|v\|=1} v^\top Cv$ = eigenvector for $\lambda_{\max}$ |
| Variance along $v$ | $v^\top Cv = \lambda$ (if $v$ is eigenvector) |
| Projection | $Z = XV_k$ |
| Reconstruction | $\hat{X} = ZV_k^\top + \mu$ |
| Explained variance | $\sum_k \lambda_k / \sum \lambda_k$ |
| Reconstruction error | $\sum_{j>k} \lambda_j$ |

---

## Exercises

### Conceptual
1. Why must data be centered before PCA?
2. If all eigenvalues are equal, what does this imply about the data?
3. Why is SVD more stable than eigendecomposition of $X^\top X$?

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
