# Eigendecomposition

## Definition

For $A \in \mathbb{R}^{n \times n}$, $\lambda$ is **eigenvalue**, $v \neq 0$ is **eigenvector** if:

$$Av = \lambda v$$

### Finding Eigenvalues

$(A - \lambda I)v = 0$ requires $\det(A - \lambda I) = 0$ (characteristic polynomial).

For $2 \times 2$:

$$\det(A - \lambda I) = \lambda^2 - \text{tr}(A)\lambda + \det(A)$$

General: $\text{tr}(A) = \sum_i \lambda_i$, $\det(A) = \prod_i \lambda_i$

---

## Properties

### General Matrices
- $n$ eigenvalues (with multiplicity), possibly complex
- Distinct eigenvalues → linearly independent eigenvectors

### Symmetric ($A = A^\top$)
- All eigenvalues real
- Distinct eigenvalues → orthogonal eigenvectors
- Always diagonalizable: $A = Q\Lambda Q^\top$, $Q$ orthogonal

---

## Decomposition

**General:** $A = V\Lambda V^{-1}$

**Symmetric:** $A = Q\Lambda Q^\top = \sum_i \lambda_i q_i q_i^\top$ (spectral decomposition)

---

## Applications

### PCA
Covariance $C = \frac{1}{n}X^\top X$:
- Eigenvectors = principal directions
- Eigenvalues = variance per direction

### Gradient Descent
For $L(w) = \frac{1}{2}w^\top Aw - b^\top w$:
- Converges if $\alpha < 2/\lambda_{\max}$
- Condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ → convergence speed

### Positive Definiteness
- PD: all $\lambda_i > 0$ ↔ $x^\top Ax > 0$ $\forall x \neq 0$
- PSD: all $\lambda_i \geq 0$

### Matrix Powers
$A^k = V\Lambda^k V^{-1}$

---

## Computation

```python
eigenvalues, eigenvectors = np.linalg.eigh(A)  # symmetric
# Returns ascending eigenvalues, columns = eigenvectors
```

---

## Key Formulas

| Property | Formula |
|----------|---------|
| Definition | $Av = \lambda v$ |
| Characteristic eq | $\det(A - \lambda I) = 0$ |
| Symmetric | $A = Q\Lambda Q^\top$ |
| Spectral | $A = \sum_i \lambda_i q_i q_i^\top$ |
| Power | $A^k = V\Lambda^k V^{-1}$ |
| Trace | $\text{tr}(A) = \sum_i \lambda_i$ |
| Determinant | $\det(A) = \prod_i \lambda_i$ |
| Inverse | $\lambda(A^{-1}) = 1/\lambda(A)$ |

---

## Exercises

1. $\lambda = 0$ → is $A$ invertible?
2. Prove $\lambda(A^\top) = \lambda(A)$
3. $A$ orthogonal → possible eigenvalues?
4. Implement power iteration
