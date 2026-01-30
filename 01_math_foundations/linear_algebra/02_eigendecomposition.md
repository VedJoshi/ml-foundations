# Eigendecomposition

## Definition

For A ∈ ℝⁿˣⁿ, λ is **eigenvalue**, v ≠ 0 is **eigenvector** if:
```
Av = λv
```

### Finding Eigenvalues

(A - λI)v = 0 requires det(A - λI) = 0 (characteristic polynomial).

For 2×2:
```
det(A - λI) = λ² - tr(A)λ + det(A)
```

General: tr(A) = Σλᵢ, det(A) = Πλᵢ

---

## Properties

### General Matrices
- n eigenvalues (with multiplicity), possibly complex
- Distinct eigenvalues → linearly independent eigenvectors

### Symmetric (A = Aᵀ)
- All eigenvalues real
- Distinct eigenvalues → orthogonal eigenvectors
- Always diagonalizable: A = QΛQᵀ, Q orthogonal

---

## Decomposition

**General:** A = VΛV⁻¹

**Symmetric:** A = QΛQᵀ = Σᵢ λᵢqᵢqᵢᵀ (spectral decomposition)

---

## Applications

### PCA
Covariance C = (1/n)XᵀX:
- Eigenvectors = principal directions
- Eigenvalues = variance per direction

### Gradient Descent
For L(w) = (1/2)wᵀAw - bᵀw:
- Converges if α < 2/λ_max
- Condition number κ = λ_max/λ_min → convergence speed

### Positive Definiteness
- PD: all λᵢ > 0 ↔ xᵀAx > 0 ∀x ≠ 0
- PSD: all λᵢ ≥ 0

### Matrix Powers
Aᵏ = VΛᵏV⁻¹

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
| Definition | Av = λv |
| Characteristic eq | det(A - λI) = 0 |
| Symmetric | A = QΛQᵀ |
| Spectral | A = Σᵢ λᵢqᵢqᵢᵀ |
| Power | Aᵏ = VΛᵏV⁻¹ |
| Trace | tr(A) = Σλᵢ |
| Determinant | det(A) = Πλᵢ |
| Inverse | λ(A⁻¹) = 1/λ(A) |

---

## Exercises

1. λ = 0 → is A invertible?
2. Prove λ(Aᵀ) = λ(A)
3. A orthogonal → possible eigenvalues?
4. Implement power iteration
