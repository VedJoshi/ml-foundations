# Singular Value Decomposition (SVD)

## Definition

For any A ∈ ℝᵐˣⁿ:
```
A = UΣVᵀ
```
- U ∈ ℝᵐˣᵐ orthogonal (left singular vectors)
- Σ ∈ ℝᵐˣⁿ diagonal, σ₁ ≥ σ₂ ≥ ... ≥ 0 (singular values)
- V ∈ ℝⁿˣⁿ orthogonal (right singular vectors)

**Thin SVD:** A = UᵣΣᵣVᵣᵀ where r = rank(A)

---

## Relation to Eigendecomposition

```
AᵀA = V(ΣᵀΣ)Vᵀ  →  V = eigenvectors of AᵀA
AAᵀ = U(ΣΣᵀ)Uᵀ  →  U = eigenvectors of AAᵀ
σᵢ² = eigenvalues of AᵀA
```

---

## Four Fundamental Subspaces

| Subspace | Basis | Dim |
|----------|-------|-----|
| col(A) | First r cols of U | r |
| row(A) | First r cols of V | r |
| null(A) | Last n-r cols of V | n-r |
| null(Aᵀ) | Last m-r cols of U | m-r |

---

## Low-Rank Approximation

Eckart-Young: Aₖ = Σᵢ₌₁ᵏ σᵢuᵢvᵢᵀ is best rank-k approximation.

```
||A - Aₖ||_F = √(σₖ₊₁² + ... + σᵣ²)
||A - Aₖ||_2 = σₖ₊₁
```

---

## Pseudoinverse

A⁺ = VΣ⁺Uᵀ (reciprocals of nonzero singular values)

- Overdetermined Ax = b: x* = A⁺b minimizes ||Ax - b||²
- Underdetermined: A⁺b gives minimum-norm solution

---

## PCA via SVD

For centered X ∈ ℝⁿˣᵈ:
```
X = UΣVᵀ
```
- Principal components: columns of V
- Variance along PC i: σᵢ²/n
- Covariance: C = (1/n)XᵀX = V(Σ²/n)Vᵀ
- Projection: Z = XVₖ = UₖΣₖ

---

## Computation

```python
U, s, Vt = np.linalg.svd(A, full_matrices=False)
# Returns Vᵀ, not V
```

---

## Key Formulas

| Item | Formula |
|------|---------|
| Full SVD | A = UΣVᵀ |
| Singular values | σᵢ = √(eigenvalue of AᵀA) |
| Rank-k approx | Aₖ = Σᵢ₌₁ᵏ σᵢuᵢvᵢᵀ |
| Frobenius norm | \|\|A\|\|_F = √(Σσᵢ²) |
| Spectral norm | \|\|A\|\|_2 = σ₁ |
| Condition number | κ(A) = σ₁/σᵣ |
| Pseudoinverse | A⁺ = VΣ⁺Uᵀ |

---

## Modern ML Connections

- **Attention:** Low-rank approximations (Linformer, Performer)
- **Compression:** W ≈ UₖΣₖVₖᵀ reduces parameters
- **Embeddings:** Matrix factorization methods

---

## Exercises

1. Implement SVD-based PCA
2. Image compression via truncated SVD
3. Implement pseudoinverse, verify least squares
4. Condition number of Hilbert matrix for n = 5, 10, 15
