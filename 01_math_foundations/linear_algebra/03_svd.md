# Singular Value Decomposition (SVD)

## Definition

For any $A \in \mathbb{R}^{m \times n}$:

$$A = U\Sigma V^\top$$

- $U \in \mathbb{R}^{m \times m}$ orthogonal (left singular vectors)
- $\Sigma \in \mathbb{R}^{m \times n}$ diagonal, $\sigma_1 \geq \sigma_2 \geq \ldots \geq 0$ (singular values)
- $V \in \mathbb{R}^{n \times n}$ orthogonal (right singular vectors)

**Thin SVD:** $A = U_r \Sigma_r V_r^\top$ where $r = \text{rank}(A)$

---

## Relation to Eigendecomposition

$$A^\top A = V(\Sigma^\top \Sigma)V^\top \quad \Rightarrow \quad V = \text{eigenvectors of } A^\top A$$
$$AA^\top = U(\Sigma \Sigma^\top)U^\top \quad \Rightarrow \quad U = \text{eigenvectors of } AA^\top$$
$$\sigma_i^2 = \text{eigenvalues of } A^\top A$$

---

## Four Fundamental Subspaces

| Subspace | Basis | Dim |
|----------|-------|-----|
| $\text{col}(A)$ | First $r$ cols of $U$ | $r$ |
| $\text{row}(A)$ | First $r$ cols of $V$ | $r$ |
| $\text{null}(A)$ | Last $n-r$ cols of $V$ | $n-r$ |
| $\text{null}(A^\top)$ | Last $m-r$ cols of $U$ | $m-r$ |

---

## Low-Rank Approximation

Eckart-Young: $A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^\top$ is best rank-$k$ approximation.

$$\|A - A_k\|_F = \sqrt{\sigma_{k+1}^2 + \ldots + \sigma_r^2}$$
$$\|A - A_k\|_2 = \sigma_{k+1}$$

---

## Pseudoinverse

$A^+ = V\Sigma^+ U^\top$ (reciprocals of nonzero singular values)

- Overdetermined $Ax = b$: $x^* = A^+ b$ minimizes $\|Ax - b\|^2$
- Underdetermined: $A^+ b$ gives minimum-norm solution

---

## PCA via SVD

For centered $X \in \mathbb{R}^{n \times d}$:

$$X = U\Sigma V^\top$$

- Principal components: columns of $V$
- Variance along PC $i$: $\sigma_i^2/n$
- Covariance: $C = \frac{1}{n}X^\top X = V(\Sigma^2/n)V^\top$
- Projection: $Z = XV_k = U_k\Sigma_k$

---

## Computation

```python
U, s, Vt = np.linalg.svd(A, full_matrices=False)
# Returns V^T, not V
```

---

## Key Formulas

| Item | Formula |
|------|---------|
| Full SVD | $A = U\Sigma V^\top$ |
| Singular values | $\sigma_i = \sqrt{\text{eigenvalue of } A^\top A}$ |
| Rank-$k$ approx | $A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^\top$ |
| Frobenius norm | $\|A\|_F = \sqrt{\sum_i \sigma_i^2}$ |
| Spectral norm | $\|A\|_2 = \sigma_1$ |
| Condition number | $\kappa(A) = \sigma_1/\sigma_r$ |
| Pseudoinverse | $A^+ = V\Sigma^+ U^\top$ |

---

## Modern ML Connections

- **Attention:** Low-rank approximations (Linformer, Performer)
- **Compression:** $W \approx U_k\Sigma_k V_k^\top$ reduces parameters
- **Embeddings:** Matrix factorization methods

---

## Exercises

1. Implement SVD-based PCA
2. Image compression via truncated SVD
3. Implement pseudoinverse, verify least squares
4. Condition number of Hilbert matrix for $n = 5, 10, 15$
