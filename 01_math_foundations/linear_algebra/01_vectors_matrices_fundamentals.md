# Vectors, Matrices, and Linear Maps

## Vectors

$v \in \mathbb{R}^n$: ordered n-tuple, column vector by default.

$$v = [v_1, v_2, \ldots, v_n]^\top$$

### Operations

| Operation | Formula |
|-----------|---------|
| Addition | $u + v = [u_1 + v_1, \ldots, u_n + v_n]^\top$ |
| Scalar mult | $\alpha v = [\alpha v_1, \ldots, \alpha v_n]^\top$ |
| Inner product | $\langle u, v \rangle = u^\top v = \sum_i u_i v_i$ |
| L2 norm | $\|v\|_2 = \sqrt{v^\top v}$ |
| Cosine similarity | $\cos(\theta) = \frac{u^\top v}{\|u\| \|v\|}$ |

Linear model: $\hat{y} = w^\top x + b$ (inner product + bias)

---

## Matrices

$A \in \mathbb{R}^{m \times n}$: m rows, n columns.

### Interpretations
1. Column vectors: $A = [a_1 | a_2 | \ldots | a_n]$
2. Row vectors: $A = [r_1^\top; r_2^\top; \ldots; r_m^\top]$
3. Linear map: $f(x) = Ax$

### Matrix-Vector Multiplication

$y = Ax$, $A \in \mathbb{R}^{m \times n}$, $x \in \mathbb{R}^n$, $y \in \mathbb{R}^m$

**Row view:** $y_i = \sum_j A_{ij} x_j$

**Column view:** $y = \sum_j x_j a_j$ (linear combination of columns)

### Matrix-Matrix Multiplication

$C = AB$, $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times p}$, $C \in \mathbb{R}^{m \times p}$

$$C_{ij} = \sum_k A_{ik} B_{kj}$$

Dimensions: $(m \times n)(n \times p) = (m \times p)$

---

## Special Matrices

| Type | Definition | Property |
|------|------------|----------|
| Identity | $I$ | $Ix = x$ |
| Diagonal | $D = \text{diag}(d_1, \ldots, d_n)$ | $(Dx)_i = d_i x_i$ |
| Symmetric | $A = A^\top$ | $a_{ij} = a_{ji}$ |
| Orthogonal | $Q^\top Q = I$ | Preserves lengths/angles |

---

## Subspaces

**Column space:** $\text{col}(A) = \{Ax : x \in \mathbb{R}^n\}$ = span of columns

**Null space:** $\text{null}(A) = \{x : Ax = 0\}$

**Rank:** $\text{rank}(A) = \dim(\text{col}(A))$

**Rank-nullity:** $\text{rank}(A) + \dim(\text{null}(A)) = n$

---

## Inverse

$A^{-1}A = AA^{-1} = I$

**Exists iff:**
- $\det(A) \neq 0$
- $\text{rank}(A) = n$
- $\text{null}(A) = \{0\}$

**Properties:**

$$(AB)^{-1} = B^{-1}A^{-1}$$
$$(A^\top)^{-1} = (A^{-1})^\top$$

**Computation:** Use `np.linalg.solve(A, b)`, not $A^{-1}$.

---

## Transpose

$(A^\top)_{ij} = A_{ji}$

$$(AB)^\top = B^\top A^\top$$
$$\langle Ax, y \rangle = \langle x, A^\top y \rangle$$

---

## Exercises

1. Implement matvec multiplication (row and column view)
2. Implement matmul
3. Verify $\det(A) \neq 0 \Rightarrow A^{-1}$ exists
