# Vectors, Matrices, and Linear Maps

## Vectors

v ∈ ℝⁿ: ordered n-tuple, column vector by default.
```
v = [v₁, v₂, ..., vₙ]ᵀ
```

### Operations

| Operation | Formula |
|-----------|---------|
| Addition | u + v = [u₁ + v₁, ..., uₙ + vₙ]ᵀ |
| Scalar mult | αv = [αv₁, ..., αvₙ]ᵀ |
| Inner product | ⟨u, v⟩ = uᵀv = Σᵢ uᵢvᵢ |
| L2 norm | \|\|v\|\|₂ = √(vᵀv) |
| Cosine similarity | cos(θ) = (uᵀv) / (\|\|u\|\| \|\|v\|\|) |

Linear model: ŷ = wᵀx + b (inner product + bias)

---

## Matrices

A ∈ ℝᵐˣⁿ: m rows, n columns.

### Interpretations
1. Column vectors: A = [a₁ | a₂ | ... | aₙ]
2. Row vectors: A = [r₁ᵀ; r₂ᵀ; ...; rₘᵀ]
3. Linear map: f(x) = Ax

### Matrix-Vector Multiplication

y = Ax, A ∈ ℝᵐˣⁿ, x ∈ ℝⁿ, y ∈ ℝᵐ

**Row view:** yᵢ = Σⱼ Aᵢⱼxⱼ

**Column view:** y = Σⱼ xⱼaⱼ (linear combination of columns)

### Matrix-Matrix Multiplication

C = AB, A ∈ ℝᵐˣⁿ, B ∈ ℝⁿˣᵖ, C ∈ ℝᵐˣᵖ
```
Cᵢⱼ = Σₖ AᵢₖBₖⱼ
```
Dimensions: (m×n)(n×p) = (m×p)

---

## Special Matrices

| Type | Definition | Property |
|------|------------|----------|
| Identity | I | Ix = x |
| Diagonal | D = diag(d₁,...,dₙ) | (Dx)ᵢ = dᵢxᵢ |
| Symmetric | A = Aᵀ | aᵢⱼ = aⱼᵢ |
| Orthogonal | QᵀQ = I | Preserves lengths/angles |

---

## Subspaces

**Column space:** col(A) = {Ax : x ∈ ℝⁿ} = span of columns

**Null space:** null(A) = {x : Ax = 0}

**Rank:** rank(A) = dim(col(A))

**Rank-nullity:** rank(A) + dim(null(A)) = n

---

## Inverse

A⁻¹A = AA⁻¹ = I

**Exists iff:**
- det(A) ≠ 0
- rank(A) = n
- null(A) = {0}

**Properties:**
```
(AB)⁻¹ = B⁻¹A⁻¹
(Aᵀ)⁻¹ = (A⁻¹)ᵀ
```

**Computation:** Use `np.linalg.solve(A, b)`, not A⁻¹.

---

## Transpose

(Aᵀ)ᵢⱼ = Aⱼᵢ

```
(AB)ᵀ = BᵀAᵀ
⟨Ax, y⟩ = ⟨x, Aᵀy⟩
```

---

## Exercises

1. Implement matvec multiplication (row and column view)
2. Implement matmul
3. Verify det(A) ≠ 0 ⟹ A⁻¹ exists
