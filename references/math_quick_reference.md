# Math Quick Reference for ML

## Linear Algebra

### Matrix Derivatives (from Matrix Cookbook)

| Expression | Derivative |
|------------|------------|
| d/dx (a'x) | a |
| d/dx (x'Ax) | (A + A')x |
| d/dX tr(AX) | A' |
| d/dX tr(X'AX) | (A + A')X |

### Key Identities

- (AB)' = B'A'
- (AB)^-1 = B^-1 A^-1
- det(AB) = det(A)det(B)
- tr(ABC) = tr(CAB) = tr(BCA)

### SVD Properties

For A = U Σ V':
- Columns of U: eigenvectors of AA'
- Columns of V: eigenvectors of A'A
- σ_i^2: eigenvalues of both AA' and A'A
- Best rank-k approximation: A_k = U_k Σ_k V_k'

---

## Calculus

### Chain Rule (Vector Form)

For f: R^n -> R composed with g: R^m -> R^n:
df/dx = df/dg * dg/dx

### Common Gradients in ML

| Function | Gradient |
|----------|----------|
| L = ||Xw - y||^2 (MSE) | ∇_w L = 2X'(Xw - y) |
| L = BCE loss | ∇_w L = X'(p - y) |
| L = Cross-entropy | ∇_w L = X'(p - y) |

### Normal Equations

For linear regression min_w ||Xw - y||^2:
w* = (X'X)^-1 X'y

---

## Probability

### Bayes' Rule

P(θ|D) = P(D|θ) P(θ) / P(D)

### Common Distributions

| Distribution | Mean | Variance |
|--------------|------|----------|
| Bernoulli(p) | p | p(1-p) |
| Gaussian(μ, σ²) | μ | σ² |
| Poisson(λ) | λ | λ |

### MLE for Common Distributions

- Bernoulli: p_hat = (1/n) Σ x_i
- Gaussian: μ_hat = (1/n) Σ x_i, σ²_hat = (1/n) Σ(x_i - μ_hat)²

---

## Optimization

### Gradient Descent Variants

**Vanilla GD:**
w_{t+1} = w_t - α ∇L(w_t)

**Momentum:**
v_{t+1} = β v_t + ∇L(w_t)
w_{t+1} = w_t - α v_{t+1}

**Adam:**
m_t = β_1 m_{t-1} + (1-β_1)∇L
v_t = β_2 v_{t-1} + (1-β_2)(∇L)²
w_t = w_{t-1} - α m_hat_t / (sqrt(v_hat_t) + ε)

---

## Neural Networks

### Activation Functions

| Name | f(x) | f'(x) |
|------|------|-------|
| Sigmoid | 1/(1+e^-x) | σ(x)(1-σ(x)) |
| Tanh | tanh(x) | 1 - tanh²(x) |
| ReLU | max(0, x) | 1 if x>0 else 0 |

### Backpropagation

For layer l: z^l = W^l a^{l-1} + b^l, a^l = σ(z^l)

δ^L = ∇_a L ⊙ σ'(z^L)
δ^l = (W^{l+1})' δ^{l+1} ⊙ σ'(z^l)
∇_{W^l} L = δ^l (a^{l-1})'
∇_{b^l} L = δ^l

---

## Information Theory

- Entropy: H(X) = -Σ p(x) log p(x)
- Cross-Entropy: H(p, q) = -Σ p(x) log q(x)
- KL Divergence: D_KL(p || q) = Σ p(x) log(p(x)/q(x))
