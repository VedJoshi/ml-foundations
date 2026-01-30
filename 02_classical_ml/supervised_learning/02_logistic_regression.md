# Logistic Regression

## Setup

Binary classification: x ∈ ℝᵈ → y ∈ {0, 1}

Model probability P(y = 1 | x):
```
z = wᵀx + b
p = σ(z) = 1/(1 + e^{-z})
```

---

## Sigmoid Properties

```
σ(z) ∈ (0, 1)
σ(0) = 0.5
σ(-z) = 1 - σ(z)
σ'(z) = σ(z)(1 - σ(z))
```

**From log-odds:**
```
log(p/(1-p)) = wᵀx + b  →  p = σ(wᵀx + b)
```

**Decision boundary:** wᵀx + b = 0 (hyperplane)

---

## Loss: Binary Cross-Entropy

Bernoulli likelihood: P(y|x; w) = p^y(1-p)^{1-y}

Log-likelihood: log L = Σᵢ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]

**BCE loss (negative log-likelihood):**
```
J(w) = -Σᵢ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
```

---

## Gradient Derivation

For single sample with p = σ(z), z = wᵀx:

```
∂L/∂p = (p - y) / [p(1-p)]
∂p/∂z = p(1-p)
∂z/∂w = x
```

Chain rule:
```
∂L/∂w = [(p - y)/p(1-p)] · [p(1-p)] · x = (p - y)x
```

**Full gradient:**
```
∇J = Σᵢ (pᵢ - yᵢ)xᵢ = Xᵀ(p - y)
```

Compare to linear regression: same structure, predictions are σ(Xw) instead of Xw.

---

## Regularization

```
J_reg = J + (λ/2)||w||²
∇J_reg = Xᵀ(p - y) + λw
```

---

## No Closed Form

σ nonlinear → Xᵀ(σ(Xw) - y) = 0 has no closed form.

Use: gradient descent, Newton's method, L-BFGS.

---

## Convexity

Hessian: H = XᵀDX where D = diag(pᵢ(1-pᵢ)) > 0

H is PSD → J convex → any local minimum is global.

---

## Multi-Class: Softmax

```
P(y = k | x) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx)
```

Categorical cross-entropy: L = -Σₖ yₖ log(pₖ)

Gradient: ∂L/∂wₖ = (pₖ - yₖ)x

---

## Numerical Stability

Stable BCE formulation:
```
L = max(z, 0) - yz + log(1 + e^{-|z|})
```

---

## Implementation

```python
class LogisticRegression:
    def __init__(self, lr=0.1, n_iter=1000, reg=0.0):
        self.lr, self.n_iter, self.lambda_ = lr, n_iter, reg

    def _sigmoid(self, z):
        return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0
        for _ in range(self.n_iter):
            p = self._sigmoid(X @ self.w + self.b)
            dw = (1/n) * X.T @ (p - y) + self.lambda_ * self.w
            db = (1/n) * np.sum(p - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self

    def predict(self, X):
        return (self._sigmoid(X @ self.w + self.b) >= 0.5).astype(int)
```

---

## Metrics

| Metric | Formula |
|--------|---------|
| Accuracy | (TP + TN) / Total |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | 2·Precision·Recall / (Precision + Recall) |

---

## Connection to Neural Networks

- Binary output: sigmoid + BCE = logistic regression on last hidden layer
- Multi-class output: softmax + cross-entropy = softmax regression

---

## Exercises

1. Derive gradient (show p(1-p) cancellation)
2. Prove Hessian is PSD
3. Derive softmax gradient
4. Implement, compare to sklearn
