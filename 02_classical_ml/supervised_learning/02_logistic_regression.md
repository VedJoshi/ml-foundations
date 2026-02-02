# Logistic Regression

## Setup

Binary classification: $x \in \mathbb{R}^d \to y \in \{0, 1\}$

Model probability $P(y = 1 | x)$:

$$z = w^\top x + b$$
$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

---

## Sigmoid Properties

$$\sigma(z) \in (0, 1)$$
$$\sigma(0) = 0.5$$
$$\sigma(-z) = 1 - \sigma(z)$$
$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

**From log-odds:**

$$\log\frac{p}{1-p} = w^\top x + b \quad \Rightarrow \quad p = \sigma(w^\top x + b)$$

**Decision boundary:** $w^\top x + b = 0$ (hyperplane)

---

## Loss: Binary Cross-Entropy

Bernoulli likelihood: $P(y|x; w) = p^y(1-p)^{1-y}$

Log-likelihood: $\log L = \sum_i [y_i \log(p_i) + (1-y_i) \log(1-p_i)]$

**BCE loss (negative log-likelihood):**

$$J(w) = -\sum_i [y_i \log(p_i) + (1-y_i) \log(1-p_i)]$$

---

## Gradient Derivation

For single sample with $p = \sigma(z)$, $z = w^\top x$:

$$\frac{\partial L}{\partial p} = \frac{p - y}{p(1-p)}$$
$$\frac{\partial p}{\partial z} = p(1-p)$$
$$\frac{\partial z}{\partial w} = x$$

Chain rule:

$$\frac{\partial L}{\partial w} = \frac{p - y}{p(1-p)} \cdot p(1-p) \cdot x = (p - y)x$$

**Full gradient:**

$$\nabla J = \sum_i (p_i - y_i)x_i = X^\top(p - y)$$

Compare to linear regression: same structure, predictions are $\sigma(Xw)$ instead of $Xw$.

---

## Regularization

$$J_{\text{reg}} = J + \frac{\lambda}{2}\|w\|^2$$
$$\nabla J_{\text{reg}} = X^\top(p - y) + \lambda w$$

---

## No Closed Form

$\sigma$ nonlinear → $X^\top(\sigma(Xw) - y) = 0$ has no closed form.

Use: gradient descent, Newton's method, L-BFGS.

---

## Convexity

Hessian: $H = X^\top DX$ where $D = \text{diag}(p_i(1-p_i)) > 0$

$H$ is PSD → $J$ convex → any local minimum is global.

---

## Multi-Class: Softmax

$$P(y = k | x) = \frac{\exp(w_k^\top x)}{\sum_j \exp(w_j^\top x)}$$

Categorical cross-entropy: $L = -\sum_k y_k \log(p_k)$

Gradient: $\frac{\partial L}{\partial w_k} = (p_k - y_k)x$

---

## Numerical Stability

Stable BCE formulation:

$$L = \max(z, 0) - yz + \log(1 + e^{-|z|})$$

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
| Accuracy | $(\text{TP} + \text{TN}) / \text{Total}$ |
| Precision | $\text{TP} / (\text{TP} + \text{FP})$ |
| Recall | $\text{TP} / (\text{TP} + \text{FN})$ |
| F1 | $2 \cdot \text{Precision} \cdot \text{Recall} / (\text{Precision} + \text{Recall})$ |

---

## Connection to Neural Networks

- Binary output: sigmoid + BCE = logistic regression on last hidden layer
- Multi-class output: softmax + cross-entropy = softmax regression

---

## Exercises

1. Derive gradient (show $p(1-p)$ cancellation)
2. Prove Hessian is PSD
3. Derive softmax gradient
4. Implement, compare to sklearn
