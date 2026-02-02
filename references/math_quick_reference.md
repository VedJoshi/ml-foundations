# Math Quick Reference for ML

## Linear Algebra

### Matrix Derivatives (from Matrix Cookbook)

| Expression | Derivative |
|------------|------------|
| $\frac{d}{dx}(a^\top x)$ | $a$ |
| $\frac{d}{dx}(x^\top A x)$ | $(A + A^\top)x$ |
| $\frac{d}{dX}\text{tr}(AX)$ | $A^\top$ |
| $\frac{d}{dX}\text{tr}(X^\top AX)$ | $(A + A^\top)X$ |

### Key Identities

- $(AB)^\top = B^\top A^\top$
- $(AB)^{-1} = B^{-1} A^{-1}$
- $\det(AB) = \det(A)\det(B)$
- $\text{tr}(ABC) = \text{tr}(CAB) = \text{tr}(BCA)$

### SVD Properties

For $A = U \Sigma V^\top$:
- Columns of $U$: eigenvectors of $AA^\top$
- Columns of $V$: eigenvectors of $A^\top A$
- $\sigma_i^2$: eigenvalues of both $AA^\top$ and $A^\top A$
- Best rank-$k$ approximation: $A_k = U_k \Sigma_k V_k^\top$

---

## Calculus

### Chain Rule (Vector Form)

For $f: \mathbb{R}^n \to \mathbb{R}$ composed with $g: \mathbb{R}^m \to \mathbb{R}^n$:

$$\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

### Common Gradients in ML

| Function | Gradient |
|----------|----------|
| $L = \|Xw - y\|^2$ (MSE) | $\nabla_w L = 2X^\top(Xw - y)$ |
| $L = $ BCE loss | $\nabla_w L = X^\top(p - y)$ |
| $L = $ Cross-entropy | $\nabla_w L = X^\top(p - y)$ |

### Normal Equations

For linear regression $\min_w \|Xw - y\|^2$:

$$w^* = (X^\top X)^{-1} X^\top y$$

---

## Probability

### Bayes' Rule

$$P(\theta|D) = \frac{P(D|\theta) P(\theta)}{P(D)}$$

### Common Distributions

| Distribution | Mean | Variance |
|--------------|------|----------|
| $\text{Bernoulli}(p)$ | $p$ | $p(1-p)$ |
| $\mathcal{N}(\mu, \sigma^2)$ | $\mu$ | $\sigma^2$ |
| $\text{Poisson}(\lambda)$ | $\lambda$ | $\lambda$ |

### MLE for Common Distributions

- Bernoulli: $\hat{p} = \frac{1}{n} \sum_i x_i$
- Gaussian: $\hat{\mu} = \frac{1}{n} \sum_i x_i$, $\hat{\sigma}^2 = \frac{1}{n} \sum_i (x_i - \hat{\mu})^2$

---

## Optimization

### Gradient Descent Variants

**Vanilla GD:**
$$w_{t+1} = w_t - \alpha \nabla L(w_t)$$

**Momentum:**
$$v_{t+1} = \beta v_t + \nabla L(w_t)$$
$$w_{t+1} = w_t - \alpha v_{t+1}$$

**Adam:**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L)^2$$
$$w_t = w_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

---

## Neural Networks

### Activation Functions

| Name | $f(x)$ | $f'(x)$ |
|------|--------|---------|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ |
| Tanh | $\tanh(x)$ | $1 - \tanh^2(x)$ |
| ReLU | $\max(0, x)$ | $\mathbf{1}_{x>0}$ |

### Backpropagation

For layer $l$: $z^l = W^l a^{l-1} + b^l$, $a^l = \sigma(z^l)$

$$\delta^L = \nabla_a L \odot \sigma'(z^L)$$
$$\delta^l = (W^{l+1})^\top \delta^{l+1} \odot \sigma'(z^l)$$
$$\nabla_{W^l} L = \delta^l (a^{l-1})^\top$$
$$\nabla_{b^l} L = \delta^l$$

---

## Information Theory

- **Entropy:** $H(X) = -\sum_x p(x) \log p(x)$
- **Cross-Entropy:** $H(p, q) = -\sum_x p(x) \log q(x)$
- **KL Divergence:** $D_{KL}(p \| q) = \sum_x p(x) \log\frac{p(x)}{q(x)}$
