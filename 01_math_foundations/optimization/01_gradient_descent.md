# Gradient Descent

## Setup

Minimize $f: \mathbb{R}^n \to \mathbb{R}$:

$$w^* = \arg\min_w f(w)$$

Gradient: $\nabla f(w) = \left[\frac{\partial f}{\partial w_1}, \ldots, \frac{\partial f}{\partial w_n}\right]^\top$

$-\nabla f(w)$ points toward steepest descent.

---

## Algorithm

$$w_{t+1} = w_t - \alpha \nabla f(w_t)$$

$\alpha > 0$: learning rate

---

## Convergence: Quadratic Case

For $f(w) = \frac{1}{2}w^\top Aw - b^\top w$, $A$ symmetric PD:

$\nabla f(w) = Aw - b$, $w^* = A^{-1}b$

Error evolution: $e_t = (I - \alpha A)^t e_0$

**Convergence condition:** $0 < \alpha < 2/\lambda_{\max}$

**Optimal $\alpha$:** $\alpha^* = 2/(\lambda_{\max} + \lambda_{\min})$

**Rate:** $\|e_t\| \leq \left(\frac{\kappa-1}{\kappa+1}\right)^t \|e_0\|$ where $\kappa = \lambda_{\max}/\lambda_{\min}$

### General Convex

$L$-smooth: $\|\nabla f(x) - \nabla f(y)\| \leq L\|x - y\|$

With $\alpha = 1/L$: $f(w_t) - f(w^*) = O(1/t)$

$\mu$-strongly convex: $f(w_t) - f(w^*) \leq (1 - \mu/L)^t (f(w_0) - f(w^*))$

---

## Learning Rate

| Too Small | Too Large |
|-----------|-----------|
| Slow convergence | Divergence |
| Stuck in local minima | Oscillation |

**Strategies:**
- Grid search: 0.001, 0.01, 0.1, 1.0
- Schedules: step decay, exponential decay, $1/t$ decay
- Line search: Armijo backtracking
- Adaptive: Adam, AdaGrad, RMSprop

---

## SGD Variants

Loss: $f(w) = \frac{1}{n} \sum_i f_i(w)$

| Variant | Gradient | Properties |
|---------|----------|------------|
| Full-batch | $\frac{1}{n} \sum_i \nabla f_i(w)$ | Exact, expensive |
| Stochastic | $\nabla f_i(w)$ random $i$ | Cheap, noisy, $O(1/\sqrt{t})$ |
| Mini-batch | $\frac{1}{b} \sum_{i \in B} \nabla f_i(w)$ | Variance $\sim 1/b$, GPU parallel |

---

## Zig-Zag Problem

High condition number → gradient points toward contour, not minimum → oscillation.

**Solutions:**
1. Momentum
2. Adaptive learning rates
3. Preconditioning

---

## Linear Regression

For $f(w) = \|Xw - y\|^2$:

$$\nabla f(w) = 2X^\top(Xw - y)$$
$$w_{t+1} = (I - 2\alpha X^\top X)w_t + 2\alpha X^\top y$$

Convergence depends on eigenvalues of $X^\top X$.

---

## Exercises

1. Implement GD for $f(x) = x^2$, vary $\alpha$
2. GD for linear regression, compare condition numbers
3. Mini-batch SGD, plot loss curves for different batch sizes
4. Implement backtracking line search
