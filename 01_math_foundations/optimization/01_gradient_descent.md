# Gradient Descent

## Setup

Minimize f: ℝⁿ → ℝ:
```
w* = argmin_w f(w)
```

Gradient: ∇f(w) = [∂f/∂w₁, ..., ∂f/∂wₙ]ᵀ

-∇f(w) points toward steepest descent.

---

## Algorithm

```
w_{t+1} = w_t - α ∇f(w_t)
```
α > 0: learning rate

---

## Convergence: Quadratic Case

For f(w) = (1/2)wᵀAw - bᵀw, A symmetric PD:

∇f(w) = Aw - b, w* = A⁻¹b

Error evolution: e_t = (I - αA)ᵗ e₀

**Convergence condition:** 0 < α < 2/λ_max

**Optimal α:** α* = 2/(λ_max + λ_min)

**Rate:** ||e_t|| ≤ ((κ-1)/(κ+1))ᵗ ||e₀|| where κ = λ_max/λ_min

### General Convex

L-smooth: ||∇f(x) - ∇f(y)|| ≤ L||x - y||

With α = 1/L: f(w_t) - f(w*) = O(1/t)

μ-strongly convex: f(w_t) - f(w*) ≤ (1 - μ/L)ᵗ (f(w₀) - f(w*))

---

## Learning Rate

| Too Small | Too Large |
|-----------|-----------|
| Slow convergence | Divergence |
| Stuck in local minima | Oscillation |

**Strategies:**
- Grid search: 0.001, 0.01, 0.1, 1.0
- Schedules: step decay, exponential decay, 1/t decay
- Line search: Armijo backtracking
- Adaptive: Adam, AdaGrad, RMSprop

---

## SGD Variants

Loss: f(w) = (1/n) Σᵢ fᵢ(w)

| Variant | Gradient | Properties |
|---------|----------|------------|
| Full-batch | (1/n) Σᵢ ∇fᵢ(w) | Exact, expensive |
| Stochastic | ∇fᵢ(w) random i | Cheap, noisy, O(1/√t) |
| Mini-batch | (1/b) Σᵢ∈B ∇fᵢ(w) | Variance ~1/b, GPU parallel |

---

## Zig-Zag Problem

High condition number → gradient points toward contour, not minimum → oscillation.

**Solutions:**
1. Momentum
2. Adaptive learning rates
3. Preconditioning

---

## Linear Regression

For f(w) = ||Xw - y||²:
```
∇f(w) = 2Xᵀ(Xw - y)
w_{t+1} = (I - 2αXᵀX)w_t + 2αXᵀy
```
Convergence depends on eigenvalues of XᵀX.

---

## Exercises

1. Implement GD for f(x) = x², vary α
2. GD for linear regression, compare condition numbers
3. Mini-batch SGD, plot loss curves for different batch sizes
4. Implement backtracking line search
