# Linear Regression

## Setup

Data: {(x₁, y₁), ..., (xₙ, yₙ)}, xᵢ ∈ ℝᵈ, yᵢ ∈ ℝ

Model: ŷ = wᵀx + b

Design matrix (with bias column): X ∈ ℝⁿˣ⁽ᵈ⁺¹⁾, predictions: ŷ = Xw

---

## Loss: MSE

```
J(w) = ||y - Xw||² = (y - Xw)ᵀ(y - Xw)
     = yᵀy - 2wᵀXᵀy + wᵀXᵀXw
```

---

## Normal Equations

∇J = -2Xᵀy + 2XᵀXw = 0

```
XᵀXw = Xᵀy
w* = (XᵀX)⁻¹Xᵀy
```

**Geometric interpretation:** Residual r = y - Xw* orthogonal to col(X): Xᵀr = 0

**(XᵀX)⁻¹ exists iff:** X has full column rank (n ≥ d, no multicollinearity)

---

## Gradient Descent

```
w_{t+1} = w_t - 2α Xᵀ(Xw_t - y)
```

Per-sample gradient: ∇(yᵢ - wᵀxᵢ)² = 2(ŷᵢ - yᵢ)xᵢ

---

## Regularization

### Ridge (L2)
```
J_ridge = ||y - Xw||² + λ||w||²
w* = (XᵀX + λI)⁻¹Xᵀy
∇J = 2Xᵀ(Xw - y) + 2λw
```

### Lasso (L1)
```
J_lasso = ||y - Xw||² + λ||w||₁
```
No closed form. Promotes sparsity.

### Elastic Net
```
J = ||y - Xw||² + λ₁||w||₁ + λ₂||w||²
```

---

## Bias-Variance Tradeoff

```
E[(y - ŷ)²] = Bias²(ŷ) + Var(ŷ) + σ²
```

| Complexity | Bias | Variance |
|------------|------|----------|
| Low | High | Low |
| High | Low | High |

Regularization: ↑bias, ↓variance

---

## Polynomial Regression

Feature expansion: x → φ(x) = [1, x, x², ..., xᵖ]

Model: ŷ = wᵀφ(x)

High degree → overfitting. Use regularization + validation.

---

## MLE Connection

Under y|x ~ N(wᵀx, σ²):
```
log L(w) = const - (1/(2σ²)) Σᵢ(yᵢ - wᵀxᵢ)²
```
max log L = min MSE

---

## Metrics

| Metric | Formula |
|--------|---------|
| R² | 1 - Σ(yᵢ - ŷᵢ)²/Σ(yᵢ - ȳ)² |
| RMSE | √((1/n)Σ(yᵢ - ŷᵢ)²) |
| MAE | (1/n)Σ\|yᵢ - ŷᵢ\| |

---

## Implementation

```python
class LinearRegression:
    def __init__(self, regularization=0.0):
        self.lambda_ = regularization

    def fit(self, X, y):
        XtX = X.T @ X
        if self.lambda_ > 0:
            XtX += self.lambda_ * np.eye(X.shape[1])
        self.w = np.linalg.solve(XtX, X.T @ y)
        return self

    def predict(self, X):
        return X @ self.w
```

---

## Exercises

1. Derive normal equations
2. Show residuals orthogonal to X
3. Derive Ridge gradient
4. Implement OLS + Ridge with GD
