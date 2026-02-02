# Linear Regression

## Setup

Data: $\{(x_1, y_1), \ldots, (x_n, y_n)\}$, $x_i \in \mathbb{R}^d$, $y_i \in \mathbb{R}$

Model: $\hat{y} = w^\top x + b$

Design matrix (with bias column): $X \in \mathbb{R}^{n \times (d+1)}$, predictions: $\hat{y} = Xw$

---

## Loss: MSE

$$J(w) = \|y - Xw\|^2 = (y - Xw)^\top(y - Xw)$$
$$= y^\top y - 2w^\top X^\top y + w^\top X^\top Xw$$

---

## Normal Equations

$$\nabla J = -2X^\top y + 2X^\top Xw = 0$$

$$X^\top Xw = X^\top y$$
$$w^* = (X^\top X)^{-1}X^\top y$$

**Geometric interpretation:** Residual $r = y - Xw^*$ orthogonal to $\text{col}(X)$: $X^\top r = 0$

**$(X^\top X)^{-1}$ exists iff:** $X$ has full column rank ($n \geq d$, no multicollinearity)

---

## Gradient Descent

$$w_{t+1} = w_t - 2\alpha X^\top(Xw_t - y)$$

Per-sample gradient: $\nabla(y_i - w^\top x_i)^2 = 2(\hat{y}_i - y_i)x_i$

---

## Regularization

### Ridge (L2)

$$J_{\text{ridge}} = \|y - Xw\|^2 + \lambda\|w\|^2$$
$$w^* = (X^\top X + \lambda I)^{-1}X^\top y$$
$$\nabla J = 2X^\top(Xw - y) + 2\lambda w$$

### Lasso (L1)

$$J_{\text{lasso}} = \|y - Xw\|^2 + \lambda\|w\|_1$$

No closed form. Promotes sparsity.

### Elastic Net

$$J = \|y - Xw\|^2 + \lambda_1\|w\|_1 + \lambda_2\|w\|^2$$

---

## Bias-Variance Tradeoff

$$\mathbb{E}[(y - \hat{y})^2] = \text{Bias}^2(\hat{y}) + \text{Var}(\hat{y}) + \sigma^2$$

| Complexity | Bias | Variance |
|------------|------|----------|
| Low | High | Low |
| High | Low | High |

Regularization: ↑bias, ↓variance

---

## Polynomial Regression

Feature expansion: $x \to \phi(x) = [1, x, x^2, \ldots, x^p]$

Model: $\hat{y} = w^\top\phi(x)$

High degree → overfitting. Use regularization + validation.

---

## MLE Connection

Under $y|x \sim \mathcal{N}(w^\top x, \sigma^2)$:

$$\log L(w) = \text{const} - \frac{1}{2\sigma^2} \sum_i(y_i - w^\top x_i)^2$$

$\max \log L = \min$ MSE

---

## Metrics

| Metric | Formula |
|--------|---------|
| $R^2$ | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ |
| RMSE | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ |
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ |

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
2. Show residuals orthogonal to $X$
3. Derive Ridge gradient
4. Implement OLS + Ridge with GD
