# Multi-Layer Perceptron and Backpropagation

---

## Architecture

A 2-layer MLP (one hidden layer):

```
Input x ∈ ℝ^{d_in}
    ↓
Hidden layer: h = σ(W₁x + b₁)
    ↓
Output layer: ŷ = W₂h + b₂
```

Where:
- W₁ ∈ ℝ^{d_h × d_in}, b₁ ∈ ℝ^{d_h} (hidden layer parameters)
- W₂ ∈ ℝ^{d_out × d_h}, b₂ ∈ ℝ^{d_out} (output layer parameters)
- σ is the activation function

For regression with MSE loss:
```
L = (1/2)||ŷ - y||²
```

### Notation

- z₁ = W₁x + b₁ (pre-activation of hidden layer)
- h = σ(z₁) (post-activation / hidden layer output)
- z₂ = W₂h + b₂ (pre-activation of output = final output for regression)
- ŷ = z₂

Full forward pass:
```
x → z₁ = W₁x + b₁ → h = σ(z₁) → z₂ = W₂h + b₂ → ŷ → L
```

---

## Goal

Compute all parameter gradients:
- ∂L/∂W₂, ∂L/∂b₂ (output layer)
- ∂L/∂W₁, ∂L/∂b₁ (hidden layer)

Then update: θ ← θ - α ∂L/∂θ where α is the learning rate.

---

## Forward Pass

```python
# Input: x of shape (d_in,)
z1 = W1 @ x + b1          # (d_h,)
h = activation(z1)        # (d_h,)
z2 = W2 @ h + b2          # (d_out,)
y_hat = z2                # (d_out,)
loss = 0.5 * np.sum((y_hat - y)**2)  # scalar
```

---

## Backward Pass Derivation

### Step 1: Gradient w.r.t. Output Layer

**∂L/∂z₂:**

For MSE loss L = (1/2)Σⱼ(ŷⱼ - yⱼ)²:
```
∂L/∂ŷⱼ = ŷⱼ - yⱼ
```

Since ŷ = z₂:
```
δ₂ ≡ ∂L/∂z₂ = ŷ - y
```

δ₂ is the output error signal.

**∂L/∂W₂:**

Since z₂ᵢ = Σₖ W₂ᵢₖhₖ + b₂ᵢ:
```
∂z₂ᵢ/∂W₂ᵢⱼ = hⱼ
```

By chain rule:
```
∂L/∂W₂ᵢⱼ = (∂L/∂z₂ᵢ)(∂z₂ᵢ/∂W₂ᵢⱼ) = δ₂ᵢ · hⱼ
```

In matrix form (outer product):
```
∂L/∂W₂ = δ₂ · hᵀ ∈ ℝ^{d_out × d_h}
```

**∂L/∂b₂:**
```
∂z₂ᵢ/∂b₂ᵢ = 1

∂L/∂b₂ = δ₂ ∈ ℝ^{d_out}
```

### Step 2: Backpropagate Through W₂

**∂L/∂h:**

z₂ depends on h:
```
∂L/∂hⱼ = Σᵢ (∂L/∂z₂ᵢ)(∂z₂ᵢ/∂hⱼ) = Σᵢ δ₂ᵢ · W₂ᵢⱼ
```

In matrix form:
```
∂L/∂h = W₂ᵀ · δ₂ ∈ ℝ^{d_h}
```

Error propagates backward through the transpose of the weight matrix.

### Step 3: Backpropagate Through Activation

**∂L/∂z₁:**

Since h = σ(z₁), by chain rule:
```
∂L/∂z₁ⱼ = (∂L/∂hⱼ)(∂hⱼ/∂z₁ⱼ) = (∂L/∂hⱼ) · σ'(z₁ⱼ)
```

Element-wise:
```
δ₁ ≡ ∂L/∂z₁ = (W₂ᵀδ₂) ⊙ σ'(z₁)
```

where ⊙ denotes element-wise (Hadamard) multiplication.

### Step 4: Gradient w.r.t. Hidden Layer

Following the same pattern as the output layer:

**∂L/∂W₁:**
```
∂L/∂W₁ = δ₁ · xᵀ ∈ ℝ^{d_h × d_in}
```

**∂L/∂b₁:**
```
∂L/∂b₁ = δ₁ ∈ ℝ^{d_h}
```

---

## Summary: The Backprop Equations

**Forward:**
```
z₁ = W₁x + b₁
h = σ(z₁)
z₂ = W₂h + b₂
ŷ = z₂
L = (1/2)||ŷ - y||²
```

**Backward:**
```
δ₂ = ŷ - y                          # Output error
∂L/∂W₂ = δ₂ · hᵀ                    # Outer product
∂L/∂b₂ = δ₂

δ₁ = (W₂ᵀ · δ₂) ⊙ σ'(z₁)           # Hidden error (backprop through W₂ and σ)
∂L/∂W₁ = δ₁ · xᵀ                    # Outer product
∂L/∂b₁ = δ₁
```

**Update:**
```
W₂ ← W₂ - α · ∂L/∂W₂
b₂ ← b₂ - α · ∂L/∂b₂
W₁ ← W₁ - α · ∂L/∂W₁
b₁ ← b₁ - α · ∂L/∂b₁
```

---

## Activation Functions

### ReLU
```
σ(z) = max(0, z)
σ'(z) = 1 if z > 0, else 0
```
No vanishing gradient for z > 0. "Dead neurons" can occur if z < 0 always.

### Sigmoid
```
σ(z) = 1/(1 + e^{-z})
σ'(z) = σ(z)(1 - σ(z))
```
Outputs in (0, 1). Vanishing gradient for |z| large, not zero-centered.

### Tanh
```
σ(z) = (e^z - e^{-z})/(e^z + e^{-z})
σ'(z) = 1 - σ(z)²
```
Zero-centered. Still vanishes for |z| large.

---

## Batch Processing

For a minibatch of m samples, let X ∈ ℝ^{d_in × m} (samples as columns).

**Forward:**
```
Z₁ = W₁X + b₁  (broadcasting b₁)    # (d_h, m)
H = σ(Z₁)                            # (d_h, m)
Z₂ = W₂H + b₂                        # (d_out, m)
Ŷ = Z₂                               # (d_out, m)
L = (1/2m) ||Ŷ - Y||²_F              # scalar (Frobenius norm)
```

**Backward:**
```
Δ₂ = (1/m)(Ŷ - Y)                    # (d_out, m)
∂L/∂W₂ = Δ₂ · Hᵀ                     # (d_out, d_h)
∂L/∂b₂ = sum over columns of Δ₂      # (d_out,)

Δ₁ = (W₂ᵀΔ₂) ⊙ σ'(Z₁)               # (d_h, m)
∂L/∂W₁ = Δ₁ · Xᵀ                     # (d_h, d_in)
∂L/∂b₁ = sum over columns of Δ₁      # (d_h,)
```

Matrix multiplications replace vector outer products. Bias gradients sum over samples.

---

## Numerical Example

**Setup:**
- d_in = 2, d_h = 2, d_out = 1
- x = [1, 2]ᵀ
- y = [1]
- W₁ = [[0.1, 0.2], [0.3, 0.4]], b₁ = [0, 0]ᵀ
- W₂ = [[0.5, 0.6]], b₂ = [0]
- ReLU activation

**Forward:**
```
z₁ = W₁x + b₁ = [[0.1, 0.2], [0.3, 0.4]] @ [1, 2]ᵀ = [0.5, 1.1]ᵀ
h = ReLU(z₁) = [0.5, 1.1]ᵀ  (both positive)
z₂ = W₂h + b₂ = [0.5, 0.6] @ [0.5, 1.1]ᵀ = [0.91]
ŷ = 0.91
L = 0.5 * (0.91 - 1)² = 0.00405
```

**Backward:**
```
δ₂ = ŷ - y = -0.09

∂L/∂W₂ = δ₂ · hᵀ = [-0.09] @ [0.5, 1.1] = [[-0.045, -0.099]]
∂L/∂b₂ = [-0.09]

∂L/∂h = W₂ᵀ · δ₂ = [[0.5], [0.6]] @ [-0.09] = [-0.045, -0.054]ᵀ
σ'(z₁) = [1, 1]ᵀ  (both z₁ > 0)
δ₁ = [-0.045, -0.054]ᵀ ⊙ [1, 1]ᵀ = [-0.045, -0.054]ᵀ

∂L/∂W₁ = δ₁ · xᵀ = [[-0.045], [-0.054]] @ [1, 2] = [[-0.045, -0.090], [-0.054, -0.108]]
∂L/∂b₁ = [-0.045, -0.054]ᵀ
```

---

## Gradient Checking

Compare analytical gradient to numerical approximation:

```python
def numerical_gradient(f, w, eps=1e-5):
    """Central difference approximation."""
    grad = np.zeros_like(w)
    for i in range(len(w)):
        w_plus = w.copy()
        w_plus[i] += eps
        w_minus = w.copy()
        w_minus[i] -= eps
        grad[i] = (f(w_plus) - f(w_minus)) / (2 * eps)
    return grad
```

Relative error check:
```
||grad_analytical - grad_numerical|| / (||grad_analytical|| + ||grad_numerical||) < 1e-5
```

---

## Chain Rule on Computational Graphs

Every computation forms a directed acyclic graph:
- Nodes: intermediate values
- Edges: operations

Backprop applies chain rule systematically:
1. Compute all forward values, caching intermediate results
2. Start from loss, propagate gradients backward
3. At each node, multiply incoming gradient by local derivative

### The Transpose Pattern

Forward through W means backward through Wᵀ.

Forward: z = Wx
Backward: ∂L/∂x = Wᵀ(∂L/∂z)

The Jacobian ∂z/∂x = W, and we left-multiply by the upstream gradient.

---

## Vanishing and Exploding Gradients

For L layers, the gradient at layer 1 involves:
```
∂L/∂W₁ ∝ Wₗᵀ · ... · W₂ᵀ · σ'(z_{L-1}) ⊙ ... ⊙ σ'(z₁)
```

### Vanishing

If |σ'(z)| < 1 (sigmoid saturates at max 0.25) and ||Wᵢ|| < 1:
- Product of many small numbers → gradient vanishes
- Early layers learn slowly

### Exploding

If ||Wᵢ|| > 1:
- Product grows exponentially
- Gradients become huge, training diverges

### Solutions

1. **ReLU:** σ'(z) = 1 for z > 0
2. **Proper initialization:** Xavier/He initialization
3. **Batch normalization:** Normalize activations
4. **Residual connections:** Skip connections allow gradient flow
5. **Gradient clipping:** Cap gradient magnitude

---

## Implementation

```python
import numpy as np

class MLP:
    """2-layer MLP from scratch."""

    def __init__(self, d_in, d_h, d_out, activation='relu'):
        # Xavier initialization
        self.W1 = np.random.randn(d_h, d_in) * np.sqrt(2.0 / d_in)
        self.b1 = np.zeros(d_h)
        self.W2 = np.random.randn(d_out, d_h) * np.sqrt(2.0 / d_h)
        self.b2 = np.zeros(d_out)
        self.activation = activation

    def _activate(self, z):
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(z)

    def _activate_deriv(self, z):
        if self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = self._activate(z)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z)**2

    def forward(self, X):
        """Forward pass. X: (n_samples, d_in)"""
        self.X = X
        self.z1 = X @ self.W1.T + self.b1     # (n, d_h)
        self.h = self._activate(self.z1)       # (n, d_h)
        self.z2 = self.h @ self.W2.T + self.b2 # (n, d_out)
        return self.z2

    def backward(self, y):
        """Backward pass. y: (n_samples, d_out)"""
        n = y.shape[0]

        # Output layer
        delta2 = (self.z2 - y) / n                      # (n, d_out)
        self.dW2 = delta2.T @ self.h                    # (d_out, d_h)
        self.db2 = delta2.sum(axis=0)                   # (d_out,)

        # Hidden layer
        delta1 = (delta2 @ self.W2) * self._activate_deriv(self.z1)  # (n, d_h)
        self.dW1 = delta1.T @ self.X                    # (d_h, d_in)
        self.db1 = delta1.sum(axis=0)                   # (d_h,)

    def update(self, lr):
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1

    def loss(self, y_pred, y):
        return 0.5 * np.mean((y_pred - y)**2)

    def fit(self, X, y, lr=0.01, epochs=1000, verbose=True):
        history = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss(y_pred, y)
            history.append(loss)
            self.backward(y)
            self.update(lr)
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        return history
```

---

## Exercises

### Derivations
1. [ ] Derive backprop for a 3-layer MLP
2. [ ] Derive the gradient for cross-entropy loss with softmax output
3. [ ] Show that sigmoid derivative is σ(1-σ)

### Implementations
1. [ ] Implement the MLP class above and train on XOR
2. [ ] Add gradient checking to verify correctness
3. [ ] Train on MNIST subset and visualize learned features
4. [ ] Compare different activations on the same problem

### Experiments
1. [ ] Visualize how loss landscape changes with depth
2. [ ] Observe vanishing gradients with sigmoid vs ReLU
3. [ ] Compare different initialization schemes
