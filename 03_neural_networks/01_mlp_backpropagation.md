# Multi-Layer Perceptron and Backpropagation

---

## Architecture

A 2-layer MLP (one hidden layer):

$$x \in \mathbb{R}^{d_{\text{in}}} \to h = \sigma(W_1 x + b_1) \to \hat{y} = W_2 h + b_2$$

Where:
- $W_1 \in \mathbb{R}^{d_h \times d_{\text{in}}}$, $b_1 \in \mathbb{R}^{d_h}$ (hidden layer parameters)
- $W_2 \in \mathbb{R}^{d_{\text{out}} \times d_h}$, $b_2 \in \mathbb{R}^{d_{\text{out}}}$ (output layer parameters)
- $\sigma$ is the activation function

For regression with MSE loss:

$$L = \frac{1}{2}\|\hat{y} - y\|^2$$

### Notation

- $z_1 = W_1 x + b_1$ (pre-activation of hidden layer)
- $h = \sigma(z_1)$ (post-activation / hidden layer output)
- $z_2 = W_2 h + b_2$ (pre-activation of output = final output for regression)
- $\hat{y} = z_2$

Full forward pass:

$$x \to z_1 = W_1 x + b_1 \to h = \sigma(z_1) \to z_2 = W_2 h + b_2 \to \hat{y} \to L$$

---

## Goal

Compute all parameter gradients:
- $\frac{\partial L}{\partial W_2}$, $\frac{\partial L}{\partial b_2}$ (output layer)
- $\frac{\partial L}{\partial W_1}$, $\frac{\partial L}{\partial b_1}$ (hidden layer)

Then update: $\theta \leftarrow \theta - \alpha \frac{\partial L}{\partial \theta}$ where $\alpha$ is the learning rate.

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

**$\frac{\partial L}{\partial z_2}$:**

For MSE loss $L = \frac{1}{2}\sum_j(\hat{y}_j - y_j)^2$:

$$\frac{\partial L}{\partial \hat{y}_j} = \hat{y}_j - y_j$$

Since $\hat{y} = z_2$:

$$\delta_2 \equiv \frac{\partial L}{\partial z_2} = \hat{y} - y$$

$\delta_2$ is the output error signal.

**$\frac{\partial L}{\partial W_2}$:**

Since $z_{2i} = \sum_k W_{2ik} h_k + b_{2i}$:

$$\frac{\partial z_{2i}}{\partial W_{2ij}} = h_j$$

By chain rule:

$$\frac{\partial L}{\partial W_{2ij}} = \frac{\partial L}{\partial z_{2i}} \cdot \frac{\partial z_{2i}}{\partial W_{2ij}} = \delta_{2i} \cdot h_j$$

In matrix form (outer product):

$$\frac{\partial L}{\partial W_2} = \delta_2 \cdot h^\top \in \mathbb{R}^{d_{\text{out}} \times d_h}$$

**$\frac{\partial L}{\partial b_2}$:**

$$\frac{\partial z_{2i}}{\partial b_{2i}} = 1$$

$$\frac{\partial L}{\partial b_2} = \delta_2 \in \mathbb{R}^{d_{\text{out}}}$$

### Step 2: Backpropagate Through $W_2$

**$\frac{\partial L}{\partial h}$:**

$z_2$ depends on $h$:

$$\frac{\partial L}{\partial h_j} = \sum_i \frac{\partial L}{\partial z_{2i}} \cdot \frac{\partial z_{2i}}{\partial h_j} = \sum_i \delta_{2i} \cdot W_{2ij}$$

In matrix form:

$$\frac{\partial L}{\partial h} = W_2^\top \cdot \delta_2 \in \mathbb{R}^{d_h}$$

Error propagates backward through the transpose of the weight matrix.

### Step 3: Backpropagate Through Activation

**$\frac{\partial L}{\partial z_1}$:**

Since $h = \sigma(z_1)$, by chain rule:

$$\frac{\partial L}{\partial z_{1j}} = \frac{\partial L}{\partial h_j} \cdot \frac{\partial h_j}{\partial z_{1j}} = \frac{\partial L}{\partial h_j} \cdot \sigma'(z_{1j})$$

Element-wise:

$$\delta_1 \equiv \frac{\partial L}{\partial z_1} = (W_2^\top \delta_2) \odot \sigma'(z_1)$$

where $\odot$ denotes element-wise (Hadamard) multiplication.

### Step 4: Gradient w.r.t. Hidden Layer

Following the same pattern as the output layer:

**$\frac{\partial L}{\partial W_1}$:**

$$\frac{\partial L}{\partial W_1} = \delta_1 \cdot x^\top \in \mathbb{R}^{d_h \times d_{\text{in}}}$$

**$\frac{\partial L}{\partial b_1}$:**

$$\frac{\partial L}{\partial b_1} = \delta_1 \in \mathbb{R}^{d_h}$$

---

## Summary: The Backprop Equations

**Forward:**

$$z_1 = W_1 x + b_1$$
$$h = \sigma(z_1)$$
$$z_2 = W_2 h + b_2$$
$$\hat{y} = z_2$$
$$L = \frac{1}{2}\|\hat{y} - y\|^2$$

**Backward:**

$$\delta_2 = \hat{y} - y \quad \text{(output error)}$$
$$\frac{\partial L}{\partial W_2} = \delta_2 \cdot h^\top \quad \text{(outer product)}$$
$$\frac{\partial L}{\partial b_2} = \delta_2$$

$$\delta_1 = (W_2^\top \cdot \delta_2) \odot \sigma'(z_1) \quad \text{(hidden error)}$$
$$\frac{\partial L}{\partial W_1} = \delta_1 \cdot x^\top \quad \text{(outer product)}$$
$$\frac{\partial L}{\partial b_1} = \delta_1$$

**Update:**

$$W_2 \leftarrow W_2 - \alpha \cdot \frac{\partial L}{\partial W_2}$$
$$b_2 \leftarrow b_2 - \alpha \cdot \frac{\partial L}{\partial b_2}$$
$$W_1 \leftarrow W_1 - \alpha \cdot \frac{\partial L}{\partial W_1}$$
$$b_1 \leftarrow b_1 - \alpha \cdot \frac{\partial L}{\partial b_1}$$

---

## Activation Functions

### ReLU

$$\sigma(z) = \max(0, z)$$
$$\sigma'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{otherwise} \end{cases}$$

No vanishing gradient for $z > 0$. "Dead neurons" can occur if $z < 0$ always.

### Sigmoid

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

Outputs in $(0, 1)$. Vanishing gradient for $|z|$ large, not zero-centered.

### Tanh

$$\sigma(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
$$\sigma'(z) = 1 - \sigma(z)^2$$

Zero-centered. Still vanishes for $|z|$ large.

---

## Batch Processing

For a minibatch of $m$ samples, let $X \in \mathbb{R}^{d_{\text{in}} \times m}$ (samples as columns).

**Forward:**

$$Z_1 = W_1 X + b_1 \quad \text{(broadcasting } b_1\text{)} \quad \in \mathbb{R}^{d_h \times m}$$
$$H = \sigma(Z_1) \quad \in \mathbb{R}^{d_h \times m}$$
$$Z_2 = W_2 H + b_2 \quad \in \mathbb{R}^{d_{\text{out}} \times m}$$
$$\hat{Y} = Z_2 \quad \in \mathbb{R}^{d_{\text{out}} \times m}$$
$$L = \frac{1}{2m} \|\hat{Y} - Y\|_F^2 \quad \text{(scalar, Frobenius norm)}$$

**Backward:**

$$\Delta_2 = \frac{1}{m}(\hat{Y} - Y) \quad \in \mathbb{R}^{d_{\text{out}} \times m}$$
$$\frac{\partial L}{\partial W_2} = \Delta_2 \cdot H^\top \quad \in \mathbb{R}^{d_{\text{out}} \times d_h}$$
$$\frac{\partial L}{\partial b_2} = \text{sum over columns of } \Delta_2 \quad \in \mathbb{R}^{d_{\text{out}}}$$

$$\Delta_1 = (W_2^\top \Delta_2) \odot \sigma'(Z_1) \quad \in \mathbb{R}^{d_h \times m}$$
$$\frac{\partial L}{\partial W_1} = \Delta_1 \cdot X^\top \quad \in \mathbb{R}^{d_h \times d_{\text{in}}}$$
$$\frac{\partial L}{\partial b_1} = \text{sum over columns of } \Delta_1 \quad \in \mathbb{R}^{d_h}$$

Matrix multiplications replace vector outer products. Bias gradients sum over samples.

---

## Numerical Example

**Setup:**
- $d_{\text{in}} = 2$, $d_h = 2$, $d_{\text{out}} = 1$
- $x = [1, 2]^\top$
- $y = [1]$
- $W_1 = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \end{bmatrix}$, $b_1 = [0, 0]^\top$
- $W_2 = [0.5, 0.6]$, $b_2 = [0]$
- ReLU activation

**Forward:**

$$z_1 = W_1 x + b_1 = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \end{bmatrix} \begin{bmatrix} 1 \\ 2 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 1.1 \end{bmatrix}$$
$$h = \text{ReLU}(z_1) = [0.5, 1.1]^\top \quad \text{(both positive)}$$
$$z_2 = W_2 h + b_2 = [0.5, 0.6] \cdot [0.5, 1.1]^\top = [0.91]$$
$$\hat{y} = 0.91$$
$$L = 0.5 \times (0.91 - 1)^2 = 0.00405$$

**Backward:**

$$\delta_2 = \hat{y} - y = -0.09$$
$$\frac{\partial L}{\partial W_2} = \delta_2 \cdot h^\top = [-0.09] \cdot [0.5, 1.1] = [-0.045, -0.099]$$
$$\frac{\partial L}{\partial b_2} = [-0.09]$$

$$\frac{\partial L}{\partial h} = W_2^\top \cdot \delta_2 = \begin{bmatrix} 0.5 \\ 0.6 \end{bmatrix} \cdot [-0.09] = [-0.045, -0.054]^\top$$
$$\sigma'(z_1) = [1, 1]^\top \quad \text{(both } z_1 > 0\text{)}$$
$$\delta_1 = [-0.045, -0.054]^\top \odot [1, 1]^\top = [-0.045, -0.054]^\top$$

$$\frac{\partial L}{\partial W_1} = \delta_1 \cdot x^\top = \begin{bmatrix} -0.045 \\ -0.054 \end{bmatrix} \cdot [1, 2] = \begin{bmatrix} -0.045 & -0.090 \\ -0.054 & -0.108 \end{bmatrix}$$
$$\frac{\partial L}{\partial b_1} = [-0.045, -0.054]^\top$$

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

$$\frac{\|\nabla_{\text{analytical}} - \nabla_{\text{numerical}}\|}{\|\nabla_{\text{analytical}}\| + \|\nabla_{\text{numerical}}\|} < 10^{-5}$$

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

Forward through $W$ means backward through $W^\top$.

Forward: $z = Wx$
Backward: $\frac{\partial L}{\partial x} = W^\top \frac{\partial L}{\partial z}$

The Jacobian $\frac{\partial z}{\partial x} = W$, and we left-multiply by the upstream gradient.

---

## Vanishing and Exploding Gradients

For $L$ layers, the gradient at layer 1 involves:

$$\frac{\partial L}{\partial W_1} \propto W_L^\top \cdot \ldots \cdot W_2^\top \cdot \sigma'(z_{L-1}) \odot \ldots \odot \sigma'(z_1)$$

### Vanishing

If $|\sigma'(z)| < 1$ (sigmoid saturates at max 0.25) and $\|W_i\| < 1$:
- Product of many small numbers → gradient vanishes
- Early layers learn slowly

### Exploding

If $\|W_i\| > 1$:
- Product grows exponentially
- Gradients become huge, training diverges

### Solutions

1. **ReLU:** $\sigma'(z) = 1$ for $z > 0$
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
3. [ ] Show that sigmoid derivative is $\sigma(1-\sigma)$

### Implementations
1. [ ] Implement the MLP class above and train on XOR
2. [ ] Add gradient checking to verify correctness
3. [ ] Train on MNIST subset and visualize learned features
4. [ ] Compare different activations on the same problem

### Experiments
1. [ ] Visualize how loss landscape changes with depth
2. [ ] Observe vanishing gradients with sigmoid vs ReLU
3. [ ] Compare different initialization schemes
