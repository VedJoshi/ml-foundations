# K-Means Clustering

---

## Problem Setup

Given: Unlabeled data {x₁, x₂, ..., xₙ} where xᵢ ∈ ℝᵈ
Goal: Partition data into K clusters

Each cluster has a centroid μₖ ∈ ℝᵈ (cluster center). Each point is assigned to exactly one cluster.

---

## Objective Function

K-means minimizes the within-cluster sum of squares (WCSS):

```
J = Σₖ Σᵢ∈Cₖ ||xᵢ - μₖ||²
```

where Cₖ is the set of points assigned to cluster k.

Using assignment indicators rᵢₖ ∈ {0, 1} where rᵢₖ = 1 if point i is assigned to cluster k:
```
J = Σᵢ Σₖ rᵢₖ ||xᵢ - μₖ||²
```

Interpretation: minimize total squared distance from points to their centroids.

---

## Algorithm

K-means alternates between two steps:

### Step 1: Assignment (Fix μ, Optimize r)

Assign each point to the nearest centroid:
```
rᵢₖ = 1 if k = argmin_j ||xᵢ - μⱼ||²
      0 otherwise
```

### Step 2: Update (Fix r, Optimize μ)

Move each centroid to the mean of its assigned points:
```
μₖ = (Σᵢ rᵢₖ xᵢ) / (Σᵢ rᵢₖ) = mean of points in cluster k
```

### Derivation of Update Step

For fixed assignments, minimize Σᵢ∈Cₖ ||xᵢ - μₖ||² with respect to μₖ.

Taking derivative and setting to zero:
```
∂/∂μₖ Σᵢ∈Cₖ ||xᵢ - μₖ||² = Σᵢ∈Cₖ -2(xᵢ - μₖ) = 0
⟹ Σᵢ∈Cₖ xᵢ = |Cₖ| · μₖ
⟹ μₖ = (1/|Cₖ|) Σᵢ∈Cₖ xᵢ
```

where |Cₖ| is the number of points in cluster k.

### Convergence

- J decreases (or stays same) each iteration
- J ≥ 0 (bounded below)
- Finite number of possible assignments
- Converges to a **local minimum**, not necessarily global

---

## Initialization

### Random Initialization

Pick K random data points as initial centroids. Simple but unreliable.

### K-Means++

1. Choose first centroid uniformly at random from data
2. For each subsequent centroid:
   - Compute D(x) = distance from x to nearest existing centroid
   - Choose next centroid with probability ∝ D(x)²
3. Points far from existing centroids are more likely to be chosen

### Multiple Restarts

Run k-means multiple times with different initializations, keep best result (lowest J).

---

## Choosing K

### Elbow Method

1. Run k-means for K = 1, 2, 3, ...
2. Plot J (WCSS) vs K
3. Look for "elbow" where adding more clusters gives diminishing returns

### Silhouette Score

For each point i:
```
a(i) = mean distance to other points in same cluster
b(i) = mean distance to points in nearest different cluster
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

s(i) ∈ [-1, 1]:
- s ≈ 1: well-clustered
- s ≈ 0: on boundary between clusters
- s < 0: possibly assigned to wrong cluster

---

## Limitations

- **Spherical clusters:** Uses Euclidean distance, so finds spherical clusters of similar size. Fails for elongated or non-convex shapes.
- **Sensitive to outliers:** A single outlier can pull a centroid away from the true cluster center.
- **Fixed K:** Must specify K in advance.
- **Local minima:** May converge to poor solutions depending on initialization.

---

## Implementation

```python
import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, n_init=10):
        self.K = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.n_init = n_init

    def _init_centroids_plusplus(self, X):
        """K-means++ initialization."""
        n = X.shape[0]
        centroids = [X[np.random.randint(n)]]

        for _ in range(1, self.K):
            # Squared distances to nearest centroid
            D2 = np.min([np.sum((X - c)**2, axis=1) for c in centroids], axis=0)
            # Sample proportional to D²
            probs = D2 / D2.sum()
            new_idx = np.random.choice(n, p=probs)
            centroids.append(X[new_idx])

        return np.array(centroids)

    def _assign_clusters(self, X, centroids):
        """Assign each point to nearest centroid."""
        distances = np.array([np.sum((X - c)**2, axis=1) for c in centroids])
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X, labels):
        """Compute new centroids as cluster means."""
        return np.array([X[labels == k].mean(axis=0) for k in range(self.K)])

    def _compute_inertia(self, X, labels, centroids):
        """Compute within-cluster sum of squares."""
        return sum(np.sum((X[labels == k] - centroids[k])**2)
                   for k in range(self.K))

    def _fit_once(self, X):
        """Single run of k-means."""
        centroids = self._init_centroids_plusplus(X)

        for _ in range(self.max_iters):
            old_centroids = centroids.copy()
            labels = self._assign_clusters(X, centroids)
            centroids = self._update_centroids(X, labels)

            if np.all(np.abs(centroids - old_centroids) < self.tol):
                break

        inertia = self._compute_inertia(X, labels, centroids)
        return labels, centroids, inertia

    def fit(self, X):
        """Run k-means with multiple initializations."""
        best_inertia = np.inf

        for _ in range(self.n_init):
            labels, centroids, inertia = self._fit_once(X)
            if inertia < best_inertia:
                best_inertia = inertia
                self.labels_ = labels
                self.centroids_ = centroids
                self.inertia_ = inertia

        return self

    def predict(self, X):
        """Assign new points to clusters."""
        return self._assign_clusters(X, self.centroids_)
```

---

## Connection to EM Algorithm

K-means is a special case of EM for Gaussian Mixture Models:

- All covariances are σ²I (spherical, equal)
- As σ → 0, soft assignments become hard (0 or 1)
- "Expectation" step → assignment step
- "Maximization" step → update step

---

## Connection to Voronoi Diagrams

K-means partitions space into Voronoi cells:
- Each cell contains points closest to one centroid
- Boundaries are hyperplanes (perpendicular bisectors between centroids)

---

## Exercises

### Conceptual
1. Derive why setting μₖ to the mean minimizes within-cluster variance.
2. What happens if a cluster becomes empty during k-means?
3. How does feature scaling affect k-means?

### Implementations
1. [ ] Implement k-means from scratch
2. [ ] Implement k-means++ initialization
3. [ ] Implement the elbow method for choosing K
4. [ ] Implement silhouette score

### Experiments
1. [ ] Visualize k-means iterations on 2D data
2. [ ] Compare random vs k-means++ initialization
3. [ ] Apply k-means to MNIST and visualize cluster centroids as images
4. [ ] Show failure cases: elongated clusters, different sizes
