"""Plotting utilities for ML Foundations.

Common visualization functions used across notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_loss_history(history, title="Training Loss", log_scale=False):
    """Plot loss over iterations.

    Args:
        history: List of loss values
        title: Plot title
        log_scale: Whether to use log scale for y-axis
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    if log_scale:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()


def plot_decision_boundary_2d(model, X, y, resolution=100, ax=None):
    """Plot decision boundary for 2D classification.

    Args:
        model: Must have .predict() method
        X: (n_samples, 2) feature matrix
        y: (n_samples,) labels
        resolution: Grid resolution
        ax: Matplotlib axis (creates new if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5)

    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k')

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    return ax


def plot_regression_fit_1d(X, y, model=None, y_pred=None, ax=None):
    """Plot 1D regression data and fit.

    Args:
        X: (n_samples, 1) or (n_samples,) feature values
        y: (n_samples,) targets
        model: If provided, uses model.predict(X)
        y_pred: Direct predictions (alternative to model)
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    X = np.asarray(X).ravel()

    # Plot data
    ax.scatter(X, y, alpha=0.6, label='Data')

    # Plot fit
    if model is not None:
        X_sorted = np.sort(X)
        X_sorted_2d = X_sorted.reshape(-1, 1)
        y_pred = model.predict(X_sorted_2d)
        ax.plot(X_sorted, y_pred, 'r-', linewidth=2, label='Fit')
    elif y_pred is not None:
        ax.plot(X, y_pred, 'r-', linewidth=2, label='Fit')

    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_gradient_descent_contour(f, grad_f, trajectory, xlim=(-3, 3), ylim=(-3, 3),
                                   resolution=100, ax=None):
    """Plot gradient descent trajectory on contour plot.

    Args:
        f: Objective function f(w) where w is 2D
        grad_f: Gradient function
        trajectory: List of [w1, w2] points visited
        xlim, ylim: Plot limits
        resolution: Contour resolution
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Create mesh
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)

    # Evaluate function on mesh
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

    # Plot contours
    ax.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
    ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.3)

    # Plot trajectory
    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', markersize=4, linewidth=1)
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start')
    ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=15, label='End')

    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.legend()

    return ax


def plot_eigenvalues(eigenvalues, cumulative=True, ax=None):
    """Plot eigenvalues and optionally cumulative variance.

    Args:
        eigenvalues: Array of eigenvalues (sorted descending)
        cumulative: Whether to show cumulative explained variance
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    n = len(eigenvalues)
    x = np.arange(1, n + 1)

    # Explained variance ratio
    total = eigenvalues.sum()
    explained_ratio = eigenvalues / total

    ax.bar(x, explained_ratio, alpha=0.7, label='Individual')

    if cumulative:
        cumulative_ratio = np.cumsum(explained_ratio)
        ax.plot(x, cumulative_ratio, 'ro-', label='Cumulative')
        ax.axhline(y=0.95, color='k', linestyle='--', alpha=0.5, label='95% threshold')

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("Scree Plot")
    ax.legend()
    ax.set_xticks(x)

    return ax


def plot_clusters_2d(X, labels, centroids=None, ax=None):
    """Plot 2D clustered data.

    Args:
        X: (n_samples, 2) data
        labels: (n_samples,) cluster assignments
        centroids: (n_clusters, 2) centroid locations
        ax: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', alpha=0.7)

    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1],
                   c='red', marker='X', s=200, edgecolors='black', linewidths=2,
                   label='Centroids')
        ax.legend()

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    return ax


def plot_mnist_samples(images, labels=None, n_rows=2, n_cols=5, figsize=(12, 5)):
    """Plot grid of MNIST images.

    Args:
        images: (n_samples, 784) or (n_samples, 28, 28) images
        labels: Optional labels to show
        n_rows, n_cols: Grid dimensions
        figsize: Figure size
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.ravel()

    n_show = min(len(images), n_rows * n_cols)

    for i in range(n_show):
        img = images[i]
        if img.ndim == 1:
            img = img.reshape(28, 28)

        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

        if labels is not None:
            axes[i].set_title(f"Label: {labels[i]}")

    # Hide remaining axes
    for i in range(n_show, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def set_plot_style():
    """Set consistent matplotlib style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (8, 5),
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })
