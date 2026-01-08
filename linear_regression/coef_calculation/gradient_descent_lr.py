import numpy as np


def gradient_descent_linear_regression(X, y, learning_rate=0.01, n_iters=1000, fit_intercept=True, verbose=False,):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if y.ndim != 1:
        raise ValueError("y must be a 1D array of shape (N,)")

    n_samples, n_features = X.shape

    if fit_intercept:
        ones = np.ones((n_samples, 1))
        X_design = np.hstack([ones, X])
        n_params = n_features + 1
    else:
        X_design = X
        n_params = n_features

    theta = np.zeros(n_params)

    losses = []

    for i in range(n_iters):
        y_hat = X_design @ theta

        errors = y_hat - y

        mse = np.mean(errors ** 2)
        losses.append(mse)

        grad = (2.0 / n_samples) * (X_design.T @ errors)

        theta -= learning_rate * grad

        if verbose and (i % max(1, n_iters // 10) == 0):
            print(f"Iteration {i:4d}/{n_iters}, MSE = {mse:.6f}")

    if fit_intercept:
        intercept = float(theta[0])
        coef = theta[1:]
    else:
        intercept = 0.0
        coef = theta

    return intercept, coef, losses


def predict_gradient_descent(X, intercept, coef):
    X = np.asarray(X, dtype=float)
    coef = np.asarray(coef, dtype=float)

    return intercept + X @ coef