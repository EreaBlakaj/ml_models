import numpy as np


def fit_normal_equation(X, y, fit_intercept=True):
    X = np.asarray(X)
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError("y must be a 1D array of shape (N,)")

    if fit_intercept:
        ones = np.ones((X.shape[0], 1))
        X_design = np.hstack([ones, X])
    else:
        X_design = X
   
    XtX = X_design.T @ X_design
    XtX_pinv = np.linalg.pinv(XtX)
    Xty = X_design.T @ y

    theta = XtX_pinv @ Xty

    if fit_intercept:
        intercept = float(theta[0])
        coef = theta[1:]
    else:
        intercept = 0.0
        coef = theta

    return intercept, coef


def predict_normal_equation(X, intercept, coef):
    X = np.asarray(X)
    coef = np.asarray(coef)

    return intercept + X @ coef