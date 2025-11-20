from typing import Iterable

import numpy as np


def sse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes of y_true {y_true.shape} and y_pred {y_pred.shape} do not match.")

    errors = y_true - y_pred
    sse = float(np.sum(errors**2))

    return sse
    