from distutils.log import error
import numpy as np
from main import GradientDescent


def test_constant_function() -> None:
    """
    Testing gradient descent algorithm for a function f(x) = 2.
    Here, x is provided as of type np.ndarray(5,) = np.array([1, 2, 3, 4, 5])

    :var func: Implements the function f.
    :var gradient: Implements gradient of the function f at a given point (vector).
    """
    func = lambda x: 2
    gradient = lambda x: 0

    model = GradientDescent(M=5, func=func, gradient=gradient, epsilon=1e-5, gamma=1e-5)
    res = model.run(np.array([1, 2, 3, 4, 5]))
    assert res["success"] and res["value"] == func(res["pt"]) and res["value"] == 2


def test_quadratic_function() -> None:
    """
    Testing gradient descent algorithm for a function f(x) = x[0]^2 + x[1]^2 + 4.
    Here, x is provided as of type np.ndarray(2,) = np.array([4, -5])

    :var func: Implements the function f.
    :var gradient: Implements gradient of the function f at a given point (vector).
    """
    func = lambda x: x[0] ** 2 + x[1] ** 2 + 4
    gradient = lambda x: np.array([2 * x[0], 2 * x[1]])
    error_threshold = 1e-3

    model = GradientDescent(M=2, func=func, gradient=gradient, epsilon=1e-5, gamma=1e-2)
    res = model.run(np.array([4, -5]))
    assert (
        res["success"]
        and res["value"] - func(res["pt"]) <= error_threshold
        and res["value"] - 4 <= error_threshold
    )
