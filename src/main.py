import numpy as np


class GradientDescent:
    """
    Class for gradient desecnt.
    """

    def __init__(self, M: int, func, gradient, epsilon: float, gamma: float) -> None:
        """
        :param M: Dimension of input vector.
        :param func: Function over which we are performing gradient descent.
        :param gradient: Function to calculate gradient of func(x) at a given x.
        :param epsilon: Threshold (tolerance) for gradient descent.
        :param gamma: Learning rate.
        """
        self.M = M
        self.func = func
        self.gradient = gradient
        self.epsilon = epsilon
        self.gamma = gamma

    def error_check(self, x0: np.ndarray) -> None:
        """
        :param x0: Starting point (initial vector) of gradient descent.
        """
        if not isinstance(x0, np.ndarray):
            raise TypeError(f"Expected ndarray; got {type(x0).__name__}")
        if x0.shape != (self.M,):
            raise ValueError(f"Expected shape {(self.M, )}; got {x0.shape}")
        if not 0 < self.gamma < 1:
            raise ValueError(f"Expected value between 0 and 1; got {self.gamma}")

    def run(self, x0: np.ndarray, max_iter=1000) -> dict:
        """
        Performs gradient decsent.

        :param x0: Starting point (initial vector) of gradient descent.
        :param max_iter: Maximum iterations.
        :return res: Dictionary with keys:values {”success” : bool, ”value” : float, ”pt” : ndarray(M,)}
        """
        self.error_check(x0)

        x_prev = x0
        i = 0

        while i < max_iter:
            x = x_prev - self.gamma * self.gradient(x_prev)
            i += 1

            if np.linalg.norm(x - x_prev) < self.epsilon:
                break
            x_prev = x

        res = {}
        res["success"] = i != max_iter
        res["value"] = self.func(x)
        res["pt"] = x

        return res
