from scipy.integrate import odeint, solve_ivp
import numpy as np

from .arrayManipulation import check_ndarray, wrapper_check_ndarray, \
    apply_function, apply_kernel

from .models import solver

# =============================================================================

class breakageModel():
    def __init__(
        self, initial_condition, times, grid, rate, kernel=None, beta=None
    ):
        self._IC  = self.__set_ICs(initial_condition)
        self.t   = self.__set_times(times)
        self.x   = self.__set_grid(grid)
        self._k   = self.__set_rate(rate)

        assert kernel != beta, "kernel and beta cannot be the same or None."

        if kernel is None:
            self._Beta = self.__set_beta(beta)
        elif beta is None:
            self._Phi  = self.__set_kernel(kernel)

# --------------------------------------------

    def solve(self, ODE):
        solved = solver(ODE, self)
        return solved.solution

# --------------------------------------------

    @wrapper_check_ndarray 
    def __set_ICs(self, initial_condition):
        self.shape = initial_condition.shape
        return initial_condition

    @wrapper_check_ndarray    
    def __set_times(self, times):        
        return times

    def __set_grid(self, grid):
        grid = grid if not isinstance(grid, np.ndarray) else [grid]
        self.dims = len(grid)
        return grid
     
    def __set_kernel(self, kernel):
        if callable(kernel):
            return apply_kernel(self.x, kernel, normalise=True)
        check_ndarray(self, kernel)
        return kernel

    def __set_beta(self, beta):
        if callable(beta):
            return apply_kernel(self.x, beta)
        check_ndarray(self, beta)
        return beta

    def __set_rate(self, rate):
        if callable(rate):
            return apply_function(self.x, rate)
        check_ndarray(self, rate)
        return rate