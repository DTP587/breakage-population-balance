from glob import glob
from warnings import warn
from importlib import util
import os

# =============================================================================

def import_cython(MODULE_NAME, module_dirs):
    if not module_dirs:
        pass
    else:
        for module_dir in module_dirs:
            if not module_dir:
                continue
            # Import the module
            module_spec = util.spec_from_file_location(
                MODULE_NAME, module_dir[0]
            )
            module = util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
            return module

    raise ValueError(f"Module {MODULE_NAME} not found.")

MODULE_NAME = "breakageODE"

module_dirs = [
    glob(path) for path in [
        os.path.join(os.path.dirname(__file__), f"{MODULE_NAME}.*.so"),
        os.path.join(
            os.path.dirname(__file__), "..", "build", "lib*", "breakage*",
            f"{MODULE_NAME}.*.so"
        )
    ]
]

breakageODE = import_cython(MODULE_NAME, module_dirs)

import numpy as np
from scipy.integrate import odeint

# =============================================================================

from .array_manipulation import check_ndarray, wrapper_check_ndarray, \
    apply_function, apply_kernel

# =============================================================================

# Currently only supports 1 Dimension in internal geometry.
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
    
    def solve_fraction(self):
        assert self._Phi is not None, \
            "kernel is undefined (did you mean solve_number?)."
        # Solve the problem
        return odeint(
            breakageODE.simple_breakage_fraction, self._IC, self.t,
            args=(
                self.x, self._k, self._Phi
            )
        )

    def solve_number(self):
        assert self._Beta is not None, \
            "beta is undefined (did you mean solve_fraction?)."
        # Solve the problem
        return odeint(
            breakageODE.simple_breakage_number, self._IC, self.t,
            args=(
                self.x, self._k, self._Beta
            )
        )

# --------------------------------------------

    @wrapper_check_ndarray 
    def __set_ICs(self, initial_condition):
        self.shape = initial_condition.shape
        self.bins  = self.shape[0]
        self.dims  = len(self.shape)
        if self.dims < 2:
            pass
        else:
            for i in range(self.dims):
                assert self.bins == self.shape[i], ("Number of classes must " +
                    "be identical between internal dimensions.")
        return initial_condition

    @wrapper_check_ndarray    
    def __set_times(self, times):        
        return times

    @wrapper_check_ndarray
    def __set_grid(self, grid):
        # Check shape is the same as initial_condition
        for i in range(self.dims):
            assert grid.shape[i] == self.bins
        return grid
     
    def __set_kernel(self, kernel):
        if callable(kernel):
            # Calculate phi
            # Phi = np.zeros([self.bins, self.bins])
            # for i in range(self.bins):
            #     for j in range(self.bins):
            #         Phi[i, j] = kernel(self.x[i], self.x[j])
            #     # Normalise array to avoid mass gain/loss
            #     row = Phi[i, :]
            #     row_total = np.sum(row)
            #     if row_total == 0:
            #          continue
            #     for j in range(self.bins):
            #         Phi[i, j] /= row_total 
            # print(Phi)
            return apply_kernel(self.x, kernel, normalise=True)
        check_ndarray(self, kernel)
        return kernel

    def __set_beta(self, beta):
        if callable(beta):
            # Calculate phi
            # Beta = np.zeros([self.bins, self.bins])
            # for i in range(self.bins):
            #     for j in range(self.bins):
            #         Beta[i, j] = beta(self.x[i], self.x[j])
            # return Beta
            return apply_kernel(self.x, beta)
        check_ndarray(self, beta)
        return beta

    def __set_rate(self, rate):
        if callable(rate):
            return apply_function(self.x, rate)
        check_ndarray(self, rate)
        return rate