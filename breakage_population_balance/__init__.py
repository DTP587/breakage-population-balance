from glob import glob
from warnings import warn
import os
from importlib import util

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


def check_ndarray(*args, **kwargs):
    if not isinstance(args[1], np.ndarray):
        raise ValueError(f"Argument '{args[1]}' is not of type numpy.ndarray.")

# def correct_flat_ndarray(*args, **kwargs):
#     """
#     Flat numpy arrays are handled by this function.
    
#     ```
#     array([1, 1, 1]) -> array([[1, 1, 1]])
#     ```

#     This way, the shape of the numpy array inputted is always standardised.
#     """
#     arg_length = np.len(args[1].shape)
#     # Reshape input array to (dims, bins)
#     if arg_length < 2:
#         arg = args[1].reshape((1, args[1].shape[0]))
#     else:
#         arg = args[1]
#     return arg


def wrapper_check_ndarray(func):
    def inner(*args, **kwargs):
        check_ndarray(*args, **kwargs)
        return func(*args, **kwargs)
    return inner

# def wrapper_correct_flat_ndarray(func):
#     def inner(*args, **kwargs):
#         args[1] = correct_flat_ndarray(*args, **kwargs)
#         return func(*args, **kwargs)
#     return inner

# def calculate_phi(kernel, indices):



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

    # @wrapper_check_ndarray 
    # # @wrapper_correct_flat_ndarray
    # def __set_ICs(self, initial_condition):
    #     self.dims = initial_condition[0]
    #     self.bins = initial_condition[1]
    #     return initial_condition

    @wrapper_check_ndarray 
    # @wrapper_correct_flat_ndarray
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
    # @wrapper_correct_flat_ndarray
    def __set_grid(self, grid):
        # Check shape is the same as initial_condition
        for i in range(self.dims):
            assert grid.shape[i] == self.bins
        return grid
     
    def __set_kernel(self, kernel):
        if callable(kernel):
            assert self.dims < 2, "Cannot calc kernels with > 2 dims."
            # Calculate phi
            Phi = np.zeros([self.bins, self.bins])
            for i in range(self.bins):
                for j in range(self.bins):
                    Phi[i, j] = kernel(self.x[i], self.x[j])
                # Normalise array to avoid mass gain/loss
                row = Phi[i, :]
                row_total = np.sum(row)
                if row_total == 0:
                     continue
                for j in range(self.bins):
                    Phi[i, j] /= row_total 
            return Phi
        check_ndarray(kernel)
        return kernel

    def __set_beta(self, beta):
        if callable(beta):
            assert self.dims < 2, "Cannot calc beta with > 2 dims."
            # Calculate phi
            Beta = np.zeros([self.bins, self.bins])
            for i in range(self.bins):
                for j in range(self.bins):
                    Beta[i, j] = beta(self.x[i], self.x[j])
            return Beta
        check_ndarray(Beta)
        return Beta

    def __set_rate(self, rate):
        #Cannot support > 1D
        assert self.dims < 2, "Cannot calculate rates with > 2 dims."
        if callable(rate):
            assert self.dims < 2, "Cannot calc rates with > 2 dims."
            return rate(self.x)
        check_ndarray(rate)
        return rate
