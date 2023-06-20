from glob import glob
from warnings import warn
import os
from importlib import util

module_name = "breakageODE"

module_path = os.path.join(
        os.path.dirname(__file__),
        f"{module_name}.*.so"
    )

module_file = glob(module_path)

if module_file:
    # Import the module
    module_spec = util.spec_from_file_location(
        module_name, module_file[0]
    )
    breakageODE = util.module_from_spec(module_spec)
    module_spec.loader.exec_module(breakageODE)
else:
    warn(f"Module {module_name} not found. Trying to find elsewhere...")

    module_path = os.path.join(
        os.path.dirname(__file__), "..", "build", "lib*", "breakage*",
        f"{module_name}.*.so"
    )

    module_file = glob(module_path)

    if module_file:
        # Import the module
        module_spec = util.spec_from_file_location(
            module_name, module_file[0]
        )
        breakageODE = util.module_from_spec(module_spec)
        module_spec.loader.exec_module(breakageODE)
    else:
        raise ValueError(f"Module {module_name} not found.")

import numpy as np
from scipy.integrate import odeint

# =============================================================================

# def import_cython(module_dir, raise_error=False):
#     # Import the module
#     module_spec = util.spec_from_file_location(
#         module_name, module_file[0]
#     )
#     module = util.module_from_spec(module_spec)
#     module_spec.loader.exec_module(module)
#     return module


def check_ndarray(*args, **kwargs):
    if not isinstance(args[1], np.ndarray):
        raise ValueError(f"Argument '{args[1]}' is not of type numpy.ndarray.")

def wrapper_check_ndarray(func):
    def wrapper(*args, **kwargs):
        check_ndarray(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper

# =============================================================================

# Currently only supports 1 Dimension in internal geometry.
class breakageModel():
    def __init__(self, initial_condition, times, grid, kernel, rate):
        self._IC  = self.__set_ICs(initial_condition)
        self._t   = self.__set_times(times)
        self._x   = self.__set_grid(grid)
        self._Phi = self.__set_kernel(kernel)
        self._k   = self.__set_rate(rate)
    
    def solve(self):
        # Solve the problem
        return odeint(
            breakageODE.simple_breakage, self._IC, self._t,
            args=(
                self._x, self._k, self._Phi
            )
        )

# --------------------------------------------

    @wrapper_check_ndarray 
    def __set_ICs(self, initial_condition):
        # Cannot support > 1D
        assert len(initial_condition.shape) < 2
        # Set the number of classes (bins for brevity)
        self._bins = initial_condition.shape[0]
        return initial_condition

    @wrapper_check_ndarray    
    def __set_times(self, times):        
        return times

    @wrapper_check_ndarray 
    def __set_grid(self, grid):
        # Cannot support > 1D
        assert len(grid.shape) <2
        assert (grid.shape[0] == self._bins)
        return grid
     
    def __set_kernel(self, kernel):
        # Cannot support > 1D
        if callable(kernel):
            # Calculate phi
            Phi = np.zeros([self._bins, self._bins])
            for i in range(self._bins):
                for j in range(self._bins):
                    Phi[i, j] = kernel(self._x[i], self._x[j])
                # Normalise array to avoid mass gain/loss
                row = Phi[i, :]
                row_total = np.sum(row)
                if row_total == 0:
                     continue
                for j in range(self._bins):
                    Phi[i, j] = Phi[i, j]/row_total 
            return Phi
        check_ndarray(kernel)
        return kernel

    def __set_rate(self, rate):
        #Cannot support > 1D
        if callable(rate):
            return rate(self._x)
        check_ndarray(rate)
        return rate
