from scipy.integrate import odeint
import numpy as np

from .cythonImports import VALID_METHODS, breakageODE

# =============================================================================

class solver():
    def __init__(
        self, name, breakageModel
    ):
        self.name = name

        try:
            getattr(breakageODE, name)
        except AttributeError:
            raise AttributeError(f"'{name}' does not exsist as a method.\n\n" \
                + f"Valid methods include: {VALID_METHODS}")

        solver_name = eval(r"self." + self.name)

        self.solution = solver_name(breakageModel)

# --------------------------------------------
# 1D solvers

    def pyf_simple_breakage(self, breakageModel):
        return odeint(
            breakageODE.pyf_simple_breakage,
            breakageModel._IC,
            breakageModel.t,
            args=(
                *breakageModel.x,
                breakageModel._k,
                breakageModel._Phi
            )
        )

    def cyf_simple_breakage(self, breakageModel):
        return odeint(
            breakageODE.cyf_simple_breakage,
            breakageModel._IC,
            breakageModel.t,
            args=(
                *breakageModel.x,
                breakageModel._k,
                breakageModel._Phi
            )
        )

    def cyf_2D_breakage(self, breakageModel):
        return breakageODE.cyf_2D_breakage(
            breakageModel._IC,
            breakageModel.t,
            breakageModel.x[0],
            breakageModel.x[1],
            breakageModel._k,
            breakageModel._Phi
        )

    def cyn_simple_breakage(self, breakageModel):
        return odeint(
            breakageODE.cyn_simple_breakage,
            breakageModel._IC,
            breakageModel.t,
            args=(
                *breakageModel.x,
                breakageModel._k,
                breakageModel._Beta
            )
        )

# --------------------------------------------