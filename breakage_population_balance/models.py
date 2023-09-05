from scipy.integrate import odeint
import numpy as np

from .cythonImports import VALID_ODES, breakageODE

# =============================================================================

class solver():
    def __init__(
        self, name, breakageModel, method
    ):
        self.name = name
        self.method = method
        self.model = breakageModel

        try:
            getattr(breakageODE, name)
        except AttributeError:
            raise AttributeError(f"'{name}' does not exsist as a ODE.\n\n" \
                + f"Valid ODEs include: {VALID_ODES}")

        self.ode = eval(r"breakageODE." + self.name)

        if method == "odeint":
            self.solution = self.odeint_dispatcher()
        elif method == "euler":
            self.solution = self.euler_dispatcher()

# --------------------------------------------
# 1D solvers
    
    def odeint_dispatcher(self):
        model = self.model

        if model.dims > 1:
            raise ValueError("odeint only works for 1D ODEs")

        if model._Beta is not None:
            args = ( *model.x, model._k, model._Beta )
        elif model._Phi is not None:
            args = ( *model.x, model._k, model._Phi )
        else:
            raise ValueError("Phi and Beta unrecognised")

        return odeint( self.ode, model._IC, model.t, args=args )

# --------------------------------------------
# 2D solvers

    def euler_dispatcher(self):
        model = self.model

        if model.dims != 2:
            raise ValueError("Only supports 2D")

        if model._Beta is not None:
            arg = model._Beta
        elif model._Phi is not None:
            arg = model._Phi
        else:
            raise ValueError("Phi and Beta unrecognised")

        return breakageODE.Euler_2D(
            model._IC,
            model.t,
            model.x[0],
            model.x[1],
            model._k,
            arg,
            self.name
        )
