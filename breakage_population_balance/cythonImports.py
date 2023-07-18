from glob import glob
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

VALID_METHODS = [ method for method in dir(breakageODE) \
    if method not in [
        'DTYPE_FLOAT', 'DTYPE_INT', '__builtins__', '__doc__', '__file__',
        '__loader__', '__name__', '__package__', '__spec__', '__test__', 'np'
    ]
]