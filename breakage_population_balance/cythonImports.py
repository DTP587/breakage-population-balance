from glob import glob
from importlib import util
import os

breakageODE = None
arrayCythFuncs = None

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

def glob_possible_dirs(MODULE_NAME):
    return [
        glob(path) for path in [
            os.path.join(os.path.dirname(__file__), f"{MODULE_NAME}.*.so"),
            os.path.join(
                os.path.dirname(__file__), "..", "build", "lib*", "breakage*",
                f"{MODULE_NAME}.*.so"
            )
        ]
    ]

for module in ["breakageODE", "arrayCythFuncs"]:
    globbed = glob_possible_dirs(module)
    exec(f"{module} = import_cython(module, globbed)")

VALID_ODES = [ method for method in dir(breakageODE) \
    if method not in [
        'DTYPE_FLOAT', 'DTYPE_INT', '__builtins__', '__doc__', '__file__',
        '__loader__', '__name__', '__package__', '__spec__', '__test__', 'np'
    ]
]