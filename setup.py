from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

#python setup.py build_ext --inplace && mv PBE_Modelling.cpython-39-darwin.so PBE_Modelling.so && rm -rf PBE_Modelling.c

setup(
    packages=['breakage_population_balance'],
    ext_modules=cythonize([
        "breakage_population_balance/breakageODE.pyx",
        "breakage_population_balance/arrayCythFuncs.pyx"
    ]),
    include_dirs=[np.get_include()],
    zip_safe=False
)
