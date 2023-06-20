import numpy as np
cimport numpy as np
import cython

# npconfig is required to disable depreciated API warning.
# see: https://stackoverflow.com/questions/25789055/cython-numpy-warning-about-npy-no-deprecated-api-when-using-memoryview
cimport npconfig

np.import_array()

DTYPE_INT = np.int64
DTYPE_FLOAT = np.float64

ctypedef np.int64_t DTYPE_INT_t
ctypedef np.float64_t DTYPE_FLOAT_t

# For use with scipy's ODEINT
@cython.boundscheck(False)
@cython.wraparound(False)
def simple_breakage(
    np.ndarray[DTYPE_FLOAT_t, ndim=1] N,
    DTYPE_FLOAT_t t,
    np.ndarray[DTYPE_FLOAT_t, ndim=1] x,
    np.ndarray[DTYPE_FLOAT_t, ndim=1] rate,
    np.ndarray[DTYPE_FLOAT_t, ndim=2] kernel
):
    
    cdef int bins = N.shape[0]

    cdef int i
    cdef int j

    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] n = np.zeros(bins)

    n -= rate * N

    for i in np.arange(bins):
        for j in np.arange(i+1, bins):
            n[i] += rate[j] * kernel[i, j] * N[j]

    return n
