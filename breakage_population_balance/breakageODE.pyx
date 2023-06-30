import numpy as np
cimport numpy as np
import cython

# npconfig is required to disable depreciated API warning.
# see: <https://stackoverflow.com/questions/25789055/cython-numpy-warning-about
#       -npy-no-deprecated-api-when-using-memoryview>
cimport npconfig

np.import_array()

DTYPE_INT = np.int64
DTYPE_FLOAT = np.float64

ctypedef np.int64_t DTYPE_INT_t
ctypedef np.float64_t DTYPE_FLOAT_t

# =========================================================================== #
# For use with scipy's ODEINT

# --------------------------------------------------------------------------- #
# Fraction based.

def pyf_simple_breakage(F, t, x, k, phi):
    dFdt = np.zeros_like(F)
    # Death breakup term
    dFdt -= F * k
    # Birth breakup term
    dFdt += np.nansum(
        phi * np.triu(
            np.tile(k*F, (phi.shape[-1], 1))
        ),
        axis=1
    )
    return dFdt

@cython.boundscheck(False)
@cython.wraparound(False)
def cyf_simple_breakage(
    np.ndarray[DTYPE_FLOAT_t, ndim=1] F,
    DTYPE_FLOAT_t t,
    np.ndarray[DTYPE_FLOAT_t, ndim=1] x,
    np.ndarray[DTYPE_FLOAT_t, ndim=1] rate,
    np.ndarray[DTYPE_FLOAT_t, ndim=2] kernel
):
    
    cdef int bins = F.shape[0]

    cdef int i
    cdef int j

    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] f = np.zeros(bins)

    f -= rate * F

    for i in np.arange(bins):
        for j in np.arange(i+1, bins):
            f[i] += rate[j] * kernel[i, j] * F[j]

    return f

# --------------------------------------------------------------------------- #
# Number based.

@cython.boundscheck(False)
@cython.wraparound(False)
def cyn_simple_breakage(
    np.ndarray[DTYPE_FLOAT_t, ndim=1] N, 
    DTYPE_FLOAT_t t,
    np.ndarray[DTYPE_FLOAT_t, ndim=1] x,
    np.ndarray[DTYPE_FLOAT_t, ndim=1] rate,
    np.ndarray[DTYPE_FLOAT_t, ndim=2] beta
):
    
    cdef int bins = N.shape[0]

    cdef int i
    cdef int j

    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] n = np.zeros(bins)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] diff = np.ediff1d(x)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] dx = \
        np.concatenate([np.array([diff[0]]), diff]).ravel()

    n -= rate * N

    for i in np.arange(bins):
        for j in np.arange(i+1, bins):
            n[i] += rate[j] * beta[i, j] * N[j] * dx[j]

    return n
