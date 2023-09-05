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

# --------------------------------------------------------------------------- #
# Fraction based.

def pyf_simple_breakage(F, t, x, k, phi, axis=1):
    dFdt = np.zeros_like(F)
    # Death breakup term
    dFdt -= F * k
    # Birth breakup term
    dFdt += np.nansum(
        phi * k * F,
        axis=axis
    )
    return dFdt

@cython.boundscheck(False)
@cython.wraparound(False)
def cyf_2D_breakage(
    np.ndarray[DTYPE_FLOAT_t, ndim=2] F,
    DTYPE_FLOAT_t t,
    np.ndarray[DTYPE_FLOAT_t, ndim=1] x,
    np.ndarray[DTYPE_FLOAT_t, ndim=1] y,
    np.ndarray[DTYPE_FLOAT_t, ndim=2] rate,
    np.ndarray[DTYPE_FLOAT_t, ndim=4] kernel
):

    cdef int bins_i = F.shape[0]
    cdef int bins_j = F.shape[1]

    cdef int i # x
    cdef int j # y
    cdef int k # x'
    cdef int l # y'

    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] f = np.zeros_like(F)

    f -= rate * F

    for i in np.arange(bins_i):
        for j in np.arange(bins_j):
            for k in np.arange(i+1, bins_i):
                for l in np.arange(j+1, bins_j):

                    f[i, j] += rate[k, l] * kernel[i, j, k, l] * F[k, l]

    return f

@cython.boundscheck(False)
@cython.wraparound(False)
def Euler_2D(
    np.ndarray[DTYPE_FLOAT_t, ndim=2] F,
    np.ndarray[DTYPE_FLOAT_t, ndim=1] t,
    np.ndarray[DTYPE_FLOAT_t, ndim=1] x,
    np.ndarray[DTYPE_FLOAT_t, ndim=1] y,
    np.ndarray[DTYPE_FLOAT_t, ndim=2] rate,
    np.ndarray[DTYPE_FLOAT_t, ndim=4] kernel,
    str ode
):
    cdef int bins_t = t.shape[0]
    cdef int ti
    cdef DTYPE_FLOAT_t dt = np.ediff1d(t)[0]

    cdef np.ndarray[DTYPE_FLOAT_t, ndim=3] sol = np.zeros((bins_t, F.shape[0], F.shape[1]))

    if ode == "cyf_2D_breakage":
        function = cyf_2D_breakage
    else:
        raise ValueError("Unrecognised ode")

    sol[0] = F.copy()

    for ti in range(1, bins_t):
        sol[ti] = sol[ti-1] + dt*function(sol[ti-1], dt*ti, x, y, rate, kernel)

    return sol

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
