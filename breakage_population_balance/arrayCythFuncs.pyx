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

@cython.boundscheck(False)
@cython.wraparound(False)
def fractional_percentage(
    np.ndarray[DTYPE_FLOAT_t, ndim=1] f,
    np.ndarray[DTYPE_FLOAT_t, ndim=1] x,
    DTYPE_FLOAT_t frac
):
    cumsum = np.cumsum(f) # Should sum to 1
    shifted = cumsum-frac # Shift lower
    i = np.where(np.diff(np.sign(shifted)))[0] # Find where 0 cross is
    sum_abs = np.abs(shifted) # get the abs value
    fraction = sum_abs[i]/(sum_abs[i] + sum_abs[i+1]) # calculate the fraction

    return ( x[i] + abs(x[i+1]-x[i])*fraction )

@cython.boundscheck(False)
@cython.wraparound(False)
def fractional_percentage_array(
    np.ndarray[DTYPE_FLOAT_t, ndim=2] f,
    np.ndarray[DTYPE_FLOAT_t, ndim=1] x,
    DTYPE_FLOAT_t frac
):
    """
    Applies fractional_percentage to a 2D array.

    f <np.ndarray, ndim=2> -> the solution array
    x <np.ndarray, ndim=1> -> the bins array
    frac <float> -> the fraction to calculate

    RETURNS:
    f_out <np.ndarray, ndim=2> -> the interpolated fraction bin corresponding
    to said frac argument.
    """
    cdef int times = f.shape[0]
    cdef int bins  = f.shape[1]
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] f_out  = f.copy()

    for time in range(times):
        cumsum = np.cumsum(f[time]) # Should sum to 1
        shifted = cumsum-frac # Shift lower
        i = np.where(np.diff(np.sign(shifted)))[0] # Find where 0 cross is
        sum_abs = np.abs(shifted) # get the abs value
        fraction = sum_abs[i]/(sum_abs[i] + sum_abs[i+1]) # calculate the fraction
        f_out[time] = ( x[i] + abs(x[i+1]-x[i])*fraction )

    return f_out

@cython.boundscheck(False)
@cython.wraparound(False)
def normalise_to_x(
    np.ndarray[DTYPE_FLOAT_t, ndim=2] N,
    np.ndarray[DTYPE_FLOAT_t, ndim=1] x
):
    cdef int times = N.shape[0]
    cdef int bins  = N.shape[1]
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] N_out = N.copy()

    cdef int time
    cdef int bin
    cdef DTYPE_FLOAT_t x_sum

    for time in range(times):
        x_sum = 0
        for bin in range(bins):
            x_sum += x[bin]*N[time, bin]
        N_out[time] = x*N[time, :]/x_sum
    return N_out