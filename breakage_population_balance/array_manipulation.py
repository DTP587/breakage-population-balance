import numpy as np
import os, inspect

# =========================================================================== #
# Checking arrays

def check_ndarray(*args, **kwargs):
    if not isinstance(args[1], np.ndarray):
        raise ValueError(f"Argument '{args[1]}' is not of type numpy.ndarray.")

def wrapper_check_ndarray(func):
    def inner(*args, **kwargs):
        check_ndarray(*args, **kwargs)
        return func(*args, **kwargs)
    return inner

# =========================================================================== #
# Calculating kernels, rates

# For a points based system
def apply_to_points(grid, func):
	grid = grid if len(grid.shape)>1 else grid[:, np.newaxis]
	return func(*(grid[...,i] for i in range(grid.shape[-1])))

# For a meshgrid system
def apply_function(grid, func):
    grid = grid if not isinstance(grid, np.ndarray) else [grid]
    if not (len(grid)==len(inspect.getfullargspec(func).args)):
        raise ValueError("Please check the Dimensions of the inputted grid " +
            "match the number of function arguments")
    return func(*grid)

def apply_kernel(grid, kernel, normalise=False):
    grid = grid if not isinstance(grid, np.ndarray) else [grid]

    out_kernel = kernel(*grid, *[g[:, np.newaxis] for g in grid])

    if normalise:
    	norms = out_kernel.sum(axis=1)
    	nonzero = norms != 0
    	out_kernel[nonzero, :] /= norms[nonzero, np.newaxis]
    
    return out_kernel.T.astype(np.float64)

# =========================================================================== #
# testing

if __name__ == "__main__":
	# do function testing
	x = np.array([1, 2, 3, 4], dtype=np.float64)

	y = np.array([0.1, 0.2, 0.3], dtype=np.float64)

	z = np.array([0.0001, 0.0002, 0.0003, 0.0004], dtype=np.float64)

	# 1D

	foo1  = lambda x: x**2
	# parent x_, child x
	kernel1  = lambda x, x_: (x)**2 + (x_)**2

	# print(apply_function(x, foo1))
	print(apply_kernel(x, kernel1, normalise=True))


	# 2D

	xy = np.meshgrid(x, y)
	foo2 = lambda x, y: x**2 - y
	# parent x_, y_, child x, y
	kernel2  = lambda x, y, x_, y_: (x/x_)**2 - y/y_

	# Example of a points based - 2D system
	grid2 = np.array(
		[
			[[1, 0.1], [1, 0.2], [1, 0.3]],
			[[2, 0.1], [2, 0.2], [2, 0.3]],
			[[3, 0.1], [3, 0.2], [3, 0.3]],
			[[4, 0.1], [4, 0.2], [4, 0.3]]
		]
	)
	# print(apply_to_points(grid2, foo2))
	# print(apply_function(xy, foo2))
	print(apply_kernel(xy, kernel2, normalise=True))


	# 3D

	xyz = np.meshgrid(x, y, z)
	foo3 = lambda x, y, z: x**2 - y + z
	# parent x_, y_, z_, child x, y, z
	kernel3 = lambda x, y, z, x_, y_, z_: x/x_**2 - y/y_ + z/z_

	# print(apply_function(xyz, foo3))
	# print(apply_kernel(xyz, kernel3))