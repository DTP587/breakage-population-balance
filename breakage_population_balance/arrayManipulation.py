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
	return func(*np.meshgrid(*grid, indexing='ij'))

def apply_kernel(grid, kernel, normalise=False):
	NUM_DIMS = len(grid)

	EXT_DIMS = [
		tuple([ i - pos for i in range(2*NUM_DIMS - 1) ]) \
			for pos in range(2*NUM_DIMS)
	]
	print(EXT_DIMS)

	out_kernel = kernel( *[ np.expand_dims(x, axis=axis) \
		for x, axis in zip(2*grid, EXT_DIMS) ] )

	if normalise:
		np.seterr(invalid='ignore')
		norms = np.linalg.norm(out_kernel, 1, axis=1, keepdims=True)
		out_kernel /= norms
		np.seterr(invalid='warn')

	return out_kernel.T.astype(np.float64)


# =========================================================================== #
# testing

if __name__ == "__main__":
	# do function testing
	x = np.array([1, 2, 3, 4], dtype=np.float64)
	y = np.array([0.1, 0.2, 0.3], dtype=np.float64)
	z = np.array([0.0001, 0.0002, 0.0003, 0.0004, 0.0005], dtype=np.float64)


	# 1D
	foo1  = lambda x: x**2
	# parent x_, child x
	kernel1  = lambda x, x_: (x)**2 + (x_)**2

	print("\nExecuting 1D tests:\n")
	print(x)
	kern1d = apply_kernel([x], kernel1, normalise=True)
	rate1d = apply_function([x], foo1)
	print(f"rate 1D:\n{rate1d}")
	print(f"kernel 1D:\n{kern1d}")
	

	# 2D
	foo2 = lambda x, y: x**2 - y
	# parent x_, y_, child x, y
	kernel2  = lambda x, y, x_, y_: (x/x_)**2 - y/y_

	print("\nExecuting 2D tests:\n")
	print(x, y)
	kern2d = apply_kernel([x, y], kernel2, normalise=True)
	rate2d = apply_function([x, y], foo2)
	print(f"rate 2D:\n{rate2d}")
	print(f"kernel 2D:\n{kern2d}")



	# 3D
	foo3 = lambda x, y, z: x**2 - y + z
	# parent x_, y_, z_, child x, y, z
	kernel3 = lambda x, y, z, x_, y_, z_: x/x_**2 - y/y_ + z/z_
	
	# print("\nExecuting 3D tests:\n")
	# print(x, y, z)
	# kern3d = apply_kernel([x, y, z], kernel3, normalise=True)
	# print(f"kernel 3D:\n{kern3d}")