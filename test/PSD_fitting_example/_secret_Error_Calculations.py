import pandas as pd
import numpy as np
from scipy.optimize import minimize as minimise
import breakage_population_balance as bpb
from breakage_population_balance import exampleIC


from Extract_UVvis import SpeedSweep

# SpeedSweep["1500"] = np.array([
# 	[ 60,      200,     20,      1e-12],
# 	[ 120,     200,     20,      1e-12],
# 	[ 300,     200,     20,      1e-12],
# 	[ 600,     200,     20,      1e-12],
# 	[ 1800,    200,     20,      1e-12]
# ])

# much better way of storing the data
PSMs = {
	name: pd.read_excel('Extract-PSD.xlsx', name, header=0, index_col=0) \
	for name in ['1500', '3800', '6500', '11500', '18500']
}



# run sim

t = np.logspace(np.log10(5), np.log10(3e3), 600+1, dtype=np.float64)
x = exampleIC.x
zero = exampleIC.IC

beas = np.array([3.85815191, 2.15005062, 4.83796328]) # params

def run_simulation(array, t=t):
	b=array[0]
	e=array[1]
	a=array[2]
	# rate function
	k = lambda x: (x/(b*1e-3))**e

	# kernel function
	def Phi(x, y):
		if x < y:
			# Spherical chickens in a vacuum
			v_ = (1/6) * np.pi * y ** 3
			v  = (1/6) * np.pi * x ** 3
			return ( v / v_ )**(a*1e-1)
		else:
			return 0
	Phi = np.vectorize(Phi, otypes=[np.float64])

	# Run and solve the breakage problem
	model = bpb.breakageModel(zero, t, x, k, kernel=Phi)
	solution = model.solve("pyf_simple_breakage")

	return solution

solution = run_simulation(beas, t=t)

y50 = bpb.arrayManipulation.fractional_percentage_array(
	solution, x, 0.5
)


def sum_fractions(solution, section_value: float):
    interpolated = None

    ix = np.where(x < section_value)[0][-1]

    fraction = (section_value - x[ix]) / (x[ix+1] - x[ix])
    lower  = np.sum(solution[1:, 0:(ix+1)], axis=1)
    difference = np.sum(solution[1:, 0:(ix+2)], axis=1) - lower

    interpolated = lower + difference*fraction

    return interpolated, fraction

def interp_time(sol, time: float, renorm=True):
    interpolated = None

    upper = np.argmax((t - time)>0)
    lower = upper - 1

    lower_frac = (time - t[lower]) / (t[upper] - t[lower])
    upper_frac = (t[upper] - time) / (t[upper] - t[lower])
    interpolated = sol[lower]*lower_frac + sol[upper]*upper_frac

    if renorm:
	    interpolated[interpolated>0] = \
	        interpolated[interpolated>0]/np.sum(interpolated)

    return interpolated

def error_calculator(sol, PSM, uv=None, cutoff=1.6e-6):
	# expect solution and PSM to be distributed into equal shaped bins

	eps_psd = 0 # zero error
	u_cutoff = PSM.index.to_numpy() < cutoff

	# Add PSD error
	for time in PSM:
		fi   = interp_time(sol, int(time))
		psd  = PSM[time].to_numpy()

		abs_err = np.abs(psd[~u_cutoff]-fi[~u_cutoff])
		not0 = psd[~u_cutoff] > 0

		eps_psd += np.sum(abs_err[not0]/psd[~u_cutoff][not0])/psd.shape[0]

	eps_psd /= len(PSM.columns) #psd[~u_cutoff].shape[0]

	eps_uv = 0
	if uv is not None:
		f = sum_fractions(sol, cutoff)[0]
		yields = uv[:, 3]/uv[:, 2]
		
		for i, time in enumerate(uv[:, 0]):
			yi = yields[i]
			yt = interp_time(f, time, renorm=False)

			eps_uv += np.sum(np.abs(yi - yt)/yi)

		eps_uv /= uv[:, 0].shape[0]

	print(eps_psd, eps_uv)

	return eps_psd+eps_uv



if __name__=="__main__":

	# interpolated = interp_time(solution, 1800)

	from matplotlib import pyplot as plt

	speed = '6500'

	PSM = PSMs[speed]

	print(PSM, SpeedSweep[speed])

	# def error_routine(array):
	# 	return error_calculator(run_simulation(array), PSM=PSM, uv=SpeedSweep[speed])

	# res = minimise(
	# 	error_routine, x0=np.array([3,2.3,4.9]),#[3.007, 2.396, 4.816]),#[5, 5, 5]),
	# 	bounds = ((1, None), (1, None), (1, None)),
	# 	options={'disp': True, 'maxiter': 10},
	# 	method='L-BFGS-B',
	# 	tol=1e-3
	# )
	# print(res.x)

	# solution = run_simulation(res.x)

	# arr = np.array([
	# 	[1500, 5.84342169, 5.68727791, 5.95542590], # [5.11195756 5.78240063 5.45470921] # 1500, uv is set False
	# 	[3800, 3.17039991, 2.97786914, 5.29315829], #
	# 	[6500, 3.03279535, 2.47731615, 5.01266793], #
	# 	[11500,2.99839725, 2.29059090, 4.89680198], #
	# 	[18500,2.86655445, 1.97819016, 4.90134592], # different ICs [2, 1, 6]
	# ])

	# plt.plot(arr[:, 0], arr[:, 1], label='b')
	# plt.plot(arr[:, 0], arr[:, 2], label='e')
	# plt.plot(arr[:, 0], arr[:, 3], label='a')
	# plt.legend(loc="upper right")
	plt.xscale('log')
	plt.show()