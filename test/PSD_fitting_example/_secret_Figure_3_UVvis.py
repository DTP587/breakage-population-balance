import os, sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import minimize as minimise
from breakage_population_balance import breakageModel
from breakage_population_balance.arrayManipulation import \
	fractional_percentage, fractional_percentage_array

from Extract_UVvis import SpeedSweep
from test_fitting_UVvis import interp
from test_fitting import \
	PSMs, error_calculator, process_args, extract_frac_psm, run_simulation, \
	x, calc_fracs
# =========================================================================== #

plt.style.use("seaborn-paper")
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

minimized_params = {
	"18500": np.array([1.70495863, 2.31787534, 4.96931008, 1.26]),
	"11500": np.array([3.00688466, 2.39594046, 4.81611375, 1.6]),
	"6500" : np.array([2.85952432, 2.87712493, 3.5529407,  0.23]),
	"3800" : np.array([7.74102875, 2.92963893, 2.73061484, 0.1]),
	"1500" : np.array([9.763026,   3.39399262, 5.2120364,  0]),
}

fit_params = {
	"18500": np.array([2.253, 1.987, 5.256, 1.68]),
	"11500": np.array([3.007, 2.396, 4.816, 1.60]),
	"6500" : np.array([3.550, 2.868, 3.849, 0.61]),
	"3800" : np.array([5.694, 3.377, 3.181, 0.36]),
	"1500" : np.array([7.410, 3.914, 2.766, 0]),
}

combined_fit = {
	# "18500": np.array([2.253,      1.987,      5.21, 1.6]),
	# "11500": np.array([3.00688466, 2.39594046, 4.82, 1.6]),
	# "6500" : np.array([2.85952432, 2.87712493, 4.8,  1.6]),
	# "3800" : np.array([2.4,        5.31,       4.26, 1.6]),
	"18500" : np.array([2.83068838, 1.90351529, 5.05717875, 1.6]),
	"11500" : np.array([3.85815191, 2.15005062, 4.83796328, 1.6]),
	"6500"  : np.array([3.00997403, 2.74385440, 4.88068714, 1.6]),
	"3800"  : np.array([4.86999577, 3.57931287, 4.10588837, 1.6]),
}
# {
# 	"18500": np.array([2,   2.2, 4.7, 1]),
# 	"11500": np.array([3,   2.4, 4.5, 1]),
# 	"6500" : np.array([3.6, 2.7, 4.3, 1]),
# 	"3800" : np.array([5.3, 3.3, 3.9, 1]),
# 	"1500" : np.array([9,     4, 3.7, 0]),
# }

fit_params = combined_fit

colors = {
	"18500": "#5F4690",
	"11500": "#38A6A5",
	"6500" : "#73AF48",
	"3800" : "#E17C05",
	"1500" : "#94346E",
}

t=np.linspace(0, 3000, 600+1, dtype=np.float64)

def plot_UVvis():
	mtoum = 1e6

	fig, ax = plt.subplots(figsize=[8,6])

	for speed, array in fit_params.items():
		color = colors[speed]

		if array[-1] == 0:
			break # Don't run 1500 speed

		solution = run_simulation(array[:-1], t=t)
		y_pred = calc_fracs(solution)
	
		interpolated = interp(solution, array[-1]*1e-6)[0]
		times = SpeedSweep[speed][:, 0]
		yields = SpeedSweep[speed][:, 3]/SpeedSweep[speed][:, 2]

		ax.loglog(
			t[1:], interpolated, color=color
		)
		ax.scatter(
			times, yields, color=color, alpha=0.4, label=f"{speed}"
		)

	ax.set_ylim([1e-5, 1e-2])
	ax.set_xlim([5, 3e3])

	ax.set_ylabel(r"Yield [-]")
	ax.set_xlabel(r"Time [$\bf{s}$]")

	plt.legend(title=r'$\Omega$ [rpm]', loc="upper left", framealpha=0.6)

	plt.tight_layout()
	# # plt.show(block=True)
	plt.savefig(os.path.splitext(__file__)[0] + '.pdf')

if __name__=="__main__":
	plot_UVvis()