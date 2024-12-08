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
plt.rc('font', size=10)          # controls default text sizes
plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=12)    # legend fontsize

fit_params = {
	"18500": np.array([2.253, 1.987, 5.256, 1.68]),
	"11500": np.array([3.007, 2.396, 4.816, 1.50]),
	"6500" : np.array([3.550, 2.868, 3.849, 0.61]),
	"3800" : np.array([5.694, 3.377, 3.181, 0.36]),
	"1500" : np.array([7.410, 3.914, 2.766, 0]),
}


combined_fit = {
	"18500": np.array([2,   2.2, 4.7, 1]),
	"11500": np.array([3,   2.4, 4.5, 1]),
	"6500" : np.array([3.6, 2.7, 4.3, 1]),
	"3800" : np.array([5.3, 3.3, 3.9, 1]),
	"1500" : np.array([9,     4, 3.7, 1]),
}

combined_fit = {
	"18500" : np.array([2.83068838, 1.90351529, 5.05717875, 1.6]),
	"11500" : np.array([3.85815191, 2.15005062, 4.83796328, 1.6]),
	"6500"  : np.array([3.00997403, 2.74385440, 4.88068714, 1.6]),
	"3800"  : np.array([4.86999577, 3.57931287, 4.10588837, 1.6]),
	"1500"  : np.array([5.08672399, 4.73651531, 3.92142023, 1.6]),
}

fit_params=combined_fit

t = np.logspace(-2, np.log(3e3), 600+1, dtype=np.float64)
# t = np.linspace(0, 3000, 600+1, dtype=np.float64)

def plot_heatmap(speed, array, y_fracs):
	b=array[0]
	e=array[1]
	a=array[2]
	s=array[3]

	SLIDER_MAX = 10

	mtoum = 1e6

	solution = run_simulation(array[:-1], t=t)
	y_pred = calc_fracs(solution)

	if s != 0:
		interpolated = interp(solution, s*1e-6)[0]
		times = SpeedSweep[speed][:, 0]
		yields = SpeedSweep[speed][:, 3]/SpeedSweep[speed][:, 2]

	fig, ax = plt.subplots(figsize=[8,6])

	ax.pcolor(
		t[1:], x*mtoum, solution[1:].T, cmap='cividis', label="Numerical Solution"
	)
	ax.loglog(
		t, y_pred["0.5"]*mtoum, linestyle="--", alpha=0.4, linewidth=3, color="red", label=r"Forecasted D(50%)"
	)
	ax.errorbar(
		y_fracs["0.5"][:, 0], y_fracs["0.5"][:, 1]*mtoum,
		yerr=[
			(y_fracs["0.5"][:, 1]-y_fracs["0.1"][:, 1])*mtoum,
			(y_fracs["0.9"][:, 1]-y_fracs["0.5"][:, 1])*mtoum
		],
		color="lime", fmt='o', alpha=0.6, capsize=4, linewidth=1.9, linestyle='', label="Particle Size Measurements"
	)

	ax.set_ylim([1, 2e3])
	ax.set_xlim([5, 3e3])
	ax.set_yscale("log")
	ax.set_xscale("log")

	ax.set_ylabel(r"Size [$\bf{\mu m}$]")
	ax.set_xlabel(r"Time [$\bf{s}$]")

	plt.legend(loc="lower left", framealpha=0.6)

	plt.tight_layout()
	# plt.show(block=True)
	plt.savefig(os.path.splitext(__file__)[0] + '.png', dpi=400)

def plot_UVvis(speed, array, y_fracs):
	b=array[0]
	e=array[1]
	a=array[2]
	s=array[3]

	SLIDER_MAX = 10

	mtoum = 1e6

	solution = run_simulation(array[:-1], t=t)
	y_pred = calc_fracs(solution)

	if s == 0:
		print("No fitting coefficient specified (probably for a reason)")
		return
	
	interpolated = interp(solution, s*1e-6)[0]
	times = SpeedSweep[speed][:, 0]
	yields = SpeedSweep[speed][:, 3]/SpeedSweep[speed][:, 2]

	fig, ax = plt.subplots(figsize=[8,6])#figsize=[8,6])

	ax.loglog(t[1:], interpolated, label="PSM prediction")
	ax.scatter(times, yields, label="Concentration", alpha=0.4)

	ax.set_ylim([1e-5, 1e-2])
	ax.set_xlim([5, 3e3])

	ax.set_ylabel(r"Size [$\bf{\mu m}$]")
	ax.set_xlabel(r"Time [$\bf{s}$]")

	plt.legend(loc="upper left", framealpha=0.6)

	plt.tight_layout()
	# # plt.show(block=True)
	plt.savefig(os.path.splitext(__file__)[0] + '_Uvvis.png', dpi=400)

if __name__=="__main__":
	speed = process_args(sys.argv)[1]

	PSM, y_fracs = extract_frac_psm(speed)
	plot_heatmap(speed, fit_params[speed], y_fracs)
	plot_UVvis(speed, fit_params[speed], y_fracs)

