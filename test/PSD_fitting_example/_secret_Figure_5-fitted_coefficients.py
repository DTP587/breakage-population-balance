import os, sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import minimize as minimise
from breakage_population_balance import breakageModel
from breakage_population_balance.arrayManipulation import \
	fractional_percentage, fractional_percentage_array

from test_fitting_UVvis import interp
from test_fitting import \
	PSMs, error_calculator, process_args, extract_frac_psm, run_simulation, \
	x, calc_fracs
# =========================================================================== #

plt.style.use("seaborn-paper")
plt.rc('font', size=14)          # controls default text sizes
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize

fitting_vals = {       # B    # E    # A    # S
	"18500": np.array([2.253, 1.987, 5.256, 1.68]),
	"11500": np.array([3.007, 2.396, 4.816, 1.60]),
	"6500" : np.array([3.550, 2.868, 3.849, 0.61]),
	"3800" : np.array([5.694, 3.377, 3.181, 0.36]),
	"1500" : np.array([7.410, 3.914, 2.766, 0.00]),
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
	# "1500"  : np.array([6.28, 4.56, 3.91, 1.6]),
}

linear_fit = {
	"18500": np.array([2.0076375, 2.0332948, 5.1428694, 1.74960230]),
	"11500": np.array([3.0362662, 2.4036750, 4.6537544, 1.31334890]),
	"6500" : np.array([4.2706994, 2.8481598, 4.0547782, 0.78981137]),
	"3800" : np.array([5.4321245, 3.2663563, 3.4912273, 0.29723744]),
	"1500" : np.array([7.4432723, 3.9905141, 2.5153708, 0]),
}

minimized_params = {
	"18500": np.array([1.70495863, 2.31787534, 4.96931008, 1.26]),
	"11500": np.array([3.00688466, 2.39594046, 4.81611375, 1.6]),
	"6500" : np.array([2.85952432, 2.87712493, 3.5529407,  0.23]),
	"3800" : np.array([7.74102875, 2.92963893, 2.73061484, 0.1]),
	"1500" : np.array([9.763026,   3.39399262, 5.2120364,  0]),
}

fitting_vals = combined_fit

if __name__ == "__main__":
	mtoum = 1e6

	fig, ax = plt.subplots(figsize=[8,6])

	BEAS   = {r"B$\times10^3$": [], "E": [], r"A$\times10^1$": [], r"S $\mu m$": []}
	linear = {r"B$\times10^3$": [], "E": [], r"A$\times10^1$": [], r"S $\mu m$": []}

	for speed, array in fitting_vals.items():
		for i, key in enumerate(BEAS.keys()):
			BEAS[key].append((float(speed), array[i]))
			linear[key].append((float(speed), linear_fit[speed][i]))

	# positions = [[0.04,0.76], [0.05,0.43], [0.9,0.55], [0.5,0.1]]
	positions = [[0.07,0.76], [0.05,0.43], [0.9,0.51], [0.45, 0.12]]
	# angles = [-26, -10, 17, 10]
	angles = [-28, -10, 8, 0]

	for key, pos, angle in zip(BEAS.keys(), positions, angles):
		BEAS[key] = np.array(BEAS[key])
		linear[key] = np.array(linear[key])

		line, = ax.plot(
			BEAS[key][:, 0], BEAS[key][:, 1],
			marker="o", markersize=10, linestyle=":", linewidth=0.9, label=key,
			alpha=0.5
		)
		p = np.polyfit(np.log(BEAS[key][:, 0]), BEAS[key][:, 1], 1)

		f = lambda i: p[0]*np.log(i) + p[1]

		# ax.plot((1e3, 3e4), (f(1e3), f(3e4)), c=line.get_color(), linestyle=":", linewidth=0.75)
		# ax.annotate(key, rotation=angle, xy=pos, c=line.get_color(), size=12,
		# 	xycoords='axes fraction', ha='left', weight="bold")

	plt.xscale("log")
	plt.ylim([  0,  10])
	plt.xlim([1e3, 3e4])
	plt.ylabel("Value [-]")
	plt.xlabel("Speed [rpm]")
	plt.legend(loc="upper right")
	plt.tight_layout()
	plt.savefig(os.path.splitext(__file__)[0] + '.png', dpi=400)

	# B*1e-3
	# E*1
	# A*1e-1
	# S*1e-6
