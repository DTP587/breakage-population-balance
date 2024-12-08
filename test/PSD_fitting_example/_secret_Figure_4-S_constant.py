import os, sys
import numpy as np
from matplotlib import pyplot as plt
from breakage_population_balance import breakageModel
from breakage_population_balance.arrayManipulation import \
	fractional_percentage, fractional_percentage_array

from test_fitting_UVvis import interp
from test_fitting import run_simulation, x, calc_fracs
# =========================================================================== #

plt.style.use("seaborn-paper")
plt.rc('font', size=14)          # controls default text sizes
plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize

S = 1

# fit_params = {       # B    # E    # A    # S
# 	"18500": np.array([2.253, 1.987, 5.256, S]),
# 	"11500": np.array([3.007, 2.396, 4.816, S]),
# 	"6500" : np.array([3.572, 2.868, 4.055, S]),

# 	# "3800" : np.array([5.694, 3.377, 3.181, S]),
# 	# "3800" : np.array([7.7437166, 2.9385023, 2.7306148, 0.36]),#0.29723744]),

# 	# "1500" : np.array([7.410, 3.914, 2.766, S]),
# 	# "1500" : np.array([9.7630260, 3.3939926, 5.2120364, 0]),
# 	# "1500" : np.array([6.5634207, 4.1794955, 2.6510477, 0]),
# 	# "1500" : np.array([30,        2.3381137, 4.6984859, 0]),
# }

linear_fit = {
	"18500": np.array([2.0076375, 2.0332948, 5.1428694, 1.74960230]),
	"11500": np.array([3.0362662, 2.4036750, 4.6537544, 1.31334890]),
	"6500" : np.array([4.2706994, 2.8481598, 4.0547782, 0.78981137]),
	"3800" : np.array([5.4321245, 3.2663563, 3.4912273, 0.29723744]),
	"1500" : np.array([7.4432723, 3.9905141, 2.5153708, 0]),
}


combined_fit = {
	"18.5" : np.array([2.83068838, 1.90351529, 5.05717875, 1.6]),
	"11.5" : np.array([3.85815191, 2.15005062, 4.83796328, 1.6]),
	"6.5"  : np.array([3.00997403, 2.74385440, 4.88068714, 1.6]),
	"3.8"  : np.array([4.86999577, 3.57931287, 4.10588837, 1.6]),
	"1.5"  : np.array([5.08672399, 4.73651531, 3.92142023, 1.6]),
}


fit_params = combined_fit

colors = {
	"18.5": "#5F4690",
	"11.5": "#38A6A5",
	"6.5" : "#73AF48",
	"3.8" : "#E17C05",
	"1.5" : "#94346E",
}

# linestyles = {
# 	"18.5": (5, 0),
# 	"11.5": (5, 2),
# 	"6.5" : (5, 4),
# 	"3.8" : (5, 6),
# 	"1.5" : (5, 8),
# }

t=np.logspace(-2, 8, 20000+1, dtype=np.float64)

if __name__=="__main__":
	mtoum = 1e6

	fig, ax = plt.subplots(figsize=[8,6])

	for speed, array in fit_params.items():
		color = colors[speed]

		if array[-1] == 0:
			break # Don't run 1500 speed

		solution = run_simulation(array[:-1], t=t)
		#y_pred = calc_fracs(solution, array=False)
	
		interpolated = interp(solution, array[-1]*1e-6)[0]

		ax.loglog(
			t[1:], interpolated, color=color,
			label=f"{speed}" #linestyle="--", dashes=linestyles[speed]
		)

	# ax.plot([0.1, 1e9], [1e-4, 1e-4]) # machine sensititvity

	ax.set_ylim([1e-6, 1])
	ax.set_xlim([0.1, 1e8])

	ax.set_ylabel(r"Yield [-]")
	ax.set_xlabel(r"Time [$\bf{s}$]")

	plt.legend(loc="upper left", framealpha=0.6, title=r"$\omega$ ($\bf{krpm}$)")

	plt.tight_layout()
	plt.show()
	# plt.savefig(os.path.splitext(__file__)[0] + '.pdf', dpi=400)