import os, sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from breakage_population_balance import breakageModel
from breakage_population_balance.arrayManipulation import \
	fractional_percentage, fractional_percentage_array

from Extract_NMR import NMR11500RAW
from test_fitting import run_simulation, x
# =========================================================================== #

mpl.rcParams.update(mpl.rcParamsDefault)
# plt.style.use("seaborn-paper")
# plt.rc('font', size=14)          # controls default text sizes
# plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=14)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=14)    # fontsize of the tick labels
# plt.rc('legend', fontsize=16)    # legend fontsize

B = 3.85815191 #3.007
E = 2.15005062 #2.396
A = 4.83796328 #4.816

B2 = 2.83068838
E2 = 1.90351529
A2 = 5.05717875

B3 = 3.00997403
E3 = 2.74385440
A3 = 4.88068714

# 3.00997403, 2.74385440, 4.88068714, 1.6
# 3.85815191, 2.15005062, 4.83796328, 1.6
# 2.83068838, 1.90351529, 5.05717875, 1.6

# Specific surface area determined from PSDs
PSD_area = np.array([
	[  10, 0.0296],
	[  20, 0.0352],
	[  30, 0.0379],
	[  40, 0.0524],
	[  50, 0.0452],
	[  60, 0.0564],
	[ 120, 0.0596],
	[ 300, 0.0815],
	[ 600, 0.0946],
	[1800, 0.1380]
])

# x used is spere of same volume
volume = (1/6)*np.pi*x**3
# Surface area of a sphere
# Equivalently the minimum surface area for the volume.
area = np.pi*x**2
# We expect a rough surface. https://doi.org/10.1007/s11249-014-0313-4
# Corresponding volume diameter to surface area is not 1:1
fractal_area = np.pi*x**2.3
# https://www.youtube.com/watch?app=desktop&v=IuMyLqnfMss&ab_channel=SalimAhmed
# I am not 100% about the measurent above. I think it is a fine assumption,
# but I believe there is a better way to derive the specific surface area.
# fractal_area = np.pi*x**2.2

# don't particularly need these as np arrays
times = list(NMR11500RAW[:, 0])
t2inv = list(NMR11500RAW[:, 1])

# t2step = lambda time: round((time*600)/3000)

t=np.logspace(-1, np.log10(3000), 600+1, dtype=np.float64)

colors = iter([plt.cm.tab10(i) for i in range(4)])

if __name__ == "__main__":
	solution = run_simulation(np.array([B, E, A]), t=t)
	# solution2 = run_simulation(np.array([B2, E2, A2]), t=t)
	# solution3 = run_simulation(np.array([B3, E3, A3]), t=t)

	def calc_areas(const, sol, area=fractal_area):
		return np.array([
			[ t[i], const*np.sum((psd/volume)*area) ] for i, psd in enumerate(sol[1:]) 
		])

	areas = calc_areas(2.8e-8, solution, area=area)
	fractal_areas = calc_areas(5.8e-7, solution)
	# fractal_areas2 = calc_areas(5.8e-7, solution2)
	# fractal_areas3 = calc_areas(5.8e-7, solution3)

	fig, ax = plt.subplots()

	# plt.style.use("seaborn-paper")
	# plt.ion()

	marker_style = dict(
		s=70, edgecolors="w", linewidths=0.6, alpha=0.6
	)

	NMR_Color = next(colors)
	# ax.loglog(
	# 	fractal_areas2[:, 0], fractal_areas2[:, 1], label=r"18500",
	# 	color=next(colors), zorder=1
	# )
	# ax.loglog(
	# 	fractal_areas3[:, 0], fractal_areas3[:, 1], label=r"6500",
	# 	color=next(colors), zorder=1
	# )
	ax.scatter(
		times, t2inv, color=NMR_Color, **marker_style, zorder=2 #, label="NMR"
	)
	ax.loglog(
		fractal_areas[:, 0], fractal_areas[:, 1], label=r"$x_v^{2.3}$",
		color=NMR_Color, zorder=1
	)
	ax.loglog(
		areas[:, 0], areas[:, 1], label=r"$x_v^{2 }$",
		color=next(colors), zorder=0, linestyle='--'
	)
	# ax.scatter(
	# 	PSD_area[:, 0], 2.8e-2*PSD_area[:, 1], label="PSD", color=next(colors), **marker_style, zorder=2
	# )

	ax.legend(loc="upper left") # , fontsize=16
	ax.set_ylabel(r"1 / $T_2$ [ms$^{-1}$]")
	ax.set_xlabel(r"Exfoliation time [s]")

	ax.set_ylim([0.7e-3, 1e-2])
	ax.set_xlim([5, 3e3])

	plt.tight_layout()
	plt.savefig('Extract_NMR.pdf')

