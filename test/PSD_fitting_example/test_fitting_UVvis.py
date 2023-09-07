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
from test_fitting import \
	PSMs, error_calculator, process_args, extract_frac_psm, run_simulation, x, t,\
	B_DEFAULT, E_DEFAULT, A_DEFAULT, calc_fracs
# =========================================================================== #

MAX_SIZE = 1600e-9  # We know that they are on average 200 nm.
                    # Changed from 1040 to 1600 nm when psds were fitted.

def interp(solution, section_value: float):
    interpolated = None

    ix = np.where(x < section_value)[0][-1]

    fraction = (section_value - x[ix]) / (x[ix+1] - x[ix])
    lower  = np.sum(solution[1:, 0:(ix+1)], axis=1)
    difference = np.sum(solution[1:, 0:(ix+2)], axis=1) - lower

    interpolated = lower + difference*fraction

    return interpolated, fraction


def plot_UVvis_with_PSDs(speed, array, y_fracs):
	b=array[0]
	e=array[1]
	a=array[2]
	s=1.6

	SLIDER_MAX = 10

	solution = run_simulation(array)
	y_pred = calc_fracs(solution)

	interpolated = interp(solution, s*1e-6)[0]

	times = SpeedSweep[speed][:, 0]
	yields = SpeedSweep[speed][:, 3]/SpeedSweep[speed][:, 2]

	slider_params = [
		('b', b, 0.01),
		('e', e, 0.01),
		('a', a, 0.01),
		('s', s, 0.01)
	]

	fig, axd = plt.subplot_mosaic(
		[
			[ "PSD",  "UV" ],
			[ "sl0", "sl0" ],
			[ "sl1", "sl1" ],
			[ "sl2", "sl2" ],
			[ "sl3", "sl3" ]
		],
		gridspec_kw = {
			'height_ratios': [50, 1, 1, 1, 1]
		}
	)
	plt.ion()

	axd["PSD"].scatter(
		y_fracs["0.5"][:, 0], y_fracs["0.5"][:, 1],
		facecolor="None", edgecolor="#335C81"
	)
	axd["PSD"].errorbar(
		y_fracs["0.5"][:, 0], y_fracs["0.5"][:, 1],
		yerr=[
			y_fracs["0.5"][:, 1]-y_fracs["0.1"][:, 1],
			y_fracs["0.9"][:, 1]-y_fracs["0.5"][:, 1]
		],
		color="#335C81", capsize=3, linewidth=0.7, linestyle=''
	)
	fill = axd["PSD"].fill_between(
		t, y_pred["0.1"], y_pred["0.9"], alpha=0.2, facecolor="#65AFFF"
	)
	line1, = axd["PSD"].loglog(
		t, y_pred["0.5"], linestyle="-.", linewidth=0.7, color="#65AFFF"
	)

	line2, = axd["UV"].loglog(t[1:], interpolated, label="PSM prediction")
	axd["UV"].scatter(times, yields, label="Concentration", alpha=0.4)

	for ax in ["UV", "PSD"]:
		axd[ax].autoscale(False)

	sliders = len(slider_params)*[None]
	for i, (param, valinit, valstep) in enumerate(slider_params):
		sliders[i] = Slider(
			axd[f"sl{i}"], param, 0, SLIDER_MAX,
			valinit=valinit, valstep=valstep
		)

	def update(val):
		solution = run_simulation(
			np.array([slider.val for slider in sliders[:-1]])
		)
		pred = calc_fracs(solution)
		dummy = axd["PSD"].fill_between(t, pred["0.1"], pred["0.9"])
		dp = dummy.get_paths()[0]
		dummy.remove()
		fill.set_paths([dp.vertices])
		line1.set_ydata(pred["0.5"])

		interpolated = interp(solution, sliders[-1].val*1e-6)[0]
		line2.set_ydata(interpolated)

		fig.canvas.draw_idle()

	for slider in sliders:
		slider.on_changed(update)

	plt.tight_layout()
	plt.show(block=True)

if __name__ == "__main__":
	speed = process_args(sys.argv)[1]

	PSM, y_fracs = extract_frac_psm(speed)
	
	def error_routine(array):
		return error_calculator(run_simulation(array), PSM=PSM)

	res = minimise(
		error_routine, x0=np.array([B_DEFAULT, E_DEFAULT, A_DEFAULT]),
		bounds = ((0, 10), (0, None), (0, None)),
		options={'tol': 1e-3, 'disp': True}
	)
	print(res.x)

	plot_UVvis_with_PSDs(
		speed, res.x, y_fracs
	)