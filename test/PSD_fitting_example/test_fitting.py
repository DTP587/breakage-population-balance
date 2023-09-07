import os, sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import minimize as minimise
from breakage_population_balance import breakageModel
from breakage_population_balance.arrayManipulation import \
	fractional_percentage, fractional_percentage_array

dirname = os.path.dirname(__file__)

FILENAMES = (
	"Extract-1500.csv",
	"Extract-3800.csv",
	"Extract-6500.csv",
	"Extract-11500.csv",
	"Extract-18500.csv"
)

FRACS = ("0.1", "0.5", "0.9")

B_DEFAULT = 3.007
E_DEFAULT = 2.396
A_DEFAULT = 4.816

CLASSES = 100
NT = 600
TF = 1800

time_to_step = lambda time: round((time*NT)/TF)

t = np.linspace(0, TF, NT+1, dtype=np.float64)

PSMs = {}

for filename in FILENAMES:
	print(f"Processing {filename}")
	PSMs[filename.strip("Extract-.csv")] = \
		pd.read_csv(os.path.join(dirname, filename), index_col=0)

	if "11500" in filename:
		PSM = PSMs["11500"]
		cols = PSM.columns.tolist()
		PSMs["11500"] = PSM[cols[2:]]
		zero = PSM[cols[1]].values
		zero = zero/np.sum(zero)
		x = PSM.index.to_numpy()*1e-6

def run_simulation(array):
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
	model = breakageModel(zero, t, x, k, kernel=Phi)
	solution = model.solve("pyf_simple_breakage")

	return solution

def error_calculator(output, PSM=None):
	solution = output

	err_sum = 0

	for time, psd in PSM:
		sol = solution[time_to_step(time)]
		err = np.sum(np.abs(psd-sol))
		err_sum += err

	return np.array(err_sum)

def calc_fracs(solution):
	pred = { frac: np.zeros_like(t, dtype=np.float64) for frac in FRACS }

	for frac in FRACS:
		for i in np.arange(NT+1):
			pred[frac][i] = fractional_percentage(
				solution[i], x, float(frac)
			)

	return pred

def plot_psds_with_sliders(speed, array, y_fracs):
	b=array[0]
	e=array[1]
	a=array[2]

	fig, axs = plt.subplots(
		4, 1, gridspec_kw={'height_ratios': [0.5, 0.5, 0.5, 10]}
	)
	plt.ion()

	SLIDER_MAX = 10

	*slideraxs, ax = axs

	slider_params = [
		('b', b, 0.01),
		('e', e, 0.01),
		('a', a, 0.01)
	]

	solution = run_simulation(array)

	y_pred = calc_fracs(solution)

	plt.title(f"{speed}")
	plt.scatter(
		y_fracs["0.5"][:, 0], y_fracs["0.5"][:, 1],
		facecolor="None", edgecolor="#335C81"
	)
	plt.errorbar(
		y_fracs["0.5"][:, 0], y_fracs["0.5"][:, 1],
		yerr=[
			y_fracs["0.5"][:, 1]-y_fracs["0.1"][:, 1],
			y_fracs["0.9"][:, 1]-y_fracs["0.5"][:, 1]
		],
		color="#335C81", capsize=3, linewidth=0.7, linestyle=''
	)
	fill = plt.fill_between(
		t, y_pred["0.1"], y_pred["0.9"], alpha=0.2, facecolor="#65AFFF"
	)
	line, = plt.loglog(
		t, y_pred["0.5"], linestyle="-.", linewidth=0.7, color="#65AFFF"
	)

	ax.autoscale(False) 

	sliders = len(slider_params)*[None]
	for i, (param, valinit, valstep) in enumerate(slider_params):
		sliders[i] = Slider(
			slideraxs[i], param, 0, SLIDER_MAX,
			valinit=valinit, valstep=valstep
		)

	def update(val):
		solution = run_simulation(
			np.array([slider.val for slider in sliders])
		)
		pred = calc_fracs(solution)
		dummy = ax.fill_between(t, pred["0.1"], pred["0.9"])
		dp = dummy.get_paths()[0]
		dummy.remove()
		fill.set_paths([dp.vertices])
		line.set_ydata(pred["0.5"])
		fig.canvas.draw_idle()

	for slider in sliders:
		slider.on_changed(update)

	plt.tight_layout()
	plt.show(block=True)

def process_args(args):
	error = ValueError(
		f"Script requires a SINGLE CL argument to run directly.\n" + \
		f"Available arguments are:\n{PSMs.keys()}"
	)

	if len(args) == 1:
		raise error
	elif len(args) > 3:
		raise error
	elif args[1] not in PSMs.keys():
		raise error

	return args

def extract_frac_psm(key):
	y_fracs = {}
	for frac in FRACS:
		y_fracs[frac] = []
		for col in PSMs[key]:
			data = PSMs[key][col].values
			data = data/np.sum(data)
			seconds = int(col.split(".", 1)[0])
			y_fracs[frac].append((
				seconds,
				fractional_percentage(data, x, float(frac))[0]
			))
		y_fracs[frac] = np.array(y_fracs[frac])

	y_psms = {}
	for speed, PSM in PSMs.items():
		y_psms[speed] = []
		for col in PSM:
			data = PSM[col].values
			data = data/np.sum(data)
			seconds = int(col.split(".", 1)[0])
			y_psms[speed].append((
				seconds,
				data
			))

	return y_psms[key], y_fracs


if __name__ == "__main__":
	speed = process_args(sys.argv)[1]

	PSM, y_fracs = extract_frac_psm(speed)

	def error_routine(array):
		return error_calculator(run_simulation(array), PSM=PSM)

	res = minimise(
		error_routine, x0=np.array([B_DEFAULT, E_DEFAULT, A_DEFAULT]),
		bounds = ((0, 8), (0, None), (0, None)),
		options={'disp': True}
	)
	print(res.x)

	solution = run_simulation(res.x)

	plot_psds_with_sliders(speed, res.x, y_fracs)


