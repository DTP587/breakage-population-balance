import numpy as np
import pandas as pd
import os, sys

dirname = os.path.dirname(__file__)

filenames = [
    "Extract-1500.csv",
    "Extract-3800.csv",
    "Extract-6500.csv",
    "Extract-11500.csv",
    "Extract-18500.csv"
]

PSMs = {}

for filename in filenames:
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


if __name__ == "__main__":
	from breakage_population_balance import breakageModel
	from breakage_population_balance.arrayManipulation \
		import fractional_percentage

	error = ValueError(
		f"Script requires a SINGLE CL argument to run directly.\n" + \
		f"Available arguments are:\n{PSMs.keys()}"
	)
	args = sys.argv

	if len(args) == 1:
		raise error
	elif len(args) > 3:
		raise error
	elif args[1] not in PSMs.keys():
		raise error

	PSM = PSMs[args[1]]

	fracs = ["0.1", "0.5", "0.9"]

	y_psms = { frac: [] for frac in fracs }

	for frac in fracs:
		for col in PSM:
			data = PSM[col].values
			data = data/np.sum(data)
			seconds = int(col.split(".", 1)[0])
			y_psms[frac].append((
				seconds,
				fractional_percentage(data, x, float(frac))[0]
			))
		y_psms[frac] = np.array(y_psms[frac])

	classes = 100
	nt = 700
	Tf = 3000

	t = np.linspace(0, Tf, nt+1, dtype=np.float64)

	B_DEFAULT = 3.007
	E_DEFAULT = 2.396
	A_DEFAULT = 4.816

	def run_simulation(b=B_DEFAULT, e=E_DEFAULT, a=A_DEFAULT):
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

		y_pred = { frac: np.zeros_like(t, dtype=np.float64) for frac in fracs }

		for frac in fracs:
			for i in np.arange(nt+1):
				y_pred[frac][i] = fractional_percentage(
					solution[i], x, float(frac)
				)

		return solution, y_pred

	from matplotlib import pyplot as plt
	from matplotlib.widgets import Slider
	from matplotlib import cm

	fig, axs = plt.subplots(4, 1, gridspec_kw={'height_ratios': [0.5, 0.5, 0.5, 10]})
	plt.ion()

	SLIDER_MAX = 10

	*slideraxs, ax = axs

	slider_params = [
		('b', B_DEFAULT, 0.01),
		('e', E_DEFAULT, 0.01),
		('a', A_DEFAULT, 0.01)
	]

	solution, y_pred = run_simulation()

	plt.title(f"{args[1]}")
	plt.scatter(
        y_psms["0.5"][:, 0], y_psms["0.5"][:, 1],
        facecolor="None", edgecolor="#335C81"
    )
	plt.errorbar(
        y_psms["0.5"][:, 0], y_psms["0.5"][:, 1],
        yerr=[
            y_psms["0.5"][:, 1]-y_psms["0.1"][:, 1],
            y_psms["0.9"][:, 1]-y_psms["0.5"][:, 1]
        ],
        color="#335C81", capsize=3, linewidth=0.7, linestyle=''
    )
	fill = plt.fill_between(
		t, y_pred["0.1"], y_pred["0.9"], alpha=0.2, facecolor="#65AFFF"
	)
	line, = ax.loglog(
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
		solution, pred = run_simulation(*[slider.val for slider in sliders])
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




