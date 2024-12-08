from all_speeds_fitted import PSMs, x, zero
from scipy.optimize import minimize as minimise
from matplotlib import pyplot as plt
from breakage_population_balance import breakageModel
from breakage_population_balance.arrayManipulation import fractional_percentage
import numpy as np

spd = "6500"

classes = 100
nt = 600
Tf = 1800

time_to_step = lambda time: round((time*nt)/Tf)

t = np.linspace(0, Tf, nt+1, dtype=np.float64)

FRACS = ("0.1", "0.5", "0.9")

B_DEFAULT = 3.007
E_DEFAULT = 2.396
A_DEFAULT = 4.816

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

PSM = y_psms[spd]

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

	y_pred = { frac: np.zeros_like(t, dtype=np.float64) for frac in FRACS }

	for frac in FRACS:
		for i in np.arange(nt+1):
			y_pred[frac][i] = fractional_percentage(
				solution[i], x, float(frac)
			)

	return solution, y_pred

def error_calculator(output):
	solution, y_pred = output

	err_sum = 0

	for time, psd in PSM:
		sol = solution[time_to_step(time)]
		err = np.sum(np.abs(psd-sol))
		err_sum += err

	return np.array(err_sum)

def error_routine(array):
	return error_calculator(run_simulation(array))

if __name__ == "__main__":
	res = minimise(
		error_routine, x0=np.array([B_DEFAULT, E_DEFAULT, A_DEFAULT]),
		bounds = ((0, 8), (0, None), (0, None)),
		options={'disp': True}
	)
	print(res.x)
	PSM = PSMs[spd]
	solution, y_pred = run_simulation(res.x)

	y_psms = { frac: [] for frac in FRACS }

	for frac in FRACS:
		for col in PSM:
			data = PSM[col].values
			data = data/np.sum(data)
			seconds = int(col.split(".", 1)[0])
			y_psms[frac].append((
				seconds,
				fractional_percentage(data, x, float(frac))[0]
			))
		y_psms[frac] = np.array(y_psms[frac])

	plt.title(f"{spd}")
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
	plt.fill_between(
		t, y_pred["0.1"], y_pred["0.9"], alpha=0.2, facecolor="#65AFFF"
	)
	plt.loglog(
		t, y_pred["0.5"], linestyle="-.", linewidth=0.7, color="#65AFFF"
	)

	plt.show()

