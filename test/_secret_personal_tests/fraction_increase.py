import numpy as np
from test_1D_number import x, t, volume

# import fraction based solution from both sources, requires similar time
# settings for both.
from test_1D_fitted import solution as fractional_solution
from test_1D_number import solution as numerical_solution

title = lambda string: string.center(30, "_")

def interp(solution, section_value):
    interpolated = None

    ix = np.where(x < section_value)[0][-1]

    fraction = (section_value - x[ix]) / (x[ix+1] - x[ix])
    lower  = np.sum(solution[1:, 0:(ix+1)], axis=1)
    difference = np.sum(solution[1:, 0:(ix+2)], axis=1) - lower

    interpolated = lower + difference*fraction

    return interpolated, fraction

t_uv = 60*np.array([
    10.0, 2.0, 5.0, 1.0, 30.0, 1.0, 2.0, 1.0, 0.16666666666666666,
    0.3333333333333333, 0.5, 0.6666666666666666, 0.8333333333333334, 1.0
])
C_uv = np.array([
    0.02396673076923077, 0.005441730769230769, 0.015530384615384613,
    0.003972628205128205, 0.04887673076923077, 0.0036785256410256416,
    0.006380961538461539, 0.003958910256410257, 0.000673782051282051,
    0.0014796153846153848, 0.0020699358974358975, 0.0028587820512820513,
    0.003363461538461538, 0.0043155128205128205
]) # mg/mL 4e-4*

ρ_g = 2.260e3  # kg/m3
M_T = 4.000e3 # mg
I_C_uv = M_T/200 # initial concentration 4000 mg / 200 mL = 20 mg/mL

plot_points = C_uv/I_C_uv # get yield (% mass)

if __name__ == "__main__":
	print(
	    title("Numerical") + "\n" + \
	    f"Starting mass: {np.sum(M_T*numerical_solution[0,  :])*1e3} kg\n" + \
	    f"Exiting mass:  {np.sum(M_T*numerical_solution[-1, :])*1e3} kg\n" + \
	    title("Fractional") + "\n" + \
	    f"Starting mass: {np.sum(M_T*fractional_solution[0,  :])*1e3} kg\n" + \
	    f"Exiting mass:  {np.sum(M_T*fractional_solution[-1, :])*1e3} kg"
	)

	import matplotlib.pyplot as plt

	v = volume(x)

	max_size = 1600e-9  # We know that they are on average 200 nm.
	                    # Changed from 1040 to 1600 nm when psds were fitted.

	ix = np.where(x < max_size)[0]

	print(
		title("Interpolation") + "\n" + \
	    f"pos: {ix[-1]}, max size = {x[ix[-1]]}"
	)
	index = t_uv.argsort()
	print(t_uv[index])
	print(C_uv[index])
	plt.plot(
		t[1:], np.sum(numerical_solution[ 1:, ix], axis=1),
		label="Number", linewidth=0.7, linestyle='-.'
	)
	# plt.plot(
	# 	t[1:], np.sum(fractional_solution[1:, ix], axis=1),
	# 	label="Fraction", linewidth=0.7, linestyle='-.'
	# )
	plt.plot(t[1:], interp(fractional_solution, max_size)[0], label="Fraction")
	plt.scatter(t_uv, plot_points, label="Concentration", alpha=0.4)
	plt.xscale("log")
	plt.yscale("log")
	plt.ylabel("Yield [-]")
	plt.xlabel("Time [s]")
	# plt.title(f"11500 rpm fitted at {max_size*1e6} μm")
	plt.tight_layout()
	plt.legend(frameon=True, facecolor='white', framealpha=0.6)
	plt.show()
