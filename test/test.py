from breakage_population_balance import breakageModel
import numpy as np

classes = 100
nt = 100
Tf = 5

x = np.logspace(-2, 1, classes)
t = np.linspace(0, Tf, nt+1)

k = lambda x: x**2

# kernel function
def Phi(x, y):
    if x < y:
        pass
    else:
        return 0
    return x / y

# Works on a fraction based approach, so number needs to be converted to
# fraction.
IC = np.exp(-x)*x

# Run and solve the breakage problem
model = breakageModel(IC, t, x, Phi, k)
solution = model.solve()

# =============================================================================

if __name__== "__main__":
	# Plot the solution against expected analytical value
	from matplotlib import pyplot as plt
	plt.style.use('seaborn-whitegrid')

	fig, axs = plt.subplot_mosaic(
		[ ['a)', 'b)'],
		  ['c)', 'c)'] ],
		sharex = True
	)

	axs['a)'].get_shared_y_axes().join(axs['a)'], axs['b)'])
	axs['b)'].set_yticklabels([]) #set_yticks([])

	fa = lambda x, t: (1+2*t*(1+x))*np.exp(-t*x*x-x)
	times = np.arange(0, nt, round(nt/5)) # only 5 plots
	analytical_solution = []
	MOC_solution = []

	for time in times:
	    analytical_solution.append(fa(x, Tf*(time/nt)))
	    MOC_solution.append(solution[time]/x)

	iterables = zip(
	    analytical_solution,
	    MOC_solution,
	    times
	)

	yc_max, y_max = (0, 0)

	for ya, ys, t in iterables:
		linestyle = ':' if t == 0 else '-'
		for ax, y in zip(['a)', 'b)'], [ya, ys]):
			# Normalise result
			axs[ax].plot(x, y/np.sum(y), linestyle=linestyle)
			y_max = np.max(y/np.sum(y))

		# Plot the percentage error
		y_c = 100*(ya/np.sum(ya)-ys/np.sum(ys))/y_max
		axs['c)'].plot(x, y_c, linestyle=linestyle, label='%.2g'%(Tf*(t/nt)))
		if np.max(y_c) > yc_max:
			yc_max = np.max(y_c)

	iterables = zip(
		['a)', 'b)', 'c)'],
		['Analytical', 'Numerical', 'Error'],
		[y_max, y_max, yc_max]
	)

	for ax, label, max_val in iterables:
		axs[ax].set_xscale("log")
		axs[ax].set_xlim([np.min(x), np.max(x)])
		axs[ax].annotate(label, xy=[0.97,0.85], xycoords='axes fraction', ha='right', style='italic')

		pad = float('%.2g'%(max_val*0.05))
		if ax == 'c)':
			axs[ax].set_ylim([-float('%.2g'%max_val)-pad, float('%.2g'%max_val)+pad])
		else:
			axs[ax].set_ylim([-pad, float('%.2g'%max_val)+pad])

	for ax, label in zip(['a)', 'c)'], ['Normalised Number', '%']):
		axs[ax].set_ylabel(label, weight="bold")

	fig.suptitle(
		"Analytical (Ziff and McGrady) vs Numerical (Method of Classes)",
		weight='bold'
	)
	plt.tight_layout()
	plt.show()

