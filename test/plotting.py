from matplotlib import pyplot as plt
import numpy as np

plt.style.use('seaborn-whitegrid')

def plot_together(solutions, models, n_plots=5, plot_as="fraction"):

	fig, axs = plt.subplot_mosaic(
		[ ['a)'],
		  ['b)'] ],
		sharex = True
	)

	iterables = zip(solutions, models, ['a)', 'b)'])

	for sol, mod, ax in iterables:
		x, = mod.x
		t = mod.t
		nt = mod.t.shape[0]

		# Same indexes, different spaces.
		times = [0, *np.round(np.logspace(0, np.log10(nt-1), n_plots, dtype=int))]

		y_max = 0

		for time in times:
			linestyle = ':' if time == 0 else '-'

			y = sol[time]/np.sum(sol[time])

			axs[ax].plot(x, y, linestyle=linestyle, label=f"{round(mod.t[time],2)} s")
		
			y_max = np.max(y) if np.max(y) > y_max else y_max

			pad = float('%.2g'%(y_max*0.05))
			axs[ax].set_ylim([-pad, float('%.2g'%y_max)+pad])
			axs[ax].set_ylabel(f"Normalised {plot_as}", weight="bold")

		axs[ax].set_xscale("log")
		axs[ax].set_xlim([np.min(x), np.max(x)])

		axs[ax].legend(
			loc="upper left", frameon=True, framealpha=0.8, facecolor="white",
			title=r"$\bf{Sample}$"
		)

	plt.tight_layout()
	plt.show()



def plot_simple(solution, model, n_plots=5, plot_as="fraction", spacing="log"):
	nt = model.t.shape[0]
	Tf = model.t[-1]
	x,  = model.x

	ax = plt.subplot(111)

	if spacing == "log":
		times = [ 0, *np.logspace(0, np.log10(nt-1), n_plots, dtype=int)]
	elif spacing == "lin":
		times = [ *np.linspace(0, nt-1, n_plots, dtype=int)]

	MOC_solution = []
	for time in times:
		if plot_as == "number":
			MOC_solution.append(solution[time])
		elif plot_as == "fraction":
			MOC_solution.append(solution[time])
		else:
			raise ValueError(f"Unrecognised plot_as: {plot_as}")

	iterables = zip(
		MOC_solution,
		times
	)

	y_max = 0

	for y, t in iterables:
		linestyle = ':' if t == 0 else '-'

		ax.plot(x, y/np.sum(y), linestyle=linestyle, label=f"{round(model.t[t],2)} s")
		
		y_max = np.max(y/np.sum(y)) if np.max(y/np.sum(y)) > y_max else y_max

	ax.set_xscale("log")
	ax.set_xlim([np.min(x), np.max(x)])

	pad = float('%.2g'%(y_max*0.05))
	ax.set_ylim([-pad, float('%.2g'%y_max)+pad])
	ax.set_ylabel(f"Normalised {plot_as}", weight="bold")

	plt.legend(
		loc="upper left", frameon=True, framealpha=0.8, facecolor="white",
		title=r"$\bf{Sample}$"
	)
	plt.tight_layout()
	plt.show()


def plot_against_analytical(solution, model, n_plots=5, plot_as="number"):
	nt = model.t.shape[0]
	Tf = model.t[-1]
	x,  = model.x

	fig, axs = plt.subplot_mosaic(
		[ ['a)', 'b)'],
		  ['c)', 'c)'] ],
		sharex = True
	)

	axs['a)'].get_shared_y_axes().join(axs['a)'], axs['b)'])
	axs['b)'].set_yticklabels([]) #set_yticks([])

	fa = lambda x, t: (1+2*t*(1+x))*np.exp(-t*x*x-x)
	times = np.arange(0, nt, round(nt/n_plots))
	analytical_solution = []
	MOC_solution = []

	for time in times:
		if plot_as == "number":
		    analytical_solution.append(fa(x, Tf*(time/nt)))
		    MOC_solution.append(solution[time])
		elif plot_as == "fraction":
		    analytical_solution.append(fa(x, Tf*(time/nt))*x)
		    MOC_solution.append(solution[time])
		else:
			raise ValueError(f"Unrecognised plot_as: {plot_as}")

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
			axs[ax].set_xlabel("Size [m]", weight="bold")
		else:
			axs[ax].set_ylim([-pad, float('%.2g'%max_val)+pad])

	for ax, label in zip(['a)', 'c)'], [f"Normalised {plot_as}", '%']):
		axs[ax].set_ylabel(label, weight="bold")
	
	axs['c)'].set_ylim(bottom=-5, top=5)
	# fig.suptitle(
	# 	"Analytical (Ziff and McGrady) vs Numerical (Method of Classes)",
	# 	weight='bold'
	# )
	plt.tight_layout()
	plt.show()

