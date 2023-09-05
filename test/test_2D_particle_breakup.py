from breakage_population_balance import breakageModel
from breakage_population_balance.arrayManipulation import apply_function
from plotting import plot_simple, plot_together
import numpy as np

# Currently not working - need to extend to higher dimensional integrals.
# 1D - simple ODE
# 2D - 4D higher dimension ODE
# 5D monte-carlo

CLASSES_x = 10
CLASSES_a = 5
TIME_STEPS = 101
TIME_FINAL = 100

x = np.linspace(0.1, 1, CLASSES_x, dtype=np.float64)
a = np.linspace(0.1, 1, CLASSES_a, dtype=np.float64)
t = np.linspace(0, TIME_FINAL, TIME_STEPS+1, dtype=np.float64)

xx, aa = np.meshgrid(x, a, indexing="ij")

k = lambda x, a: 0.001 * x**2 / a**2

# kernel function
def Phi(x, a, y, b):
    if x < y:
        return x / y
    elif x == y:
        if b < a:
            return x / y
        else:
            0.
    else:
        return 0.
Phi = np.vectorize(Phi, otypes=[np.float64])

IC  = np.exp(-xx)* ((1/6) * np.pi * xx ** 3)
IC /= np.sum(IC)

# Run and solve the breakage problem
model = breakageModel(IC, t, [x, a], k, kernel=Phi)

print(model._Phi[0, :, :, :])
print(model._k)

sol = model.solve("cyf_2D_breakage")




# plotting

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import cm

fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 10]})
plt.ion()

sliderax, ax = axs

sindex = Slider(sliderax, 't_step', 0, TIME_STEPS-1, valinit=0, valstep=1)

ax.pcolor(xx, aa, sol[0], cmap=cm.viridis, vmin=0, vmax=0.3)
# ax.plot_surface(xx, aa, IC, cmap=cm.viridis, linewidth=0, antialiased=False)

ax.set_xlim([np.min(x), np.max([x])])
ax.set_ylim([np.min(a), np.max([a])])

ax.set_xlabel("x")
ax.set_ylabel("a")

def update(val):
    i = sindex.val
    ax.pcolor(xx, aa, sol[i], cmap=cm.viridis)
    fig.canvas.draw_idle()

sindex.on_changed(update)
plt.tight_layout()
plt.show(block=True)