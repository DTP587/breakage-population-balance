import numpy as np
import sys, os
import pandas as pd
import breakage_population_balance as bpb
from breakage_population_balance.arrayManipulation import fractional_percentage

dirname = os.path.dirname(__file__)
PSM = pd.read_csv(os.path.join(dirname, "Extract-11500.csv"), index_col=0)
cols = PSM.columns.tolist()
zero = PSM[cols[1]].values
zero = zero/np.sum(zero)
x = PSM.index.to_numpy()*1e-6

# Importing test plotting functions
sys.path.append(os.path.join(dirname, '..'))
from plotting import plot_simple, plot_together

classes = 100
nt = 700
Tf = 1800

# x = np.logspace(-8, -2, classes, dtype=np.float64)
t = np.linspace(0, Tf, nt+1, dtype=np.float64)

k = lambda x: (x/0.003007)**2.396

# kernel function
def Phi(x, y):
    if x < y:
        # Spherical chickens in a vacuum
        v_ = (1/6) * np.pi * y ** 3
        v  = (1/6) * np.pi * x ** 3
        return ( v / v_ )**(0.4816)
    else:
        return 0
Phi = np.vectorize(Phi, otypes=[np.float64])

IC = zero

# Run and solve the breakage problem
model = bpb.breakageModel(IC, t, x, k, kernel=Phi)
solution = model.solve("pyf_simple_breakage")

# =============================================================================

fracs = ["0.1", "0.5", "0.9"]

y_pred = { frac: np.zeros_like(t, dtype=np.float64) for frac in fracs }
y_psms = { frac: [] for frac in fracs }

plot_PSM = PSM[cols[2:]]

for frac in fracs:
    for i in np.arange(nt+1):
        y_pred[frac][i] = fractional_percentage(solution[i], x, float(frac))

for frac in fracs:
    for col in plot_PSM:
        data = plot_PSM[col].values
        data = data/np.sum(data)
        seconds = int(col.split(".", 1)[0])
        y_psms[frac].append((
            seconds,
            fractional_percentage(data, x, float(frac))[0]
        ))
    y_psms[frac] = np.array(y_psms[frac])

# =============================================================================

if __name__ == "__main__":

    import pickle

    with open("y_psms.pkl", 'wb') as f:
        pickle.dump(y_psms, f)
    with open("IC.pkl", 'wb') as f:
        pickle.dump((x, IC), f)

    import matplotlib.pyplot as plt

    plt.fill_between(t, y_pred["0.1"], y_pred["0.9"], alpha=0.2, facecolor="#65AFFF")
    plt.loglog(t, y_pred["0.5"], linestyle="-.", linewidth=0.7, color="#65AFFF")
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

    plt.show()

# =============================================================================

    print(
    	f"Mass before:\t{np.sum(IC)}\n" +
    	f"------------\n" +
    	f"Mass after:\t{np.sum(solution[-1])}"
    )

    # plot_simple(solution, model, n_plots=5, plot_as="fraction")