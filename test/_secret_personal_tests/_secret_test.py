import sys, os
from breakage_population_balance import apply_function, apply_kernel, breakageModel
import numpy as np

# Importing test plotting functions
dirname = os.path.dirname(__file__)
sys.path.append(os.path.join(dirname, '..'))
from plotting import plot_simple

X_BINS = 10
A_BINS = 20
Z_BINS = 7

nt = 100
Tf = 5

x = np.logspace(-2, 1, X_BINS, dtype=np.float64)
a = np.logspace(-2, 0, A_BINS, dtype=np.float64)
z = np.logspace( 0, 1, Z_BINS, dtype=np.float64)

t = np.linspace(0, Tf, nt+1, dtype=np.float64)





f_k = lambda x: (x**2)

# kernel function
def Phi(x, x_):
    if x < x_:
        return (x) / (x_)
    else:
        return 0.
Phi = np.vectorize(Phi, otypes=[np.float64])

phi1 = apply_kernel([x], Phi, normalise=True)
k1   = apply_function([x], f_k)
IC1  = apply_function([x], f_k)






f_k = lambda x, a: (x**2)/a

# kernel function
def Phi(x, a, x_, a_):
    if x < x_:
        return (a_*x) / (a*x_)
    else:
        return 0.
Phi = np.vectorize(Phi, otypes=[np.float64])

phi2 = apply_kernel([x, a], Phi, normalise=True)
k2   = apply_function([x, a], f_k)
IC2  = apply_function([x, a], f_k)






# f_k = lambda x, a, z: (x**2)/a + z

# # kernel function
# def Phi(x, a, z, x_, a_, z_):
#     if x < x_:
#         return (a_*x) / (a*x_) + z
#     else:
#         return 0.
# Phi = np.vectorize(Phi, otypes=[np.float64])

# phi3 = apply_kernel([x, a, z], Phi, normalise=True)
# k3   = apply_function([x, a, z], f_k)
# IC3  = apply_function([x, a, z], f_k)





# print(phi1*np.broadcast_to(k1*IC1, phi1.shape))
# print("---")

# print((phi1).size)
# print(phi1)
# print((k2*IC2).size)
# print((phi2).size)
# print((k2*IC2).size)
# print(phi2.shape)

def pyf_simple_breakage(F, t, x, k, phi, ax=-1):
    dFdt = np.zeros_like(F)
    # Death breakup term
    dFdt -= F * k
    # Birth breakup term
    dFdt += np.nansum(
        phi * np.triu(
            np.multiply(phi, k*F)
        ),
        axis=ax # (-2, -1)
    )
    return dFdt

def create_axis(ndims):
    return tuple(i for i in reversed(range(-ndims, 0)))

ax1 = create_axis(len(IC1.shape))
ax2 = create_axis(len(IC2.shape))

print(ax1, ax2)

# pyf_simple_breakage(IC2, 0, 0, k2, phi2, ax=)
# print(pyf_simple_breakage(IC1, 0, 0, k1, phi1, ax=1))
# print(pyf_simple_breakage(IC1, 0, 0, k1, phi1, ax=ax1))

# # model = breakageModel(IC, t, xa, k, kernel=Phi)

# # print(model._Phi)

# pyf_simple_breakage(IC, 0.1, [x, a], k, phi)

# # Run and solve the breakage problem
# model = breakageModel(IC, t, xa, k, kernel=Phi)
# solution = model.solve("pyf_simple_breakage")

# # =============================================================================

# print(
# 	f"Mass before:\t{np.sum(IC)}\n" +
# 	f"------------\n" +
# 	f"Mass after:\t{np.sum(solution[-1])}"
# )

# plot_against_analytical(solution, model, plot_as="fraction")
# # plot_against_analytical(solution/x, model, plot_as="number")
