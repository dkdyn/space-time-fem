import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 20,
    "pgf.rcfonts": False,
})

data = np.loadtxt("results_hyper_space_time_2order.csv", delimiter=",", skiprows=1)
st2_t = data[:, 0]
st2_u = data[:, 1]

data = np.loadtxt("results_hyper_space_time_bc.csv", delimiter=",", skiprows=1)
stbc_t = data[:, 0]
stbc_u = data[:, 1]

data = np.loadtxt("results_hyper_time_stepping.csv", delimiter=",", skiprows=1)
ts_t = data[:, 0]
ts_u = data[:, 1]

t = np.linspace(0, 1, 100)
u = np.cos(np.pi * t)

plt.plot(t, u, "gray", linewidth=2, label="exact solution")
plt.plot(ts_t, ts_u, "rx", linewidth=2, label="time-stepping")
plt.plot(st2_t, st2_u, "go", linewidth=2, label="space-time IC")
plt.plot(stbc_t, stbc_u, "b+", linewidth=2, label="space-time BC")
plt.xlabel("$t$")
plt.ylabel("$u$")
plt.tight_layout()
plt.legend()
plt.savefig("plot_hyper_results.pgf")
#plt.show()

print(ts_t[0], st2_t[0], stbc_t[0])
print(ts_t[1], st2_t[1], stbc_t[1])