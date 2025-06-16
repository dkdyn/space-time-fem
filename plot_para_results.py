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

data = np.loadtxt("results_para_space_time.csv", delimiter=",", skiprows=1)
st_t = data[:, 0]
st_u = data[:, 1]

data = np.loadtxt("results_para_time_stepping.csv", delimiter=",", skiprows=1)
ts_t = data[:, 0]
ts_u = data[:, 1]

t = np.linspace(0, 1, 100)
u = np.exp(-(np.pi**2) * t)

plt.plot(t, u, "gray", linewidth=2, label="exact solution")
plt.plot(ts_t, ts_u, "rx", linewidth=2, label="time-stepping")
plt.plot(st_t, st_u, "go", linewidth=2, label="space-time")
plt.xlabel("$t$")
plt.ylabel("$u$")
plt.tight_layout()
plt.legend()
plt.savefig("plot_para_results.pgf")
#plt.show()

print(st_t[0], ts_t[0])
print(st_t[1], ts_t[1])