import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16
})

# Create a grid of x and t values
x = np.linspace(0, 1, 100)
t = np.linspace(0, 1, 100)
X, T = np.meshgrid(x, t, indexing='ij')

# Compute the function values
#U = np.sin(np.pi*X) * np.exp(-T*np.pi**2)
U = np.sin(np.pi*X) * np.cos(T*np.pi)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the colored surface
surf = ax.plot_surface(X, T, U, cmap='viridis', edgecolor='none', alpha=0.95)

# Overlay the wireframe grid lines (coarser for clarity)
ax.plot_wireframe(X, T, U, color='k', linewidth=0.5, rstride=10, cstride=10)

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$t$')
ax.set_zlabel(r'$u$')
ax.view_init(elev=28, azim=51)

# Hide the background panes but keep grid lines
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.pane.set_edgecolor('w')
    axis.pane.set_alpha(0)

plt.tight_layout()
#plt.savefig("plot_para_exact.pdf")
plt.savefig("plot_hyper_exact.pdf")
plt.show()
