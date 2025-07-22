import numpy as np
import pyvista as pv

# Create a grid of x and t values
x = np.linspace(0, 1, 100)
t = np.linspace(0, 1, 100)
X, T = np.meshgrid(x, t, indexing='ij')

# Compute the function values
#U = np.sin(np.pi*X) * np.cos(np.pi*T)   # hyper
U = np.sin(np.pi*X) * np.exp(-(np.pi**2)*T)  # exponential decay

# Prepare points for PyVista StructuredGrid
points = np.zeros((X.size, 3))
points[:, 0] = X.ravel(order="F")
points[:, 1] = T.ravel(order="F")
points[:, 2] = U.ravel(order="F")

# Create the StructuredGrid
grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = [X.shape[0], X.shape[1], 1]
grid["u"] = U.ravel(order="F")

# Plot with PyVista
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="u", cmap="viridis", show_edges=False)
#plotter.view_xy()
plotter.show_grid()
plotter.add_axes()
plotter.show_bounds(
    xlabel="x",
    ylabel="t",
    zlabel="u"
)
#plotter.show_bounds
#plotter.renderer.SetYAxisLabel("t")
#plotter.renderer.SetZAxisLabel("u")
plotter.show(title="Exact solution: $u(x,t) = \sin(\pi x)\cos(\pi t)$")
