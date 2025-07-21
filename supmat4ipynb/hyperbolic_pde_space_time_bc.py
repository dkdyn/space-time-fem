"""
Solve non-dimensional hyperbolic PDE in space-time with Dirichlet boundary conditions 
in space and time, i.e. an initial and a final conditions.
This is not an initial value problem and done for comparison only.
"""
from mpi4py import MPI
from dolfinx import mesh, geometry
import numpy as np
#import matplotlib.pyplot as plt
import pyvista as pv
import ufl
from dolfinx.fem import functionspace
from dolfinx import default_scalar_type
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx import plot

domain = mesh.create_unit_square(MPI.COMM_WORLD, 4, 8, mesh.CellType.quadrilateral)

V = functionspace(domain, ("Lagrange", 1))

uD = fem.Function(V)
uD.interpolate(lambda x: np.sin(np.pi*x[0])*np.cos(np.pi*x[1]))

# Create facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

f = fem.Constant(domain, default_scalar_type(0.0))
K = fem.Constant(domain, np.array([[1, 0], [0, -1]], dtype=np.float64))

a = ufl.dot(K*ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx


problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

print(pv.global_theme.jupyter_backend)

#pv.start_xvfb()   # troublemaker
domain.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)

grid = pv.UnstructuredGrid(topology, cell_types, geometry)

plotter = pv.Plotter()
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
u_warped = u_grid.warp_by_scalar()
plotter.add_mesh(u_warped, show_edges=True)
#u_plotter.view_xy()
if not pv.OFF_SCREEN:
    plotter.show()


