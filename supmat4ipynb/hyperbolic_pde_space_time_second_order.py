""" Solving the non-dimensional hyperbolic PDE in space-time using a second-order formulation. """
import numpy as np
import ufl
from dolfinx import geometry
from dolfinx import fem, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import pyvista
import matplotlib.pyplot as plt

# --- 1. Problem Setup ---
# Space-time mesh
nx = 4
nt = 8
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([1, 1])],
    [nx, nt],
    cell_type=mesh.CellType.triangle,
)

# Function space (using quadratic elements for better accuracy)
V = fem.functionspace(domain, ("Lagrange", 1))

# --- 2. Define Boundary Conditions ---
# Locate facets for boundary conditions (initial time and left/right boundaries)
fdim = domain.topology.dim - 1
facets_t0 = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
facets_space = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))
)

# Define Dirichlet boundary conditions
# Initial displacement u(x, 0) = u_0(x)
def u0_func(x):
    return np.sin(np.pi * x[0])

u_initial = fem.Function(V)
u_initial.interpolate(u0_func)
dofs_t0 = fem.locate_dofs_topological(V, fdim, facets_t0)
bc_t0 = fem.dirichletbc(u_initial, dofs_t0)

# Spatial boundary u(0,t) = u(1,t) = 0
dofs_space = fem.locate_dofs_topological(V, fdim, facets_space)
bc_space = fem.dirichletbc(ScalarType(0.0), dofs_space, V)

bcs = [bc_t0, bc_space]

# --- 3. Define Variational Problem ---
u = ufl.TrialFunction(V)
phi = ufl.TestFunction(V)

# Initial velocity v(x, 0) = v_0(x)
# This will be integrated over the t=0 boundary
v0 = fem.Function(V)
v0.interpolate(lambda x: np.zeros_like(x[0]))

# Source term f
f = fem.Constant(domain, ScalarType(0.0))

# Define the boundary integral for the initial velocity
# Mark the t=0 boundary with tag 1
T0_MARKER = 1
facet_tags = mesh.meshtags(domain, fdim, facets_t0, np.full_like(facets_t0, T0_MARKER))
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

# Bilinear form (left-hand side)
a = (u.dx(0) * phi.dx(0) - u.dx(1) * phi.dx(1)) * ufl.dx

# Linear form (right-hand side)
L = f * phi * ufl.dx + v0 * phi * ds(T0_MARKER)

# --- 4. Solve the Linear System ---
problem = LinearProblem(
    a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
uh = problem.solve()

# --- 5. Post-processing and Visualization ---

u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pyvista.Plotter()
u_warped = u_grid.warp_by_scalar()
u_plotter.add_mesh(u_warped, show_edges=True)
#u_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    u_plotter.show()

