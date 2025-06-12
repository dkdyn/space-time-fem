# Author: Gemini
# Date: June 11, 2025
# Description:
# This script provides a minimal implementation of the space-time finite element
# method (FEM) for the one-dimensional wave equation using FEniCSx.
# The formulation includes velocity as an additional variable to allow for
# the straightforward prescription of initial conditions in time.
#
# Mathematical Formulation:
# -------------------------
# The 1D wave equation is given by:
#
#   d^2u/dt^2 - c^2 * d^2u/dx^2 = f   in (0, L) x (0, T)
#
# where u is the displacement, c is the wave speed, f is a source term,
# x is the spatial coordinate, and t is the time coordinate.
#
# To apply initial conditions, we introduce the velocity v = du/dt.
# The second-order PDE is then rewritten as a system of two first-order PDEs:
#
#   1. du/dt - v = 0
#   2. dv/dt - c^2 * d^2u/dx^2 = f
#
# The space-time domain is Omega = (0, L) x (0, T).
#
# Weak Form:
# -----------
# We define a mixed function space W = V x V, where V is the function space
# for u and v. Let (w_u, w_v) be the test functions corresponding to (u, v).
#
# After multiplying the equations by test functions and integrating by parts
# over the space-time domain Omega, the variational problem is:
# Find (u, v) in W such that for all (w_u, w_v) in W:
#
#   Integral_Omega( (du/dt * w_u - v * w_u) * dV ) = 0
#   Integral_Omega( (dv/dt * w_v + c^2 * du/dx * dw_v/dx) * dV ) = Integral_Omega( f * w_v * dV )
#
# We solve this coupled system over the entire space-time domain at once.

import dolfinx
from dolfinx.fem.petsc import LinearProblem
from dolfinx import fem, mesh, plot, geometry
from dolfinx.fem import dirichletbc, locate_dofs_geometrical
from dolfinx import default_scalar_type
from mpi4py import MPI
import ufl
import numpy as np
# Import element creation functions from basix.ufl
from basix.ufl import element, mixed_element

# --- 1. Problem Parameters ---
L = 1.0  # Length of the spatial domain
T = 1.0  # Total time
#c = 1.0  # Wave speed
nx = 100  # Number of elements in space
# Increase nt to satisfy the CFL condition (dt < dx/c)
# dx = 1/50 = 0.02. dt should be smaller. T/nt < dx/c -> 2/nt < 0.02 -> nt > 100.
nt = 100 # Number of elements in time

# --- 2. Create the Space-Time Mesh ---
# The domain is a 2D rectangle [0, L] x (0, T)
domain = mesh.create_rectangle(MPI.COMM_WORLD,
                               [np.array([0, 0]), np.array([L, T])],
                               [nx, nt],
                               mesh.CellType.quadrilateral)

# We need to distinguish between spatial and temporal derivatives
# x[0] is space, x[1] is time
x = ufl.SpatialCoordinate(domain)


# --- 3. Define the Function Space ---
# We use a mixed function space for (u, v) using basix.ufl
P1 = element("Lagrange", domain.basix_cell(), 2)
W_element = mixed_element([P1, P1])
W = fem.functionspace(domain, W_element)
U = W.sub(0).collapse()[0]  # Function space for u
V = W.sub(1).collapse()[0]  # Function space for u

# Define Trial and Test functions
(u, v) = ufl.TrialFunctions(W)
(Du, Dv) = ufl.TestFunctions(W)


# Define the source term. fem.Constant requires the domain as the first argument.
f = fem.Constant(domain, dolfinx.default_scalar_type(0.0)) # Source term

# Equation 1: du/dt - v = 0
a1 = (u.dx(1) - v) * Dv * ufl.dx
L1 = fem.Constant(domain, dolfinx.default_scalar_type(0.0)) * Dv * ufl.dx

# Equation 2: dv/dt - c^2 * d^2u/dx^2 = f
# Integrate by parts in space: -c^2 * d^2u/dx^2 -> c^2 * du/dx * dw_v/dx
a2 = (v.dx(1) * Du + u.dx(0) * Du.dx(0)) * ufl.dx
L2 = f * Du * ufl.dx

# Combine into a single system
a = a1 + a2
L_form = L1 + L2


# Locate DOFs for u (component 0) and v (component 1) at x=0 and x=L
dofs_u_left = locate_dofs_geometrical(U, lambda x: np.isclose(x[0], 0))
dofs_u_right = locate_dofs_geometrical(U, lambda x: np.isclose(x[0], 1))
dofs_u_initial = locate_dofs_geometrical(U, lambda x: np.isclose(x[1], 0))
#dofs_v_left = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
#dofs_v_right = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1))
dofs_v_initial = locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0))

u_init = fem.Function(U)
u_init.interpolate(lambda x: np.sin(np.pi * x[0]))

# Create Dirichlet BCs for u and v at both boundaries
bc_u_left = dirichletbc(default_scalar_type(0), dofs_u_left, W.sub(0))
bc_u_right = dirichletbc(default_scalar_type(0), dofs_u_right, W.sub(0))
bc_u_initial = dirichletbc(u_init, dofs_u_initial)
#bc_v_left = dirichletbc(default_scalar_type(0), dofs_v_left, W.sub(1))
#bc_v_right = dirichletbc(default_scalar_type(0), dofs_v_right, W.sub(1))
bc_v_initial = dirichletbc(default_scalar_type(0), dofs_v_initial, W.sub(1))

# Collect all BCs in a list
bcs = [bc_u_left, bc_u_right, bc_u_initial, bc_v_initial]

# 1. Set up the linear problem
problem = LinearProblem(a, L_form, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

# 2. Solve for the solution (mixed function: (u, v))
w_sol = problem.solve()

A = fem.petsc.assemble_matrix(fem.form(a), bcs)
A.assemble()
print("Matrix norm:", A.norm())

# Extract u and v components
u_sol = w_sol.sub(0).collapse()
v_sol = w_sol.sub(1).collapse()

# Print statistics for u
print("u min:", np.nanmin(u_sol.x.array))
print("u max:", np.nanmax(u_sol.x.array))
print("u mean:", np.nanmean(u_sol.x.array))
print("u contains NaN:", np.isnan(u_sol.x.array).any())

# Print statistics for v
print("v min:", np.nanmin(v_sol.x.array))
print("v max:", np.nanmax(v_sol.x.array))
print("v mean:", np.nanmean(v_sol.x.array))
print("v contains NaN:", np.isnan(v_sol.x.array).any())


# post# --- 4. Post-processing with PyVista ---
import pyvista as pv
import numpy as np

# Extract the u component (displacement) from the mixed solution
u_sol = w_sol.sub(0).collapse()

# Get the coordinates of the mesh nodes
coords = u_sol.function_space.mesh.geometry.x
u_vals = u_sol.x.array

# Create a PyVista structured grid
# Assumes a structured mesh: sort coordinates for grid construction
sort_idx = np.lexsort((coords[:, 1], coords[:, 0]))
coords_sorted = coords[sort_idx]
u_sorted = u_vals[sort_idx]

# Get unique x and t values
x_unique = np.unique(coords_sorted[:, 0])
t_unique = np.unique(coords_sorted[:, 1])

# Reshape u to (nx+1, nt+1) for grid
nx_pts = len(x_unique)
nt_pts = len(t_unique)
u_grid = u_sorted.reshape((nx_pts, nt_pts))

# Create the meshgrid for plotting
X, T = np.meshgrid(x_unique, t_unique, indexing='ij')

# Create a PyVista grid
grid = pv.StructuredGrid()
grid.points = np.c_[X.ravel(), T.ravel(), np.zeros(X.size)]
grid.dimensions = [nx_pts, nt_pts, 1]
grid["u"] = u_grid.ravel(order="F")  # Fortran order for PyVista

# Plot the contour
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="u", cmap="viridis", show_edges=False)
plotter.view_xy()
plotter.add_axes()
plotter.show_grid()
plotter.show(title="Displacement u(x, t) contour")