from mpi4py import MPI
import ufl
from dolfinx import fem, mesh, plot, default_scalar_type
from dolfinx.fem import dirichletbc, locate_dofs_geometrical
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary
from basix.ufl import element, mixed_element

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

# --- 1. Problem Parameters ---
c = 1.0  # Wave speed
nx = 4  # Number of elements in space
# Increase nt to satisfy the CFL condition (dt < dx/c)
# dx = 1/50 = 0.02. dt should be smaller. T/nt < dx/c -> 2/nt < 0.02 -> nt > 100.
nt = 8 # Number of elements in time

# --- 2. Create the Space-Time Mesh ---
# The domain is a 2D rectangle [0, L] x (0, T)
domain = mesh.create_rectangle(MPI.COMM_WORLD,
                               [np.array([0, 0]), np.array([1.0, 1.0])],
                               [nx, nt],
                               mesh.CellType.quadrilateral)

# x[0] is space, x[1] is time
x = ufl.SpatialCoordinate(domain)


# --- 3. Define the Function Space ---
P1 = element("Lagrange", domain.basix_cell(), 1)
W_element = mixed_element([P1, P1])
W = fem.functionspace(domain, W_element)

(u, v) = ufl.TrialFunctions(W)
(Du, Dv) = ufl.TestFunctions(W)


# Equation 1: du/dt - v = 0
a1 = (u.dx(1) * Dv - v * Dv) * ufl.dx
L1 = fem.Constant(domain, default_scalar_type(0.0)) * Dv * ufl.dx

# Equation 2: dv/dt - c^2 * d^2u/dx^2 = f
# Integrate by parts in space: -c^2 * d^2u/dx^2 -> c^2 * du/dx * dw_v/dx
f = fem.Constant(domain, default_scalar_type(0.0)) # Source term
a2 = (v.dx(1) * Du + c**2 * u.dx(0) * Du.dx(0)) * ufl.dx
L2 = f * Du * ufl.dx

# Combine into a single system
a = a1 + a2 
L_form = L1 + L2

# For x=0 (left boundary)
facets_left = locate_entities_boundary(domain, 1, lambda x: np.isclose(x[0], 0.0))
dofs_u_left = fem.locate_dofs_topological(W.sub(0), 1, facets_left)

# For x=1 (right boundary)
facets_right = locate_entities_boundary(domain, 1, lambda x: np.isclose(x[0], 1.0))
dofs_u_right = fem.locate_dofs_topological(W.sub(0), 1, facets_right)

# For t=0 (initial time)
facets_t0 = locate_entities_boundary(domain, 1, lambda x: np.isclose(x[1], 0.0))
dofs_u_initial = fem.locate_dofs_topological(W.sub(0), 1, facets_t0)
dofs_v_initial = fem.locate_dofs_topological(W.sub(1), 1, facets_t0)

# For t=1 (final time)
#facets_t1 = locate_entities_boundary(domain, 1, lambda x: np.isclose(x[1], 1.0))
#dofs_v_final = fem.locate_dofs_topological(W.sub(1), 1, facets_t1)

u_init = fem.Function(W.sub(0).collapse()[0])
u_init.interpolate(lambda x: np.sin(np.pi * x[0]))    # x[0]*(1-x[0])

# Create Dirichlet BCs for u and v at both boundaries
bc_u_left = dirichletbc(default_scalar_type(0), dofs_u_left, W.sub(0))
bc_u_right = dirichletbc(default_scalar_type(0), dofs_u_right, W.sub(0))
bc_u_initial = dirichletbc(u_init, dofs_u_initial)
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

"""   POST-PROCESSING AND PLOTTING   """
# Get the coordinates of all DOFs for u
u_coords = u_sol.function_space.tabulate_dof_coordinates()
u_vals = u_sol.x.array

# Select DOFs at t=0 (within a tolerance)
tol = 1e-10
mask_t0 = np.abs(u_coords[:, 1]) < tol
x_t0 = u_coords[mask_t0, 0]
u_t0 = u_vals[mask_t0]

# Sort by x for a proper line plot
sort_idx = np.argsort(x_t0)
x_t0_sorted = x_t0[sort_idx]
u_t0_sorted = u_t0[sort_idx]

# Compute the initial condition for comparison
u_init_exact = np.sin(np.pi * x_t0_sorted)

# Plot
plt.figure(figsize=(7, 4))
plt.plot(x_t0_sorted, u_t0_sorted, 'o-', label="FEM $u(x, t=0)$")
plt.plot(x_t0_sorted, u_init_exact, '--', label="Initial $\sin(\pi x)$")
plt.xlabel("x")
plt.ylabel("u(x, t=0)")
plt.legend()
plt.title("Displacement at $t=0$ vs Initial Condition")
plt.grid(True)
plt.tight_layout()
plt.show()

# Get the coordinates of all DOFs for v
v_coords = v_sol.function_space.tabulate_dof_coordinates()
v_vals = v_sol.x.array

# Select DOFs at t=0 (within a tolerance)
tol = 1e-10
mask_t0_v = np.abs(v_coords[:, 1]) < tol
x_t0_v = v_coords[mask_t0_v, 0]
v_t0 = v_vals[mask_t0_v]

# Sort by x for a proper line plot
sort_idx_v = np.argsort(x_t0_v)
x_t0_v_sorted = x_t0_v[sort_idx_v]
v_t0_sorted = v_t0[sort_idx_v]

# Compute the initial condition for comparison (if you have one, e.g., v=0)
v_init_exact = np.zeros_like(x_t0_v_sorted)

# Plot
plt.figure(figsize=(7, 4))
plt.plot(x_t0_v_sorted, v_t0_sorted, 'o-', label="FEM $v(x, t=0)$")
plt.plot(x_t0_v_sorted, v_init_exact, '--', label="Initial $v(x,0)=0$")
plt.xlabel("x")
plt.ylabel("v(x, t=0)")
plt.legend()
plt.title("Velocity at $t=0$ vs Initial Condition")
plt.grid(True)
plt.tight_layout()
plt.show()

print("Plotter u")
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(W.sub(0).collapse()[0])
u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = u_sol.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pv.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
if not pv.OFF_SCREEN:
    u_plotter.show()

