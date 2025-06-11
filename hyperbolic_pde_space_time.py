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
from dolfinx import fem, mesh, plot
from mpi4py import MPI
import ufl
import numpy as np
# Import element creation functions from basix.ufl
from basix.ufl import element, mixed_element

# --- 1. Problem Parameters ---
L = 1.0  # Length of the spatial domain
T = 4.0  # Total time
c = 1.0  # Wave speed
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
P1 = element("Lagrange", domain.basix_cell(), 1)
W_element = mixed_element([P1, P1])
W = fem.functionspace(domain, W_element)

# Define Trial and Test functions
(u, v) = ufl.TrialFunctions(W)
(Du, Dv) = ufl.TestFunctions(W)


# --- 4. Define the Variational Problem (Weak Form) ---
# In UFL, the derivative of a function 'h' with respect to the
# i-th coordinate is given by h.dx(i).
# Here, u.dx(0) is the spatial derivative (du/dx)
# and u.dx(1) is the temporal derivative (du/dt).

# Define the source term. fem.Constant requires the domain as the first argument.
f = fem.Constant(domain, dolfinx.default_scalar_type(0.0)) # Source term

# Equation 1: du/dt - v = 0
a1 = (u.dx(1) - v) * Du * ufl.dx
L1 = fem.Constant(domain, dolfinx.default_scalar_type(0.0)) * Du * ufl.dx

# Equation 2: dv/dt - c^2 * d^2u/dx^2 = f
# Integrate by parts in space: -c^2 * d^2u/dx^2 -> c^2 * du/dx * dw_v/dx
a2 = (v.dx(1) * Dv + c**2 * u.dx(0) * Dv.dx(0)) * ufl.dx
L2 = f * Dv * ufl.dx

# Combine into a single system
a = a1 + a2
L_form = L1 + L2


# --- 5. Initial and Boundary Conditions (Simplified) ---
# Define functions to locate the boundaries
def initial_boundary(x):
    # This function identifies the boundary at t=0
    return np.isclose(x[1], 0)

def spatial_boundary(x):
    # This function identifies the boundaries at x=0 and x=L
    return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], L))

# Find all facet entities on the boundaries first
initial_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, initial_boundary)
spatial_facets = mesh.locate_entities_boundary(domain, domain.topology.dim - 1, spatial_boundary)

# Define a constant for the value zero
u_zero = fem.Constant(domain, dolfinx.default_scalar_type(0.0))

# -- 1. Spatial Boundary Condition: u(0,t) = u(L,t) = 0 --
u_spatial_dofs = fem.locate_dofs_topological(W.sub(0), domain.topology.dim - 1, spatial_facets)
bc_u_spatial = fem.dirichletbc(u_zero, u_spatial_dofs, W.sub(0))

# -- 2. Initial Condition for Displacement: u(x,0) = f(x) --
#u_initial_dofs = fem.locate_dofs_topological(W.sub(0), domain.topology.dim - 1, initial_facets)
#bc_u_initial = fem.dirichletbc(u_zero, u_initial_dofs, W.sub(0))
# Define the initial velocity function (e.g., a Gaussian pulse)
class InitialDisplacement:
    def __init__(self):
        pass
    def __call__(self, x):
        # A Gaussian pulse as the initial velocity
        return np.sin(np.pi * x[0] / L)

# Create a fem.Function on the collapsed subspace for interpolation
U1, _ = W.sub(0).collapse()
u_initial_val = fem.Function(U1)
u_initial_val.interpolate(InitialDisplacement())

# Locate dofs for 'u' on the initial boundary (t=0)
u_initial_dofs = fem.locate_dofs_topological(W.sub(0), domain.topology.dim - 1, initial_facets)
# Create the BC object. When using a fem.Function as the value, the FunctionSpace is inferred
# and should not be passed as an argument.
bc_u_initial = fem.dirichletbc(u_initial_val, u_initial_dofs)


# -- 3. Initial Condition for Velocity: v(x,0) = g(x) --
# Define the initial velocity function (e.g., a Gaussian pulse)
class InitialVelocity:
    def __init__(self):
        pass
    def __call__(self, x):
        # A Gaussian pulse as the initial velocity
        return np.sin(2*np.pi * x[0] / L)

# Create a fem.Function on the collapsed subspace for interpolation
V1, _ = W.sub(1).collapse()
v_initial_val = fem.Function(V1)
v_initial_val.interpolate(InitialVelocity())

# Locate dofs for 'v' on the initial boundary (t=0)
v_initial_dofs = fem.locate_dofs_topological(W.sub(1), domain.topology.dim - 1, initial_facets)
# Create the BC object. When using a fem.Function as the value, the FunctionSpace is inferred
# and should not be passed as an argument.
bc_v_initial = fem.dirichletbc(v_initial_val, v_initial_dofs)


# Collect all boundary conditions into a list
bcs = [bc_u_spatial, bc_u_initial, bc_v_initial]


# --- 6. Assemble and Solve the Linear System ---
# The problem is linear, so we can directly solve the system Ax = b
problem = LinearProblem(a, L_form, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()


# --- 7. Post-processing and Visualization (Final Corrected Version) ---
# The solution 'uh' contains both displacement and velocity.
# We can split them to analyze them separately.
u_sol = uh.sub(0).collapse()
v_sol = uh.sub(1).collapse()

# Diagnostic check: Print the norms of the solution to check for NaNs
norm_u = u_sol.x.petsc_vec.norm()
norm_v = v_sol.x.petsc_vec.norm()
if MPI.COMM_WORLD.rank == 0:
    print(f"L2 norm of displacement solution (u): {norm_u}")
    print(f"L2 norm of velocity solution (v): {norm_v}")
    if np.isnan(norm_u) or np.isnan(norm_v) or np.isinf(norm_u) or np.isinf(norm_v):
        print("\nERROR: Solution contains NaN/inf values. The simulation is unstable.")
        print("This is likely due to the mesh discretization (CFL condition).")
        print("Try increasing 'nt' further or decreasing 'nx'.")

# To visualize the solution, we can plot the fields u and v
# over the entire space-time domain.
try:
    import pyvista
    #pyvista.start_xvfb()

    # Create a plotter with two subplots (one for u, one for v)
    plotter = pyvista.Plotter(shape=(1, 2), window_size=[1200, 600])

    # --- Plot displacement u(x,t) ---
    plotter.subplot(0, 0)
    # Create the VTK topology and geometry
    V_u, _ = W.sub(0).collapse()
    topology, cell_types, geometry = plot.vtk_mesh(V_u)
    # Create the pyvista grid object
    grid_u = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    # Attach the solution values to the grid
    grid_u.point_data["u"] = u_sol.x.array
    # Add the grid to the plotter
    plotter.add_mesh(grid_u, show_edges=True, scalars="u")
    plotter.add_text("Displacement u(x,t)", font_size=16)
    plotter.view_xy()
    plotter.camera.zoom(1.2)


    # --- Plot velocity v(x,t) ---
    plotter.subplot(0, 1)
    # Create the VTK topology and geometry
    V_v, _ = W.sub(1).collapse()
    topology, cell_types, geometry = plot.vtk_mesh(V_v)
    # Create the pyvista grid object
    grid_v = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    # Attach the solution values to the grid
    grid_v.point_data["v"] = v_sol.x.array
    # Add the grid to the plotter
    plotter.add_mesh(grid_v, show_edges=True, scalars="v")
    plotter.add_text("Velocity v(x,t)", font_size=16)
    plotter.view_xy()
    plotter.camera.zoom(1.2)

    # Show the plots
    plotter.show()


except (ImportError, ModuleNotFoundError):
    print("PyVista is not installed or other visualization error. Skipping visualization.")

