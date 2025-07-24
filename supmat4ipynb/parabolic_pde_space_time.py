"""
FEniCSx program for solving the non-dimensional heat equation in a 1D bar
using a full space-time finite element method.

The PDE is:
    du/dt - d^2u/dx^2 = 0   for (x, t) in (0, 1) x (0, 1]

Boundary and Initial Conditions are all treated as Dirichlet BCs on the
boundary of the space-time domain:
    u(x, 0) = sin(pi*x)         (Initial condition on the bottom boundary)
    u(0, t) = 0                 (BC on the left boundary)
    u(1, t) = 0                 (BC on the right boundary)

The weak form is derived by multiplying by a test function v(x,t) and
integrating over the entire space-time domain:
∫(du/dt * v + du/dx * dv/dx) dx dt = 0
"""
import numpy as np
import ufl
from dolfinx.fem import (Constant, Function, functionspace, dirichletbc, form, assemble_scalar, Expression)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import locate_dofs_geometrical
from dolfinx import plot
from dolfinx import default_scalar_type
from dolfinx import mesh 
from mpi4py import MPI
import matplotlib.pyplot as plt
import pyvista as pv


# ---   1. Create the space-time mesh and define function space   ---
comm = MPI.COMM_WORLD
nx = 4 # Number of spatial elements
nt = 8 # Number of temporal elements
order = 1  # Polynomial order

# Create a 2D mesh for the space-time domain: (x, t) in [0,1]x[0,1]
domain = mesh.create_rectangle(comm,
                        [np.array([0, 0]), np.array([1, 1])],
                        [nx, nt],
                        cell_type=mesh.CellType.quadrilateral) # TODO unit square

# Use Lagrange polynomials of degree 1 on the space-time mesh
V = functionspace(domain, ("Lagrange", order))

# ---   2. Define boundary and initial conditions   ---

# Define the initial, i.e. time boundary, condition u(x, 0)
def initial_condition_func(x):
    return np.sin(np.pi * x[0])

u_initial = Function(V)
u_initial.interpolate(initial_condition_func)
initial_dofs = locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0))
bc_initial = dirichletbc(u_initial, initial_dofs)

# Define the spatial boundary condition u(0, t)
left_boundary_dofs = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
bc_left = dirichletbc(np.float64(0.0), left_boundary_dofs, V)

# Define the spatial boundary condition u(1, t) 
right_boundary_dofs = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1))
bc_right = dirichletbc(np.float64(0.0), right_boundary_dofs, V)

# Combine all boundary conditions (space and time)
bcs = [bc_initial, bc_left, bc_right]

# ---   3. Define the variational problem   ---
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Weak form: ∫(du/dt*v + alpha*du/dx*dv/dx) dV = 0
# du/dx = grad(u)[0] and du/dt = grad(u)[1]
# The domain of integration dV is now the entire space-time mesh.
a = (ufl.grad(u)[1] * v + ufl.grad(u)[0] * ufl.grad(v)[0]) * ufl.dx
L = Constant(domain, np.float64(0.0)) * v * ufl.dx # Homogeneous PDE


# ---   4. Set up solver and solve   ---
# This solves for the entire space-time solution at once.
problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
uh.name = "u_spacetime"


# ---   5. postprocessing (here plot only)   ---
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
u_warped = u_grid.warp_by_scalar()
u_plotter = pv.Plotter()
u_plotter.add_mesh(u_warped, show_edges=True)
if not pv.OFF_SCREEN:
    u_plotter.show()

