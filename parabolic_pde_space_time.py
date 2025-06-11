"""
FEniCSx program for solving the transient heat equation in a 1D bar
using a full space-time finite element method.

The PDE is:
    du/dt - alpha * d^2u/dx^2 = 0   for (x, t) in (0, 1) x (0, T]

Boundary and Initial Conditions are all treated as Dirichlet BCs on the
boundary of the space-time domain:
    u(x, 0) = sin(pi*x)         (Initial condition on the bottom boundary)
    u(0, t) = 0                 (BC on the left boundary)
    u(1, t) = 0                 (BC on the right boundary)

The weak form is derived by multiplying by a test function v(x,t) and
integrating over the entire space-time domain:
∫(du/dt * v + alpha * du/dx * dv/dx) dx dt = 0
"""
import numpy as np
import ufl

from dolfinx.fem import (Constant, Function, functionspace, dirichletbc,
                         form)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import locate_dofs_geometrical
from dolfinx.io import VTKFile
from dolfinx import mesh 
from mpi4py import MPI
import matplotlib.pyplot as plt
import os

# --- 1. Define problem parameters ---
T = 2.0           # Total simulation time
alpha_val = 0.005 # Thermal diffusivity

# --- 2. Create the space-time mesh and define function space ---
comm = MPI.COMM_WORLD
nx = 100 # Number of spatial elements
nt = 100 # Number of temporal elements

# Create a 2D mesh for the space-time domain: (x, t) in [0,1]x[0,T]
mesh = mesh.create_rectangle(comm,
                        [np.array([0, 0]), np.array([1, T])],
                        [nx, nt],
                        cell_type=mesh.CellType.quadrilateral)

# Use Lagrange polynomials of degree 1 on the space-time mesh
V = functionspace(mesh, ("Lagrange", 1))

# --- 3. Define boundary and initial conditions ---
# In space-time FEM, all are treated as Dirichlet BCs on the domain boundary.

# Define the initial condition u(x, 0) = sin(pi*x) on the t=0 boundary
def initial_condition_func(x):
    return np.sin(np.pi * x[0])

u_initial = Function(V)
u_initial.interpolate(initial_condition_func)
initial_dofs = locate_dofs_geometrical(V, lambda x: np.isclose(x[1], 0))
bc_initial = dirichletbc(u_initial, initial_dofs)

# Define the spatial boundary condition u(0, t) = 0
left_boundary_dofs = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
bc_left = dirichletbc(np.float64(0.0), left_boundary_dofs, V)

# Define the spatial boundary condition u(1, t) = 0
right_boundary_dofs = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1))
bc_right = dirichletbc(np.float64(0.0), right_boundary_dofs, V)

# Combine all boundary conditions
bcs = [bc_initial, bc_left, bc_right]

# --- 4. Define the variational problem ---
# Define trial and test functions on the space-time domain
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# The gradient on the 2D mesh is grad(u) = (du/dx, du/dt)
# So, du/dx = grad(u)[0] and du/dt = grad(u)[1]
alpha = Constant(mesh, np.float64(alpha_val))

# Weak form: ∫(du/dt*v + alpha*du/dx*dv/dx) dV = 0
# The domain of integration dV is now the entire space-time domain.
a = (ufl.grad(u)[1] * v + alpha * ufl.grad(u)[0] * ufl.grad(v)[0]) * ufl.dx
L = Constant(mesh, np.float64(0.0)) * v * ufl.dx # Homogeneous PDE


# --- 5. Set up solver and solve ---
# This solves for the entire space-time solution at once.
print("Setting up and solving the space-time problem...")
problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
uh.name = "u_spacetime"
print("Solve complete.")

# --- 6. Save and plot results ---
# Save the full 2D space-time solution to a VTK file for ParaView
results_dir = 'heat_spacetime_fenicsx'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

with VTKFile(comm, os.path.join(results_dir, "solution.pvd"), "w") as vtk:
    vtk.write_function(uh, 0.0)

# --- 7. Plot the solution at the final time T ---
print("Extracting and plotting solution at final time T...")
# We need to evaluate the 2D solution `uh` on the line where t=T.
# Create points on the final time slice
num_plot_points = 500
x_plot = np.linspace(0, 1, num_plot_points)
t_final_points = np.zeros((num_plot_points, 3))
t_final_points[:, 0] = x_plot
t_final_points[:, 1] = T

# # Plot the extracted 1D solution
# plt.figure(figsize=(10, 6))
# plt.plot(np.array(points_on_proc)[:, 0], u_final_values)
# plt.title(f'Space-Time FEM: Temperature distribution at t = {T:.2f}')
# plt.xlabel('Position (x)')
# plt.ylabel('Temperature (u)')
# plt.grid(True)
# plt.show()

# V2 = fem.functionspace(domain, ("Lagrange", 2))
# uex = fem.Function(V2)
# uex.interpolate(fem.Expression(u_ex, V2.element.interpolation_points()))

# L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
# error_local = fem.assemble_scalar(L2_error)
# error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

# error_max = numpy.max(numpy.abs(uD.x.array-uh.x.array))
# # Only print the error on one process
# if domain.comm.rank == 0:
#     print(f"Error_L2 : {error_L2:.2e}")
#     print(f"Error_max : {error_max:.2e}")


# print(pv.global_theme.jupyter_backend)

# print("Plotter 1...")
# u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
# u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
# u_grid.point_data["u"] = uh.x.array.real
# u_grid.set_active_scalars("u")
# u_plotter = pv.Plotter()
# u_plotter.add_mesh(u_grid, show_edges=True)
# u_plotter.view_xy()
# if not pv.OFF_SCREEN:
#     u_plotter.show()

