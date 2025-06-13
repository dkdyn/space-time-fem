"""
FEniCSx program for solving the transient heat equation in a 1D bar
with Dirichlet boundary conditions.

This is a direct translation of the legacy FEniCS code to the FEniCSx API.

The PDE is:
    du/dt = alpha * d^2u/dx^2   for x in (0, 1), t in (0, T]
    u(x, 0) = u_0(x)            (Initial condition)
    u(0, t) = 0                 (Boundary condition at x=0)
    u(1, t) = 0                 (Boundary condition at x=1)

The discretization uses the Finite Element Method for space and the
mid-point (Crank-Nicolson) rule for time.
"""
import numpy as np
import ufl
#import basix.ufl
from dolfinx import geometry
from dolfinx.fem import (Constant, Function, functionspace, dirichletbc,
                         form)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import locate_dofs_geometrical
from dolfinx.io import VTKFile
from dolfinx.mesh import create_interval
from mpi4py import MPI
import matplotlib.pyplot as plt
import os

# --- 1. Define problem parameters ---
T = 1.0           # Total simulation time
num_steps = 8   # Number of time steps
dt_val = T / num_steps # Time step size
alpha_val = 1.0     # Thermal diffusivity

# --- 2. Create mesh and define function space ---
comm = MPI.COMM_WORLD
nx = 4 # Number of elements in the mesh
mesh = create_interval(comm, nx, [0, 1])

# Use Lagrange polynomials of degree 1 for the function space
# This is the corrected way to define the FunctionSpace in modern FEniCSx
#element = basix.ufl.element("Lagrange", "triangle", 1)
V = functionspace(mesh, ("Lagrange", 1))

# --- 3. Define initial condition ---
# The initial temperature distribution is u(x, 0) = sin(pi*x)
# In FEniCSx, we use a function to define expressions
def initial_condition(x):
    return np.sin(np.pi * x[0])

# u_n will store the solution at the previous time step (t_n)
u_n = Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

# --- 4. Define boundary conditions ---
# Define Dirichlet boundary conditions: temperature is fixed at 0.0 at both ends.
# First, locate the degrees of freedom on the boundary
boundary_dofs_L = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
boundary_dofs_R = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1))

# Create the boundary condition objects
bc_L = dirichletbc(np.float64(0.0), boundary_dofs_L, V)
bc_R = dirichletbc(np.float64(0.0), boundary_dofs_R, V)
bcs = [bc_L, bc_R]

# --- 5. Define the variational problem ---
# Define trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Define constants used in the form
dt = Constant(mesh, np.float64(dt_val))
alpha = Constant(mesh, np.float64(alpha_val))

# The mid-point rule is: (u - u_n)/dt = alpha * d^2(0.5*(u + u_n))/dx^2
# Weak form: ∫(u - u_n)/dt * v dx + ∫alpha * grad(0.5*(u + u_n)) ⋅ grad(v) dx = 0
# Rearranging for a linear system a(u, v) = L(v):
# a(u,v) = ∫(u*v + 0.5*dt*alpha*dot(grad(u), grad(v)))dx
# L(v)   = ∫(u_n*v - 0.5*dt*alpha*dot(grad(u_n), grad(v)))dx
a = u * v * ufl.dx + 0.5 * dt * alpha * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = u_n * v * ufl.dx - 0.5 * dt * alpha * ufl.dot(ufl.grad(u_n), ufl.grad(v)) * ufl.dx

# Compile the forms
problem_a = form(a)
problem_L = form(L)

# --- 6. Set up for time-stepping and results storage ---
# Create a directory for results if it doesn't exist
#if not os.path.exists('heat_transient_1d_fenicsx'):
#    os.makedirs('heat_transient_1d_fenicsx')

# Create VTK file for saving the solution
#vtkfile = VTKFile(comm, "heat_transient_1d_fenicsx/solution.pvd", "w")

# Create the solution function `uh` which will be solved for
uh = Function(V)
uh.name = "u"

# Save initial state
#vtkfile.write_function(u_n, 0.0)

# --- 7. Set up the solver and time-stepping loop ---
problem = LinearProblem(problem_a, problem_L, bcs=bcs, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

t_values = []
u_values = []
t_values.append(0.0)
u_values.append(1.0)
point = np.array([[0.5,0,0]], dtype=np.float64)  # shape (1, 1) for 1D

# Build a bounding box tree for the mesh
bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)

# Find the cell that contains the point
cell_candidates = geometry.compute_collisions_points(bb_tree, point)
colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, point)
cell = colliding_cells.links(0)[0]

t = 0.0
print("Starting time-stepping loop...")
for n in range(num_steps):
    t += dt_val

    # Solve the linear problem for the current time step
    problem.solve()

    # Save solution to file
    #vtkfile.write_function(uh, t)

    # Update the previous solution for the next time step
    u_n.x.array[:] = uh.x.array

    t_values.append(t)
    u_value = u_n.eval(point, np.array([cell], dtype=np.int32))
    u_values.append(u_value[0]) 

print("Simulation finished.")
#vtkfile.close()

plt.plot(t_values, u_values, "k", linewidth=2, label="u")
plt.show()

# Save t_values and u_values to a CSV file
np.savetxt("results_para_time_stepping.csv", np.column_stack([t_values, u_values]), delimiter=",", header="t,u", comments='')


# --- 8. Plot the final solution ---
# Get the coordinates of the degrees of freedom for plotting
x_coords = V.tabulate_dof_coordinates()[:, 0]
# Get the solution vector
u_values = uh.x.array
# Sort the values by the x-coordinate for a clean plot
sort_order = np.argsort(x_coords)

print("Plotting final solution.")
plt.figure(figsize=(10, 6))
plt.plot(x_coords[sort_order], u_values[sort_order])
plt.title(f'Temperature distribution at t = {T:.2f}')
plt.xlabel('Position (x)')
plt.ylabel('Temperature (u)')
plt.grid(True)
plt.show()

# Plot the initial condition for comparison
u_initial_values = Function(V)
u_initial_values.interpolate(initial_condition)
u_initial_array = u_initial_values.x.array

plt.figure(figsize=(10, 6))
plt.plot(x_coords[sort_order], u_initial_array[sort_order])
plt.title('Initial temperature distribution (t = 0.0)')
plt.xlabel('Position (x)')
plt.ylabel('Temperature (u)')
plt.grid(True)
plt.show()
