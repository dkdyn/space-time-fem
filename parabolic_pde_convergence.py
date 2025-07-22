"""
FEniCSx program for solving the non-dimensional heat equation in a 1D bar
using finite elements in space and time-stepping by  the mid-point rule (Crank-Nicolson).

The PDE is:
    du/dt - d^2u/dx^2 = 0  for x in (0, 1), t in (0, T]
    u(x, 0) = u_0(x)            (Initial condition)
    u(0, t) = u(1, t) = 0       (Boundary conditions)
"""
import numpy as np
import ufl
#import basix.ufl
from dolfinx import geometry
from dolfinx.fem import (Constant, Function, functionspace, dirichletbc,
                         form)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import locate_dofs_geometrical

from dolfinx.mesh import create_interval
from mpi4py import MPI
#import matplotlib.pyplot as plt
import pyvista as pv

comm = MPI.COMM_WORLD

def time_stepping(nx,nt, T):
    dt_val = T / nt     # Time step size    
    domain = create_interval(comm, nx, [0, 1])

    # Use Lagrange polynomials of degree 1 for the function space
    V = functionspace(domain, ("Lagrange", 1))

    # --- 3. Define initial condition ---
    # The initial temperature distribution is u(x, 0) = sin(pi*x)
    def initial_condition(x):
        return np.sin(np.pi * x[0])

    # u_n will store the solution at the previous time step (t_n)
    u_n = Function(V)
    u_n.name = "u_n"
    u_n.interpolate(initial_condition)

    # --- 4. Define boundary conditions ---
    boundary_dofs_L = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0))
    boundary_dofs_R = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1))

    # Create the boundary condition objects
    bc_L = dirichletbc(np.float64(0.0), boundary_dofs_L, V)
    bc_R = dirichletbc(np.float64(0.0), boundary_dofs_R, V)
    bcs = [bc_L, bc_R]

    # --- 5. Define the variational problem ---
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    dt = Constant(domain, np.float64(dt_val))

    # The mid-point rule is: (u - u_n)/dt = alpha * d^2(0.5*(u + u_n))/dx^2
    # Weak form: ∫(u - u_n)/dt * v dx + ∫alpha * grad(0.5*(u + u_n)) ⋅ grad(v) dx = 0
    # Rearranging for a linear system a(u, v) = L(v):
    # a(u,v) = ∫(u*v + 0.5*dt*alpha*dot(grad(u), grad(v)))dx
    # L(v)   = ∫(u_n*v - 0.5*dt*alpha*dot(grad(u_n), grad(v)))dx
    a = u * v * ufl.dx + 0.5 * dt  * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx - 0.5 * dt * ufl.dot(ufl.grad(u_n), ufl.grad(v)) * ufl.dx

    # Compile the forms
    problem_a = form(a)
    problem_L = form(L)

    # --- 6. Set up for time-stepping and results storage ---
    # Create the solution function `uh` which will be solved for
    uh = Function(V)
    uh.name = "u"


    # --- 7. Set up the solver and time-stepping loop ---
    problem = LinearProblem(problem_a, problem_L, bcs=bcs, u=uh, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})


    t = 0.0
    u_sol = np.zeros((nt+1, nx+1))
    u_sol[0, :] = u_n.x.array
    print("Starting time-stepping loop...")
    for n in range(nt):
        print(t)
        t += dt_val

        # Solve the linear problem for the current time step
        problem.solve()

        # Update the previous solution for the next time step
        u_n.x.array[:] = uh.x.array
        
        x_coords = V.tabulate_dof_coordinates()[:, 0]
        sort_order = np.argsort(x_coords)
        #print(x_coords[sort_order])
        u_values = uh.x.array
        u_sol[n+1, :] =  u_values[sort_order]
    
    return u_sol

T = 1.0             # Total simulation time
nt = 8              # Number of time steps
nx = 4 # Number of elements in the spatial mesh

u_ts = time_stepping(nx, nt, T)



xt = np.meshgrid(np.linspace(0, 1, nx+1), np.linspace(0, T, nt+1), indexing='ij')
X, T = xt  # X: space, T: time, both shape (nx+1, nt+1)

#TODO get here error from exact solution!
u_grid = u_ts.T  # shape (nx+1, nt+1)
print(u_grid.shape)

#TODO for space-time
# for i in range(u_grid.n_points):
#     coord = u_grid.points[i]
#     value = u_grid.point_data["u"][i]
#     print(f"Point {i}: (x, t) = {coord}, u = {value}")

points = np.zeros((X.size, 3))
points[:, 0] = X.ravel(order="F")  # x
points[:, 1] = T.ravel(order="F")  # t
points[:, 2] = u_grid.ravel(order="F")  # u as height

# Create the structured grid
grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = [X.shape[0], X.shape[1], 1]
grid["u"] = u_grid.ravel(order="F")

# Plot the surface
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="u", cmap="viridis", show_edges=True)
#plotter.view_xy()
plotter.show_grid()
plotter.add_axes()
plotter.show(title="u(x, t) surface plot")