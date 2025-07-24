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
from dolfinx import plot
from dolfinx.mesh import create_interval
from dolfinx import mesh
from mpi4py import MPI
import matplotlib.pyplot as plt
import pyvista as pv


comm = MPI.COMM_WORLD

def space_time_fem(nx, nt, order, T):
    # Create a 2D mesh for the space-time domain: (x, t) in [0,1]x[0,1]
    domain = mesh.create_rectangle(comm,
                            [np.array([0, 0]), np.array([1, 1])],
                            [nx, nt],
                            cell_type=mesh.CellType.quadrilateral)

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

    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
    u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["u"] = uh.x.array.real

    return u_grid

def time_stepping(nx,nt, order, T):
    dt_val = T / nt     # Time step size    
    domain = create_interval(comm, nx, [0, 1])

    # Use Lagrange polynomials of degree 1 for the function space
    V = functionspace(domain, ("Lagrange", order))

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
    x_coords = V.tabulate_dof_coordinates()[:, 0]
    sort_order = np.argsort(x_coords)
    u_sol = np.zeros((nt+1, order*nx+1))
    u_sol[0, :] = u_n.x.array[sort_order]
    #print(u_sol[0, :])
    #print("Starting time-stepping loop...")
    for n in range(nt):
        #print(t)
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
        #print(u_values[sort_order])
    return u_sol

T = 1.0             # Total simulation time
nt = 8              # Number of time steps
nx = 4 # Number of elements in the spatial mesh
order = 2  # Polynomial order for the finite element space
u_ts = time_stepping(nx, nt, order, T)

x = np.linspace(0, 1, nx*order+1)  # Spatial mesh points
t = np.linspace(0, 1, nt+1)
XX, TT = np.meshgrid(x, t, indexing='ij')
U = np.sin(np.pi*XX) * np.exp(-(np.pi**2)*TT)  # exponential decay

u_e = np.abs(u_ts.T - U)  
#print(u_st.shape)

points = np.zeros((XX.size, 3))
points[:, 0] = XX.ravel(order="F")  # x
points[:, 1] = TT.ravel(order="F")  # t
#points[:, 2] = u_st.ravel(order="F")  # u as height

# Create the structured grid
grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = [XX.shape[0], XX.shape[1], 1]
grid["u"] = u_e.ravel(order="F")

# Plot the surface
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="u", cmap="viridis", show_edges=True)   # , clim=[0, 0.1]
plotter.view_xy()
plotter.show_grid()
plotter.add_axes()
plotter.show(title="error in time-stepping")

max_error = np.max(u_e)
print(f"Maximum error time-stepping: {max_error:.4e}")


"""   ---   error plot for space-time FEM   ---"""
u_st = space_time_fem(nx, nt, 2, T)

for i in range(u_st.n_points):
    coord = u_st.points[i]
    value = u_st.point_data["u"][i]
    error = np.abs(value - np.sin(np.pi * coord[0])*np.exp(-coord[1]*np.pi**2))
    u_st.point_data["u"][i] = error

# Plot the surface
plotter = pv.Plotter()
plotter.add_mesh(u_st, scalars="u", cmap="viridis", show_edges=True)   # , clim=[0, 0.1]
plotter.view_xy()
plotter.show_grid()
plotter.add_axes()
plotter.show(title="error in space-time FEM")

max_error = np.max(u_st.point_data["u"])
print(f"Maximum error space-time: {max_error:.4e}")


""" convergence for both methods"""
N=6

ts_error = np.zeros(N)
for i in range(N):
    u_ts = time_stepping((2**i)*nx, (2**i)*nt, order, T)
    x = np.linspace(0, 1, (2**i)*nx*order+1)
    t = np.linspace(0, 1, (2**i)*nt+1)
    XX, TT = np.meshgrid(x, t, indexing='ij')
    U = np.sin(np.pi*XX) * np.exp(-(np.pi**2)*TT)  # exponential decay
    u_st = np.abs(u_ts.T - U)  # shape (nx+1, nt+1)
    max_error = np.max(u_st)
    ts_error[i] = max_error

st_error = np.zeros(N)
for i in range(N):
    u_st = space_time_fem((i+1)*nx, (i+1)*nt, order, T)
    for j in range(u_st.n_points):
        coord = u_st.points[j]
        value = u_st.point_data["u"][j]
        error = np.abs(value - np.sin(np.pi * coord[0])*np.exp(-coord[1]*np.pi**2))
        u_st.point_data["u"][j] = error
    max_error = np.max(u_st.point_data["u"])
    st_error[i] = max_error


plt.semilogy(ts_error, 'ro-')
plt.semilogy(st_error, 'go-')
plt.legend(['TS'+str(order), 'ST'+str(order)])
plt.show()