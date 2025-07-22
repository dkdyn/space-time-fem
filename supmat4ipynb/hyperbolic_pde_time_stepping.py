"""
Solves the non-dimensional wave equation in 1D using the leap-frog method.
"""
import dolfinx
from dolfinx import geometry
from dolfinx.fem import Function, form
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_interval
from dolfinx.io import VTXWriter
from ufl import TrialFunction, TestFunction, dx, inner, grad, sin, pi, SpatialCoordinate
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
#import matplotlib.pyplot as plt
import pyvista as pv


nx = 4  # Number of spatial cells
nt = 8
T = 1.0  # Total time (unit interval)
dt = T/nt  # Time step on unit interval


# 1. Mesh and Function Space
domain = create_interval(MPI.COMM_WORLD, nx, [0.0, 1.0])
#V = dolfinx.fem.function.functionspace(mesh, ("CG", 1))   # unit length
V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))

# 2. Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# 3. Time-stepping variables
u_n = Function(V)  # u at current time step (n)
u_n_minus_1 = Function(V)  # u at previous time step (n-1)
u_n_plus_1 = Function(V)  # u at next time step (n+1)

# 4. Initial conditions
x = SpatialCoordinate(domain)
u_n.interpolate(lambda x: np.sin(np.pi * x[0]))
u_n_minus_1.interpolate(lambda x: np.sin(np.pi * x[0])) # Assuming zero initial velocity, i.a. same displacements on the time before start

# 5. Weak form for Leap-Frog method
# u^{n+1} = 2u^n - u^{n-1} + dt^2 * c^2 * div(grad(u^n))
# Weak form:
# inner(u_n_plus_1, v) * dx = inner(2*u_n - u_n_minus_1, v) * dx - dt**2 * c**2 * inner(grad(u_n), grad(v)) * dx

# Define the LHS and RHS for the LinearProblem
a = inner(u, v) * dx  # This will be the mass matrix
# The RHS depends on u_n and u_n_minus_1, so it will be updated in the loop
L_expr = (2 * u_n - u_n_minus_1) * v * dx - dt**2  * inner(grad(u_n), grad(v)) * dx

# 6. Boundary Conditions (Homogeneous Dirichlet at x=0 and x=Lx)
left_boundary = lambda x: np.isclose(x[0], 0.0)
right_boundary = lambda x: np.isclose(x[0], 1.0)

left_facets = dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim - 1, left_boundary)
right_facets = dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim - 1, right_boundary)

left_dofs = dolfinx.fem.locate_dofs_topological(V, domain.topology.dim - 1, left_facets)
right_dofs = dolfinx.fem.locate_dofs_topological(V, domain.topology.dim - 1, right_facets)

bcs = [dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), left_dofs, V),
    dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), right_dofs, V)]

# 7. Setup LinearProblem
# We pass 'a' and 'L_expr' to LinearProblem. The solution 'u_n_plus_1'
# will be updated by the solver.
# For a direct solver (LU factorization), we set ksp_type to "preonly" and pc_type to "lu".
# Using "mumps" for the factor solver type is generally a good robust choice if available.
problem = LinearProblem(a, L_expr, bcs=bcs, u=u_n_plus_1,
                        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})

# 8. Time-stepping loop
t = 0.0
u_sol = np.zeros((nt+1, nx+1))
u_sol[0, :] = u_n.x.array
for n in range(nt):
    t += dt

    # The LinearProblem object's 'L' form will be re-assembled automatically
    # when problem.solve() is called because it depends on u_n and u_n_minus_1,
    # which are updated in each time step.
    problem.solve()

    # Update for next time step
    u_n_minus_1.x.array[:] = u_n.x.array
    u_n.x.array[:] = u_n_plus_1.x.array

    x_coords = V.tabulate_dof_coordinates()[:, 0]
    sort_order = np.argsort(x_coords)
    u_values = u_n.x.array
    u_sol[n+1, :] =  u_values[sort_order]

# --- 8. Plot solution ---
xt = np.meshgrid(np.linspace(0, 1, nx+1), np.linspace(0, T, nt+1), indexing='ij')
X, T = xt  # X: space, T: time, both shape (nx+1, nt+1)

u_grid = u_sol.T  # shape (nx+1, nt+1)

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