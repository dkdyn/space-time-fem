import dolfinx
from dolfinx.fem import Function, form
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_interval
from dolfinx.io import VTXWriter
from ufl import TrialFunction, TestFunction, dx, inner, grad, sin, pi, SpatialCoordinate
from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

Lx = 1.0  # Length of the domain
Nx = 100  # Number of spatial cells
dt = 0.001  # Time step size
T = 1.0  # Total simulation time
c = 1.0  # Wave speed


"""
Solves the 1D wave equation using the leap-frog method in FEniCSx,
employing dolfinx.fem.petsc.LinearProblem for simplified solving.

Args:
    Lx (float): Length of the 1D domain.
    Nx (int): Number of spatial cells.
    dt (float): Time step size.
    T (float): Total simulation time.
    c (float): Wave speed.
    output_filepath (str): File path for saving the solution (e.g., "wave_solution_leapfrog_linearproblem.bp").
"""

# 1. Mesh and Function Space
mesh = create_interval(MPI.COMM_WORLD, Nx, [0.0, Lx])
V = dolfinx.fem.function.functionspace(mesh, ("CG", 1))

# 2. Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# 3. Time-stepping variables
u_n = Function(V)  # u at current time step (n)
u_n_minus_1 = Function(V)  # u at previous time step (n-1)
u_n_plus_1 = Function(V)  # u at next time step (n+1)

# 4. Initial conditions
x = SpatialCoordinate(mesh)
u_n.interpolate(lambda x: np.sin(np.pi * x[0] / Lx))
u_n_minus_1.interpolate(lambda x: np.sin(np.pi * x[0] / Lx)) # Assuming zero initial velocity

# 5. Weak form for Leap-Frog method
# u^{n+1} = 2u^n - u^{n-1} + dt^2 * c^2 * div(grad(u^n))
# Weak form:
# inner(u_n_plus_1, v) * dx = inner(2*u_n - u_n_minus_1, v) * dx - dt**2 * c**2 * inner(grad(u_n), grad(v)) * dx

# Define the LHS and RHS for the LinearProblem
a = inner(u, v) * dx  # This will be the mass matrix
# The RHS depends on u_n and u_n_minus_1, so it will be updated in the loop
L_expr = (2 * u_n - u_n_minus_1) * v * dx - dt**2 * c**2 * inner(grad(u_n), grad(v)) * dx

# 6. Boundary Conditions (Homogeneous Dirichlet at x=0 and x=Lx)
left_boundary = lambda x: np.isclose(x[0], 0.0)
right_boundary = lambda x: np.isclose(x[0], Lx)

left_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, left_boundary)
right_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, right_boundary)

left_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, left_facets)
right_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, right_facets)

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
num_steps = int(T / dt)
t = 0.0

# Create VTK writer for visualization
#vtx_writer = VTXWriter(mesh.comm, output_filepath, u_n, engine="BP4")
#vtx_writer.write(t)

print(f"Starting simulation with {num_steps} steps...")
for i in range(num_steps):
    t += dt

    # The LinearProblem object's 'L' form will be re-assembled automatically
    # when problem.solve() is called because it depends on u_n and u_n_minus_1,
    # which are updated in each time step.
    problem.solve()

    # Update for next time step
    u_n_minus_1.x.array[:] = u_n.x.array
    u_n.x.array[:] = u_n_plus_1.x.array

    # Write solution to file
    #if i % 10 == 0:  # Write every 10 steps for efficiency
    #    vtx_writer.write(t)
    #    print(f"Time: {t:.4f}/{T:.4f}, Step: {i+1}/{num_steps}")

#vtx_writer.close()
#print("Simulation finished.")
# To visualize the results, you can use ParaView and open the "wave_solution_leapfrog_linearproblem.bp" file.

plt.plot(u_n.x.array)
plt.show()

