from mpi4py import MPI
from dolfinx import mesh, geometry
import numpy as np
#import matplotlib.pyplot as plt
import pyvista as pv
import ufl
import dolfinx
from dolfinx.fem import functionspace
from dolfinx import default_scalar_type
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx import plot
import basix.ufl
from dolfinx.fem import Function, form
from dolfinx.mesh import create_unit_interval
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
from ufl import TrialFunction, TestFunction, dx, inner, grad, sin, pi, SpatialCoordinate
import matplotlib.pyplot as plt
import sys

def space_time_bc(nx, nt, order, T):
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([1, T])],
        [nx, nt],
        cell_type=mesh.CellType.triangle,
    )

    V = functionspace(domain, ("Lagrange", order))

    uD = fem.Function(V)
    uD.interpolate(lambda x: np.sin(np.pi*x[0])*np.cos(np.pi*x[1]))

    # Create facet to cell connectivity required to determine boundary facets
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)

    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = fem.Constant(domain, default_scalar_type(0.0))
    K = fem.Constant(domain, np.array([[1, 0], [0, -1]], dtype=np.float64))

    a = ufl.dot(K*ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    uh.name = "u_spacetime"

    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
    u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["u"] = uh.x.array.real

    return u_grid


def space_time_2(nx, nt, order, T):
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([1, T])],
        [nx, nt],
        cell_type=mesh.CellType.triangle,
    )   

    # Function space (using quadratic elements for better accuracy)
    V = fem.functionspace(domain, ("Lagrange", order))

    # --- 2. Define Boundary Conditions ---
    # Locate facets for boundary conditions (initial time and left/right boundaries)
    fdim = domain.topology.dim - 1
    facets_t0 = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
    facets_space = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))
    )

    # Define Dirichlet boundary conditions
    # Initial displacement u(x, 0) = u_0(x)
    def u0_func(x):
        return np.sin(np.pi * x[0])

    u_initial = fem.Function(V)
    u_initial.interpolate(u0_func)
    dofs_t0 = fem.locate_dofs_topological(V, fdim, facets_t0)
    bc_t0 = fem.dirichletbc(u_initial, dofs_t0)

    # Spatial boundary u(0,t) = u(1,t) = 0
    dofs_space = fem.locate_dofs_topological(V, fdim, facets_space)
    bc_space = fem.dirichletbc(ScalarType(0.0), dofs_space, V)

    bcs = [bc_t0, bc_space]

    # --- 3. Define Variational Problem ---
    u = ufl.TrialFunction(V)
    phi = ufl.TestFunction(V)

    # Initial velocity v(x, 0) = v_0(x)
    # This will be integrated over the t=0 boundary
    v0 = fem.Function(V)
    v0.interpolate(lambda x: np.zeros_like(x[0]))

    # Source term f
    f = fem.Constant(domain, ScalarType(0.0))

    # Define the boundary integral for the initial velocity
    # Mark the t=0 boundary with tag 1
    T0_MARKER = 1
    facet_tags = mesh.meshtags(domain, fdim, facets_t0, np.full_like(facets_t0, T0_MARKER))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

    # Bilinear form (left-hand side)
    a = (u.dx(0) * phi.dx(0) - u.dx(1) * phi.dx(1)) * ufl.dx

    # Linear form (right-hand side)
    L = f * phi * ufl.dx + v0 * phi * ds(T0_MARKER)

    # --- 4. Solve the Linear System ---
    problem = LinearProblem(
        a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
    )
    uh = problem.solve()

    # --- 5. Post-processing and Visualization ---

    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
    u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["u"] = uh.x.array.real
    u_grid.set_active_scalars("u")
    return u_grid


def space_time_1(nx, nt, order, T):
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([1, T])],
        [nx, nt],
        dolfinx.mesh.CellType.quadrilateral)
    el_u = basix.ufl.element("Lagrange", domain.basix_cell(), order)
    el_v = basix.ufl.element("Lagrange", domain.basix_cell(), order)
    el_mixed = basix.ufl.mixed_element([el_u, el_v])

    W = dolfinx.fem.functionspace(domain, el_mixed)
    u, v = ufl.TrialFunctions(W)
    Du, Dv = ufl.TestFunctions(W)

    # Equation 1: du/dt - v = 0
    a1 = (u.dx(1) * Dv - v * Dv) * ufl.dx
    L1 = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(0.0)) * Dv * ufl.dx

    # Equation 2: dv/dt - c^2 * d^2u/dx^2 = f
    # Integrate by parts in space:   d^2u/dx^2   -->  du/dx * dw_v/dx
    a2 = (v.dx(1) * Du + u.dx(0) * Du.dx(0)) * ufl.dx
    f = dolfinx.fem.Constant(domain, dolfinx.default_scalar_type(0.0)) # Source term
    L2 = f * Du * ufl.dx

    # Combine into a single system
    a = a1 + a2 
    L_form = L1 + L2

    def left_right_marker(x):
        return np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))
    left_right_facets = dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim - 1, left_right_marker)

    def initial_time_marker(x):
        return np.isclose(x[1], 0.0)
    initial_time_facets = dolfinx.mesh.locate_entities_boundary(domain, domain.topology.dim - 1, initial_time_marker)

    W0 = W.sub(0)
    U, U_to_W0 = W0.collapse()

    W1 = W.sub(1)
    V, V_to_W1 = W1.collapse()

    leftright_u_dofs   = dolfinx.fem.locate_dofs_topological((W0,U), domain.topology.dim - 1, left_right_facets)
    initial_u_dofs = dolfinx.fem.locate_dofs_topological((W0,U), domain.topology.dim - 1, initial_time_facets)
    initial_v_dofs = dolfinx.fem.locate_dofs_topological((W1,V), domain.topology.dim - 1, initial_time_facets)

    leftright_u = dolfinx.fem.Function(U)
    leftright_u.x.array[:] = 0.0
    leftright_u_bc = dolfinx.fem.dirichletbc(leftright_u, leftright_u_dofs, W0)

    initial_u = dolfinx.fem.Function(U)
    def initial_u_f(x):
        values = np.zeros((1, x.shape[1]))
        values[0, :] = np.sin(np.pi * x[0])
        return values
    initial_u.interpolate(initial_u_f)
    initial_u_bc = dolfinx.fem.dirichletbc(initial_u, initial_u_dofs, W0)

    initial_v = dolfinx.fem.Function(V)
    def initial_v_f(x):
        values = np.zeros((1, x.shape[1]))
        values[0, :] = 0.0*x[0]
        return values
    initial_v.interpolate(initial_v_f)
    initial_v_bc = dolfinx.fem.dirichletbc(initial_v, initial_v_dofs, W1)

    bcs = [leftright_u_bc, initial_u_bc, initial_v_bc]
    problem = LinearProblem(a, L_form, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    w_sol = problem.solve()

    u_plot = w_sol.sub(0).collapse()
    #v_plot = w_sol.sub(1).collapse()
    u_grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u_plot.function_space))
    u_grid.point_data["u"] = u_plot.x.array
    return u_grid


def time_stepping(nx, nt, order, T):
    
    dt = T/nt  # Time step on unit interval

    # 1. Mesh and Function Space
    domain = create_unit_interval(MPI.COMM_WORLD, nx)   # TODO
    #V = dolfinx.fem.function.functionspace(mesh, ("CG", 1))   # unit length
    V = dolfinx.fem.functionspace(domain, ("Lagrange", order))

    # 2. Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # 3. Time-stepping variables
    u_n = Function(V)  # u at current time step (n)
    u_n_minus_1 = Function(V)  # u at previous time step (n-1)
    u_n_plus_1 = Function(V)  # u at next time step (n+1)

    # 4. Initial conditions
    #x = SpatialCoordinate(domain)
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
    x_coords = V.tabulate_dof_coordinates()[:, 0]
    sort_order = np.argsort(x_coords)
    u_sol = np.zeros((nt+1, order*nx+1))
    u_sol[0, :] = u_n.x.array[sort_order]
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
    return u_sol

"""
    MAIN SCRIPT
"""

nx = 4
nt = 16     # must be greater than 2*nx*order (on unit square) for stability
order = 1
T = 2.0

"""   ---  Error in Time-Stepping   ---   """
u_ts = time_stepping(nx, nt, order, T)

x = np.linspace(0, 1, nx*order+1)  # Spatial mesh points
t = np.linspace(0, T, nt+1)
XX, TT = np.meshgrid(x, t, indexing='ij')
U = np.sin(np.pi*XX) * np.cos(np.pi*TT)  # exact solution

u_err = np.abs(u_ts.T - U)  
#print(u_st.shape)

points = np.zeros((XX.size, 3))
points[:, 0] = XX.ravel(order="F")  # x
points[:, 1] = TT.ravel(order="F")  # t
#points[:, 2] = u_st.ravel(order="F")  # u as height (warping)

# Create the structured grid
grid = pv.StructuredGrid()
grid.points = points
grid.dimensions = [XX.shape[0], XX.shape[1], 1]
grid["u"] = u_err.ravel(order="F")

# Plot the surface
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="u", cmap="viridis", show_edges=True, clim=[0, 0.4], show_scalar_bar=False)   
plotter.view_xy()
#plotter.camera_position = [(0.5, 0.5, 10), (0.5, 0.5, 0), (0, 1, 0)]
plotter.show_bounds(xlabel="space",ylabel="time",use_2d=True,n_xlabels=2,n_ylabels=2, font_size=10, bold=False, fmt='%d')
#plotter.reset_camera()
plotter.show(title="error in time-stepping")
plotter.screenshot("error_TS.png")

max_error = np.max(u_err)
print(f"Maximum error time-stepping: {max_error:.4e}")

sys.exit()
"""   ---   Error in Space-Time BC formulation   ---   """
u_stbc = space_time_bc(nx, nt, order, T)

for i in range(u_stbc.n_points):
    coord = u_stbc.points[i]
    value = u_stbc.point_data["u"][i]
    error = np.abs(value - np.sin(np.pi * coord[0])*np.cos(coord[1]*np.pi))
    u_stbc.point_data["u"][i] = error

# Plot the surface
plotter = pv.Plotter()
plotter.add_mesh(u_stbc, scalars="u", cmap="viridis", show_edges=True, clim=[0, 0.65], scalar_bar_args={"vertical": True, "title": "error"})   
plotter.show_bounds(xtitle="x",ytitle="t")
plotter.show_grid()
plotter.add_axes()
plotter.show(title="error in space-time FEM BC")

plotter.screenshot("error_STbc.png")

max_error = np.max(u_stbc.point_data["u"])
print(f"Maximum error space-time BC: {max_error:.4e}")


"""   ---   Error in Space-Time 1.order formulation   ---   """
u_st1 = space_time_1(nx, nt, order, T)

for i in range(u_st1.n_points):
    coord = u_st1.points[i]
    value = u_st1.point_data["u"][i]
    error = np.abs(value - np.sin(np.pi * coord[0])*np.cos(coord[1]*np.pi))
    u_st1.point_data["u"][i] = error

# Plot the surface
plotter = pv.Plotter()
plotter.add_mesh(u_st1, scalars="u", cmap="viridis", show_edges=True, clim=[0, 0.65], show_scalar_bar=False)   
plotter.show_bounds(xtitle="x",ytitle="t")
plotter.show_grid()
plotter.add_axes()
plotter.show(title="error in space-time FEM 1.order")
plotter.screenshot("error_ST1o.png")

max_error = np.max(u_st1.point_data["u"])
print(f"Maximum error space-time 1.order: {max_error:.4e}")


"""   ---   Error in Space-Time 2.order formulation   ---   """
u_st2 = space_time_2(nx, nt, order, T)

for i in range(u_st2.n_points):
    coord = u_st2.points[i]
    value = u_st2.point_data["u"][i]
    error = np.abs(value - np.sin(np.pi * coord[0])*np.cos(coord[1]*np.pi))
    u_st2.point_data["u"][i] = error

# Plot the surface
plotter = pv.Plotter()
plotter.add_mesh(u_st2, scalars="u", cmap="viridis", show_edges=True, clim=[0, 0.65], show_scalar_bar=False)   
plotter.show_bounds(xtitle="x",ytitle="t")
plotter.show_grid()
plotter.add_axes()
plotter.show(title="error in space-time FEM BC 2.order")
plotter.screenshot("error_ST2o.png")

max_error = np.max(u_st2.point_data["u"])
print(f"Maximum error space-time 2.order: {max_error:.4e}")


""" convergence for all methods"""
N=3   # mesh refinements

ts_error = np.zeros(N)
for i in range(N):
    u_ts = time_stepping((2**i)*nx, (2**i)*nt, order, T)
    x = np.linspace(0, 1, (2**i)*nx*order+1)
    t = np.linspace(0, T, (2**i)*nt+1)
    XX, TT = np.meshgrid(x, t, indexing='ij')
    U = np.sin(np.pi*XX) * np.cos(np.pi*TT)  # exponential decay
    u_st = np.abs(u_ts.T - U)  # shape (nx+1, nt+1)
    max_error = np.max(u_st)
    ts_error[i] = max_error

stbc_error = np.zeros(N)
st1_error = np.zeros(N)
st2_error = np.zeros(N)
for i in range(N):
    u_stbc = space_time_bc((2**i)*nx, (2**i)*nt, order, T)
    for j in range(u_stbc.n_points):
        coord = u_stbc.points[j]
        value = u_stbc.point_data["u"][j]
        error = np.abs(value - np.sin(np.pi * coord[0])*np.cos(coord[1]*np.pi))
        u_stbc.point_data["u"][j] = error
    max_error = np.max(u_stbc.point_data["u"])
    stbc_error[i] = max_error

    u_st1 = space_time_1((2**i)*nx, (2**i)*nt, order, T)
    for j in range(u_st1.n_points):
        coord = u_st1.points[j]
        value = u_st1.point_data["u"][j]
        error = np.abs(value - np.sin(np.pi * coord[0])*np.cos(coord[1]*np.pi))
        u_st1.point_data["u"][j] = error
    max_error = np.max(u_st1.point_data["u"])
    st1_error[i] = max_error

    u_st2 = space_time_2((2**i)*nx, (2**i)*nt, order, T)
    for j in range(u_st2.n_points):
        coord = u_st2.points[j]
        value = u_st2.point_data["u"][j]
        error = np.abs(value - np.sin(np.pi * coord[0])*np.cos(coord[1]*np.pi))
        u_st2.point_data["u"][j] = error
    max_error = np.max(u_st2.point_data["u"])
    st2_error[i] = max_error

labels = ["$2^0$", "$2^1$", "$2^2$"]  # or any custom labels, length N
plt.semilogy(ts_error, 'ko-')
plt.semilogy(stbc_error, 'ro-')
plt.semilogy(st1_error, 'go-')
plt.semilogy(st2_error, 'bo-')
plt.legend(['TS', 'STbc'+str(order), 'ST1o'+str(order), 'ST2o'+str(order)])
plt.xticks(ticks=np.arange(len(labels)), labels=labels)
plt.xlabel("refinement factor")
plt.ylabel("error")
plt.title("convergence")
plt.show()