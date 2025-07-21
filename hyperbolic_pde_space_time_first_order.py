"""non-dimensional hyperbolic PDE in space-time, first order formulation"""
from mpi4py import MPI
import dolfinx
from dolfinx.fem.petsc import LinearProblem
import basix.ufl
import ufl
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from petsc4py.PETSc import ScalarType
from dolfinx import default_scalar_type

domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 8, dolfinx.mesh.CellType.quadrilateral)

el_u = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
el_v = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
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
plotter_u = pv.Plotter()
u_warped = u_grid.warp_by_scalar()
plotter_u.add_mesh(u_warped, show_edges=True)
#plotter_u.view_xy()
plotter_u.show()
