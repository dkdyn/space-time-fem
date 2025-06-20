from mpi4py import MPI
import dolfinx
from dolfinx.fem.petsc import LinearProblem
import basix.ufl
import ufl
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

def visualize_mixed(mixed_function: dolfinx.fem.Function, scale=1.0):
    u_c = mixed_function.sub(0).collapse()
    v_c = mixed_function.sub(1).collapse()

    u_grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(u_c.function_space))
    u_grid.point_data["u"] = u_c.x.array
    plotter_u = pv.Plotter()
    plotter_u.add_mesh(u_grid, show_edges=False)
    plotter_u.view_xy()
    plotter_u.show()
    
    p_grid = pv.UnstructuredGrid(*dolfinx.plot.vtk_mesh(v_c.function_space))
    p_grid.point_data["v"] = v_c.x.array
    plotter_p = pv.Plotter()
    plotter_p.add_mesh(p_grid, show_edges=False)
    plotter_p.view_xy()
    plotter_p.show()

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 80, 40, dolfinx.mesh.CellType.quadrilateral)

el_u = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
el_v = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
el_mixed = basix.ufl.mixed_element([el_u, el_v])

W = dolfinx.fem.functionspace(mesh, el_mixed)
u, v = ufl.TrialFunctions(W)
Du, Dv = ufl.TestFunctions(W)


# Equation 1: du/dt - v = 0
a1 = (u.dx(1) * Dv - v * Dv) * ufl.dx
L1 = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)) * Dv * ufl.dx

# Equation 2: dv/dt - c^2 * d^2u/dx^2 = f
# Integrate by parts in space: -c^2 * d^2u/dx^2 -> c^2 * du/dx * dw_v/dx
f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)) # Source term
a2 = (v.dx(1) * Du + u.dx(0) * Du.dx(0)) * ufl.dx
L2 = f * Du * ufl.dx

# Combine into a single system
a = a1 + a2 
L_form = L1 + L2

#ds = ufl.Measure("ds", domain=mesh)
#mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

def left_marker(x):
    return np.isclose(x[0], 0.0)
left_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, left_marker)

def right_marker(x):
    return np.isclose(x[0], 1.0)
right_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, right_marker)

def bottom_marker(x):
    return np.isclose(x[1], 0.0)
bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, bottom_marker)

def top_marker(x):
    return np.isclose(x[1], 1.0)
top_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1, top_marker)

W0 = W.sub(0)
U, U_to_W0 = W0.collapse()

W1 = W.sub(1)
V, V_to_W1 = W1.collapse()

left_u_dofs   = dolfinx.fem.locate_dofs_topological((W0,U), mesh.topology.dim - 1, left_facets)
right_u_dofs  = dolfinx.fem.locate_dofs_topological((W0,U), mesh.topology.dim - 1, right_facets)
bottom_u_dofs = dolfinx.fem.locate_dofs_topological((W0,U), mesh.topology.dim - 1, bottom_facets)
bottom_v_dofs = dolfinx.fem.locate_dofs_topological((W1,V), mesh.topology.dim - 1, bottom_facets)

left_u = dolfinx.fem.Function(U)
def left_f(x):
    values = np.zeros((1, x.shape[1]))
    values[0, :] = 0.0*x[0]
    return values
left_u.interpolate(left_f)
left_u_bc = dolfinx.fem.dirichletbc(left_u, left_u_dofs, W0)

right_u = dolfinx.fem.Function(U)
def right_f(x):
    values = np.zeros((1, x.shape[1]))
    values[0, :] = 0.0*x[0]
    return values
right_u.interpolate(right_f)
right_u_bc = dolfinx.fem.dirichletbc(right_u, right_u_dofs, W0)

bottom_u = dolfinx.fem.Function(U)
def bottom_f(x):
    values = np.zeros((1, x.shape[1]))
    values[0, :] = np.sin(np.pi * x[0])
    return values
bottom_u.interpolate(bottom_f)
bottom_u_bc = dolfinx.fem.dirichletbc(bottom_u, bottom_u_dofs, W0)

bottom_v = dolfinx.fem.Function(V)
def bottom_ff(x):
    values = np.zeros((1, x.shape[1]))
    values[0, :] = 0.0*x[0]
    return values
bottom_v.interpolate(bottom_ff)
bottom_v_bc = dolfinx.fem.dirichletbc(bottom_v, bottom_v_dofs, W1)

bcs = [left_u_bc, right_u_bc, bottom_u_bc, bottom_v_bc]
problem = LinearProblem(a, L_form, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
w_sol = problem.solve()

visualize_mixed(w_sol)
