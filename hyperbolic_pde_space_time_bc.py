from mpi4py import MPI
from dolfinx import mesh, geometry
import numpy
import matplotlib.pyplot as plt

domain = mesh.create_unit_square(MPI.COMM_WORLD, 4, 8, mesh.CellType.quadrilateral)

from dolfinx.fem import functionspace
V = functionspace(domain, ("Lagrange", 1))

from dolfinx import fem
uD = fem.Function(V)
uD.interpolate(lambda x: numpy.sin(numpy.pi*x[0])*numpy.cos(numpy.pi*x[1]))


# Create facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)

boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

import ufl
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

from dolfinx import default_scalar_type
f = fem.Constant(domain, default_scalar_type(0.0))
K = fem.Constant(domain, numpy.array([[1, 0], [0, -1]], dtype=numpy.float64))

a = ufl.dot(K*ufl.grad(u), ufl.grad(v)) * ufl.dx
#a = (ufl.grad(u)[0]*ufl.grad(v)[0] - ufl.grad(u)[1]*ufl.grad(v)[1] ) * ufl.dx
#a = (u.dx(0)*v.dx(0) - u.dx(1)*v.dx(1)) * ufl.dx
L = f * v * ufl.dx

from dolfinx.fem.petsc import LinearProblem
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

V2 = fem.functionspace(domain, ("Lagrange", 2))
uex = fem.Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

error_max = numpy.max(numpy.abs(uD.x.array-uh.x.array))
# Only print the error on one process
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")

import pyvista as pv
print(pv.global_theme.jupyter_backend)

from dolfinx import plot
#pv.start_xvfb()   # troublemaker
domain.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)

grid = pv.UnstructuredGrid(topology, cell_types, geometry)

# print("Plotter 1...")
# plotter = pv.Plotter(off_screen=False)
# plotter.add_mesh(grid, show_edges=True)
# plotter.view_xy()
# if not pv.OFF_SCREEN:
#     plotter.show(interactive=True)
# else:
#     figure = plotter.screenshot("fundamentals_mesh.png")

print("Plotter 2...")
u_topology, u_cell_types, u_geometry = plot.vtk_mesh(V)
u_grid = pv.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
u_plotter = pv.Plotter()
u_plotter.add_mesh(u_grid, show_edges=True)
u_plotter.view_xy()
if not pv.OFF_SCREEN:
    u_plotter.show()

# print("Plotter 3...")
# warped = u_grid.warp_by_scalar()
# plotter2 = pv.Plotter()
# plotter2.add_mesh(warped, show_edges=True, show_scalar_bar=True)
# if not pv.OFF_SCREEN:
#     plotter2.show()

# print("Export...")
# from dolfinx import io
# from pathlib import Path
# results_folder = Path("results")
# results_folder.mkdir(exist_ok=True, parents=True)
# filename = results_folder / "fundamentals"
# with io.VTXWriter(domain.comm, filename.with_suffix(".bp"), [uh]) as vtx:
#     vtx.write(0.0)
# with io.XDMFFile(domain.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
#     xdmf.write_mesh(domain)
#     xdmf.write_function(uh)

# Get the coordinates of all DOFs for u
u_coords = uh.function_space.tabulate_dof_coordinates()
u_vals = uh.x.array

tol = 1e-10
mask_xc = numpy.abs(u_coords[:, 0]-0.5) < tol
tc = u_coords[mask_xc, 1]
uc = u_vals[mask_xc]

# Sort by x for a proper line plot
sort_idx = numpy.argsort(tc)
t_values = tc[sort_idx]
u_values = uc[sort_idx]


plt.plot(t_values, u_values, "k", linewidth=2, label="u")
plt.show()

# Save t_values and u_values to a CSV file
numpy.savetxt("results_hyper_space_time_bc.csv", numpy.column_stack([t_values, u_values]), delimiter=",", header="t,u", comments='')
