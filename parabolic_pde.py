from fenics import *

mesh = UnitSquareMesh(20, 200)
dx = Measure("dx", mesh)

V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)

a = u.dx(1) * v * dx + u.dx(0) * v.dx(0) * dx
L = Constant(0) * v * dx

left = CompiledSubDomain("on_boundary && near(x[0], 0, tol)", tol=DOLFIN_EPS)
right = CompiledSubDomain("on_boundary && near(x[0], 1, tol)", tol=DOLFIN_EPS)
initial = CompiledSubDomain("on_boundary && near(x[1], 0, tol)", tol=DOLFIN_EPS)

bc_left = DirichletBC(V, Constant(0), left)
bc_right = DirichletBC(V, Constant(0), right)
expr = Expression("exp(-10*pow(x[0] - 0.5, 2))", degree=2)
bc_initial = DirichletBC(V, expr, initial)
bcs = [bc_left, bc_right, bc_initial]

u = Function(V)
solve(a == L, u, bcs)
