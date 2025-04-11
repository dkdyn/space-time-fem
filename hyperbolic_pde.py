from fenics import *
import numpy as np
import matplotlib.pyplot as plt

# space time [x,t]
# note that fenics imposes BCs on all boundaries
# but for initial value problems there would be both BCs, Dirichlet and Neumann, 
# at start while there are no BCs at the end
#
# p-q: 
#    u.dx(0) is \partial u /\partial x and u.dx(1) is \partial u / \partial y.
#    disable end-boundary with zero-making Robin-BC or gradient-dot-normal=0

# Create mesh and define function space
x0 = 0.0
t0 = 0.0
x1 = 1.0
t1 = 2.25
nx = 10
nt = 20   

dt = (t1-t0)/nt

mesh = RectangleMesh(Point(x0, t0), Point(x1, t1), nx, nt)  
V = FunctionSpace(mesh, 'P', 2)

# Define boundary condition
#u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
initial_position = Expression('0.2*sin(1*pi*x[0]) + 0.0*sin(2*pi*x[0])', degree=3)   # no dirichlet BC means neumann BC
final_position = Expression('0.2*sin(1*pi*x[0]) + 0.0*sin(2*pi*x[0])', degree=3)   # no dirichlet BC means neumann BC

zero = Constant(0)
sF = Constant(np.array([[-1, 0], [0, 1]]))   # symplectic matrix

#def boundary(x, on_boundary):
#    return on_boundary
tol = 1e-14

def left(x):
    return near(x[0], x0, tol) 

def right(x):
    return near(x[0], x1, tol) 

def start(x):
    return near(x[1], t0, tol)  

def end(x):
    return near(x[1], t1, tol)  

ic = DirichletBC(V, initial_position, start)
fc = DirichletBC(V, final_position, end)
bc_left = DirichletBC(V, zero, left)
bc_right = DirichletBC(V, zero, right)
bc = [bc_left, bc_right, ic, fc]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0.0)
a = dot(sF*grad(u), grad(v))*dx
L = f*v*dx
# Compute solution
u = Function(V)
solve(a == L, u, bc)
# Plot solution and mesh
#plt.figure()
plot(u)
#plot(mesh)

plt.figure()
x_grid = np.linspace(x0+tol, x1-tol)
t_samples = np.linspace(t0, t1, 11)
for tx in t_samples:
    spaceline = [ (xx, tx) for xx in x_grid ]
    ux = np.array([u(point) for point in spaceline])
    plt.plot(x_grid, ux)
    plt.xlabel("$x$")
    plt.ylabel("$u(x)$")
plt.legend(t_samples)

plt.figure()
t_grid = np.linspace(t0+tol, t1-tol)
dx = 0.25*(x1-x0)
x_samples = np.linspace(x0+dx, x1-dx, 3)
for xt in x_samples:
    timeline = [ (xt, tt) for tt in t_grid ]
    ut = np.array([u(point) for point in timeline])
    plt.plot(t_grid, ut)
    plt.xlabel("$t$")
    plt.ylabel("$u(t)$")
plt.legend(x_samples)    


# Save solution to file in VTK format
vtkfile = File('hyperbolic_pde_solution.pvd')
vtkfile << u
