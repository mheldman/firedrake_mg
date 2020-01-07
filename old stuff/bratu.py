from firedrake import *
from firedrakegmg import *
from gridtransfers import mrestrict
from time import time
import matplotlib.pyplot as plt

'''
Tests multigrid solvers for the obstacle Bratu problem, a nonlinear complementarity problem:

                    0 <= F(u),    psi <= u,    (F(u), u) = 0,
                    
with

                       F(u) = -u_xx - u_yy + e^(psi - u).
                       
Default test case is a square domain

                          Omega = [x1, x2] x [y1, y2]

with x1 = y1 = 0, x2 = y2 = 1, psi = C is a constant function with C < 0, and homogeneous Dirichlet boundary conditions. The solution u is forced downward by the exponential nonlinearity, but comes in contact with the obstacle psi.
                          
                    
'''


#coarse_mx, coarse_my = 2, 2
numlevels            = 7
preiters             = 1
postiters            = 1
maxiters             = 50
cycle                = 'W'
rtol                 = 1e-10
atol                 = 1e-15
coarse_mx            = 2
coarse_my            = 2
mgtype               = 'pfas'
innits               = 1
L                    = 1
psi                  = Constant(-.1)
k                    = Constant(2.5)
quad                 = True

levels = []

def build_levels(numlevels):

  coarse = RectangleMesh(coarse_mx, coarse_my, L, L, quadrilateral=quad)
  mh = MeshHierarchy(coarse, numlevels - 1)
  z = 0
  for mesh in mh:
    V  = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    bcs = DirichletBC(V, 0.0, 'on_boundary')
    
    uf  = Function(V)
    uf.vector()[:] = np.arange(len(uf.vector()[:]))
    
    if z > 0:
      inject(uf, uc)
    
    H, v = Function(V), TestFunction(V)
    a    = inner(grad(H), grad(v))*dx + k*exp(psi-H)*v*dx
    A    = None #Jacobian matrix to be assembled in the solver
    b    = H*v*dx
    B    = None
    u    = TrialFunction(V)
    m    = u*v*dx
    M    = assemble(m, bcs=bcs)
    lvl = licplevel(mesh, V, a, b, A, B, bcs, H, findices=None, M=M)
    if z > 0:
      levels[z - 1].findices = uc.vector()[:]
      levels[z - 1].findices = np.rint(levels[z - 1].findices)
      levels[z - 1].findices  = levels[z - 1].findices.astype('int32')
    levels.append(lvl)
    uc    = Function(V)
    z += 1

  levels.reverse()
  return levels

z = 0
levels = []
print('building multigrid hierarchy...')
tstart = time()
levels = build_levels(numlevels)
print('time to build hierarchy:', time() - tstart)
V = levels[0].fspace
bcs = levels[0].bcs
mesh = levels[0].mesh

x, y = SpatialCoordinate(mesh)
v = TestFunction(V)
g = Function(V)
g.interpolate(psi)
u = Function(V)
u.interpolate(x**2*(1 - y**2)*(1-x**2)*y**2)
f = Function(V)
f.interpolate(Constant(0.0))

if mgtype == 'gmg':
  mgsolver = nlobstacle_gmg_solver(levels, preiters, postiters)
else:
  mgsolver = nlobstacle_pfas_solver(levels, preiters, postiters)

tstart = time()
if mgtype == 'gmg':
  mgsolver.solve(u, g, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters, innerits=innits)
  mgsolver = nlobstacle_pfas_solver(levels, preiters, postiters)
  mgsolver.solve(u, f, g, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters)
else:
  mgsolver.solve(u, f, g, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters)
print('time for gmg solve:', time() - tstart)

plot(u, plot3d=True)
plt.savefig('test results/ncp/bratu/' +mgtype + 'bratusol.png')

a = replace(levels[0].a, {levels[0].H : u})
b = replace(levels[0].b, {levels[0].H : u})
ra, rb = Function(V), Function(V)
if mgtype == 'gmg':
  ra.assign(assemble(a, bcs=levels[0].bcs))
elif mgtype == 'pfas':
  ra.assign(assemble(a, bcs=levels[0].bcs))
rb.assign(u + psi)
r = Function(levels[0].fspace)
r.interpolate(Min(ra, rb))

plot(r)
plt.savefig('test results/ncp/bratu/' + mgtype + 'braturesidual.png')

plot(ra)
plt.savefig('test results/ncp/bratu/' +mgtype + 'bratuFresidual.png')

plot(rb)
plt.savefig('test results/ncp/bratu/' +mgtype + 'bratusolresidual.png')



  



