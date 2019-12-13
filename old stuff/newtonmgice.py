from firedrake import *
from firedrakegmg import *
from time import time
import matplotlib.pyplot as plt

coarse_mx, coarse_my = 2, 2
numlevels            = 10
presmooth            = 'sor'
postsmooth           = 'sor'
preiters             = 1
postiters            = 1
coarsesolve          = 'lu'
maxiters             = 100
cycle                = 'V'
rtol                 = 1e-8


levels = []
'''
def build_levels(coarse_mx, coarse_my, numlevels):

  mesh = Mesh('UnitCircle.msh')
  plot(mesh)
  plt.savefig('meshplot.png')
  mh = MeshHierarchy(coarse, numlevels - 1)
  mx, my = coarse_mx, coarse_my
  z = 0
  for mesh in mh:
    V  = FunctionSpace(mesh, "CG", 1)
    uf = Function(V)
    if z > 0:
      restrict(uf, uc) #precompute restriction operators
      prolong(uc, uf)
    bcs = DirichletBC(V, 0.0, 'on_boundary')
    
    u, v = TrialFunction(V), TestFunction(V)
    a    = inner(grad(u), grad(v))*dx
    A    = assemble(a, bcs=bcs)
    lvl = level(presmooth, postsmooth, mesh, V, a, A, bcs)
    levels.append(lvl)
    uc = Function(V)
    z = 1

  levels.reverse()
  levels[-1].presmooth = 'lu'
  return level
'''
mesh = Mesh('UnitCircle.msh')
V = FunctionSpace(mesh, "CG", 1)
bcs = DirichletBC(V, 0.0, [13])
plot(mesh)
plt.savefig('meshplot.png')
rstar  = 0.697965148223374
rstar2 = rstar**2
u, v = TrialFunction(V), TestFunction(V)
a    = inner(grad(u), grad(v))*dx
A    = assemble(a, bcs=bcs)
ai, aj, av = A.M.handle.getValuesCSR()
b = u*v*dx
B   = assemble(b, bcs=bcs)
bi, bj, bv = B.M.handle.getValuesCSR()
psi = Function(V)
x, y = SpatialCoordinate(mesh)
psi.interpolate(sqrt(Max(1.0 - x**2 - y**2, 0.0)))
psi = assemble(psi*v*dx, bcs=bcs)
plot(psi)
plt.savefig('obstacle.png')
u = Function(V)
u.interpolate(0.0*x)
plot(u)
plt.savefig('initguess.png')
N = len(u.vector().array())
f = assemble(Constant(0.0)*v*dx, bcs=bcs)


for i in range(10000):
  uold = u.copy()
  projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, f.vector().array(), psi.vector().array())
class exact(Expression):
    def eval(self, value, x):
        if x[0]**2 + x[1]**2 > rstar2:
            value[:] = -rstar2*ln(sqrt(x[0]**2 + x[1]**2)/2.) / sqrt(1. - rstar2)
        else:
          value[:] = sqrt(1.0 - (x[0]**2 + x[1]**2))

w = exact()

uexact = Function(V)
uexact.interpolate(w)
#uexact = uexact.vector().array()
#sol = Function(V)
#sol.vector()[:] = u
print(norm(u - uexact))
print(norm(f - Max(assemble(action(a, u), bcs=bcs), psi - assemble(action(b, u), bcs=bcs))))
#ufunc = Function(V)
#ufunc.vector()[:] = u
plot(u)
plt.savefig('smoothedu.png')
#u = Function(levels[0].fspace)
#u.dat[:] =

'''
gmg_solver = gmg_solver(levels, preiters, postiters)
print('solving varational problem...')
tstart = time()
u          = gmg_solver.solve(Function(levels[0].fspace), assemble(L, bcs=levels[0].bcs), resid_norm, maxiters=maxiters, cycle=cycle, rtol=rtol)
print('time to solution:', time() - tstart)
uexact = Function(levels[0].fspace)
uexact.interpolate(sin(2.0*pi*x)*sin(2.0*pi*y))
#plot(u)
#plt.savefig('solution.png')
#plot(b)
#plt.savefig('exactsolution.png')
print('solution error:', norm(u - uexact))
'''
