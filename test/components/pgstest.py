from firedrake import *
from relaxation import *
from time import time

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

def build_levels(coarse_mx, coarse_my, numlevels):

  coarse = CircleMesh(Point(0.0, 0.0), 1.0, 0.5)
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



print('building multigrid hierarchy...')
tstart = time()
levels = build_levels(coarse_mx, coarse_my, numlevels)
print('time to build hierarchy:', time() - tstart)

rstar  = .6979651482
rstar2 = rstar**2
v    = TestFunction(levels[0].fspace)
psi = Function(levels[0].fspace)
x, y = SpatialCoordinate(mesh)
psi.assign(sqrt(Max(1 - x**2 - y**2, 0.0)))

ai, aj, av = levels[0].A.M.handle.getValuesCSR()
u = Function(levels[0].fspace)

b = Constant(0.0).vector().array()
N = len(u.vector().array())
for i in range(1000):
  uold = u.copy()
  projected_gauss_seidel(ai, aj, av, np.arange(0,N), np.ones(N), np.ones(N), u.vector().dat[:], b, psi.vector().array())
uexact = Function(levels[0].fspace)
uexact.interpolate(-rstar2*log(sqrt(x**2 + y**2)/2.)**2 / np.sqrt(1. - rstar2))
uexact = uexact.vector().array()

print(np.linalg.norm(uexact - u, np.inf))
ufunc = Function(levels[0].fspace)
ufunc.vector()[:] = u
plot(ufunc)
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
