from firedrake import *
from firedrakegmg import *
from time import time

coarse_mx, coarse_my = 2, 2
numlevels            = 8
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

  coarse = UnitSquareMesh(coarse_mx, coarse_my)
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
  return levels


print('building multigrid hierarchy...')
tstart = time()
levels = build_levels(coarse_mx, coarse_my, numlevels)
print('time to build hierarchy:', time() - tstart)
v    = TestFunction(levels[0].fspace)
x, y = SpatialCoordinate(levels[0].mesh)
L    = 8.0*pi**2*sin(2.0*pi*x)*sin(2.0*pi*y)*v*dx

#solve(levels[0].op == L, uf, bcs=bcs) #what does ksp_type richardson, pc_type preonly, do,
def resid_norm(u):
  return norm(assemble(L - inner(grad(u), grad(v))*dx, bcs=levels[0].bcs))

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
