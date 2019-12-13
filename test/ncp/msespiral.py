from firedrake import *
from firedrakegmg import *
from gridtransfers import mrestrict
from time import time
import matplotlib.pyplot as plt
from numpy import arctan2

#coarse_mx, coarse_my = 2, 2
numlevels            = 4
preiters             = 2
postiters            = 2
maxiters             = 20
cycle                = 'W'
rtol                 = 1e-8
atol                 = 1e-15
k   = Constant(0.0)
coarsemx = 2
coarsemy = 2
quad = True
mg_type = 'gmg'

levels = []

def build_levels(numlevels):

  coarse = RectangleMesh(coarsemx, coarsemy, 2, 2, quadrilateral=quad)
  mh = MeshHierarchy(coarse, numlevels - 1)
  z = 0
  for mesh in mh:
    V  = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    bcs = DirichletBC(V, 0.0, 'on_boundary')
    uf  = Function(V)
    uf.vector()[:] = np.arange(len(uf.vector()[:]))
    u, v = Function(V), TestFunction(V)
    bndry = np.rint(assemble(Constant(0.0)*v*dx, bcs=DirichletBC(V, 1.0, 'on_boundary')).vector().array()).astype('bool')
    bvals = np.zeros(len(uf.vector()[:]))[bndry]
    a    = inner(grad(u), grad(v))/sqrt(1. + inner(grad(u),grad(u)))*dx
    A    = None
    H    = TrialFunction(V)
    b    = H*v*dx# + k*inner(grad(u), grad(v))*dx
    B    = None
    m    = H*v*dx
    M    = assemble(m, bcs=bcs)
    lvl = licplevel(mesh, V, a, b, A, B, bcs, u, findices=None, M=M, bindices=bndry, bvals=bvals)
    if z > 0:
      inject(uf, uc)
      levels[z - 1].findices = uc.vector()[:]
      levels[z - 1].findices = np.rint(levels[z - 1].findices)
      levels[z - 1].findices = levels[z-1].findices.astype('int32')
    levels.append(lvl)
    uc    = Function(V)
    z += 1

  levels.reverse()
  return levels


print('building multigrid hierarchy...')
tstart = time()
levels = build_levels(numlevels)
print('time to build hierarchy:', time() - tstart)
v = TestFunction(levels[0].fspace)
x, y = SpatialCoordinate(levels[0].mesh)
b = assemble(Constant(0.0)*v*dx, bcs=levels[0].bcs)

g = Function(levels[0].fspace)
f = Function(levels[0].fspace)
class obst(Expression):
    def eval(self, value, x):
        r = (x[0] - 1.)**2 + (x[1] - 1.)**2
        if r > 1e-14:
            r = sqrt(r)
            value[:] = sin(2.*pi/r + pi/2. - arctan2(x[0] - 1., x[1] - 1.)) + r*(r + 1.)/(r - 2.) - 3.*r + 3.6
        else:
          value[:] = 3.6

psi = obst()
g.interpolate(psi)
#plot(c, plot3d=True)
#plt.savefig('test results/lcp/spiral/spiralpsi')
u = assemble(Constant(0.0)*v*dx, bcs=levels[0].bcs)
#u.interpolate(sin(pi*x)*sin(pi*y))
tstart = time()
if mg_type == 'gmg':
  gmgsolver = nlobstacle_gmg_solver(levels, preiters, postiters)
  gmgsolver.solve(u, g, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters)
else:
  gmgsolver = nlobstacle_pfas_solver(levels, preiters, postiters)
  gmgsolver.solve(u, f, g, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters)
print('time for gmg solve:', time() - tstart)


plot(u, plot3d=True)
plt.savefig('test results/ncp/msespiral/' + mg_type + 'spiralu')
