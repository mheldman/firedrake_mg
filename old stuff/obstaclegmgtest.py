from firedrake import *
from firedrakegmg import *
from gridtransfers import mrestrict
from time import time
import matplotlib.pyplot as plt

coarse_mx, coarse_my = 3, 3
numlevels            = 6
preiters             = 1
postiters            = 1
maxiters             = 15
cycle                = 'V'
rtol                 = 1e-8

levels = []

rstar  = 0.697965148223374
rstar2 = rstar**2

def build_levels(coarse_mx, coarse_my, numlevels):

  coarse = RectangleMesh(coarse_mx, coarse_my, 4, 4)
  mh = MeshHierarchy(coarse, numlevels - 1)
  mx, my = coarse_mx, coarse_my
  z = 0
  for mesh in mh:
    V  = FunctionSpace(mesh, "CG", 1)
    x = SpatialCoordinate(mesh)
    if z == numlevels - 1:
      g = -rstar2 * ln( sqrt((x[0] - 2.)**2 + (x[1] - 2.)**2)/2.) / sqrt(1. - rstar2)
      bcs = DirichletBC(V, g, 'on_boundary')
    else:
      bcs = DirichletBC(V, 0.0, 'on_boundary')
    
    uf  = Function(V)
    uf.vector()[:] = np.arange(len(uf.vector()[:]))
    
    if z > 0:
      inject(uf, uc)
    
    u, v = TrialFunction(V), TestFunction(V)
    a    = inner(grad(u), grad(v))*dx
    A    = assemble(a, bcs=bcs)
    m    = u*v*dx
    M    = assemble(m, bcs=bcs)
    lvl = licplevel(mesh, V, a, m, A, M, bcs, findices=None, M=M)
    if z > 0:
      levels[z - 1].findices = uc.vector()[:].astype('int32')
    levels.append(lvl)
    uc    = Function(V)
    z += 1

  levels.reverse()
  levels[-1].presmooth = 'lu'
  return levels


print('building multigrid hierarchy...')
tstart = time()
levels = build_levels(coarse_mx, coarse_my, numlevels)
print('time to build hierarchy:', time() - tstart)
v = TestFunction(levels[0].fspace)
x, y = SpatialCoordinate(levels[0].mesh)
b = assemble(Constant(0.0)*v*dx, bcs=levels[0].bcs)

class obst(Expression):
    def eval(self, value, x):
        r2 = (x[0] - 2.)**2 + (x[1] - 2.)**2
        if r2 < 1.0:
            value[:] = sqrt(1. - r2)
        else:
          value[:] = -100.0

psi = obst()
#parabola = 1.0 - (x - 2.0)**2 - (y - 2.0)**2
v = TestFunction(levels[0].fspace)
#plot(assemble(parabola*v*dx), plot3d=True)
#plt.savefig('parabola.png')
#ball = Max(1.0 - (x - 2.0)**2 - (y - 2.0)**2, 0.0)
#plot(assemble(ball*v*dx), plot3d=True)
#plt.savefig('ball.png')
c = Function(levels[0].fspace)
c.interpolate(psi)
#g = Function(levels[0].fspace)
u = assemble(Constant(0.0)*v*dx, bcs=levels[0].bcs)
#plot(u, plot3d=True)
#plt.savefig('uinit.png')

class exact(Expression):
    def eval(self, value, x):
        if (x[0] - 2)**2 + (x[1] - 2)**2 > rstar2:
            value[:] = -rstar2 * ln( sqrt((x[0] - 2.)**2 + (x[1] - 2.)**2)/2.) / sqrt(1. - rstar2)
        else:
          value[:] = sqrt(1.0 - (x[0] - 2.0)**2 - (x[1] - 2.0)**2)

w = exact()
#u.interpolate(w)




#u.vector()[:] = np.random.rand(len(u.vector().array()))

#u *= g.interpolate(sin(x-4)*sin(y-4)*sin(x)*sin(y))
#u += g.interpolate(Max(Min(-rstar2 * ln(sqrt((x - 2) ** 2 + (y - 2) ** 2) / 2.) / sqrt(1. - rstar2), 10.0), -10.0))
#plot(u)
#plt.savefig('uinit.png', plot3d=True)
ai, aj, av = levels[0].A.M.handle.getValuesCSR()
bi, bj, bv = levels[0].B.M.handle.getValuesCSR()
#for i in range(10):
#  projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array())
gmgsolver = obstacle_gmg_solver(levels, preiters, postiters)
gmgsolver.solve(u, b, c, cycle='V', rtol=1e-12, maxiters=maxiters)

plot(u, plot3d=True)
plt.savefig('obstaclegmgsol')
plot(c, plot3d=True)
plt.savefig('obstacle')
uexact = Function(levels[0].fspace)
uexact.interpolate(w)
print(norm(u - uexact))
plot(u)
plt.savefig('ugmgsol.png')
u -= uexact
plot(uexact, plot3d=True)
plt.savefig('uexactgmgsol.png')
r = Function(levels[0].fspace)
r = assemble(Max(b - assemble(action(levels[0].a, u), bcs=levels[0].bcs), c - assemble(action(levels[0].b, u), bcs=levels[0].bcs))*v*dx)
plot(r)
plt.savefig('residplot.png')
