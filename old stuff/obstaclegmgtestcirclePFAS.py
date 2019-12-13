from firedrake import *
from firedrakegmg import *
from gridtransfers import mrestrict
from time import time
import matplotlib.pyplot as plt

#coarse_mx, coarse_my = 2, 2
numlevels            = 6
preiters             = 1
postiters            = 1
maxiters             = 1000
cycle                = 'V'
rtol                 = 1e-8
k   = Constant(0.0)

levels = []

rstar  = 0.697965148223374
rstar2 = rstar**2

def build_levels(numlevels):

  coarse = Mesh('UnitCircle.msh')
  mh = MeshHierarchy(coarse, numlevels - 1)
  z = 0
  for mesh in mh:
    V  = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    if z == numlevels-1:
      #g =  -rstar2 * ln( sqrt(x**2 + y**2)/2.) / sqrt(1. - rstar2)
      g = Function(V)
      #g.interpolate(-rstar2 * ln( sqrt(x**2 + y**2)/2.) / sqrt(1. - rstar2))
      bcs = DirichletBC(V, 0.0, [13])
    else:
      bcs = DirichletBC(V, 0.0, [13])
    uf  = Function(V)
    uf.vector()[:] = np.arange(len(uf.vector()[:]))
    
    if z > 0:
      inject(uf, uc)
    
    u, v = TrialFunction(V), TestFunction(V)
    a    = inner(grad(u), grad(v))*dx
    A    = assemble(a, bcs=bcs)
    b    = u*v*dx + k*inner(grad(u), grad(v))*dx
    B    = assemble(b, bcs=bcs)
    m    = u*v*dx
    M    = assemble(m, bcs=bcs)
    lvl = licplevel(mesh, V, a, b, A, B, bcs, H=u, findices=None, M=M)
    if z > 0:
      levels[z - 1].findices = uc.vector()[:].astype('int32')
    levels.append(lvl)
    uc = Function(V)
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
class obst(Expression):
    def eval(self, value, x):
        r2 = (x[0])**2 + (x[1])**2
        if r2 < 1.0:
            value[:] = sqrt(1. - r2)
        else:
          value[:] = -100.0

psi = obst()
g.interpolate(psi)
u = Function(levels[0].fspace)
u.interpolate(Max(g, 0.0)

g = assemble(g*v*dx)

f = Function(levels[0].fspace)

#plot(c)
#plt.savefig('obstacleplot.png')
#g = Function(levels[0].fspace)
#u = Function(levels[0].fspace)


#plot(levels[0].mesh)
#plt.savefig('finemesh.png')




#u.vector()[:] = np.random.rand(len(u.vector().array()))

#u *= g.interpolate(sin(x-4)*sin(y-4)*sin(x)*sin(y))
#u += g.interpolate(Max(Min(-rstar2 * ln(sqrt((x - 2) ** 2 + (y - 2) ** 2) / 2.) / sqrt(1. - rstar2), 10.0), -10.0))
#plot(u)
#plt.savefig('uinit.png', plot3d=True)
ai, aj, av = levels[0].A.M.handle.getValuesCSR()
bi, bj, bv = levels[0].B.M.handle.getValuesCSR()
ra, rb = Function(levels[0].fspace), Function(levels[0].fspace)

pfassolver = nlobstacle_pfas_solver(levels, preiters, postiters)
tstart = time()
pfassolver.solve(u, f, g, cycle=cycle, rtol=rtol, maxiters=maxiters)
print('time for gmg solve:', time() - tstart)

'''
#PURE PGS:
#bi, bj, bv = np.arange(ai.shape[0],dtype='int32'), np.arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
for i in range(1000):
  projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array())
  ra.assign(b - assemble(action(levels[0].a, u), bcs=levels[0].bcs))
  rb.assign(c - assemble(action(levels[0].b, u), bcs=levels[0].bcs))
  print(np.linalg.norm(np.maximum(ra.vector().array(), rb.vector().array()),np.inf))
'''

plot(u, plot3d=True)
plt.savefig('obstaclegmgsol')
#plot(c, plot3d=True)
#plt.savefig('obstacle')
uexact = Function(levels[0].fspace)

class exact(Expression):
    def eval(self, value, x):
        if (x[0])**2 + (x[1])**2 > rstar2:
            value[:] = -rstar2 * ln( sqrt(x[0]**2 + x[1]**2)/2.) / sqrt(1. - rstar2)
        else:
          value[:] = sqrt(1.0 - x[0]**2 - x[1]**2)

w = exact()

uexact.interpolate(w)
print(norm(u - uexact))
#plot(u)
#plt.savefig('ugmgsol.png')
u -= uexact
plot(u)
plt.savefig('uexactgmgsol.png')
r = Function(levels[0].fspace)
r = assemble(Max(b - assemble(action(levels[0].a, u), bcs=levels[0].bcs), c - assemble(action(levels[0].b, u), bcs=levels[0].bcs))*v*dx)
plot(r)
plt.savefig('residplot.png')
