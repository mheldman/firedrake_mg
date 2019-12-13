from firedrake import *
from firedrakegmg import *
from gridtransfers import mrestrict
from time import time
import matplotlib.pyplot as plt

numlevels            = 6
preiters             = 2
postiters            = 2
maxiters             = 50
cycle                = 'V'
rtol                 = 1e-8
atol                 = 1e-15
coarsemx             = 2
coarsemy             = 2
quad                 = True
mg_type              = 'gmg'

levels = []
np.set_printoptions(precision=16)
def build_levels(numlevels):

  coarse = RectangleMesh(coarsemx, coarsemy, 1, 1, quadrilateral=quad)
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
    
    u, v = TrialFunction(V), TestFunction(V)
    a    = inner(grad(u), grad(v))*dx + 8.0*v*dx
    A    = None
    b    = u*v*dx# + k*inner(grad(u), grad(v))*dx
    B    = None
    m    = u*v*dx
    M    = assemble(m, bcs=bcs)
    lvl = licplevel(mesh, V, a, b, A, B, bcs, u, findices=None, M=M)
    if z > 0:
      levels[z - 1].findices = uc.vector()[:]
      levels[z - 1].findices = np.rint(levels[z - 1].findices)
      levels[z - 1].findices = levels[z - 1].findices.astype('int32')
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
          value[:] = -min(min(x[0], x[1]), min(1.0 - x[0], 1.0 - x[1]))

psi = obst()
g.interpolate(psi)
#plot(c, plot3d=True)
#plt.savefig('test results/lcp/spiral/spiralpsi')
u = Function(levels[0].fspace)
#u.interpolate(sin(pi*x)*sin(pi*y))
tstart = time()
if mg_type == 'gmg':
  gmgsolver = nlobstacle_gmg_solver(levels, preiters, postiters)
  gmgsolver.solve(u, g, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters)
else:
  gmgsolver = nlobstacle_pfas_solver(levels, preiters, postiters)
  gmgsolver.solve(u, f, g, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters)
print('time for gmg solve:', time() - tstart)
'''
bi, bj, bv = np.arange(ai.shape[0],dtype='int32'), np.arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
for i in range(1000):
  projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array())
  ra.assign(b - assemble(action(levels[0].a, u), bcs=levels[0].bcs))
  rb.assign(c - u)
  print(np.linalg.norm(np.maximum(ra.vector().array(), rb.vector().array()),np.inf))
'''

plot(u, plot3d=True)
plt.savefig('test results/lcp/elasto/' + mg_type + 'elastou')
