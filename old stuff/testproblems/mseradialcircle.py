from firedrake import *
from firedrakegmg import *
from gridtransfers import mrestrict
from time import time
import matplotlib.pyplot as plt

#coarse_mx, coarse_my = 2, 2
numlevels            = 2
preiters             = 2
postiters            = 2
maxiters             = 30
cycle                = 'W'
rtol                 = 1e-10
atol                 = 1e-15
k   = Constant(0.0)
mg_type = 'pfas'
innits = 3

levels = []

rstar  = 0.697965148223374
rstar2 = rstar**2

def build_levels(numlevels):

  coarse = Mesh('UnitCircle.msh')
  #coarse._radius = 2.0
  mh = MeshHierarchy(coarse, numlevels - 1)
  z = 0
  for mesh in mh:
    V  = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    bcs = DirichletBC(V, 0.0, [13])
    '''
    why isn't this working?
    if z == numlevels-1:
      #g =  -rstar2 * ln( sqrt(x**2 + y**2)/2.) / sqrt(1. - rstar2)
      g = Function(V)
      #g.interpolate(-rstar2 * ln( sqrt(x**2 + y**2)/2.) / sqrt(1. - rstar2))
      bcs = DirichletBC(V, g, [13])
    else:
      bcs = DirichletBC(V, 0.0, [13])
    '''
    uf  = Function(V)
    uf.vector()[:] = np.arange(len(uf.vector()[:]))
    
    if z > 0:
      inject(uf, uc)
    
    u, v = TrialFunction(V), TestFunction(V)

    
    a    = inner(grad(u), grad(v))/sqrt(1 + inner(grad(u), grad(u)))*dx
    A    = None
    b    = u*v*dx #+ k*inner(grad(u), grad(v))*dx
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

g = Function(levels[0].fspace)
f = Function(levels[0].fspace)
bndry = Function(levels[0].fspace)
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

tstart = time()
if mg_type == 'gmg':
  gmgsolver = nlobstacle_gmg_solver(levels, preiters, postiters)
  gmgsolver.solve(u, g, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters, innerits=innits)
else:
  f = assemble(f, bcs=levels[0].bcs)
  gmgsolver = nlobstacle_pfas_solver(levels, preiters, postiters)
  gmgsolver.solve(u, f, g, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters)



print('time for gmg solve:', time() - tstart)
if numlevels < 5:
  u.assign(u)
  plot(u, plot3d=True)
  plt.savefig('test results/ncp/mseradialcircle/' + mg_type + 'radialu')
uexact = Function(levels[0].fspace)
class exact(Expression): #can't achieve high accuracy by refining the mesh using the coarse grid boundary approximation
    def eval(self, value, x):
        if (x[0])**2 + (x[1])**2 > rstar2:
            value[:] = -rstar2 * ln( sqrt(x[0]**2 + x[1]**2)/2.) / sqrt(1. - rstar2)
        else:
          value[:] = sqrt(1.0 - x[0]**2 - x[1]**2)

w = exact()

uexact.interpolate(w)
print('|u - uexact|_2: ', norm(u - uexact))
