from firedrake import *
from firedrakegmg import *
from gridtransfers import mrestrict
from time import time
import matplotlib.pyplot as plt
from relaxation import projected_gauss_seidel
from firedrake.functionspacedata import get_boundary_nodes

#coarse_mx, coarse_my = 2, 2
numlevels            = 6
preiters             = 2
postiters            = 2
maxiters             = 30
cycle                = 'FV'
rtol                 = 1e-8
atol                 = 1e-15
coarsemx = 2
coarsemy = 2
k   = Constant(0.0)
mg_type = 'pfas'
fmg     = True
quad = True
eps = 0.0
innits = 1
levels = []

rstar  = 0.697965148223374
rstar2 = rstar**2
eps = 1e-8
p = 4
q = 2



def build_levels(numlevels):

  coarse = RectangleMesh(coarsemx, coarsemy, 4, 4, quadrilateral=quad)
  mh = MeshHierarchy(coarse, numlevels - 1)
  z = 0
  for mesh in mh:
    V  = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    
    g = Function(V)
    g.interpolate(-rstar2 * ln( sqrt((x - 2.)**2 + (y - 2.)**2)/2.) / sqrt(1. - rstar2))
    bcs = DirichletBC(V, g, 'on_boundary')
    
    uf  = Function(V)
    uf.vector()[:] = np.arange(len(uf.vector()[:]))
    
    v = TestFunction(V)
    bndry = np.rint(assemble(Constant(0.0)*v*dx, bcs=DirichletBC(V, 1.0, 'on_boundary')).vector().array()).astype('bool')
    bvals = g.vector().array()[bndry]
    u = Function(V)
    a    = (inner(grad(u), grad(u))**((p-2)/2))*u**q*inner(grad(u), grad(v))*dx
    A    = None
    H = TrialFunction(V)
    b    = H*v*dx #+ k*inner(grad(u), grad(v))*dx
    B    = None
    m    = H*v*dx
    M    = assemble(m)
    lvl = licplevel(mesh, V, a, b, A, B, bcs, u, findices=None, M=M, bindices=bndry, bvals=bvals)
    
    if z > 0:
      inject(uf, uc)
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

'''
plot(levels[0].mesh)
plt.savefig('test results/ncp/mseradialsquare/finemesh')
plot(levels[-1].mesh)
plt.savefig('test results/ncp/mseradialsquare/coarsemesh')
'''
print('time to build hierarchy:', time() - tstart)
v = TestFunction(levels[0].fspace)
x, y = SpatialCoordinate(levels[0].mesh)
g = Function(levels[0].fspace)
f = Function(levels[0].fspace)
bndry = Function(levels[0].fspace)
class obst(Expression):
    def eval(self, value, x):
        r2 = (x[0] - 2.)**2 + (x[1] - 2.)**2
        if r2 < 1.0:
            value[:] = sqrt(1. - r2)
        else:
          value[:] = -100.0
'''
g.interpolate(-rstar2 * ln( sqrt((x - 2.)**2 + (y - 2.)**2)/2.) / sqrt(1. - rstar2))
plot(g)
plt.savefig('test results/lcp/radialsquare/' + mg_type + 'bc')

g.assign(u - g)
plot(g)
'''
u = assemble(Constant(0.0)*v*dx)
'''
plt.savefig('test results/ncp/mseradialsquare/' + mg_type + 'bcdiff')
plot(u, plot3d=True)
plt.savefig('test results/ncp/mseradialsquare/' + mg_type + 'uinit')
'''
f = Constant(0.0)
psi = obst()
g.interpolate(psi)

tstart = time()
if mg_type == 'gmg':
  gmgsolver = nlobstacle_gmg_solver(levels, preiters, postiters)
  if fmg:
    u = gmgsolver.fmgsolve(psi, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters, innerits=1, j=2)
  else:
    u = gmgsolver.solve(u, g, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters, innerits=innits)
else:
  gmgsolver = nlobstacle_pfas_solver(levels, preiters, postiters)
  if fmg:
    u = gmgsolver.fmgsolve(f, psi, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters, j=2)
  else:
    f = assemble(f, bcs=levels[0].bcs)
    gmgsolver.solve(u, f, g, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters)


print('time for gmg solve:', time() - tstart)
'''
ai, aj, av = levels[0].A.M.handle.getValuesCSR()
bi, bj, bv = np.arange(ai.shape[0],dtype='int32'), np.arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
b = assemble(Constant(0.0)*v*dx, bcs=levels[0].bcs)
ra, rb = Function(levels[0].fspace), Function(levels[0].fspace)
r = Function(levels[0].fspace)
levels[0].a = replace(levels[0].a, { levels[0].H : u })
ra = assemble(levels[0].a)
rb.assign(u - g)

for i in range(0):
  projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), g.vector().array())
  levels[0].a = replace(levels[0].a, { levels[0].H : u })
  ra = assemble(levels[0].a)
  rb.assign(u - g)
w = np.minimum(ra.vector().array(), rb.vector().array())
w[levels[0].bindices] = 0.0
r.vector()[:] = w
plot(r)
plt.savefig('test results/lcp/radialsquare/' + mg_type + 'residual')
plot(ra)
plt.savefig('test results/lcp/radialsquare/' + mg_type + 'residuala')
plot(rb)
plt.savefig('test results/lcp/radialsquare/' + mg_type + 'residualb')
'''
w = assemble(Constant(0.0)*v*dx, bcs=levels[0].bcs).vector().array()
if numlevels < 8:
  
  #u.vector()[levels[0].bindices] = w[levels[0].bindices]
  plot(u, plot3d=True)
  plt.savefig('test results/ncp/pradialsquare/' + mg_type + 'radialu')

plot(u)
plt.savefig('test results/ncp/pradialsquare/' + mg_type + 'radialucolor')

'''
collects boundary nodes? save for later..
def topological_boundary_nodes(V, sub_domain):
    section = V.dm.getDefaultSection()
    dm = V.mesh()._plex
    ids = dm.getLabelIdIS("Face Sets").indices
    if sub_domain == "on_boundary":
        indices = ids
    else:
        assert not set(sub_domain).difference(ids)
        indices = sub_domain
    maxsize = 0
    for j in indices:
        for i in dm.getStratumIS("Face Sets", j).indices:
            if dm.getLabelValue("exterior_facets", i) == -1:
                continue
            maxsize += section.getDof(i)
    nodes = np.full(maxsize, -1, dtype=int)
    idx = 0
    for j in indices:
        for i in dm.getStratumIS("Face Sets", j).indices:
            if dm.getLabelValue("exterior_facets", i) == -1:
                continue
            off = section.getOffset(i)
            for k in range(section.getDof(i)):
                nodes[idx] = off + k
                idx += 1
    return np.unique(nodes[:idx])
'''
