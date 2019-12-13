from firedrake import *
from firedrakegmg import *
from gridtransfers import mrestrict
from time import time
import matplotlib.pyplot as plt
from relaxation import projected_gauss_seidel
from firedrake.functionspacedata import get_boundary_nodes

#coarse_mx, coarse_my = 2, 2
numlevels            = 7
preiters             = 1
postiters            = 1
maxiters             = 10
cycle                = 'V'
rtol                 = 1e-10
atol                 = 1e-15
coarsemx = 2
coarsemy = 2
k   = Constant(0.0)
mg_type = 'gmg'
quad = True

levels = []

rstar  = 0.697965148223374
rstar2 = rstar**2



def build_levels(numlevels):

  coarse = RectangleMesh(coarsemx, coarsemy, 4, 4, quadrilateral=quad)
  mh = MeshHierarchy(coarse, numlevels - 1)
  z = 0
  for mesh in mh:
    V  = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    
    if z == numlevels-1:
      g = Function(V)
      g.interpolate(-rstar2 * ln( sqrt((x - 2.)**2 + (y - 2.)**2)/2.)/ sqrt(1. - rstar2))
      bcs = DirichletBC(V, g, 'on_boundary')
    else:
      bcs = DirichletBC(V, 0.0, 'on_boundary')
    
    uf  = Function(V)
    uf.vector()[:] = np.arange(len(uf.vector()[:]))
    
    u, v = TrialFunction(V), TestFunction(V)
    bndry = np.rint(assemble(Constant(0.0)*v*dx, bcs=DirichletBC(V, 1.0, 'on_boundary')).vector().array()).astype('bool')
    if z == numlevels-1:
      bvals = g.vector().array()[bndry]
    else:
      bvals = np.zeros(len(uf.vector()[:]))[bndry]
    
    a    = inner(grad(u), grad(v))*dx
    A    = None
    b    = u*v*dx #+ k*inner(grad(u), grad(v))*dx
    B    = None
    m    = u*v*dx
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
plt.savefig('test results/lcp/radialsquare/finemesh')
plot(levels[-1].mesh)
plt.savefig('test results/lcp/radialsquare/coarsemesh')
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
          value[:] = -1.0
'''
g.interpolate(-rstar2 * ln( sqrt((x - 2.)**2 + (y - 2.)**2)/2.) / sqrt(1. - rstar2))
plot(g)
plt.savefig('test results/lcp/radialsquare/' + mg_type + 'bc')

g.assign(u - g)
plot(g)
'''
u = assemble(Constant(0.0)*v*dx)
plt.savefig('test results/lcp/radialsquare/' + mg_type + 'bcdiff')
plot(u, plot3d=True)
plt.savefig('test results/lcp/radialsquare/' + mg_type + 'uinit')


psi = obst()
g.interpolate(psi)

tstart = time()
if mg_type == 'gmg':
  gmgsolver = nlobstacle_gmg_solver(levels, preiters, postiters)
  gmgsolver.solve(u, g, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters)
else:
  f = assemble(f, bcs=levels[0].bcs)
  gmgsolver = nlobstacle_pfas_solver(levels, preiters, postiters)
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
  plt.savefig('test results/lcp/radialsquare/' + mg_type + 'radialu')

plot(u)
plt.savefig('test results/lcp/radialsquare/' + mg_type + 'radialucolor')

uexact = Function(levels[0].fspace)
class exact(Expression): #can't achieve high accuracy by refining the mesh using the coarse grid boundary approximation
    def eval(self, value, x):
        if (x[0] - 2.)**2 + (x[1] - 2.)**2 > rstar2:
            value[:] = -rstar2 * ln( sqrt((x[0] - 2.)**2 + (x[1] - 2.)**2)/2.) / sqrt(1. - rstar2)
        else:
          value[:] = sqrt(1.0 - (x[0] - 2.)**2 - (x[1] - 2.)**2)

w = exact()
uexact.interpolate(w)
e = Function(levels[0].fspace)

e.assign(u - uexact)
plot(e)
plt.savefig('test results/lcp/radialsquare/' + mg_type + 'error')
print('|u - uexact|_inf: ', np.linalg.norm(assemble(e*v*dx, bcs=DirichletBC(levels[0].fspace, 0.0, 'on_boundary')).vector().array(), np.inf))
print('|u - uexact|_2: ', np.linalg.norm(assemble(e*v*dx, bcs=DirichletBC(levels[0].fspace, 0.0, 'on_boundary')).vector().array()))


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
