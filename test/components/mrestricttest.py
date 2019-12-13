from firedrake import *
from firedrakegmg import *
from gridtransfers import mrestrict
from time import time
import matplotlib.pyplot as plt

coarse_mx, coarse_my = 7, 7
numlevels            = 2
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

  coarse = UnitIntervalMesh(coarse_mx)
  mh = MeshHierarchy(coarse, numlevels - 1)
  mx, my = coarse_mx, coarse_my
  z = 0
  for mesh in mh:
    V  = FunctionSpace(mesh, "CG", 1)
    #bcs = DirichletBC(V, 0.0, 'on_boundary')
    uf  = Function(V)
    uf.vector()[:] = np.arange(len(uf.vector()[:]))
    
    if z > 0:
      inject(uf, uc)
    u, v = TrialFunction(V), TestFunction(V)
    a    = inner(grad(u), grad(v))*dx
    A    = assemble(a)
    m    = u*v*dx
    M    = assemble(m)
    
    lvl = licplevel(presmooth, postsmooth, mesh, V, a, m, A, M, bcs, findices=None, M=M)
    if z > 0:
      levels[z - 1].findices = uc.vector()[:].astype('int32')
      print(levels[z - 1].findices)
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

psi = Function(levels[0].fspace)
psi.vector()[:] = np.random.rand(len(psi.vector().array()))
psic = Function(levels[1].fspace)
mi, mj, mv = levels[0].M.M.handle.getValuesCSR()
mrestrict(mi, mj, mv, levels[1].findices, psi.dat.data, psic.dat.data)
plot(psi)
plt.savefig('finepsi')
plot(psic)
plt.savefig('coarsepsi')
