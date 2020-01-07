from firedrake import *
from firedrakegmg import *
from gridtransfers import mrestrict
from relaxation import projected_gauss_seidel
from time import time
import matplotlib.pyplot as plt

#coarse_mx, coarse_my = 2, 2
numlevels            = 8
preiters             = 2
postiters            = 2
maxiters             = 26
cycle                = 'W'
rtol                 = 1e-50
atol                 = 1e-15
coarsemx            = 4
coarsemy            = 4
a1                   = .01
a2                   = .03
n                    = 3
R                    = a2/(a1 + a2)
L = 4
mgtype               = 'pfas'
innits               = 1
quad = True
fmg = True
fmgj = 2
eps=1e-4
delt = 1e-4
'''
L = 2
g = 9.81
rho = 910.
secpera = 31556926.
A = 1.0e-16 / secpera
Gamma =  2. * A * (rho * g)**3. / 5.
D0 = 1.0
eps = .0 #eps = 0.0 solves uniformly elliptic p-Laplace obstacle problem, eps = 1.0 solves the SIA approximation to a steady sheet ice sheet model, a generalization of the p-Laplace problem in which the Jacobian of the nonlinear PDE governing the flow on the ice covered region degenerates near the free boundary
rpl = 2.0 #regularization for p-Laplace term
rpm = 20.0
t = .01
domeL   = 2.
domeR   = 1.
domeH0  = 100.
domeCx  = 2.
domeCy  = 2.
n  = 3.0
pp = 1.0 / n
CC = Gamma * domeH0**(2.0*n+2.0) / (2.0 * domeR * (1.0-1.0/n)) ** n
'''
#Gamma = 1.0
psih = 0.0 #obstacle height

'''
class cmb(Expression):
  def eval(self, value, x):
      xc = x[0] - domeCx
      yc = x[1] - domeCy
      r = sqrt(xc**2 + yc**2)
      r = max(r, .01)
      r = min(r, domeR - .01)
      s = r / domeR
      tmp1 = s**pp + (1.0 - s)**pp - 1.0
      tmp2 = 2.0 * s**pp + (1.0 - 2.0*s)*(1.0 - s)**(pp - 1.0) - 1.0
      value[:] = (CC / r) * tmp1**(n-1.0) * tmp2
'''

def build_levels(numlevels):
  levels = []
  coarse = RectangleMesh(coarsemx, coarsemy, L, L, quadrilateral=quad)
  mh = MeshHierarchy(coarse, numlevels - 1)
  z = 0
  for mesh in mh:
    V  = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    
    g = Function(V)
    bcs = DirichletBC(V, 0.0, 'on_boundary')
    
    uf  = Function(V)
    uf.vector()[:] = np.arange(len(uf.vector()[:]))
    
    u, v = TrialFunction(V), TestFunction(V)
    bndry = np.rint(assemble(Constant(0.0)*v*dx, bcs=DirichletBC(V, 1.0, 'on_boundary')).vector().array()).astype('bool')
    bvals = g.vector().array()[bndry]
    u = Function(V)
    a    = .2*(3./8.)**3*inner(grad(u), grad(u))*inner(grad(u), grad(v))*dx
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

print('time to build hierarchy:', time() - tstart)
v = TestFunction(levels[0].fspace)
x, y = SpatialCoordinate(levels[0].mesh)
g = Function(levels[0].fspace)
f = Function(levels[0].fspace)

psi = Constant(0.0)

class psi(Expression):
  def eval(self, value, x):
    value[:] = psih*sin(pi*x[0]/4.)*sin(pi*x[1]/4.)
class cmb(Expression):
  def eval(self, value, x):
      r = sqrt((x[0] - L/2.)**2 + (x[1] - L/2.)**2)
      if r <= R:
        value[:] = a1
      else:
        value[:] = -a2
psi = psi()
f = cmb()
g.interpolate(psi)
b = Function(levels[-fmgj].fspace)
b.interpolate(f)

H = (40.*a1*R)**(1./8.)
class exact(Expression):
  def eval(self, value, x):
      r = sqrt((x[0] - L/2.)**2 + (x[1] - L/2.)**2)
      if r < R:
          value[:] = H*(1. - (1. + a1/a2)**(1./3.)*(r)**(4./3.))**(3./8.)
      elif R <= r <= 1.:
        value[:] = H*(1. + a2/a1)**(1./8.)*(1. - r)**(1./2.)
      else:
        value[:] = 0.0
        
uexact = Function(levels[0].fspace)
uexact.interpolate(exact())
#w = Function(levels[0].fspace)
#w.interpolate(f)
#F = assemble(.2*uexact**5*inner(grad(uexact), grad(uexact))*inner(grad(uexact), grad(v))*dx - w*v*dx)
#plot(F)
plt.savefig('test results/ncp/icedome/Fexact')
'''
if numlevels < 8:

  plot(uexact, plot3d=True, num_sample_points=1)
  plt.savefig('test results/ncp/icedome/uexact')
'''
if numlevels < 7:
  plot(b, plot3d=True)
  plt.savefig('test results/ncp/icedome/cmb')
#use as initial guess solution to some problem obtained by picard iteration

if fmg:
  class init(Expression):
    def eval(self, value, x):
        r = sqrt((x[0] - L/2.)**2 + (x[1] - L/2.)**2)
        if r <= 1.5:
          value[:] = .86
        else:
          value[:] = 0.0

  u = Function(levels[-fmgj].fspace)
  u.interpolate(init())
  '''
  H = TrialFunction(levels[-fmgj].fspace)
  v = TestFunction(levels[-fmgj].fspace)
  A = assemble(inner(grad(H), grad(v))*dx)
  ai, aj, av = A.M.handle.getValuesCSR()
  g = Function(levels[-fmgj].fspace)
  g.interpolate(psi)
  bi, bj, bv = np.arange(ai.shape[0],dtype='int32'), np.arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
  u = Function(levels[-fmgj].fspace)
  uold = Function(levels[-fmgj].fspace)
  uold.assign(u + 1.0)
  b = assemble(b*v*dx) #after applying the mass matrix this can be uniformly negative.. need to make the grid fine enough that the coarse solution is not the zero function
  while norm(u - uold) > 1e-14:
    uold = u.copy(deepcopy=True)
    projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), g.vector().array(),  np.arange(len(u.vector().array()), dtype='int32')[~levels[-fmgj].bindices])
    #A = assemble(inner(grad(H), grad(v))*dx) #inner(grad(u), grad(u))
    #ai, aj, av = A.M.handle.getValuesCSR()
'''
tstart = time()
if mgtype == 'gmg':
  for level in levels:
    v = TestFunction(level.fspace)
    m = Function(level.fspace)
    m.interpolate(f)
    level.a = level.a - m*v*dx
  gmgsolver = nlobstacle_gmg_solver(levels, preiters, postiters)
  if fmg:
    #u = Function(levels[-fmgj].fspace)
    #u.interpolate(exact())
    u = gmgsolver.fmgsolve(psi, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters, innerits=1, j=fmgj, u0=u)
  else:
    u = Function(levels[0].fspace)
    u.interpolate(f)
    u.assign(u)
    u = gmgsolver.solve(u, g, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters, innerits=innits)
else:
  gmgsolver = nlobstacle_pfas_solver(levels, preiters, postiters)
  if fmg:
    #u = Function(levels[-fmgj].fspace)
    #u.interpolate(exact())
    u = gmgsolver.fmgsolve(f, psi, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters, j=fmgj, u0=u)
  else:
    u = Function(levels[0].fspace)
    u.interpolate(f)
    f = assemble(u*v*dx, bcs=levels[0].bcs)
    gmgsolver.solve(u, f, g, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters)
u.assign(u**(3./8.))
print('time for gmg solve:', time() - tstart)
print('max val u:', np.max(u.vector()[:]))
if numlevels < 8: #change this to number of unknowns, or just fix the plotting so it doesn't take so long
  plot(u, plot3d=True, num_sample_points=1)
  plt.savefig('test results/ncp/icedome/' + mgtype + 'regu')
plot(u)
plt.savefig('test results/ncp/icedome/regu')




print('L2 error: ', norm(uexact - u))

  



