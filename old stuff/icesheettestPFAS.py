from firedrake import *
from firedrakegmg import *
from gridtransfers import mrestrict
from time import time
import matplotlib.pyplot as plt

#coarse_mx, coarse_my = 2, 2
numlevels            = 6
preiters             = 2
postiters            = 2
maxiters             = 50
cycle                = 'V'
rtol                 = 1e-12
atol                 = 1e-15
coarse_mx            = 2
coarse_my            = 2
a1                   = .01
a2                   = .02
n                    = 3
R                    = a2/(a1 + a2)
mgtype               = 'gmg'
innits               = 1
L = 2
g = 9.81
rho = 910.0
secpera = 31556926.
A = 1.0e-16 / secpera
Gamma =  2. * A * (rho * g)**3. / 5.
D0 = 1.0
eps = 0.0 #eps = 0.0 solves uniformly elliptic p-Laplace obstacle problem, eps = 1.0 solves the SIA approximation to a steady sheet ice sheet model, a generalization of the p-Laplace problem in which the Jacobian of the nonlinear PDE governing the flow on the ice covered region degenerates near the free boundary
r = 1e-4 #regularization for p-Laplace term
t = .01
#Gamma = 0.0
psie = Constant(.0) #obstacle height


levels = []

def build_levels(numlevels):

  coarse = RectangleMesh(coarse_mx, coarse_my, 2*L, 2*L)
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
    
    H, v = Function(V), TestFunction(V)
    a    = inner((Gamma*H**5*inner(grad(H), grad(H)) + r**2)*grad(H), grad(v))*dx
    #a    = inner(grad(H), grad(v))*dx
    A    = None #Jacobian matrix to be assembled in the solver
    b    = H*v*dx #+ 10.*H*inner(grad(H), grad(v))*dx
    B    = None
    u    = TrialFunction(V)
    m    = u*v*dx
    M    = assemble(m, bcs=bcs)
    lvl = licplevel(mesh, V, a, b, A, B, bcs, H, findices=None, M=M)
    if z > 0:
      levels[z - 1].findices = uc.vector()[:].astype('int32')
    levels.append(lvl)
    uc    = Function(V)
    z += 1

  levels.reverse()
  return levels

z = 0
levels = []
print('building multigrid hierarchy...')
tstart = time()
levels = build_levels(numlevels)
print('time to build hierarchy:', time() - tstart)
V = levels[0].fspace
bcs = levels[0].bcs
mesh = levels[0].mesh

class aaf(Expression):
      def eval(self, value, x):
          if sqrt((x[0] - L)**2 + (x[1] - L)**2) < R:
              value[:] = a1
          else:
            value[:] = -a2

if mgtype == 'gmg':
  for level in levels:
    m, v = Function(level.fspace), TestFunction(level.fspace)
    m.interpolate(aaf())
    f = m*v*dx
    level.a = level.a - f
  m, v = Function(V), TestFunction(V)
  m.interpolate(aaf())
  f = Function(V)
  f = assemble(m*v*dx, bcs=bcs)
else:
  m, v = Function(V), TestFunction(V)
  m.interpolate(aaf())
  f = assemble(m*v*dx, bcs=bcs)

g = Function(V)
x, y = SpatialCoordinate(levels[0].mesh)
g.interpolate(psie*sin(pi*x/(2.*L))*sin(pi*y/(2.*L)))
#g.interpolate(Constant(0.0))


if Gamma == 12.0:
    W = (40.*a1*R)**(1/8)
    class exact(Expression):
      def eval(self, value, x):
          r = sqrt((x[0] - L)**2 + (x[1] - L)**2)
          if r < R:
              value[:] = W*(1. - (1. + a1/a2)**(1/3)*(r/L)**(4/3))**(3/8)
          elif R <= r <= 1.:
            value[:] = W*(1. + a2/a1)**(1/8)*(1. - r/L)**(1/2)
          else:
            value[:] = 0.0
    u = Function(V)
    u.interpolate(exact())
else:
  u = Function(V)
  u.interpolate((y/(2*L))**4*(1. - (x/(2.*L))**4.)*(1. - (y/(2.*L))**4)*(x/(2*L))**4)


if mgtype == 'gmg':
  mgsolver = nlobstacle_gmg_solver(levels, preiters, postiters)
else:
  mgsolver = nlobstacle_pfas_solver(levels, preiters, postiters)

while eps <= 1.0:

  tstart = time()
  if mgtype == 'gmg':
    mgsolver.solve(u, g, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters, innerits=innits)
  else:
    mgsolver.solve(u, f, g, cycle=cycle, rtol=rtol, atol=atol, maxiters=maxiters)
  print('time for gmg solve:', time() - tstart)

  print('epsilon = ', eps, '\n')
  print('psie = ', psie, '\n')
  plot(u, plot3d=True)
  plt.savefig(mgtype + 'icesol.png')

  if Gamma == 1.0:
    W = (40.*a1*R)**(1/8)
    print(W)
    class exact(Expression):
      def eval(self, value, x):
          r = sqrt((x[0] - L)**2 + (x[1] - L)**2)
          if r < R:
              value[:] = W*(1. - (1. + a1/a2)**(1/3)*(r/L)**(4/3))**(3/8)
          elif R <= r <= 1.:
            value[:] = W*(1. + a2/a1)**(1/8)*(1. - r/L)**(1/2)
          else:
            value[:] = 0.0
    uexact = Function(V)
    uexact.interpolate(exact())
    print('L2 error: ', norm(uexact - u))
    plot(uexact, plot3d=True)
    plt.savefig('iceexactsol.png')
  if eps == 1.0:
    break
  eps = min(eps + t, 1.0)
  z += 1

  for level in mgsolver.levels:
      H, v = level.H, TestFunction(level.fspace)
      if mgtype == 'gmg':
        m, v = Function(level.fspace), TestFunction(level.fspace)
        m.interpolate(aaf())
        level.a = inner( (eps*H**(n+2) + (1. - eps)*D0)*Gamma*grad(H)*(inner(grad(H), grad(H)) + r**2), grad(v))/(n+2)*dx - m*v*dx
      else:
        level.a = inner( (eps*H**(n+2) + (1. - eps)*D0)*Gamma*grad(H)*(inner(grad(H), grad(H)) + r**2), grad(v))/(n+2)*dx


  a = replace(levels[0].a, {levels[0].H : u})
  b = replace(levels[0].b, {levels[0].H : u})
  ra, rb = Function(V), Function(V)
  if mgtype == 'gmg':
    ra.assign(assemble(a, bcs=levels[0].bcs))
  elif mgtype == 'pfas':
    ra.assign(assemble(a, bcs=levels[0].bcs) - f)
  rb.assign(u)
  r = Function(levels[0].fspace)
  r.interpolate(Min(ra, rb))
  plot(r)
  plt.savefig(mgtype + 'residual.png')


  



