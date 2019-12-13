from firedrake import *
from firedrakegmg import *
from gridtransfers import mrestrict
from time import time
import matplotlib.pyplot as plt

#coarse_mx, coarse_my = 2, 2
numlevels            = 2
preiters             = 1
postiters            = 1
maxiters             = 50
cycle                = 'V'
rtol                 = 1e-12
coarse_mx            = 2
coarse_my            = 2
a1                   = .01
a2                   = .03
n                    = 3
R                    = a2/(a1 + a2)

levels = []

def build_levels(numlevels):

  coarse = RectangleMesh(coarse_mx, coarse_my, 4, 4)
  mh = MeshHierarchy(coarse, numlevels - 1)
  z = 0
  for mesh in mh:
    V  = FunctionSpace(mesh, "CG", 2)
    x, y = SpatialCoordinate(mesh)
    bcs = DirichletBC(V, 0.0, 'on_boundary')
    uf  = Function(V)
    uf.vector()[:] = np.arange(len(uf.vector()[:]))
    
    if z > 0:
      inject(uf, uc)
      
    class aaf(Expression):
      def eval(self, value, x):
          if sqrt((x[0] - 2.0)**2 + (x[1] - 2.0)**2) < R:
              value[:] = a1
          else:
            value[:] = -a2
    m = Function(V)
    m.interpolate(aaf())
    H, v = Function(V), TestFunction(V)
    a    = inner(grad(H), grad(v))/5.*dx - m*v*dx
    #a    = inner(grad(H), grad(v))*dx
    A    = None #Jacobian matrix to be assembled in the solver
    b    = None
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


print('building multigrid hierarchy...')
tstart = time()
levels = build_levels(numlevels)
print('time to build hierarchy:', time() - tstart)
x, y = SpatialCoordinate(levels[0].mesh)
u = Function(levels[0].fspace)
u.interpolate(sin(pi*x)*sin(pi*y))
u.vector()[:] = u.vector().array()*np.random.rand(len(u.vector().array()))
v = TestFunction(levels[0].fspace)
x, y = SpatialCoordinate(levels[0].mesh)
b = assemble(Constant(0.0)*v*dx, bcs=levels[0].bcs)
c = Function(levels[0].fspace)
c = assemble(Constant(0.0)*v*dx)

gmgsolver = nlobstacle_gmg_solver(levels, preiters, postiters)
tstart = time()
gmgsolver.solve(u, c, cycle=cycle, rtol=rtol, maxiters=maxiters)
print('time for gmg solve:', time() - tstart)

'''

ra, rb = Function(levels[0].fspace), Function(levels[0].fspace)
b = Function(levels[0].fspace)
w = Function(levels[0].fspace)
v = TestFunction(levels[0].fspace)
u = Function(levels[0].fspace)
u.interpolate(10.*sin(10.*pi*x)*sin(10.*pi*y))
f = Function(levels[0].fspace)
f.interpolate(sin(2*pi*x)*sin(2*pi*y))
H = Function(levels[0].fspace)
class aaf(Expression):
  def eval(self, value, x):
      if sqrt((x[0] - 2.0)**2 + (x[1] - 2.0)**2) < R:
          value[:] = a1
      else:
        value[:] = -a2
m = Function(levels[0].fspace)
m.interpolate(aaf())

F = 1e-22*inner(H**(n+2)/(n+2)*inner(grad(H), grad(H))*grad(H), grad(v))*dx - m*v*dx
F = replace(F, {H : u})
ra.assign(assemble(F, bcs=levels[0].bcs))
rb.assign(u - c)
#F = inner(grad(u), grad(v))*dx - f*v*dx
#c.interpolate(Max(.5 - (x - 2.0)**2 - (y - 2.0)**2, 0.0))
J = derivative(F, u)
b.assign(assemble(action(J, u) - F, bcs=levels[0].bcs))
A = assemble(J, bcs=levels[0].bcs)
ai, aj, av = A.M.handle.getValuesCSR()
bi, bj, bv = np.arange(ai.shape[0],dtype='int32'), np.arange(ai.shape[0] - 1,dtype='int32'), np.ones(ai.shape[0])
z = 0
print(np.linalg.norm(np.minimum(ra.vector().array(), rb.vector().array()),np.inf))
while np.linalg.norm(np.minimum(ra.vector().array(), rb.vector().array()),np.inf) > 1e-8 and z < maxiters:
  ai, aj, av = A.M.handle.getValuesCSR()
  b.assign(b/max(av))
  projected_gauss_seidel(ai, aj, av, bi, bj, bv, u.dat.data, b.vector().array(), c.vector().array())
x
  solve(A, u, b, solver_parameters = {'ksp_type': 'richardson',
                                          'ksp_max_it': 1,
                                          'ksp_convergence_test': 'skip',
                                          'ksp_initial_guess_nonzero': True,
                                          'ksp_richardson_scale': 3/4,
                                          'pc_type': 'jacobi'})
  
  u.interpolate(Max(u, c))
x
  ra.assign(assemble(F, bcs=levels[0].bcs))
  rb.assign(u - c)
  #F = inner(grad(u), grad(v))*dx
  b.assign(assemble(action(J, u) - F, bcs=levels[0].bcs))
  A = assemble(J, bcs=levels[0].bcs)
  ai, aj, av = A.M.handle.getValuesCSR()
  z += 1
  print(np.linalg.norm(np.minimum(ra.vector().array(), rb.vector().array()),np.inf))
'''
class aaf(Expression):
      def eval(self, value, x):
          if sqrt((x[0] - 2.0)**2 + (x[1] - 2.0)**2) < R:
              value[:] = a1
          else:
            value[:] = -a2
m = Function(levels[0].fspace)
m.interpolate(aaf())

plot(u, plot3d=True)
plt.savefig('icesol.png')
F = inner(u**(n+2)/(n+2)*inner(grad(u), grad(u))*grad(u), grad(v))*dx - m*v*dx

ra, rb = Function(levels[0].fspace), Function(levels[0].fspace)
ra.assign(assemble(F, bcs=levels[0].bcs))
rb.assign(u - c)
J = derivative(F, u)
r = Function(levels[0].fspace)
r.interpolate(Min(ra, rb))
plot(r)
plt.savefig('residplot.png')


