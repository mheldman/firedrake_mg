from firedrake import *
from firedrakegmg import *
from gridtransfers import inject1
import numpy as np
from time import time
import matplotlib.pyplot as plt

coarse = Mesh('UnitCircle.msh')
mh = MeshHierarchy(coarse, 2)
Vf = FunctionSpace(mh[-1], "CG", 1)
Vc = FunctionSpace(mh[-2], "CG", 1)

uf1 = Function(Vf)
uf2 = Function(Vf)
uc1 = Function(Vc)
uc2 = Function(Vc)

uf  = Function(Vf)
uc  = Function(Vc)

''''
uf.vector()[:] = np.arange(len(uf.vector()[:]))
inject(uf, uc)
findices = uc.vector()[:]#.astype('int32')
findices = np.rint(findices)
findices = findices.astype('int32')
'''
'''
class a(Expression):
  def eval(self, value, x):
    if x[0] > 0.:
      value[:] = 1.0
    else:
      value[:] = 0.0

uf1.interpolate(a())
uf2.assign(1. - uf1)

print(np.linalg.norm(np.minimum(uf1.vector().array(), uf2.vector().array())))

inject1(uf1.dat.data, uc1.dat.data, findices)
#inject(uf2, uc2)
inject1(uf2.dat.data, uc2.dat.data, findices)
#print(norm(uf1), norm(uf2))
#print(norm(uc1), norm(uc2))

print(np.linalg.norm(np.minimum(uc1.vector().array(), uc2.vector().array())))

plot(uc1, plot3d=True)
plt.savefig('prolongtestfigs/uc1.png')
plot(uc2, plot3d=True)
plt.savefig('prolongtestfigs/uc2.png')
'''
x, y = SpatialCoordinate(mh[-2])
uc.interpolate(sin(6.*x)*sin(6.*y))
prolong(uc, uf1)
restrict(uf1, uc)
prolong(uc, uf2)
plot(uf1, plot3d=True)
plt.savefig('test results/other/prolongtestfigs/uf1.png')
uf2.assign(uf2/4.)
plot(uf2, plot3d=True)
plt.savefig('test results/other/prolongtestfigs/uf2.png')
uf.assign(uf1 - uf2/4.)
plot(uf, plot3d=True)
plt.savefig('test results/other/prolongtestfigs/difference.png')
print(norm(uf))
'''
x, y = SpatialCoordinate(mh[-2])
uc.interpolate(sin(x)*sin(y))
#prolong(uc1, uf1)
inject1(uf.dat.data, uc.dat.data, findices)
#inject(uf, uc)
prolong(uc, uf2)
print(norm(uf - uf2))
'''
