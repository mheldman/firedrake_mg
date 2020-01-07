# python test.py -d zzradial.py -o ./testresults/radialplaplace/ --mgtype pfas --fmgc 2 --numlevels 10 --plotexact3d true --rtol 1e-12 --eps 0.0 --maxiters 10000 --fmg true

from firedrake import *

rstar  = 0.697965148223374
rstar2 = rstar**2
a = 0.5879603630392823
C1 = -a**(4./3.)/sqrt(1. - a**2)
C2 = -1.5*2**(2/3)*C1
eps = 1e-4
'''
a = 2.**(1./3.) #make sure this value of a is correct
C1 = -a**3/sqrt(1. - a**2)
C2 = -.5*C1
'''
coarsemx = 3 #these have to specified in the user file along with element types, which is annoying
coarsemy = 3
cmesh = RectangleMesh(coarsemx, coarsemy, 4, 4, quadrilateral=True)
p = 4

def g(x, y):
  #return -rstar2 * ln( sqrt((x - 2.)**2 + (y - 2.)**2)/2.) / sqrt(1. - rstar2)
  return 1.5*C1*((x-2.)**2 + (y-2.)**2)**(1./3.) + C2
  #return conditional((x-2.)**2 + (y-2.)**2 > 1., -C1/sqrt((x-2.)**2 + (y-2.)**2) + C2, 0.0)

def f(x, y):
  return Constant(0.0)

def psi(x, y):
  return conditional( (x - 2.)**2 + (y - 2.)**2 < 1.0, sqrt(1. - ((x - 2.)**2 + (y - 2.)**2)), -100.0)

def init(x, y):
  return Constant(0.0)

def exact(x, y):
  return conditional( sqrt((x - 2.)**2 + (y - 2.)**2) > a, 1.5*C1*((x-2.)**2 + (y-2.)**2)**(1./3.) + C2, sqrt(1.0 - (x - 2.)**2 - (y - 2.)**2))

def D(u):
  return (inner(grad(u), grad(u)))**((p - 2.)/2.)
transform = None


   #return conditional( sqrt((x - 2.)**2 + (y - 2.)**2) > a, 1.5*C1*((x-2.)**2 + (y-2.)**2)**(1./3.) + C2, sqrt(1.0 - (x - 2.)**2 - (y - 2.)**2))
   
#conditional( (x - 2.)**2 + (y - 2.)**2 > rstar2, -rstar2 * ln( sqrt((x - 2.)**2 + (y - 2.)**2)/2.) / sqrt(1. - rstar2), sqrt(1.0 - (x - 2.)**2 - (y - 2.)**2))

