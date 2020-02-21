#python test.py -d zztorsion.py -o ./testresults/lcp/torsion/ --mgtype gmg --maxiters 50 --atol 1e-14 --rtol 1e-12 --eps 0.0 --numlevels 10 --preiters 2 --postiters 2 --fmgc 2 --cycle FV


from firedrake import *

coarsemx = 2 #these have to specified in the user file along with element types, which is annoying
coarsemy = 2
cmesh = RectangleMesh(coarsemx, coarsemy, 1, 1, quadrilateral=True)

def g(x, y):
  return Constant(0.0)

def f(x, y):
  return Constant(-8.0)

def psi(x, y):
  return  -Min(Min(x, y), Min(1.0 - x, 1.0 - y))

def init(x, y):
  return Constant(0.0)
  
def form(u, v):
  return inner(grad(u), grad(v))*dx

transform = None
exact = None
