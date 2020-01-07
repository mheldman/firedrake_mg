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

exact = None
