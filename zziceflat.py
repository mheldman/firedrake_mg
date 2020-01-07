from firedrake import *

a1                   = .01
a2                   = .03
R                    = a2/(a1 + a2)

cmesh = RectangleMesh(2, 2, 4, 4, quadrilateral=True)

def g(x, y):
  return Constant(0.0)

def f(x, y):
  r = sqrt((x - 2.)**2 + (y - 2.)**2)
  return -rho*g*(1 - f)

def psi(x, y):
  return Constant(0.0)

def init(x, y):
  return sin(pi*x/4.)*sin(pi*y/4.)

exact = None
