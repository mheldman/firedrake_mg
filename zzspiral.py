
from firedrake import *
from ufl.mathfunctions import Atan2
from numpy import arctan2

coarsemx = 2 #these have to specified in the user file along with element types, which is annoying
coarsemy = 2
cmesh = RectangleMesh(coarsemx, coarsemy, 2, 2, quadrilateral=True)

def g(x, y):
  return Constant(0.0)

def f(x, y):
  return Constant(0.0)

def psi(x, y):
  r = sqrt((x - 1.)**2 + (y - 1.)**2)
  return conditional( r > 1e-15, sin(2.*pi/r + pi/2. - Atan2(x - 1., y - 1.)) + r*(r + 1.)/(r - 2.) - 3.*r + 3.6, 3.6)

def init(x, y):
  return Constant(0.0)

exact = None
