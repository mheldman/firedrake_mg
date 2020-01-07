from firedrake import *
from firedrakegmg import *
from gridtransfers import mrestrict
from time import time
import matplotlib.pyplot as plt
from relaxation import projected_gauss_seidel
from firedrake.functionspacedata import get_boundary_nodes


coarsemx = 2 #these have to specified in the user file along with element types, which is annoying
coarsemy = 2
cmesh = RectangleMesh(coarsemx, coarsemy, 16, 24, quadrilateral=True)

def f(x, y):
  return Constant(1.0)

def psi(x, y):
  return Constant(0.0)

c = 4.0
a, b = 16.0, 24.0
def g(x, y):
    return conditional( x < 1e-16, .5*(b - y) ** 2, conditional(y < 1e-16, ((a - x)*b**2 + x*c**2)/(2*a), conditional(24.0 - y < 1e-16, 0.0, conditional(y > c, 0.0, conditional(16.0 - x < 1e-16, .5*(c - y)**2, 0.0)))))

def init(x, y):
  return Constant(0.0)

exact = None
