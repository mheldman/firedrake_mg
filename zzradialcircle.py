
from firedrake import *
from firedrakegmg import *
from gridtransfers import mrestrict
from time import time
import matplotlib.pyplot as plt
from relaxation import projected_gauss_seidel
from firedrake.functionspacedata import get_boundary_nodes

rstar  = 0.697965148223374
rstar2 = rstar**2
cmesh = Mesh('UnitCircle.msh')


def g(x, y):
  return -rstar2 * ln( sqrt(x**2 + y**2)/2.) / sqrt(1. - rstar2)

def f(x, y):
  return Constant(0.0)

def psi(x, y):
  return conditional( x**2 + y**2 < 1.0, sqrt(1. - (x**2 + y**2)), -100.0)

def init(x, y):
  return Constant(0.0)

def exact(x, y):
   return conditional( x**2 + y**2 > rstar2, -rstar2 * ln( sqrt(x**2 + y**2)/2.) / sqrt(1. - rstar2), sqrt(1.0 - x**2 - y**2))
