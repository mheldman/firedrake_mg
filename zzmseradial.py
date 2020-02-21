#python test.py -d zzmseradial.py -o ./testresults/lcp/radial/ --mgtype pfas --maxiters 50 --atol 1e-14 --rtol 1e-12 --eps 0.0 --numlevels 10 --preiters 3 --postiters 3 --cycle FV --fmg true 

#python test.py -d zzmseradial.py -o ./testresults/lcp/mseradial/ --mgtype gmg --maxiters 50 --atol 1e-14 --rtol 1e-12 --eps 0.0 --numlevels 10 --preiters 1 --postiters 1  --cycle FV --atol 1e-10 --fmg true --fmgc 2

from firedrake import *
from firedrakegmg import *
from gridtransfers import mrestrict
from time import time
import matplotlib.pyplot as plt
from relaxation import projected_gauss_seidel
from firedrake.functionspacedata import get_boundary_nodes

rstar  = 0.697965148223374
rstar2 = rstar**2
coarsemx, coarsemy = 2, 2
cmesh = RectangleMesh(coarsemx, coarsemy, 4, 4, quadrilateral=False)

def g(x, y):
  return conditional( (x - 2.0)**2 + (y - 2.0)**2 > 1.0, -rstar2 * ln( sqrt((x - 2.0)**2 + (y - 2.0)**2)/2.) / sqrt(1. - rstar2), 0.0)

def f(x, y):
  return Constant(0.0)

def psi(x, y):
  return conditional( (x - 2.0)**2 + (y - 2.0)**2 < 1.0, sqrt(1. - ((x - 2.0)**2 + (y - 2.0)**2)), -100.0)

def init(x, y):
  return Constant(0.0)
  
def form(u, v):
  return (1/sqrt(1 + inner(grad(u), grad(u))))*inner(grad(u), grad(v))*dx

def exact(x, y):
   return conditional( (x - 2.0)**2 + (y - 2.0)**2 > rstar2, -rstar2 * ln( sqrt((x - 2.0)**2 + (y - 2.0)**2)/2.) / sqrt(1. - rstar2), sqrt(1.0 - (x - 2.0)**2 - (y - 2.0)**2))

def transform(u): #if you qnat to monitor the errors
  return u

