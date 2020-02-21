#python test.py -d zzbratu.py -o ./testresults/lcp/bratu/ --mgtype pfas --maxiters 50 --atol 1e-14 --rtol 1e-12 --eps 0.0 --numlevels 10 --preiters 2 --postiters 2 --fmgc 2 --cycle V --fmg true

#python test.py -d zzbratu.py -o ./testresults/lcp/bratu/ --mgtype gmg --maxiters 50 --atol 1e-14 --rtol 1e-12 --eps 0.0 --numlevels 10 --preiters 1 --postiters 1 --fmgc 2 --cycle V



from firedrake import *
from firedrakegmg import *
from gridtransfers import mrestrict
from time import time
import matplotlib.pyplot as plt
from relaxation import projected_gauss_seidel
from firedrake.functionspacedata import get_boundary_nodes


coarsemx = 2 #these have to specified in the user file along with element types, which is annoying
coarsemy = 2
cmesh = RectangleMesh(coarsemx, coarsemy, 1, 1, quadrilateral=True)

def f(x, y):
  return Constant(0.0)

def psi(x, y):
  return Constant(-.1)

def g(x, y):
    return Constant(0.0)

def init(x, y):
  return Constant(0.0)
  
def form(u, v):
  return inner(grad(u), grad(v))*dx + 2.5*exp(-u)*v*dx

transform = None
exact = None
