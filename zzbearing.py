#python test.py -d zzbearing.py -o ./testresultspfas/lcp/bearing/ --mgtype pfas --maxiters 50 --atol 1e-14 --rtol 1e-12 --eps 0.0 --numlevels 10 --preiters 4 --postiters 4 --cycle FV


#python test.py -d zzbearing.py -o ./testresultsgmg/lcp/bearing/ --mgtype pfas --maxiters 50 --atol 1e-14 --rtol 1e-10 --eps 0.0 --numlevels 10 --preiters 1 --postiters 1 --cycle FV --fmg true --fmgc 2  --plotsolution3d true

from firedrake import *
from firedrakegmg import *
from gridtransfers import mrestrict
from time import time
import matplotlib.pyplot as plt
from relaxation import projected_gauss_seidel
from firedrake.functionspacedata import get_boundary_nodes

b = 10.
coarsemx, coarsemy = 2, 2
cmesh = RectangleMesh(coarsemx, coarsemy, 2*pi, 2*b, quadrilateral=True)
eps = .99 #should be in (0, 1)

def g(x, y):
  return Constant(0.0)

def f(x, y):
  return eps*sin(x)

def psi(x, y):
  return Constant(0.0)

def init(x, y):
  return Constant(0.0)
  
def form(u, v):
  x, y = SpatialCoordinate(u)
  return (1 + eps*cos(x))**3*inner(grad(u), grad(v))*dx

exact = None
def transform(u): #if you qnat to monitor the errors
  return u

