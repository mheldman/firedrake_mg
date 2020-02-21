#python test.py -d zzspiral.py -o ./testresults/lcp/spiral/ --mgtype pfas --maxiters 50 --atol 1e-14 --rtol 1e-12 --eps 0.0 --numlevels 8 --preiters 2 --postiters 2 --cycle V --fmg true --fmgc 2  --atol 1e-10

#python test.py -d zzspiral.py -o ./testresults/lcp/spiral/ --mgtype gmg --maxiters 50 --atol 1e-14 --rtol 1e-12 --eps 0.0 --numlevels 8 --preiters 1 --postiters 1 --cycle V --atol 1e-10

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
  return Constant(1.0)

def form(u, v):
  return inner(grad(u), grad(v))*dx

exact = None
transform=None
