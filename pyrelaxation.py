import numpy as np
from firedrake import *
from time import time

def nlpgs(F, f, u, c, indices):
    #A = assemble(derivative(F, u))
    #ai, aj, av = A.M.handle.getValuesCSR()
    #ftime = 0.0
    #dftime = 0.0
    #tottime = 0.0
    #tstart1 = time()
    w = Function(u.function_space())
    z = 0
    eps = 1e-8
    for i in indices:
       w.vector()[i] = 1.0 #form ith basis function
       #tstart = time()
       dFi = assemble(action(derivative(F, u, w), w)) #compute diagoanl element of Jacobian about u
       #dftime += time() - tstart
       if abs(dFi + eps) > 1e-15:
        #tstart = time()
        #v = TestFunction(u.function_space())
        #Fi = assemble(action(F - f*v*dx, w)) #compute the action of F on the ith basis function (i.e., F_i(u))
        Fi = action(F, w)
        correct = (assemble(Fi) - eps*u.vector()[i] - f.vector()[i])/(dFi + eps)
        #ftime += time() - tstart
        u.vector()[i] = max(u.vector()[i] - correct, c.vector()[i])
        z += 1
        w.vector()[i] = 0.0
        #print(av[ai[i]:ai[i+1]], dFi)
    #print('total time: ', time() - tstart1)
    #print('Ftime: ', ftime)
    #print('dftime: ', dftime)
