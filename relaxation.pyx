# python setup.py build_ext -i

cimport numpy as np
from libc.math cimport sin

def gauss_seidel(np.ndarray[int, ndim=1] Ap, np.ndarray[int, ndim=1] Aj, np.ndarray[double, ndim=1] Ax, np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] b, np.ndarray[int, ndim=1] indices):
    cdef unsigned int i, j, jj, start, end
    cdef double rsum, diag

    for i in indices:
       start = Ap[i]
       end   = Ap[i + 1]
       rsum = 0.0
       for jj in xrange(start, end):
            diag, j = 0.0, Aj[jj]
            if i == j:
              diag = Ax[jj]
            else:
              rsum += Ax[jj]*x[j]
       if diag != 0.0:
            x[i] = (b[i] - rsum)/diag
       else:
            x[i] = b[i]

def projected_gauss_seidel(np.ndarray[int, ndim=1] Ap, np.ndarray[int, ndim=1] Aj, np.ndarray[double, ndim=1] Ax, np.ndarray[int, ndim=1] Bp, np.ndarray[int, ndim=1] Bj, np.ndarray[double, ndim=1] Bx, np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] b, np.ndarray[double, ndim=1] c, np.ndarray[int, ndim=1] indices):
    cdef unsigned int i, j, jj, start, end
    cdef double rsum, diag, val, eps, omega

    eps = 1e-8#ensures nonzeros on the diagonal of the jacobian approximation
    omega =  1.0 #relaxation parameter
    for i in indices:

       start = Ap[i]
       end   = Ap[i + 1]
       rsum, diag = 0.0, 0.0
       for jj in xrange(start, end):
            j = Aj[jj]
            if i == j:
              diag  = Ax[jj]
            else:
              rsum += Ax[jj]*x[j]
       if abs(diag + eps) > 1e-16:
          x[i] = (1. - omega)*x[i] + omega*(b[i] + eps*x[i] - rsum)/(diag + eps)
          #x[i] = (b[i] + eps*x[i] - (1. - eps)*rsum)/((1. - eps)*diag + eps)
          #x[i] = x[i] + (b[i]  - rsum - diag*x[i])

       start = Bp[i]
       end   = Bp[i + 1]
       rsum, diag = 0.0, 0.0
       for jj in xrange(start, end):
          j = Bj[jj]
          if i == j:
            diag  = Bx[jj]
          else:
            rsum += Bx[jj]*x[j]
       if diag != 0.0:
          val = (c[i] - rsum)/diag

       x[i] = max(val, x[i])

def symmetric_pgs(np.ndarray[int, ndim=1] Ap, np.ndarray[int, ndim=1] Aj, np.ndarray[double, ndim=1] Ax, np.ndarray[int, ndim=1] Bp, np.ndarray[int, ndim=1] Bj, np.ndarray[double, ndim=1] Bx, np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] b, np.ndarray[double, ndim=1] c, np.ndarray[int, ndim=1] indices):


    projected_gauss_seidel(Ap, Aj, Ax, Bp, Bj, Bx, x, b, c, indices)
    projected_gauss_seidel(Ap, Aj, Ax, Bp, Bj, Bx, x, b, c, indices[len(indices) - 1:-1:-1])

    #doesn't seem to work

def rs_gauss_seidel(np.ndarray[int, ndim=1] Ap, np.ndarray[int, ndim=1] Aj, np.ndarray[double, ndim=1] Ax, np.ndarray[int, ndim=1] Bp, np.ndarray[int, ndim=1] Bj, np.ndarray[double, ndim=1] Bx, np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] b, np.ndarray[double, ndim=1] c, np.ndarray[int, ndim=1] indices):
    cdef unsigned int i, j, jj, start, end
    cdef double rsum, diag, val

    for i in indices:

       start = Ap[i]
       end   = Ap[i + 1]
       rsum, diag = 0.0, 0.0
       for jj in xrange(start, end):
            j = Aj[jj]
            if i == j:
              diag  = Ax[jj]
            else:
              rsum += Ax[jj]*x[j]
       if abs(diag) > 1e-15:
           x[i] = (b[i] - rsum)/diag #do unconstrained gauss-seidel on the reduced space

def rsc_gauss_seidel(np.ndarray[int, ndim=1] Ap, np.ndarray[int, ndim=1] Aj, np.ndarray[double, ndim=1] Ax, np.ndarray[int, ndim=1] Bp, np.ndarray[int, ndim=1] Bj, np.ndarray[double, ndim=1] Bx, np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] b, np.ndarray[double, ndim=1] c, np.ndarray[int, ndim=1] indices):
    cdef unsigned int i, j, jj, start, end
    cdef double rsum, diag, val

    for i in indices:

       start = Ap[i]
       end   = Ap[i + 1]
       rsum, diag = 0.0, 0.0
       for jj in xrange(start, end):
            j = Aj[jj]
            if i == j:
              diag  = Ax[jj]
            rsum += Ax[jj]*x[j]
       if b[i] - rsum > 0.0 or x[i] > c[i]:
            if abs(diag) > 1e-16:
              x[i] = (b[i] - rsum + diag*x[i])/diag

       x[i] = max(x[i], c[i])

'''
def nl_pgs(fd.ufl_expr F, fd.Function u, np.ndarray[double, ndim=1] c, np.ndarray[int, ndim=1] indices):
    cdef unsigned int i
    cdef fd.ufl_expr F, dF
    cdef fd.Function w
    cdef float Fi, dFi

    for i in indices:
       w = fd.Function(u.function_space())
       w.vector()[i] = 1.0
       #F  = F.replace({H : u, v : w})
       dF = fd.derivative(F, u, w)
       Fi = fd.assemble(Fi)
       dFi = fd.assemble(dFi)
       if dFi > 0.0:
        u.vector()[i] = u.vector()[i] - Fi/dFi
        u.vector()[i] = max(u.vector()[i], c[i])
'''
