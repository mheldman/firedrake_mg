cimport numpy as np
def mrestrict(np.ndarray[int, ndim=1] Bp, np.ndarray[int, ndim=1] Bj, np.ndarray[double, ndim=1] Bx, np.ndarray[int, ndim=1] findices, np.ndarray[double, ndim=1] psi, np.ndarray[double, ndim=1] psic):
    cdef unsigned int i, j, jj, start, end
    cdef double val

    for i in xrange(0, psic.shape[0]):

       start = Bp[findices[i]]
       end   = Bp[findices[i] + 1]
       val   = -1000.0
       for jj in xrange(start, end):
          j = Bj[jj]
          if abs(Bx[jj]) > 1e-13:
            val = max(val, psi[j])
       psic[i] = val

def inject1(np.ndarray[double, ndim=1] u, np.ndarray[double, ndim=1] uc, np.ndarray[int, ndim=1] findices):
    cdef unsigned int i

    for i in xrange(0, uc.shape[0]):
       uc[i] = u[findices[i]]

