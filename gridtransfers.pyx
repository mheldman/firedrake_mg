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
       
def restrict_cp(np.ndarray[double, ndim=1] u, np.ndarray[double, ndim=1] uc, np.ndarray[double, ndim=1] g, np.ndarray[double, ndim=1] gc, np.ndarray[double, ndim=1] r, np.ndarray[double, ndim=1] rc, np.ndarray[int, ndim=1] findices, np.ndarray[int, ndim=1] indices):
    cdef unsigned int i
    for i in indices:
      if abs(r[findices[i]]) + abs(u[findices[i]] - g[findices[i]]) < 1e-2*(u.shape[0])**(.5): #if point is near the free boundary
        rc[i] = r[findices[i]]
        uc[i] = u[findices[i]]
        '''
        if abs(min(rc[i], uc[i] - gc[i])) > 1e-15: #if coarse grid complementarity condition is not satisfied at that point
          if u[findices[i]] - g[findices[i]] < 1e-15:
            uc[i] = gc[i] #if the fine grid function is in contact with the obstacle, force the same on the coarse grid function
          else: #if the fine grid residual is zero, use injection to restrict to the coarse grid residual
            rc[i] = 0.0 #if the fine grid function at the node is equal to the fine grid obstacle, force the same to hold on the coarse grid
        '''
        

