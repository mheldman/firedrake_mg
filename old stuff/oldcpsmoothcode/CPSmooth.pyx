from petsc4py.PETSc cimport Vec,  PetscVec
from petsc4py.PETSc cimport Mat,  PetscMat
import numpy as np
cimport numpy as np

from petsc4py.PETSc import Error

cdef extern from "CPSmoothimpl.h":
    int PGS(PetscMat A, PetscMat B, np.ndarray[np.float64_t, ndim=2] b, np.ndarray[np.float64_t, ndim=2] c, np.ndarray[np.float64_t, ndim=2] x)

def PGS(Mat A, Mat B, np.ndarray[np.float64_t, ndim=2] b, np.ndarray[np.float64_t, ndim=2] c, np.ndarray[np.float64_t, ndim=2] x):
    cdef int ierr
    ierr = PGS(A, B, b, c, x)
    if ierr != 0: raise Error(ierr)
