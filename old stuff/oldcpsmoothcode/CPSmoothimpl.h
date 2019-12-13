#ifndef CPSmooth_H
#define CPSmooth_H

#include <petsc.h>

#if PETSC_VERSION_(3,1,0)
#include <petscvec.h>
#include <petscmat.h>
#endif

PetscErrorCode PGS(Mat A, Mat B, double b[], double c[], double x[]);

#endif /* !CPSmooth_H */
