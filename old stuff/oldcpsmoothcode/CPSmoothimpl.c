/* ------------------------------------------------------------------------

    Solid Fuel Ignition (SFI) problem.  This problem is modeled by the
    partial differential equation

            -Laplacian(u) - lambda * exp(u) = 0,  0 < x,y,z < 1,

    with boundary conditions

             u = 0  for  x = 0, x = 1, y = 0, y = 1, z = 0, z = 1

    A finite difference approximation with the usual 7-point stencil
    is used to discretize the boundary value problem to obtain a
    nonlinear system of equations. The problem is solved in a 3D
    rectangular domain, using distributed arrays (DAs) to partition
    the parallel grid.

  ------------------------------------------------------------------------- */

#include "CPSmoothimpl.h"

#undef  __FUNCT__
#define __FUNCT__ "PGS"
PetscErrorCode PGS(int A[], Mat B, double b[], double c[], double x[])
{
  PetscInt Istart, Iend;
  PetscErrorCode ierr;
  
  ierr = MatGetOwnershipRange(A, &Istart, &Iend);
  for(int i = Istart; i < Iend; i++){
    const *colsA;
    PetscInt ncolsA;
    const PetscScalar *valsA;
    PetscScalar diagA;
    PetscScalar rsumA = 0.0;
    MatGetRow(A, i, &ncolsA, &colsA, &valsA);
    for(int j = 0; j < ncolsA; j++){
      int jj = colsA[j];
      if(jj == i)
        diagA = valsA[jj];
      else
        rsumA += x[jj]*valsA[jj];
    }
    if(diagA != 0.0)
      x[i] = (b[i] - rsumA)/diagA;
    MatRestoreRow(A, i, &ncolsA, &colsA, &valsA);
    
    const *colsB;
    PetscInt ncolsB;
    const PetscScalar *valsB;
    PetscScalar diagB, val;
    PetscScalar rsumB = 0.0;
    MatGetRow(B, i, &ncolsB, &colsB, &valsB);
    for(int j = 0; j < ncolsB; j++){
      int jj = colsB[j];
      if(jj == i)
        diagB = valsB[jj];
      else
        rsumA += x[jj]*valsB[jj];
    }
    if(diagB != 0.0){
      val = (c[i] - rsumB)/diagB;
      if(val > x[i])
        x[i] = val;
    }
    MatRestoreRow(B, i, &ncolsB, &colsB, &valsB);
  }
  PetscFunctionReturn(0);
}
