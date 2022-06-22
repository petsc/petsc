/* file: del2mat.h */

#ifndef DEL2MAT_H
#define DEL2MAT_H

#include <petsc.h>
#include <petscvec.h>
#include <petscmat.h>

/* external Fortran 90 subroutine */
#define Del2Apply del2apply_
EXTERN_C_BEGIN
extern void Del2Apply(int*,double*,const double*,double*);
EXTERN_C_END

/* user data structure and routines
 * defining the matrix-free operator */

typedef struct {
  PetscInt    N;
  PetscScalar *F;
} Del2Mat;

/* y <- A * x */
PetscErrorCode Del2Mat_mult(Mat A, Vec x, Vec y)
{
  Del2Mat *ctx;
  const PetscScalar *xx;
  PetscScalar *yy;
  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A,&ctx));
  /* get raw vector arrays */
  PetscCall(VecGetArrayRead(x,&xx));
  PetscCall(VecGetArray(y,&yy));
  /* call external Fortran subroutine */
  Del2Apply(&ctx->N,ctx->F,xx,yy);
  /* restore raw vector arrays */
  PetscCall(VecRestoreArrayRead(x,&xx));
  PetscCall(VecRestoreArray(y,&yy));
  PetscFunctionReturn(0);
}

/*D_i <- A_ii */
PetscErrorCode Del2Mat_diag(Mat A, Vec D)
{
  PetscFunctionBegin;
  PetscCall(VecSet(D,6.0));
  PetscFunctionReturn(0);
}

#endif
