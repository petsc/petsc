/* file: del2mat.h */

#ifndef DEL2MAT_H
#define DEL2MAT_H

#include <petsc.h>
#include <petscvec.h>
#include <petscmat.h>

/* external Fortran 90 subroutine */
#define Del2Apply del2apply_
extern void Del2Apply(int*,double*,double*,double*);

/* user data structure and routines 
 * defining the matrix-free operator */

typedef struct {
  PetscInt    N;
  PetscScalar *F;
} Del2Mat;

#undef  __FUNCT__  
#define __FUNCT__ "Del2Mat_mult"
/* y <- A * x */
PetscErrorCode Del2Mat_mult(Mat A, Vec x, Vec y) 
{
  Del2Mat *ctx;
  PetscScalar *xx,*yy;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatShellGetContext(A,(void**)&ctx);CHKERRQ(ierr);
  /* get raw vector arrays */
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  /* call external Fortran subroutine */
  Del2Apply(&ctx->N,ctx->F,xx,yy);
  /* restore raw vector arrays */
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef  __FUNCT__  
#define __FUNCT__ "Del2Mat_diag"
/*D_i <- A_ii */
PetscErrorCode Del2Mat_diag(Mat A, Vec D) 
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecSet(D,6.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif
