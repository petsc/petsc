#define PETSCMAT_DLL

/*
   Defines a Cholesky factorization preconditioner with BlockSolve95 interface.

   Note that BlockSolve95 works with a scaled and permuted preconditioning matrix.
   If the linear system matrix and preconditioning matrix are the same, we then
   work directly with the permuted and scaled linear system:
      - original system:  Ax = b
      - permuted and scaled system:   Cz = f, where
             C = P D^{-1/2} A D^{-1/2}
             z = P D^{1/2} x
             f = P D^{-1/2} b
             D = diagonal of A
             P = permutation matrix determined by coloring
   In this case, we use pre-solve and post-solve phases to handle scaling and
   permutation, and by default the scaled residual norm is monitored for the
   ILU/ICC preconditioners.  Use the option
     -ksp_truemonitor
   to print both the scaled and unscaled residual norms.
*/

#include "petsc.h"

#include "src/mat/impls/rowbs/mpi/mpirowbs.h"

#undef __FUNCT__  
#define __FUNCT__ "MatScaleSystem_MPIRowbs"
PetscErrorCode MatScaleSystem_MPIRowbs(Mat mat,Vec x,Vec rhs)
{
  Mat_MPIRowbs *bsif  = (Mat_MPIRowbs*)mat->data;
  Vec          v = bsif->xwork;
  PetscScalar  *xa,*rhsa,*va;
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  /* Permute and scale RHS and solution vectors */
  if (x) {
    ierr = VecGetArray(x,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(v,&va);CHKERRQ(ierr);
    BSperm_dvec(xa,va,bsif->pA->perm);CHKERRBS(0);
    ierr = VecRestoreArray(x,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(v,&va);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(v,bsif->diag,x);CHKERRQ(ierr);
  }

  if (rhs) {
    ierr = VecGetArray(rhs,&rhsa);CHKERRQ(ierr);
    ierr = VecGetArray(v,&va);CHKERRQ(ierr);
    BSperm_dvec(rhsa,va,bsif->pA->perm);CHKERRBS(0);
    ierr = VecRestoreArray(rhs,&rhsa);CHKERRQ(ierr);
    ierr = VecRestoreArray(v,&va);CHKERRQ(ierr);
    ierr = VecPointwiseMult(v,bsif->diag,rhs);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatUnScaleSystem_MPIRowbs"
PetscErrorCode MatUnScaleSystem_MPIRowbs(Mat mat,Vec x,Vec rhs)
{
  Mat_MPIRowbs *bsif  = (Mat_MPIRowbs*)mat->data;
  Vec          v = bsif->xwork;
  PetscScalar  *xa,*va,*rhsa;
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  /* Unpermute and unscale the solution and RHS vectors */
  if (x) {
    ierr = VecPointwiseMult(x,bsif->diag,v);CHKERRQ(ierr);
    ierr = VecGetArray(v,&va);CHKERRQ(ierr);
    ierr = VecGetArray(x,&xa);CHKERRQ(ierr);
    BSiperm_dvec(va,xa,bsif->pA->perm);CHKERRBS(0);
    ierr = VecRestoreArray(x,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(v,&va);CHKERRQ(ierr);
  }
  if (rhs) {
    ierr = VecPointwiseDivide(rhs,bsif->diag,v);CHKERRQ(ierr);
    ierr = VecGetArray(rhs,&rhsa);CHKERRQ(ierr);
    ierr = VecGetArray(v,&va);CHKERRQ(ierr);
    BSiperm_dvec(va,rhsa,bsif->pA->perm);CHKERRBS(0);
    ierr = VecRestoreArray(rhs,&rhsa);CHKERRQ(ierr);
    ierr = VecRestoreArray(v,&va);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatUseScaledForm_MPIRowbs"
PetscErrorCode MatUseScaledForm_MPIRowbs(Mat mat,PetscTruth scale)
{
  Mat_MPIRowbs *bsif  = (Mat_MPIRowbs*)mat->data;

  PetscFunctionBegin;
  bsif->vecs_permscale = scale;
  PetscFunctionReturn(0);
}

