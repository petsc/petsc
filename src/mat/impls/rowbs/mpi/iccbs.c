#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: iccbs.c,v 1.32 1999/01/26 17:13:30 bsmith Exp bsmith $";
#endif
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
     -ksp_bsmonitor
   to print both the scaled and unscaled residual norms.

   If the preconditioning matrix differs from the linear system matrix, then we
   work directly ith the original linear system, and just do the scaling and
   permutation within PCApply().
*/

#include "petsc.h"

#if defined(HAVE_BLOCKSOLVE) && !defined(USE_PETSC_COMPLEX)
#include "src/mat/impls/rowbs/mpi/mpirowbs.h"

#undef __FUNC__  
#define __FUNC__ "MatScaleSystem_MPIRowbs"
int MatScaleSystem_MPIRowbs(Mat mat,Vec x,Vec rhs)
{
  Mat_MPIRowbs *bsif  = (Mat_MPIRowbs *) mat->data;
  Vec          v = bsif->xwork;
  Scalar       *xa, *rhsa, *va;
  int          ierr;

  PetscFunctionBegin;  
  /* Permute and scale RHS and solution vectors */
  ierr = VecGetArray(x,&xa); CHKERRQ(ierr);
  ierr = VecGetArray(v,&va); CHKERRQ(ierr);
  BSperm_dvec(xa,va,bsif->pA->perm); CHKERRBS(0);
  ierr = VecRestoreArray(x,&xa); CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&va); CHKERRQ(ierr);
  ierr = VecPointwiseDivide(v,bsif->diag,x); CHKERRQ(ierr);
  ierr = VecGetArray(rhs,&rhsa); CHKERRQ(ierr);
  ierr = VecGetArray(v,&va); CHKERRQ(ierr);
  BSperm_dvec(rhsa,va,bsif->pA->perm); CHKERRBS(0);
  ierr = VecRestoreArray(rhs,&rhsa); CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&va); CHKERRQ(ierr);
  ierr = VecPointwiseMult(v,bsif->diag,rhs); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatUnScaleSystem_MPIRowbs"
int MatUnScaleSystem_MPIRowbs(Mat mat,Vec x, Vec rhs)
{
  Mat_MPIRowbs *bsif  = (Mat_MPIRowbs *) mat->data;
  Vec          v = bsif->xwork;
  Scalar       *xa, *va, *rhsa;
  int          ierr;

  PetscFunctionBegin;  
  /* Unpermute and unscale the solution and RHS vectors */
  ierr = VecPointwiseMult(x,bsif->diag,v); CHKERRQ(ierr);

  ierr = VecGetArray(v,&va); CHKERRQ(ierr);
  ierr = VecGetArray(x,&xa); CHKERRQ(ierr);
  BSiperm_dvec(va,xa,bsif->pA->perm); CHKERRBS(0);
  ierr = VecRestoreArray(x,&xa); CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&va); CHKERRQ(ierr);
  ierr = VecPointwiseDivide(rhs,bsif->diag,v); CHKERRQ(ierr);
  ierr = VecGetArray(rhs,&rhsa); CHKERRQ(ierr);
  ierr = VecGetArray(v,&va); CHKERRQ(ierr);
  BSiperm_dvec(va,rhsa,bsif->pA->perm); CHKERRBS(0);
  ierr = VecRestoreArray(rhs,&rhsa); CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&va); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatUseScaledForm_MPIRowbs"
int MatUseScaledForm_MPIRowbs(Mat mat,PetscTruth scale)
{
  Mat_MPIRowbs *bsif  = (Mat_MPIRowbs *) mat->data;

  PetscFunctionBegin;
  bsif->vecs_permscale = scale;
  PetscFunctionReturn(0);
}

#include "sles.h"

/* 
   KSPMonitor_MPIRowbs - Prints the actual (unscaled) residual norm as
   well as the scaled residual norm.  The default residual monitor for 
   ICC/ILU with BlockSolve95 prints only the scaled residual norm.

   Options Database Keys:
$  -ksp_bsmonitor
 */
#undef __FUNC__  
#define __FUNC__ "KSPMonitor_MPIRowbs"
int KSPMonitor_MPIRowbs(KSP ksp,int n,double rnorm,void *dummy)
{
  Mat_MPIRowbs *bsif;
  int          ierr;
  Vec          resid;
  double       scnorm;
  PC           pc;
  Mat          mat;
  MPI_Comm     comm;

  PetscFunctionBegin;  
  ierr = KSPGetPC(ksp,&pc); CHKERRQ(ierr);
  ierr = PCGetOperators(pc,&mat,0,0); CHKERRQ(ierr);
  bsif = (Mat_MPIRowbs *) mat->data;
  ierr = KSPBuildResidual(ksp,0,bsif->xwork,&resid); CHKERRQ(ierr);
  ierr = VecPointwiseDivide(resid,bsif->diag,resid); CHKERRQ(ierr); 
  ierr = VecNorm(resid,NORM_2,&scnorm); CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)ksp,&comm);CHKERRQ(ierr);
  PetscPrintf(comm,"%d Preconditioned %14.12e True %14.12e\n",n,rnorm,scnorm); 
  PetscFunctionReturn(0);
}

#else
#undef __FUNC__  
#define __FUNC__ "MatNull_MPIRowbs"
int MatNull_MPIRowbs(void)
{
  PetscFunctionBegin;  
  PetscFunctionReturn(0);
}
#endif

