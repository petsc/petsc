#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: iccbs.c,v 1.29 1998/04/03 23:15:15 bsmith Exp bsmith $";
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
#include "src/pc/pcimpl.h"            /*I "pc.h" I*/
#include "src/pc/impls/icc/icc.h"
#include "src/ksp/kspimpl.h"
#include "mpirowbs.h"

#undef __FUNC__  
#define __FUNC__ "PCDestroy_ICC_MPIRowbs"
static int PCDestroy_ICC_MPIRowbs(PC pc)
{
  PC_ICC *icc = (PC_ICC *) pc->data;
  PCiBS  *iccbs = (PCiBS *) icc->implctx; 
  int    ierr;

  PetscFunctionBegin;  
  PetscFree(iccbs);
  ierr = MatDestroy(icc->fact); CHKERRQ(ierr);
  PetscFree(icc);
  PetscFunctionReturn(0);
}

/* Note:  We only call PCPreSolve_MPIRowbs() if both
   the linear system matrix and preconditioning matrix
   are stored in the MATMPIROWBS format */
#undef __FUNC__  
#define __FUNC__ "PCPreSolve_MPIRowbs"
int PCPreSolve_MPIRowbs(PC pc,KSP ksp)
{
  Mat_MPIRowbs *bsif = (Mat_MPIRowbs *) pc->pmat->data;
  Mat_MPIRowbs *bsifa = (Mat_MPIRowbs *) pc->mat->data;
  Vec          rhs, x, v = bsif->xwork;
  Scalar       *xa, *rhsa, *va;
  int          ierr;

  PetscFunctionBegin;  
  /* Permute and scale RHS and solution vectors */
  ierr = KSPGetSolution(ksp,&x); CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp,&rhs); CHKERRQ(ierr);
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
  bsif->vecs_permscale  = 1;
  bsifa->vecs_permscale = 1;
  PetscFunctionReturn(0);
}

/* Note:  We only call PCPostSolve_MPIRowbs() if both
   the linear system matrix and preconditioning matrix
   are stored in the MATMPIROWBS format */
#undef __FUNC__  
#define __FUNC__ "PCPostSolve_MPIRowbs"
int PCPostSolve_MPIRowbs(PC pc,KSP ksp)
{
  Mat_MPIRowbs *bsif = (Mat_MPIRowbs *) pc->pmat->data;
  Mat_MPIRowbs *bsifa = (Mat_MPIRowbs *) pc->mat->data;
  Vec          x, rhs, v = bsif->xwork;
  Scalar       *xa, *va, *rhsa;
  int          ierr;

  PetscFunctionBegin;  
  /* Unpermute and unscale the solution and RHS vectors */
  ierr = KSPGetSolution(ksp,&x); CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp,&rhs); CHKERRQ(ierr);
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
  bsif->vecs_permscale  = 0;
  bsifa->vecs_permscale = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetUp_ICC_MPIRowbs"
int PCSetUp_ICC_MPIRowbs(PC pc)
{
  PC_ICC       *icc = (PC_ICC *) pc->data;
  PCiBS        *iccbs;
  MatStructure pflag;
  Mat          Amat, Pmat;
  int          ierr;

  PetscFunctionBegin;  
  ierr = PCGetOperators(pc,&Amat,&Pmat,&pflag); CHKERRQ(ierr);
  if (Amat != Pmat && Amat->type == MATMPIROWBS) {
    SETERRQ(PETSC_ERR_ARG_INCOMP,0,"Does not support different Amat and\n\
      Pmat with MATMPIROWBS format for both.  Use a different format for\n\
      Amat (e.g., MATMPIAIJ) and keep Pmat the same.");
  }

  pc ->destroy        = PCDestroy_ICC_MPIRowbs;
  icc->implctx        = (void *) (iccbs = PetscNew(PCiBS)); CHKPTRQ(iccbs);
  PLogObjectMemory(pc,sizeof(PCiBS));

  if (icc->bs_iter) { /* Set BlockSolve iterative solver defaults */
    SETERRQ(PETSC_ERR_SUP,0,"BS iterative solvers not currently supported");
/*    iccbs->blocksize  = 1;
    iccbs->pre_option = PRE_STICCG;
    iccbs->rtol       = 1.e-5;
    iccbs->max_it     = 10000;
    iccbs->rnorm      = 0.0;
    iccbs->guess_zero = 1; */
  } else {
    iccbs->blocksize  = 0;
    iccbs->pre_option = 0;
    iccbs->rtol       = 0;
    iccbs->max_it     = 0;
    iccbs->rnorm      = 0.0;
    iccbs->guess_zero = 0;
    if (Amat->type == MATMPIROWBS) {
      pc->presolve    = PCPreSolve_MPIRowbs;
      pc->postsolve   = PCPostSolve_MPIRowbs;
    }
  }
  PetscFunctionReturn(0);
}

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

  PetscFunctionBegin;  
  ierr = KSPGetPC(ksp,&pc); CHKERRQ(ierr);
  ierr = PCGetOperators(pc,&mat,0,0); CHKERRQ(ierr);
  bsif = (Mat_MPIRowbs *) mat->data;
  ierr = KSPBuildResidual(ksp,0,bsif->xwork,&resid); CHKERRQ(ierr);
  ierr = VecPointwiseDivide(resid,bsif->diag,resid); CHKERRQ(ierr); 
  ierr = VecNorm(resid,NORM_2,&scnorm); CHKERRQ(ierr);
  PetscPrintf(ksp->comm,"%d Preconditioned %14.12e True %14.12e\n",n,rnorm,scnorm); 
  PetscFunctionReturn(0);
}
  

#undef __FUNC__  
#define __FUNC__ "PCBSIterSetFromOptions"
/* @
  PCBSIterSetFromOptions - Sets various options for the BlockSolve 
  iterative solvers.

  Input Parameter:
. pc - the PC context

  Notes:
  These iterative solvers can be used only with the MATMPIROWBS matrix data 
  structure for symmetric matrices.  They are intended primarily for
  comparison with the SLES/KSP interface, which we recommend for general use.
@ */
int PCBSIterSetFromOptions(PC pc)
{
  PC_ICC *icc = (PC_ICC *) pc->data;
  PCiBS  *iccbs;
  int    ierr,flg;

  PetscFunctionBegin;  
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->pmat->type != MATMPIROWBS) PetscFunctionReturn(0);
  iccbs = (PCiBS *) icc->implctx;
  ierr = OptionsGetInt(pc->prefix,"-pc_bs_max_it",&iccbs->max_it,&flg);CHKERRQ(ierr);
  ierr = OptionsGetInt(pc->prefix,"-pc_bs_blocksize",&iccbs->blocksize,&flg);CHKERRQ(ierr);
  ierr = OptionsGetDouble(pc->prefix,"-pc_bs_rtol",&iccbs->rtol,&flg);CHKERRQ(ierr);
  ierr = OptionsHasName(pc->prefix,"-pc_bs_guess_zero",&flg);CHKERRQ(ierr); 
  if (flg) { 
    iccbs->guess_zero = 1;
  }
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

