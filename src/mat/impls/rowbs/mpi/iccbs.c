#ifndef lint
static char vcid[] = "$Id: iccbs.c,v 1.4 1995/12/01 22:29:37 curfman Exp $";
#endif
/*
   Defines a Cholesky factorization preconditioner with BlockSolve interface.
*/
#if defined(HAVE_BLOCKSOLVE) && !defined(__cplusplus)
#include "src/pc/pcimpl.h"
#include "src/pc/impls/icc/icc.h"
#include "ksp/kspimpl.h"
#include "matimpl.h"
#include "mpirowbs.h"
#include "BSprivate.h"

static int PCDestroy_ICC_MPIRowbs(PetscObject obj)
{
  PC     pc = (PC) obj;
  PC_ICC *icc = (PC_ICC *) pc->data;
  PCiBS  *iccbs = (PCiBS *) icc->implctx; 
  int    ierr;
  PetscFree(iccbs);
  ierr = MatDestroy(icc->fact); CHKERRQ(ierr);
  PetscFree(icc);
  return 0;
}

int PCPreSolve_MPIRowbs(PC pc,KSP ksp)
{
  Mat_MPIRowbs *bsif = (Mat_MPIRowbs *) pc->pmat->data;
  Vec          rhs, x, v = bsif->xwork;
  Scalar       *xa, *rhsa, *va;
  int          ierr;

  /* Permute RHS and solution vectors */
  ierr = KSPGetSolution(ksp,&x); CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp,&rhs); CHKERRQ(ierr);
  ierr = VecGetArray(rhs,&rhsa); CHKERRQ(ierr);
  ierr = VecGetArray(x,&xa); CHKERRQ(ierr);
  ierr = VecGetArray(v,&va); CHKERRQ(ierr);
  BSperm_dvec(xa,va,bsif->pA->perm); CHKERRBS(0);
  ierr = VecPDiv(v,bsif->diag,x); CHKERRQ(ierr);
  BSperm_dvec(rhsa,va,bsif->pA->perm); CHKERRBS(0);
  ierr = VecPMult(v,bsif->diag,rhs); CHKERRQ(ierr);
  ierr = VecRestoreArray(rhs,&rhsa); CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&xa); CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&va); CHKERRQ(ierr);
  bsif->vecs_permscale = 1;
  return 0;
}

int PCPostSolve_MPIRowbs(PC pc,KSP ksp)
{
  Mat_MPIRowbs *bsif = (Mat_MPIRowbs *) pc->pmat->data;
  Vec          x, rhs, v = bsif->xwork;
  Scalar       *xa, *va, *rhsa;
  int          ierr;

  /* Unpermute and unscale the solution and RHS vectors */
  ierr = KSPGetSolution(ksp,&x); CHKERRQ(ierr);
  ierr = KSPGetRhs(ksp,&rhs); CHKERRQ(ierr);
  ierr = VecGetArray(v,&va); CHKERRQ(ierr);
  ierr = VecGetArray(x,&xa); CHKERRQ(ierr);
  ierr = VecGetArray(rhs,&rhsa); CHKERRQ(ierr);
  ierr = VecPMult(x,bsif->diag,v); CHKERRQ(ierr);
  BSiperm_dvec(va,xa,bsif->pA->perm); CHKERRBS(0);
  ierr = VecPDiv(rhs,bsif->diag,v); CHKERRQ(ierr);
  BSiperm_dvec(va,rhsa,bsif->pA->perm); CHKERRBS(0);
  ierr = VecRestoreArray(rhs,&rhsa); CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&xa); CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&va); CHKERRQ(ierr);
  bsif->vecs_permscale = 0;
  return 0;
}

int PCSetUp_ICC_MPIRowbs(PC pc)
{
  PC_ICC *icc = (PC_ICC *) pc->data;
  PCiBS  *iccbs;

  pc ->destroy        = PCDestroy_ICC_MPIRowbs;
  icc->implctx        = (void *) (iccbs = PetscNew(PCiBS)); CHKPTRQ(iccbs);
  if (icc->bs_iter) { /* Set BlockSolve iterative solver defaults */
    SETERRQ(1,"PCSetUp_ICC_MPIRowbs: BS iterative solvers not currently supported");
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
    pc->presolve      = PCPreSolve_MPIRowbs;
    pc->postsolve     = PCPostSolve_MPIRowbs;
  }
  return 0;
}

/* 
   KSPMonitor_MPIRowbs - Monitors the actual (unscaled) residual.  The
   default residual monitor for PCICC with BlockSolve prints the scaled 
   residual.

   Question: Should this routine really be here? 
 */
int KSPMonitor_MPIRowbs(KSP itP,int n,double rnorm,Mat mat)
{
  Mat_MPIRowbs *bsif = (Mat_MPIRowbs *) mat->data;
  int          ierr;
  Vec          resid;
  double       scnorm;

  ierr = KSPBuildResidual(itP,0,bsif->xwork,&resid); CHKERRQ(ierr);
  ierr = VecPMult(resid,bsif->diag,resid); CHKERRQ(ierr);
  ierr = VecNorm(resid,NORM_2,&scnorm); CHKERRQ(ierr);
  MPIU_printf(itP->comm,"%d %14.12e \n",n,scnorm); 
  return 0;
}
  
/* @ 
    PCBSIterSolve - Solves a linear system using the BlockSolve iterative
    solvers instead of the usual SLES/KSP solvers.  

    Input Parameters:
.   pc - the PC context
.   b - right-hand-side vector
.   x - solution vector

    Output Parameter:
.   its - number of iterations until termination

    Notes:
    This routine is intended primarily for comparison with the SLES/KSP
    interface.  We recommend using the SLES interface for general use.
@ */
int PCBSIterSolve(PC pc,Vec b,Vec x,int *its)
{
/*  PC_ICC       *icc = (PC_ICC *) pc->data;
  PCiBS        *iccbs = (PCiBS *) icc->implctx; 
  Mat_MPIRowbs *amat = (Mat_MPIRowbs *) pc->mat->data;
  Scalar       *xa, *ba; */

  SETERRQ(1,"PCBSIterSolve: Currently out of commission.");
  /* Note: The vectors x and b are permuted within BSpar_solve */
/*
  if (amat != pc->pmat->data) SETERRQ(1,"PCBSIterSolve:Need same pre and matrix");
  if (pc->mat->type != MATMPIROWBS) SETERRQ(1,"PCBSIterSolve:MATMPIROWBS only");
  VecGetArray(b,&ba); VecGetArray(x,&xa);
  *its = BSpar_solve(iccbs->blocksize,amat->pA,amat->fpA,amat->comm_pA,ba,xa,
             iccbs->pre_option,iccbs->rtol,iccbs->max_it,&(iccbs->rnorm),
             iccbs->guess_zero,amat->procinfo); CHKERRQ(0);  
  MPIU_printf(pc->mat->comm,"method=%d, final residual = %e\n",
              iccbs->pre_option,iccbs->rnorm); 
  VecRestoreArray(b,&ba); VecRestoreArray(x,&xa);
*/
  return 0;
}

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

  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  if (pc->pmat->type != MATMPIROWBS) return 0;
  iccbs = (PCiBS *) icc->implctx;
  OptionsGetInt(pc->prefix,"-pc_bs_max_it",&iccbs->max_it);
  OptionsGetInt(pc->prefix,"-pc_bs_blocksize",&iccbs->blocksize);
  OptionsGetDouble(pc->prefix,"-pc_bs_rtol",&iccbs->rtol);
  if (OptionsHasName(pc->prefix,"-pc_bs_guess_zero")) 
    iccbs->guess_zero = 1;
/*  if (OptionsHasName(pc->prefix,"-pc_bs_ssor")) 
    iccbs->pre_option = PRE_SSOR;
  else if (OptionsHasName(pc->prefix,"-pc_bs_bjacobi")) 
    iccbs->pre_option = PRE_BJACOBI;
  else if (OptionsHasName(pc->prefix,"-pc_bs_diag")) 
    iccbs->pre_option = PRE_DIAG; */
  return 0;
}

/*@
   PCBSIterSetBlockSolve - Sets flag so that BlockSolve iterative solver is
   used instead of default KSP routines.  This routine should be called
   before PCSetUp().

   Input Parameter:
.  pc - the preconditioner context

   Note:
   This option is valid only when the MATMPIROWBS data structure
   is used for the preconditioning matrix.
@*/
int PCBSIterSetBlockSolve(PC pc)
{
  PC_ICC *icc = (PC_ICC *) pc->data;
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  SETERRQ(1,"PCBSIterSetBlockSolve: Not currently supported.");
  if (pc->setupcalled) SETERRQ(1,"PCBSIterSetBlockSolve:Must call before PCSetUp");
  if (pc->type != PCICC) return 0;
  icc->bs_iter = 1;
  return 0;
}

#else
int MatNull_MPIRowbs()
{return 0;}
#endif

