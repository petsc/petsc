#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: jacobi.c,v 1.40 1998/04/03 23:14:09 bsmith Exp curfman $";
#endif
/*

/*  -------------------------------------------------------------------- 

     This file implements a Jacobi preconditioner for any implementation 
     of the preconditioner matrix, A, that uses the Mat interface.

     The following basic routines are required for each preconditioner.
          PCCreate_XXX()          - Creates a preconditioner context
          PCSetFromOptions_XXX()  - Sets runtime options
          PCApply_XXX()           - Applies the preconditioner
          PCDestroy_XXX()         - Destroys the preconditioner context
     where the suffix "_XXX" denotes a particular implementation, in
     this case we use _Jacobi (e.g., PCCreate_Jacobi, PCApply_Jacobi).
     These routines are actually called via the common user interface
     routines PCCreate(), PCSetFromOptions(), PCApply(), and PCDestroy(), 
     so the application code interface remains identical for all 
     preconditioners.  

     Another key routine is:
          PCSetUp_XXX()           - Prepares for the use of a preconditioner
     by setting data structures and options.   The interface routine PCSetUp()
     is not usually called directly by the user, but instead is called by
     PCApply() if necessary.

     Additional basic routines are:
          PCPrintHelp_XXX()       - Prints preconditioner runtime options
          PCView_XXX()            - Prints details of runtime options that
                                    have actually been used.
     These are called by application codes via the interface routines
     PCPrintHelp() and PCView().

     The various types of solvers (preconditioners, Krylov subspace methods,
     nonlinear solvers, timesteppers) are all organized similarly, so the
     above description applies to these categories also.  One exception is
     that the analogues of PCApply() for these components are KSPSolve(), 
     SNESSolve(), and TSSolve().

     Additional optional functionality unique to preconditioners is left and
     right symmetric preconditioner application via PCApplySymmetricLeft() 
     and PCApplySymmetricRight().  The Jacobi implementation is 
     PCApplySymmetricLeftOrRight_Jacobi().

    -------------------------------------------------------------------- */

/* 
   Include files needed for the Jacobi preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners 
*/

#include "src/pc/pcimpl.h"   /*I "pc.h" I*/
#include <math.h>

/* 
   Private context for the Jacobi preconditioner.  
 */
typedef struct {
  Vec diag;      /* vector containing the reciprocal of the diagonal of the
                    preconditioner matrix */
  Vec diagsqrt;  /* vector containing the square root of the reciprocal of 
                    the diagonal of the preconditioner matrix (used only for
                    symmetric preconditioner application) */
} PC_Jacobi;

/* -------------------------------------------------------------------------- */
/*
   PCSetUp_Jacobi - Prepares for the use of the Jacobi preconditioner
   by setting data structures and options.   

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
#undef __FUNC__  
#define __FUNC__ "PCSetUp_Jacobi"
static int PCSetUp_Jacobi(PC pc)
{
  int        ierr, i, n,zeroflag = 0;
  PC_Jacobi  *jac = (PC_Jacobi *) pc->data;
  Vec        diag, diagsqrt;
  Scalar     *x;

  PetscFunctionBegin;
  /* We set up both regular and symmetric preconditioning. Perhaps there
     actually should be an option to use only one or the other? */
  if (pc->setupcalled == 0) {
    ierr = VecDuplicate(pc->vec,&diag); CHKERRQ(ierr);
    PLogObjectParent(pc,diag);
    ierr = VecDuplicate(pc->vec,&diagsqrt); CHKERRQ(ierr);
    PLogObjectParent(pc,diagsqrt);
  } else {
    diag     = jac->diag;
    diagsqrt = jac->diagsqrt;
  }
  ierr = MatGetDiagonal(pc->pmat,diag); CHKERRQ(ierr);
  ierr = VecCopy(diag,diagsqrt); CHKERRQ(ierr);
  ierr = VecReciprocal(diag); CHKERRQ(ierr);
  ierr = VecGetLocalSize(diag,&n); CHKERRQ(ierr);
  ierr = VecGetArray(diag,&x); CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    if (x[i] == 0.0) {
      x[i]     = 1.0;
      zeroflag = 1;
    }
  }
  ierr = VecRestoreArray(diag,&x); CHKERRQ(ierr);
  ierr = VecGetArray(diagsqrt,&x); CHKERRQ(ierr);
  for ( i=0; i<n; i++ ) {
    if (x[i] != 0.0) x[i] = 1.0/sqrt(PetscAbsScalar(x[i]));
    else x[i] = 1.0;
  }
  jac->diag     = diag;
  jac->diagsqrt = diagsqrt;

  if (zeroflag) {
    PLogInfo(pc,"PCSetUp_Jacobi:WARNING: Zero detected in diagonal while building Jacobi preconditioner\n");
  }
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCApply_Jacobi - Applies the Jacobi preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routine: PCApply()
 */
#undef __FUNC__  
#define __FUNC__ "PCApply_Jacobi"
static int PCApply_Jacobi(PC pc,Vec x,Vec y)
{
  PC_Jacobi *jac = (PC_Jacobi *) pc->data;
  int       ierr;

  PetscFunctionBegin;
  ierr = VecPointwiseMult(x,jac->diag,y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCApplySymmetricLeftOrRight_Jacobi - Applies the left or right part of a
   symmetric preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routines: PCApplySymmetricLeft(), PCApplySymmetricRight()
*/
#undef __FUNC__  
#define __FUNC__ "PCApplySymmetricLeftOrRight_Jacobi"
static int PCApplySymmetricLeftOrRight_Jacobi(PC pc,Vec x,Vec y)
{
  PC_Jacobi *jac = (PC_Jacobi *) pc->data;

  PetscFunctionBegin;
  VecPointwiseMult(x,jac->diagsqrt,y);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCDestroy_Jacobi - Destroys the private context for the Jacobi preconditioner
   that was created with PCCreate_Jacobi().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
#undef __FUNC__  
#define __FUNC__ "PCDestroy_Jacobi"
static int PCDestroy_Jacobi(PC pc)
{
  PC_Jacobi *jac = (PC_Jacobi *) pc->data;
  int       ierr;

  PetscFunctionBegin;
  if (jac->diag)     {ierr = VecDestroy(jac->diag);CHKERRQ(ierr);}
  if (jac->diagsqrt) {ierr = VecDestroy(jac->diagsqrt);CHKERRQ(ierr);}
  PetscFree(jac);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCCreate_Jacobi - Creates a Jacobi preconditioner context, PC_Jacobi, 
   and sets this as the private data within the generic preconditioning 
   context, PC, that was created within PCCreate().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCCreate()
*/
#undef __FUNC__  
#define __FUNC__ "PCCreate_Jacobi"
int PCCreate_Jacobi(PC pc)
{
  PC_Jacobi *jac = PetscNew(PC_Jacobi); CHKPTRQ(jac);

  PetscFunctionBegin;
  PLogObjectMemory(pc,sizeof(PC_Jacobi));

  jac->diag          = 0;
  pc->apply          = PCApply_Jacobi;
  pc->setup          = PCSetUp_Jacobi;
  pc->destroy        = PCDestroy_Jacobi;
  pc->data           = (void *) jac;
  pc->view           = 0;
  pc->applyrich      = 0;
  pc->applysymmetricleft  = PCApplySymmetricLeftOrRight_Jacobi;
  pc->applysymmetricright = PCApplySymmetricLeftOrRight_Jacobi;
  PetscFunctionReturn(0);
}


