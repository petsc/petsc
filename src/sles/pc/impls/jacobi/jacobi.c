#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: jacobi.c,v 1.36 1997/11/18 19:29:56 bsmith Exp bsmith $";
#endif
/*
   Defines a  Jacobi preconditioner for any Mat implementation
*/
#include "src/pc/pcimpl.h"   /*I "pc.h" I*/
#include <math.h>

typedef struct {
  Vec diag;
  Vec diagsqrt;
} PC_Jacobi;

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
    diag = jac->diag;
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
    PLogInfo(pc,"WARNING: Zero detected in diagonal while building Jacobi preconditioner\n");
  }
  PetscFunctionReturn(0);
}

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

#undef __FUNC__  
#define __FUNC__ "PCApplySymmetricLeftOrRight_Jacobi"
static int PCApplySymmetricLeftOrRight_Jacobi(PC pc,Vec x,Vec y)
{
  PC_Jacobi *jac = (PC_Jacobi *) pc->data;

  PetscFunctionBegin;
  VecPointwiseMult(x,jac->diagsqrt,y);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCDestroy_Jacobi"
static int PCDestroy_Jacobi(PetscObject obj)
{
  PC        pc = (PC) obj;
  PC_Jacobi *jac = (PC_Jacobi *) pc->data;
  int       ierr;

  PetscFunctionBegin;
  if (jac->diag)     {ierr = VecDestroy(jac->diag);CHKERRQ(ierr);}
  if (jac->diagsqrt) {ierr = VecDestroy(jac->diagsqrt);CHKERRQ(ierr);}
  PetscFree(jac);
  PetscFunctionReturn(0);
}

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
  pc->type           = PCJACOBI;
  pc->data           = (void *) jac;
  pc->view           = 0;
  pc->applyrich      = 0;
  pc->applysymmetricleft  = PCApplySymmetricLeftOrRight_Jacobi;
  pc->applysymmetricright = PCApplySymmetricLeftOrRight_Jacobi;
  PetscFunctionReturn(0);
}


