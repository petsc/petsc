#ifndef lint
static char vcid[] = "$Id: jacobi.c,v 1.10 1995/04/05 20:31:15 bsmith Exp bsmith $";
#endif
/*
   Defines a  Jacobi preconditioner for any Mat implementation
*/
#include "pcimpl.h"

typedef struct {
  Vec diag;
} PC_Jacobi;

int PCSetUp_Jacobi(PC pc)
{
  int       ierr;
  PC_Jacobi *jac = (PC_Jacobi *) pc->data;
  Vec       diag;
  if (pc->setupcalled == 0) {
    if ((ierr = VecDuplicate(pc->vec,&diag))) SETERR(ierr,0);
    PLogObjectParent(pc,diag);
  }
  else {
    diag = jac->diag;
  }
  if ((ierr = MatGetDiagonal(pc->pmat,diag))) SETERR(ierr,0);
  if ((ierr = VecReciprocal(diag))) SETERR(ierr,0);
  jac->diag = diag;
  return 0;
}

int PCApply_Jacobi(PC pc,Vec x,Vec y)
{
  PC_Jacobi *jac = (PC_Jacobi *) pc->data;
  Vec       diag = jac->diag;
  VecPMult(x,diag,y);
  return 0;
}

int PCDestroy_Jacobi(PetscObject obj)
{
  PC pc = (PC) obj;
  PC_Jacobi *jac = (PC_Jacobi *) pc->data;
  if (jac->diag) VecDestroy(jac->diag);
  FREE(jac);
  PLogObjectDestroy(pc);
  PETSCHEADERDESTROY(pc);
  return 0;
}

int PCCreate_Jacobi(PC pc)
{
  PC_Jacobi *jac = NEW(PC_Jacobi); CHKPTR(jac);
  jac->diag   = 0;
  pc->apply   = PCApply_Jacobi;
  pc->setup   = PCSetUp_Jacobi;
  pc->destroy = PCDestroy_Jacobi;
  pc->type    = PCJACOBI;
  pc->data    = (void *) jac;
  return 0;
}
