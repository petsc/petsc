#ifndef lint
static char vcid[] = "$Id: jacobi.c,v 1.16 1995/08/04 01:51:45 bsmith Exp bsmith $";
#endif
/*
   Defines a  Jacobi preconditioner for any Mat implementation
*/
#include "pcimpl.h"

typedef struct {
  Vec diag;
} PC_Jacobi;

static int PCSetUp_Jacobi(PC pc)
{
  int       ierr;
  PC_Jacobi *jac = (PC_Jacobi *) pc->data;
  Vec       diag;
  if (pc->setupcalled == 0) {
    ierr = VecDuplicate(pc->vec,&diag); CHKERRQ(ierr);
    PLogObjectParent(pc,diag);
  }
  else {
    diag = jac->diag;
  }
  ierr = MatGetDiagonal(pc->pmat,diag); CHKERRQ(ierr);
  ierr = VecReciprocal(diag); CHKERRQ(ierr);
  jac->diag = diag;
  return 0;
}

static int PCApply_Jacobi(PC pc,Vec x,Vec y)
{
  PC_Jacobi *jac = (PC_Jacobi *) pc->data;
  Vec       diag = jac->diag;
  VecPMult(x,diag,y);
  return 0;
}

static int PCDestroy_Jacobi(PetscObject obj)
{
  PC pc = (PC) obj;
  PC_Jacobi *jac = (PC_Jacobi *) pc->data;
  if (jac->diag) VecDestroy(jac->diag);
  PetscFree(jac);
  return 0;
}

int PCCreate_Jacobi(PC pc)
{
  PC_Jacobi *jac = PetscNew(PC_Jacobi); CHKPTRQ(jac);
  jac->diag   = 0;
  pc->apply   = PCApply_Jacobi;
  pc->setup   = PCSetUp_Jacobi;
  pc->destroy = PCDestroy_Jacobi;
  pc->type    = PCJACOBI;
  pc->data    = (void *) jac;
  pc->view    = 0;
  return 0;
}
