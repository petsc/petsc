#ifndef lint
static char vcid[] = "$Id: jacobi.c,v 1.7 1995/03/17 04:56:22 bsmith Exp bsmith $";
#endif
/*
   Defines a  Jacobi preconditioner for any Mat implementation
*/
#include "pcimpl.h"

typedef struct {
  Vec diag;
} PCiJacobi;

int PCiJacobiSetup(PC pc)
{
  int       ierr;
  PCiJacobi *jac = (PCiJacobi *) pc->data;
  Vec       diag;
  if (pc->setupcalled == 0) {
    if ((ierr = VecCreate(pc->vec,&diag))) SETERR(ierr,0);
  }
  else {
    diag = jac->diag;
  }
  if ((ierr = MatGetDiagonal(pc->pmat,diag))) SETERR(ierr,0);
  if ((ierr = VecReciprocal(diag))) SETERR(ierr,0);
  jac->diag = diag;
  return 0;
}

int PCiJacobiApply(PC pc,Vec x,Vec y)
{
  PCiJacobi *jac = (PCiJacobi *) pc->data;
  Vec       diag = jac->diag;
  VecPMult(x,diag,y);
  return 0;
}

int PCiJacobiDestroy(PetscObject obj)
{
  PC pc = (PC) obj;
  PCiJacobi *jac = (PCiJacobi *) pc->data;
  if (jac->diag) VecDestroy(jac->diag);
  FREE(jac);
  PLogObjectDestroy(pc);
  PETSCHEADERDESTROY(pc);
  return 0;
}

int PCiJacobiCreate(PC pc)
{
  PCiJacobi *jac = NEW(PCiJacobi); CHKPTR(jac);
  jac->diag = 0;
  pc->apply = PCiJacobiApply;
  pc->setup = PCiJacobiSetup;
  pc->destroy = PCiJacobiDestroy;
  pc->type  = PCJACOBI;
  pc->data  = (void *) jac;
  return 0;
}
