/*
   Defines a  Jacobi preconditioner for any Mat implementation
*/
#include "pcimpl.h"

typedef struct {
  Vec diag;
} PCiJacobi;

int PCiJacobiSetup(PC pc)
{
  int ierr;
  PCiJacobi *jac = (PCiJacobi *) pc->data;
  Vec       diag;
  if (ierr = VecCreate(pc->vec,&diag)) SETERR(ierr,0);
  if (ierr = MatGetDiagonal(pc->mat,diag)) SETERR(ierr,0);
  if (ierr = VecReciprocal(diag)) SETERR(ierr,0);
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

int PCiJacobiCreate(PC pc)
{
  PCiJacobi *jac = NEW(PCiJacobi); CHKPTR(jac);
  jac->diag = 0;
  pc->apply = PCiJacobiApply;
  pc->setup = PCiJacobiSetup;
  pc->type  = PCJACOBI;
  pc->data  = (void *) jac;
  return 0;
}
