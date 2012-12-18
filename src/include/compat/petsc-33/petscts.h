#include "petsc-private/tsimpl.h"

#undef __FUNCT__
#define __FUNCT__ "TSSetConvergedReason"
static PetscErrorCode
TSSetConvergedReason(TS ts,TSConvergedReason reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->reason = reason;
  PetscFunctionReturn(0);
}

#define TSSolve(ts,u) TSSolve((ts),(u),PETSC_NULL)
