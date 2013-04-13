#include <private/tsimpl.h>

#define TSSolve(ts,u) TSSolve((ts),(u),PETSC_NULL)

#define TSROSW "rosw"

#undef __FUNCT__
#define __FUNCT__ "TSSetConvergedReason"
static PetscErrorCode TSSetConvergedReason(TS ts,TSConvergedReason reason)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->reason = reason;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetSNESIterations"
static PetscErrorCode TSGetSNESIterations(TS ts,PetscInt *nits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(nits,2);
  *nits = ts->nonlinear_its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetKSPIterations"
static PetscErrorCode TSGetKSPIterations(TS ts,PetscInt *lits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(lits,2);
  *lits = ts->linear_its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetStepRejections"
static PetscErrorCode TSGetStepRejections(TS ts,PetscInt *rejects)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(rejects,2);
  *rejects = ts->reject;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "TSGetSNESFailures"
static PetscErrorCode TSGetSNESFailures(TS ts,PetscInt *fails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(fails,2);
  *fails = ts->num_snes_failures;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetMaxStepRejections"
static PetscErrorCode TSSetMaxStepRejections(TS ts,PetscInt rejects)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->max_reject = rejects;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetMaxSNESFailures"
static PetscErrorCode TSSetMaxSNESFailures(TS ts,PetscInt fails)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->max_snes_failures = fails;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetErrorIfStepFails()"
static PetscErrorCode TSSetErrorIfStepFails(TS ts,PetscBool err)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ts->errorifstepfailed = err;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetTolerances"
static PetscErrorCode TSSetTolerances(TS ts,PetscReal atol,Vec vatol,PetscReal rtol,Vec vrtol)
{

  PetscFunctionBegin;
  SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_SUP,"Upgrade PETSc to 3.3 or later for adaptive time stepping");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetTolerances"
static PetscErrorCode TSGetTolerances(TS ts,PetscReal *atol,Vec *vatol,PetscReal *rtol,Vec *vrtol)
{

  PetscFunctionBegin;
  SETERRQ(((PetscObject)ts)->comm,PETSC_ERR_SUP,"Upgrade PETSc to 3.3 or later for adaptive time stepping");
  PetscFunctionReturn(0);
}
