#include "petsc-private/tsimpl.h"

#define TSEIMEX "eimex"

typedef enum {
  TS_EQ_UNSPECIFIED               = -1,
  TS_EQ_EXPLICIT                  = 0,
  TS_EQ_ODE_EXPLICIT              = 1,
  TS_EQ_DAE_SEMI_EXPLICIT_INDEX1  = 100,
  TS_EQ_DAE_SEMI_EXPLICIT_INDEX2  = 200,
  TS_EQ_DAE_SEMI_EXPLICIT_INDEX3  = 300,
  TS_EQ_DAE_SEMI_EXPLICIT_INDEXHI = 500,
  TS_EQ_IMPLICIT                  = 1000,
  TS_EQ_ODE_IMPLICIT              = 1001,
  TS_EQ_DAE_IMPLICIT_INDEX1       = 1100,
  TS_EQ_DAE_IMPLICIT_INDEX2       = 1200,
  TS_EQ_DAE_IMPLICIT_INDEX3       = 1300,
  TS_EQ_DAE_IMPLICIT_INDEXHI      = 1500
} TSEquationType;

#undef  __FUNCT__
#define __FUNCT__ "TSGetEquationType"
static PetscErrorCode
TSGetEquationType(TS ts,TSEquationType* et)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(PETSC_ERR_SUP);
}
#undef  __FUNCT__
#define __FUNCT__ "TSSetEquationType"
static PetscErrorCode
TSSetEquationType(TS ts,TSEquationType et)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(PETSC_ERR_SUP);
}

#undef  __FUNCT__
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
#define TSGetSolveTime(ts,t) TSGetTime(ts,t)
