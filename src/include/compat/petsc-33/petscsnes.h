#include "petsc-private/snesimpl.h"

#define SNESNEWTONLS     SNESLS
#define SNESNEWTONTR     SNESTR
#define SNESVINEWTONRSLS SNESVIRS
#define SNESVINEWTONSSLS SNESVISS
#define SNESNASM         "nasm"
#define SNESANDERSON     "anderson"
#define SNESASPIN        "aspin"

#define SNESConvergedDefault SNESDefaultConverged

#undef  __FUNCT__
#define __FUNCT__ "SNESLogConvergenceHistory"
static PetscErrorCode
SNESLogConvergenceHistory(SNES snes, PetscReal fnorm, PetscInt lits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  SNESLogConvHistory(snes,fnorm,lits);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "SNESSetObjective"
static PetscErrorCode
SNESSetObjective(SNES snes,PetscErrorCode (*SNESObjectiveFunction)(SNES,Vec,PetscReal *,void*),void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(PETSC_ERR_SUP);
}

#undef  __FUNCT__
#define __FUNCT__ "SNESGetObjective"
static PetscErrorCode
SNESGetObjective(SNES snes,PetscErrorCode (**SNESObjectiveFunction)(SNES,Vec,PetscReal *,void*),void **ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(PETSC_ERR_SUP);
}

#undef  __FUNCT__
#define __FUNCT__ "SNESComputeObjective"
static PetscErrorCode SNESComputeObjective(SNES snes,Vec x,PetscReal *obj)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,__FUNCT__"() not supported in this PETSc version");
  PetscFunctionReturn(PETSC_ERR_SUP);
}
