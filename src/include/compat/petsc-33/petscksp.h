#include "petsc-private/kspimpl.h"

#undef __FUNCT__
#define __FUNCT__ "KSPLogResidualHistory_Compat"
static PetscErrorCode
KSPLogResidualHistory_Compat(KSP ksp,PetscReal norm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  KSPLogResidualHistory(ksp,rnorm);
  PetscFunctionReturn(0);
}
#undef  KSPLogResidualHistory
#define KSPLogResidualHistory KSPLogResidualHistory_Compat
