#include "petsc-private/kspimpl.h"

#define KSP_DIVERGED_NANORINF KSP_DIVERGED_NAN

#undef __FUNCT__
#define __FUNCT__ "KSPLogResidualHistory_Compat"
static PetscErrorCode
KSPLogResidualHistory_Compat(KSP ksp,PetscReal norm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,1);
  KSPLogResidualHistory(ksp,norm);
  PetscFunctionReturn(0);
}
#undef  KSPLogResidualHistory
#define KSPLogResidualHistory KSPLogResidualHistory_Compat
