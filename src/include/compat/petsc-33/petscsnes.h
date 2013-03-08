#include "petsc-private/snesimpl.h"

#define SNESNEWTONLS     SNESLS
#define SNESNEWTONTR     SNESTR
#define SNESVINEWTONRSLS SNESVIRS
#define SNESVINEWTONSSLS SNESVISS

#define SNESConvergedDefault SNESDefaultConverged

#undef __FUNCT__
#define __FUNCT__ "SNESLogConvergenceHistory"
static PetscErrorCode
SNESLogConvergenceHistory(SNES snes, PetscReal fnorm, PetscInt lits)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  SNESLogConvHistory(snes,fnorm,lits);
  PetscFunctionReturn(0);
}
