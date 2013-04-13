#include "private/snesimpl.h"

#define SNESNEWTONLS     SNESLS
#define SNESNEWTONTR     SNESTR


#define SNESNRICHARDSON  "nrichardson"
#define SNESVINEWTONRSLS "virs"
#define SNESVINEWTONSSLS "viss"
#define SNESQN           "qn"
#define SNESSHELL        "shell"
#define SNESGS           "gs"
#define SNESNCG          "ncg"
#define SNESFAS          "fas"
#define SNESMS           "ms"
#define SNESNASM         "nasm"
#define SNESANDERSON     "anderson"
#define SNESASPIN        "aspin"

#define SNESConvergedDefault SNESDefaultConverged

#define SNES_CONVERGED_SNORM_RELATIVE SNES_CONVERGED_FNORM_RELATIVE
#define SNES_DIVERGED_INNER           SNES_DIVERGED_FUNCTION_DOMAIN

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

#undef __FUNCT__
#define __FUNCT__ "SNESSetIterationNumber"
static PetscErrorCode
SNESSetIterationNumber(SNES snes, PetscInt its)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = its;
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFunctionNorm"
static PetscErrorCode
SNESSetFunctionNorm(SNES snes, PetscReal fnorm)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->norm = fnorm;
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
