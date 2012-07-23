#include "private/snesimpl.h"

#define SNESNRICHARDSON  "nrichardson"
#define SNESVIRS         "virs"
#define SNESVISS         "viss"
#define SNESQN           "qn"
#define SNESSHELL        "shell"
#define SNESGS           "gs"
#define SNESNCG          "ncg"
#define SNESFAS          "fas"
#define SNESMS           "ms"

#define SNES_CONVERGED_SNORM_RELATIVE SNES_CONVERGED_FNORM_RELATIVE
#define SNES_DIVERGED_INNER           SNES_DIVERGED_FUNCTION_DOMAIN

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
