
#include <petsc-private/tsimpl.h>     /*I  "petscts.h"  I*/
EXTERN_C_BEGIN
extern PetscErrorCode  TSCreate_Euler(TS);
extern PetscErrorCode  TSCreate_BEuler(TS);
extern PetscErrorCode  TSCreate_Pseudo(TS);
extern PetscErrorCode  TSCreate_Sundials(TS);
extern PetscErrorCode  TSCreate_CN(TS);
extern PetscErrorCode  TSCreate_Theta(TS);
extern PetscErrorCode  TSCreate_Alpha(TS);
extern PetscErrorCode  TSCreate_GL(TS);
extern PetscErrorCode  TSCreate_SSP(TS);
extern PetscErrorCode  TSCreate_RK(TS);
extern PetscErrorCode  TSCreate_ARKIMEX(TS);
extern PetscErrorCode  TSCreate_RosW(TS);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "TSRegisterAll"
/*@C
  TSRegisterAll - Registers all of the timesteppers in the TS package. 

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: TS, timestepper, register, all
.seealso: TSCreate(), TSRegister(), TSRegisterDestroy(), TSRegisterDynamic()
@*/
PetscErrorCode  TSRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TSRegisterAllCalled = PETSC_TRUE;

  ierr = TSRegisterDynamic(TSEULER,           path, "TSCreate_Euler",    TSCreate_Euler);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TSBEULER,          path, "TSCreate_BEuler",   TSCreate_BEuler);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TSCN,              path, "TSCreate_CN",       TSCreate_CN);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TSPSEUDO,          path, "TSCreate_Pseudo",   TSCreate_Pseudo);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TSGL,              path, "TSCreate_GL",       TSCreate_GL);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TSSSP,             path, "TSCreate_SSP",      TSCreate_SSP);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TSTHETA,           path, "TSCreate_Theta",    TSCreate_Theta);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TSALPHA,           path, "TSCreate_Alpha",    TSCreate_Alpha);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SUNDIALS)
  ierr = TSRegisterDynamic(TSSUNDIALS,        path, "TSCreate_Sundials", TSCreate_Sundials);CHKERRQ(ierr);
#endif
  ierr = TSRegisterDynamic(TSRK,              path, "TSCreate_RK",       TSCreate_RK);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TSARKIMEX,         path, "TSCreate_ARKIMEX",  TSCreate_ARKIMEX);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TSROSW,            path, "TSCreate_RosW",     TSCreate_RosW);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

