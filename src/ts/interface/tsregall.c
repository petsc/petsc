#define PETSCTS_DLL

#include "src/ts/tsimpl.h"     /*I  "petscts.h"  I*/
EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSCreate_Euler(TS);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSCreate_BEuler(TS);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSCreate_Pseudo(TS);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSCreate_PVode(TS);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSCreate_CN(TS);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSCreate_Rk(TS);
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
PetscErrorCode PETSCTS_DLLEXPORT TSRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TSRegisterAllCalled = PETSC_TRUE;

  ierr = TSRegisterDynamic(TS_EULER,           path, "TSCreate_Euler", TSCreate_Euler);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TS_BEULER,          path, "TSCreate_BEuler",TSCreate_BEuler);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TS_CRANK_NICHOLSON, path, "TSCreate_CN", TSCreate_CN);CHKERRQ(ierr);
  ierr = TSRegisterDynamic(TS_PSEUDO,          path, "TSCreate_Pseudo", TSCreate_Pseudo);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PVODE)
  ierr = TSRegisterDynamic(TS_PVODE,           path, "TSCreate_PVode", TSCreate_PVode);CHKERRQ(ierr);
#endif
  ierr = TSRegisterDynamic(TS_RUNGE_KUTTA,     path, "TSCreate_Rk", TSCreate_Rk);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

