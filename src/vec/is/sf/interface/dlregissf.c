#include <petsc/private/sfimpl.h>

PetscClassId PETSCSF_CLASSID;

static PetscBool PetscSFPackageInitialized = PETSC_FALSE;

PetscBool PetscSFRegisterAllCalled;

/*@C
   PetscSFInitializePackage - Initialize SF package

   Logically Collective

   Level: developer

.seealso: PetscSFFinalizePackage()
@*/
PetscErrorCode PetscSFInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscSFPackageInitialized) PetscFunctionReturn(0);
  PetscSFPackageInitialized = PETSC_TRUE;

  ierr = PetscClassIdRegister("Star Forest Graph",&PETSCSF_CLASSID);CHKERRQ(ierr);
  ierr = PetscSFRegisterAll();CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SFSetGraph"     , PETSCSF_CLASSID, &PETSCSF_SetGraph);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SFBcastBegin"   , PETSCSF_CLASSID, &PETSCSF_BcastBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SFBcastEnd"     , PETSCSF_CLASSID, &PETSCSF_BcastEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SFReduceBegin"  , PETSCSF_CLASSID, &PETSCSF_ReduceBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SFReduceEnd"    , PETSCSF_CLASSID, &PETSCSF_ReduceEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SFFetchOpBegin" , PETSCSF_CLASSID, &PETSCSF_FetchAndOpBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("SFFetchOpEnd"   , PETSCSF_CLASSID, &PETSCSF_FetchAndOpEnd);CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscSFFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscSFFinalizePackage - Finalize PetscSF package, it is called from PetscFinalize()

   Logically Collective

   Level: developer

.seealso: PetscSFInitializePackage()
@*/
PetscErrorCode PetscSFFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFunctionListDestroy(&PetscSFList);CHKERRQ(ierr);
  PetscSFPackageInitialized = PETSC_FALSE;
  PetscSFRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}
