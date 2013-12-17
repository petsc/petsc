#include <petsc-private/sfimpl.h>

PetscClassId PETSCSF_CLASSID;

static PetscBool PetscSFPackageInitialized = PETSC_FALSE;

PetscBool PetscSFRegisterAllCalled;

#undef __FUNCT__
#define __FUNCT__ "PetscSFInitializePackage"
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

  ierr = PetscClassIdRegister("Bipartite Graph",&PETSCSF_CLASSID);CHKERRQ(ierr);
  ierr = PetscSFRegisterAll();CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PetscSFSetGraph"     , PETSCSF_CLASSID, &PETSCSF_SetGraph);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PetscSFBcastBegin"   , PETSCSF_CLASSID, &PETSCSF_BcastBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PetscSFBcastEnd"     , PETSCSF_CLASSID, &PETSCSF_BcastEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PetscSFReduceBegin"  , PETSCSF_CLASSID, &PETSCSF_ReduceBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PetscSFReduceEnd"    , PETSCSF_CLASSID, &PETSCSF_ReduceEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PetscSFFetchOpBegin" , PETSCSF_CLASSID, &PETSCSF_FetchAndOpBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PetscSFFetchOpEnd"   , PETSCSF_CLASSID, &PETSCSF_FetchAndOpEnd);CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscSFFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSFFinalizePackage"
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
