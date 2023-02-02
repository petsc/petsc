#include <petsc/private/vecimpl.h> /*I "petscvec.h" I*/

PetscClassId VEC_TAGGER_CLASSID;

static PetscBool VecTaggerPackageInitialized = PETSC_FALSE;

PetscBool VecTaggerRegisterAllCalled;

/*@C
   VecTaggerInitializePackage - Initialize VecTagger package

   Logically Collective

   Level: developer

.seealso: `VecTaggerFinalizePackage()`
@*/
PetscErrorCode VecTaggerInitializePackage(void)
{
  PetscFunctionBegin;
  if (VecTaggerPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  VecTaggerPackageInitialized = PETSC_TRUE;

  PetscCall(PetscClassIdRegister("Vector Indices Tagger", &VEC_TAGGER_CLASSID));
  PetscCall(VecTaggerRegisterAll());
  PetscCall(PetscRegisterFinalize(VecTaggerFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   VecTaggerFinalizePackage - Finalize VecTagger package, it is called from PetscFinalize()

   Logically Collective

   Level: developer

.seealso: `VecTaggerInitializePackage()`
@*/
PetscErrorCode VecTaggerFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&VecTaggerList));
  VecTaggerPackageInitialized = PETSC_FALSE;
  VecTaggerRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
