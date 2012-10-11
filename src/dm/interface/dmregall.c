
#include <petscdm.h>     /*I  "petscdm.h"  I*/
EXTERN_C_BEGIN
extern PetscErrorCode  DMCreate_DA(DM);
extern PetscErrorCode  DMCreate_Composite(DM);
extern PetscErrorCode  DMCreate_Sliced(DM);
extern PetscErrorCode  DMCreate_Shell(DM);
extern PetscErrorCode  DMCreate_ADDA(DM);
extern PetscErrorCode  DMCreate_Redundant(DM);
extern PetscErrorCode  DMCreate_Complex(DM);
extern PetscErrorCode  DMCreate_Patch(DM);
#ifdef PETSC_HAVE_SIEVE
extern PetscErrorCode  DMCreate_Mesh(DM);
extern PetscErrorCode  DMCreate_Cartesian(DM);
#endif
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "DMRegisterAll"
/*@C
  DMRegisterAll - Registers all of the DM components in the DM package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: DM, register, all
.seealso:  DMRegister(), DMRegisterDestroy(), DMRegisterDynamic()
@*/
PetscErrorCode  DMRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  DMRegisterAllCalled = PETSC_TRUE;
  ierr = DMRegisterDynamic(DMDA,        path, "DMCreate_DA",        DMCreate_DA);CHKERRQ(ierr);
  ierr = DMRegisterDynamic(DMCOMPOSITE, path, "DMCreate_Composite", DMCreate_Composite);CHKERRQ(ierr);
  ierr = DMRegisterDynamic(DMSLICED,    path, "DMCreate_Sliced",    DMCreate_Sliced);CHKERRQ(ierr);
  ierr = DMRegisterDynamic(DMSHELL,     path, "DMCreate_Shell",     DMCreate_Shell);CHKERRQ(ierr);
  ierr = DMRegisterDynamic(DMADDA,      path, "DMCreate_ADDA",      DMCreate_ADDA);CHKERRQ(ierr);
  ierr = DMRegisterDynamic(DMREDUNDANT, path, "DMCreate_Redundant", DMCreate_Redundant);CHKERRQ(ierr);
  ierr = DMRegisterDynamic(DMCOMPLEX,   path, "DMCreate_Complex",   DMCreate_Complex);CHKERRQ(ierr);
  ierr = DMRegisterDynamic(DMPATCH,     path, "DMCreate_Patch",     DMCreate_Patch);CHKERRQ(ierr);
#ifdef PETSC_HAVE_SIEVE
  ierr = DMRegisterDynamic(DMMESH,      path, "DMCreate_Mesh",      DMCreate_Mesh);CHKERRQ(ierr);
  ierr = DMRegisterDynamic(DMCARTESIAN, path, "DMCreate_Cartesian", DMCreate_Cartesian);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

