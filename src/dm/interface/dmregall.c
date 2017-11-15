
#include <petscdm.h>     /*I  "petscdm.h"  I*/
PETSC_EXTERN PetscErrorCode DMCreate_DA(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Composite(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Sliced(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Shell(DM);
PETSC_EXTERN PetscErrorCode DMCreate_ADDA(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Redundant(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Plex(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Patch(DM);
#if defined(PETSC_HAVE_SIEVE)
PETSC_EXTERN PetscErrorCode DMCreate_Mesh(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Cartesian(DM);
#endif
PETSC_EXTERN PetscErrorCode  DMCreate_Moab(DM);

#undef __FUNCT__
#define __FUNCT__ "DMRegisterAll"
/*@C
  DMRegisterAll - Registers all of the DM components in the DM package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: DM, register, all
.seealso:  DMRegister(), DMRegisterDestroy()
@*/
PetscErrorCode  DMRegisterAll()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  DMRegisterAllCalled = PETSC_TRUE;

  ierr = DMRegister(DMDA,         DMCreate_DA);CHKERRQ(ierr);
  ierr = DMRegister(DMCOMPOSITE,  DMCreate_Composite);CHKERRQ(ierr);
  ierr = DMRegister(DMSLICED,     DMCreate_Sliced);CHKERRQ(ierr);
  ierr = DMRegister(DMSHELL,      DMCreate_Shell);CHKERRQ(ierr);
  ierr = DMRegister(DMADDA,       DMCreate_ADDA);CHKERRQ(ierr);
  ierr = DMRegister(DMREDUNDANT,  DMCreate_Redundant);CHKERRQ(ierr);
  ierr = DMRegister(DMPLEX,       DMCreate_Plex);CHKERRQ(ierr);
  ierr = DMRegister(DMPATCH,      DMCreate_Patch);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SIEVE)
  ierr = DMRegister(DMMESH,       DMCreate_Mesh);CHKERRQ(ierr);
  ierr = DMRegister(DMCARTESIAN,  DMCreate_Cartesian);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MOAB)
  ierr = DMRegister(DMMOAB,       DMCreate_Moab);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

