
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
  ierr = DMRegisterDynamic(DMPLEX,      path, "DMCreate_Plex",      DMCreate_Plex);CHKERRQ(ierr);
  ierr = DMRegisterDynamic(DMPATCH,     path, "DMCreate_Patch",     DMCreate_Patch);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SIEVE)
  ierr = DMRegisterDynamic(DMMESH,      path, "DMCreate_Mesh",      DMCreate_Mesh);CHKERRQ(ierr);
  ierr = DMRegisterDynamic(DMCARTESIAN, path, "DMCreate_Cartesian", DMCreate_Cartesian);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MOAB)
  ierr = DMRegisterDynamic(DMMOAB,      path, "DMCreate_Moab",      DMCreate_Moab);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

