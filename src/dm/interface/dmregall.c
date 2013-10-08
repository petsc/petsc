
#include <petscdm.h>     /*I  "petscdm.h"  I*/
PETSC_EXTERN PetscErrorCode DMCreate_DA(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Composite(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Sliced(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Shell(DM);
PETSC_EXTERN PetscErrorCode DMCreate_ADDA(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Redundant(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Plex(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Patch(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Moab(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Circuit(DM);

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
#if defined(PETSC_HAVE_MOAB)
  ierr = DMRegister(DMMOAB,       DMCreate_Moab);CHKERRQ(ierr);
#endif
  ierr = DMRegister(DMCIRCUIT,    DMCreate_Circuit);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#include <petscfe.h>     /*I  "petscfe.h"  I*/

PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Polynomial(PetscSpace);
PETSC_EXTERN PetscErrorCode PetscSpaceCreate_DG(PetscSpace);

#undef __FUNCT__
#define __FUNCT__ "PetscSpaceRegisterAll"
/*@C
  PetscSpaceRegisterAll - Registers all of the PetscSpace components in the PetscFE package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: PetscSpace, register, all
.seealso:  PetscSpaceRegister(), PetscSpaceRegisterDestroy()
@*/
PetscErrorCode PetscSpaceRegisterAll()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscSpaceRegisterAllCalled = PETSC_TRUE;

  ierr = PetscSpaceRegister(PETSCSPACEPOLYNOMIAL, PetscSpaceCreate_Polynomial);CHKERRQ(ierr);
  ierr = PetscSpaceRegister(PETSCSPACEDG,         PetscSpaceCreate_DG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscDualSpaceCreate_Lagrange(PetscDualSpace);

#undef __FUNCT__
#define __FUNCT__ "PetscDualSpaceRegisterAll"
/*@C
  PetscDualSpaceRegisterAll - Registers all of the PetscDualSpace components in the PetscFE package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: PetscDualSpace, register, all
.seealso:  PetscDualSpaceRegister(), PetscDualSpaceRegisterDestroy()
@*/
PetscErrorCode PetscDualSpaceRegisterAll()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscDualSpaceRegisterAllCalled = PETSC_TRUE;

  ierr = PetscDualSpaceRegister(PETSCDUALSPACELAGRANGE, PetscDualSpaceCreate_Lagrange);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscFECreate_Basic(PetscFE);
#ifdef PETSC_HAVE_OPENCL
PETSC_EXTERN PetscErrorCode PetscFECreate_OpenCL(PetscFE);
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscFERegisterAll"
/*@C
  PetscFERegisterAll - Registers all of the PetscFE components in the PetscFE package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: PetscFE, register, all
.seealso:  PetscFERegister(), PetscFERegisterDestroy()
@*/
PetscErrorCode PetscFERegisterAll()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFERegisterAllCalled = PETSC_TRUE;

  ierr = PetscFERegister(PETSCFEBASIC,  PetscFECreate_Basic);CHKERRQ(ierr);
#ifdef PETSC_HAVE_OPENCL
  ierr = PetscFERegister(PETSCFEOPENCL, PetscFECreate_OpenCL);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
