
#include <petscdm.h>     /*I  "petscdm.h"  I*/
#include <petscdmplex.h> /*I  "petscdmplex.h"  I*/
#include <petsc/private/dmimpl.h>
#include <petsc/private/dmpleximpl.h>
#include <petsc/private/petscfeimpl.h>
#include <petsc/private/petscfvimpl.h>
#include <petsc/private/petscdsimpl.h>
PETSC_EXTERN PetscErrorCode DMCreate_DA(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Composite(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Sliced(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Shell(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Redundant(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Plex(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Patch(DM);
#if defined(PETSC_HAVE_MOAB)
PETSC_EXTERN PetscErrorCode DMCreate_Moab(DM);
#endif
PETSC_EXTERN PetscErrorCode DMCreate_Network(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Forest(DM);
#if defined(PETSC_HAVE_P4EST)
PETSC_EXTERN PetscErrorCode DMCreate_p4est(DM);
PETSC_EXTERN PetscErrorCode DMCreate_p8est(DM);
#endif

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
  if (DMRegisterAllCalled) PetscFunctionReturn(0);
  DMRegisterAllCalled = PETSC_TRUE;

  ierr = DMRegister(DMDA,         DMCreate_DA);CHKERRQ(ierr);
  ierr = DMRegister(DMCOMPOSITE,  DMCreate_Composite);CHKERRQ(ierr);
  ierr = DMRegister(DMSLICED,     DMCreate_Sliced);CHKERRQ(ierr);
  ierr = DMRegister(DMSHELL,      DMCreate_Shell);CHKERRQ(ierr);
  ierr = DMRegister(DMREDUNDANT,  DMCreate_Redundant);CHKERRQ(ierr);
  ierr = DMRegister(DMPLEX,       DMCreate_Plex);CHKERRQ(ierr);
  ierr = DMRegister(DMPATCH,      DMCreate_Patch);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MOAB)
  ierr = DMRegister(DMMOAB,       DMCreate_Moab);CHKERRQ(ierr);
#endif
  ierr = DMRegister(DMNETWORK,    DMCreate_Network);CHKERRQ(ierr);
  ierr = DMRegister(DMFOREST,     DMCreate_Forest);CHKERRQ(ierr);
#if defined(PETSC_HAVE_P4EST)
  ierr = DMRegister(DMP4EST,      DMCreate_p4est);CHKERRQ(ierr);
  ierr = DMRegister(DMP8EST,      DMCreate_p8est);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_Chaco(PetscPartitioner);
PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_ParMetis(PetscPartitioner);
PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_Shell(PetscPartitioner);
PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_Simple(PetscPartitioner);
PETSC_EXTERN PetscErrorCode PetscPartitionerCreate_Gather(PetscPartitioner);

#undef __FUNCT__
#define __FUNCT__ "PetscPartitionerRegisterAll"
/*@C
  PetscPartitionerRegisterAll - Registers all of the PetscPartitioner components in the DM package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: PetscPartitioner, register, all
.seealso:  PetscPartitionerRegister(), PetscPartitionerRegisterDestroy()
@*/
PetscErrorCode PetscPartitionerRegisterAll()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscPartitionerRegisterAllCalled) PetscFunctionReturn(0);
  PetscPartitionerRegisterAllCalled = PETSC_TRUE;

  ierr = PetscPartitionerRegister(PETSCPARTITIONERCHACO,    PetscPartitionerCreate_Chaco);CHKERRQ(ierr);
  ierr = PetscPartitionerRegister(PETSCPARTITIONERPARMETIS, PetscPartitionerCreate_ParMetis);CHKERRQ(ierr);
  ierr = PetscPartitionerRegister(PETSCPARTITIONERSHELL,    PetscPartitionerCreate_Shell);CHKERRQ(ierr);
  ierr = PetscPartitionerRegister(PETSCPARTITIONERSIMPLE,   PetscPartitionerCreate_Simple);CHKERRQ(ierr);
  ierr = PetscPartitionerRegister(PETSCPARTITIONERGATHER,   PetscPartitionerCreate_Gather);CHKERRQ(ierr);
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
  if (PetscSpaceRegisterAllCalled) PetscFunctionReturn(0);
  PetscSpaceRegisterAllCalled = PETSC_TRUE;

  ierr = PetscSpaceRegister(PETSCSPACEPOLYNOMIAL, PetscSpaceCreate_Polynomial);CHKERRQ(ierr);
  ierr = PetscSpaceRegister(PETSCSPACEDG,         PetscSpaceCreate_DG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscDualSpaceCreate_Lagrange(PetscDualSpace);
PETSC_EXTERN PetscErrorCode PetscDualSpaceCreate_Simple(PetscDualSpace);

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
  if (PetscDualSpaceRegisterAllCalled) PetscFunctionReturn(0);
  PetscDualSpaceRegisterAllCalled = PETSC_TRUE;

  ierr = PetscDualSpaceRegister(PETSCDUALSPACELAGRANGE, PetscDualSpaceCreate_Lagrange);CHKERRQ(ierr);
  ierr = PetscDualSpaceRegister(PETSCDUALSPACESIMPLE,   PetscDualSpaceCreate_Simple);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscFECreate_Basic(PetscFE);
PETSC_EXTERN PetscErrorCode PetscFECreate_Nonaffine(PetscFE);
PETSC_EXTERN PetscErrorCode PetscFECreate_Composite(PetscFE);
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
  if (PetscFERegisterAllCalled) PetscFunctionReturn(0);
  PetscFERegisterAllCalled = PETSC_TRUE;

  ierr = PetscFERegister(PETSCFEBASIC,     PetscFECreate_Basic);CHKERRQ(ierr);
  ierr = PetscFERegister(PETSCFENONAFFINE, PetscFECreate_Nonaffine);CHKERRQ(ierr);
  ierr = PetscFERegister(PETSCFECOMPOSITE, PetscFECreate_Composite);CHKERRQ(ierr);
#ifdef PETSC_HAVE_OPENCL
  ierr = PetscFERegister(PETSCFEOPENCL, PetscFECreate_OpenCL);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
#include <petscfv.h>     /*I  "petscfv.h"  I*/

PETSC_EXTERN PetscErrorCode PetscLimiterCreate_Sin(PetscLimiter);
PETSC_EXTERN PetscErrorCode PetscLimiterCreate_Zero(PetscLimiter);
PETSC_EXTERN PetscErrorCode PetscLimiterCreate_None(PetscLimiter);
PETSC_EXTERN PetscErrorCode PetscLimiterCreate_Minmod(PetscLimiter);
PETSC_EXTERN PetscErrorCode PetscLimiterCreate_VanLeer(PetscLimiter);
PETSC_EXTERN PetscErrorCode PetscLimiterCreate_VanAlbada(PetscLimiter);
PETSC_EXTERN PetscErrorCode PetscLimiterCreate_Superbee(PetscLimiter);
PETSC_EXTERN PetscErrorCode PetscLimiterCreate_MC(PetscLimiter);

#undef __FUNCT__
#define __FUNCT__ "PetscLimiterRegisterAll"
/*@C
  PetscLimiterRegisterAll - Registers all of the PetscLimiter components in the PetscFV package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: PetscLimiter, register, all
.seealso:  PetscLimiterRegister(), PetscLimiterRegisterDestroy()
@*/
PetscErrorCode PetscLimiterRegisterAll()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscLimiterRegisterAllCalled) PetscFunctionReturn(0);
  PetscLimiterRegisterAllCalled = PETSC_TRUE;

  ierr = PetscLimiterRegister(PETSCLIMITERSIN,       PetscLimiterCreate_Sin);CHKERRQ(ierr);
  ierr = PetscLimiterRegister(PETSCLIMITERZERO,      PetscLimiterCreate_Zero);CHKERRQ(ierr);
  ierr = PetscLimiterRegister(PETSCLIMITERNONE,      PetscLimiterCreate_None);CHKERRQ(ierr);
  ierr = PetscLimiterRegister(PETSCLIMITERMINMOD,    PetscLimiterCreate_Minmod);CHKERRQ(ierr);
  ierr = PetscLimiterRegister(PETSCLIMITERVANLEER,   PetscLimiterCreate_VanLeer);CHKERRQ(ierr);
  ierr = PetscLimiterRegister(PETSCLIMITERVANALBADA, PetscLimiterCreate_VanAlbada);CHKERRQ(ierr);
  ierr = PetscLimiterRegister(PETSCLIMITERSUPERBEE,  PetscLimiterCreate_Superbee);CHKERRQ(ierr);
  ierr = PetscLimiterRegister(PETSCLIMITERMC,        PetscLimiterCreate_MC);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscFVCreate_Upwind(PetscFV);
PETSC_EXTERN PetscErrorCode PetscFVCreate_LeastSquares(PetscFV);

#undef __FUNCT__
#define __FUNCT__ "PetscFVRegisterAll"
/*@C
  PetscFVRegisterAll - Registers all of the PetscFV components in the PetscFV package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: PetscFV, register, all
.seealso:  PetscFVRegister(), PetscFVRegisterDestroy()
@*/
PetscErrorCode PetscFVRegisterAll()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscFVRegisterAllCalled) PetscFunctionReturn(0);
  PetscFVRegisterAllCalled = PETSC_TRUE;

  ierr = PetscFVRegister(PETSCFVUPWIND,       PetscFVCreate_Upwind);CHKERRQ(ierr);
  ierr = PetscFVRegister(PETSCFVLEASTSQUARES, PetscFVCreate_LeastSquares);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#include <petscds.h>     /*I  "petscds.h"  I*/

PETSC_EXTERN PetscErrorCode PetscDSCreate_Basic(PetscDS);

#undef __FUNCT__
#define __FUNCT__ "PetscDSRegisterAll"
/*@C
  PetscDSRegisterAll - Registers all of the PetscDS components in the PetscDS package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: PetscDS, register, all
.seealso:  PetscDSRegister(), PetscDSRegisterDestroy()
@*/
PetscErrorCode PetscDSRegisterAll()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscDSRegisterAllCalled) PetscFunctionReturn(0);
  PetscDSRegisterAllCalled = PETSC_TRUE;

  ierr = PetscDSRegister(PETSCDSBASIC, PetscDSCreate_Basic);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
