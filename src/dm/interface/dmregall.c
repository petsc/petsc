
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
PETSC_EXTERN PetscErrorCode DMCreate_Swarm(DM);
#if defined(PETSC_HAVE_MOAB)
PETSC_EXTERN PetscErrorCode DMCreate_Moab(DM);
#endif
PETSC_EXTERN PetscErrorCode DMCreate_Network(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Forest(DM);
#if defined(PETSC_HAVE_P4EST)
PETSC_EXTERN PetscErrorCode DMCreate_p4est(DM);
PETSC_EXTERN PetscErrorCode DMCreate_p8est(DM);
#endif
PETSC_EXTERN PetscErrorCode DMCreate_Product(DM);
PETSC_EXTERN PetscErrorCode DMCreate_Stag(DM);

/*@C
  DMRegisterAll - Registers all of the DM components in the DM package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.seealso:  DMRegister(), DMRegisterDestroy()
@*/
PetscErrorCode  DMRegisterAll(void)
{
  PetscFunctionBegin;
  if (DMRegisterAllCalled) PetscFunctionReturn(0);
  DMRegisterAllCalled = PETSC_TRUE;

  CHKERRQ(DMRegister(DMDA,       DMCreate_DA));
  CHKERRQ(DMRegister(DMCOMPOSITE,DMCreate_Composite));
  CHKERRQ(DMRegister(DMSLICED,   DMCreate_Sliced));
  CHKERRQ(DMRegister(DMSHELL,    DMCreate_Shell));
  CHKERRQ(DMRegister(DMREDUNDANT,DMCreate_Redundant));
  CHKERRQ(DMRegister(DMPLEX,     DMCreate_Plex));
  CHKERRQ(DMRegister(DMPATCH,    DMCreate_Patch));
  CHKERRQ(DMRegister(DMSWARM,    DMCreate_Swarm));
#if defined(PETSC_HAVE_MOAB)
  CHKERRQ(DMRegister(DMMOAB,     DMCreate_Moab));
#endif
  CHKERRQ(DMRegister(DMNETWORK,  DMCreate_Network));
  CHKERRQ(DMRegister(DMFOREST,   DMCreate_Forest));
#if defined(PETSC_HAVE_P4EST)
  CHKERRQ(DMRegister(DMP4EST,    DMCreate_p4est));
  CHKERRQ(DMRegister(DMP8EST,    DMCreate_p8est));
#endif
  CHKERRQ(DMRegister(DMPRODUCT,  DMCreate_Product));
  CHKERRQ(DMRegister(DMSTAG,     DMCreate_Stag));
  PetscFunctionReturn(0);
}

#include <petscfe.h>     /*I  "petscfe.h"  I*/

PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Polynomial(PetscSpace);
PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Ptrimmed(PetscSpace);
PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Tensor(PetscSpace);
PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Sum(PetscSpace);
PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Point(PetscSpace);
PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Subspace(PetscSpace);
PETSC_EXTERN PetscErrorCode PetscSpaceCreate_WXY(PetscSpace);

/*@C
  PetscSpaceRegisterAll - Registers all of the PetscSpace components in the PetscFE package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.seealso:  PetscSpaceRegister(), PetscSpaceRegisterDestroy()
@*/
PetscErrorCode PetscSpaceRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscSpaceRegisterAllCalled) PetscFunctionReturn(0);
  PetscSpaceRegisterAllCalled = PETSC_TRUE;

  CHKERRQ(PetscSpaceRegister(PETSCSPACEPOLYNOMIAL, PetscSpaceCreate_Polynomial));
  CHKERRQ(PetscSpaceRegister(PETSCSPACEPTRIMMED,   PetscSpaceCreate_Ptrimmed));
  CHKERRQ(PetscSpaceRegister(PETSCSPACETENSOR,     PetscSpaceCreate_Tensor));
  CHKERRQ(PetscSpaceRegister(PETSCSPACESUM,        PetscSpaceCreate_Sum));
  CHKERRQ(PetscSpaceRegister(PETSCSPACEPOINT,      PetscSpaceCreate_Point));
  CHKERRQ(PetscSpaceRegister(PETSCSPACESUBSPACE,   PetscSpaceCreate_Subspace));
  CHKERRQ(PetscSpaceRegister(PETSCSPACEWXY,        PetscSpaceCreate_WXY));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscDualSpaceCreate_Lagrange(PetscDualSpace);
PETSC_EXTERN PetscErrorCode PetscDualSpaceCreate_Lagrange_BDM(PetscDualSpace);
PETSC_EXTERN PetscErrorCode PetscDualSpaceCreate_Simple(PetscDualSpace);
PETSC_EXTERN PetscErrorCode PetscDualSpaceCreate_Refined(PetscDualSpace);

/*@C
  PetscDualSpaceRegisterAll - Registers all of the PetscDualSpace components in the PetscFE package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.seealso:  PetscDualSpaceRegister(), PetscDualSpaceRegisterDestroy()
@*/
PetscErrorCode PetscDualSpaceRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscDualSpaceRegisterAllCalled) PetscFunctionReturn(0);
  PetscDualSpaceRegisterAllCalled = PETSC_TRUE;

  CHKERRQ(PetscDualSpaceRegister(PETSCDUALSPACELAGRANGE, PetscDualSpaceCreate_Lagrange));
  CHKERRQ(PetscDualSpaceRegister(PETSCDUALSPACEBDM,      PetscDualSpaceCreate_Lagrange));
  CHKERRQ(PetscDualSpaceRegister(PETSCDUALSPACESIMPLE,   PetscDualSpaceCreate_Simple));
  CHKERRQ(PetscDualSpaceRegister(PETSCDUALSPACEREFINED,  PetscDualSpaceCreate_Refined));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscFECreate_Basic(PetscFE);
PETSC_EXTERN PetscErrorCode PetscFECreate_Nonaffine(PetscFE);
PETSC_EXTERN PetscErrorCode PetscFECreate_Composite(PetscFE);
#if defined(PETSC_HAVE_OPENCL)
PETSC_EXTERN PetscErrorCode PetscFECreate_OpenCL(PetscFE);
#endif

/*@C
  PetscFERegisterAll - Registers all of the PetscFE components in the PetscFE package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.seealso:  PetscFERegister(), PetscFERegisterDestroy()
@*/
PetscErrorCode PetscFERegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscFERegisterAllCalled) PetscFunctionReturn(0);
  PetscFERegisterAllCalled = PETSC_TRUE;

  CHKERRQ(PetscFERegister(PETSCFEBASIC,     PetscFECreate_Basic));
  CHKERRQ(PetscFERegister(PETSCFECOMPOSITE, PetscFECreate_Composite));
#if defined(PETSC_HAVE_OPENCL)
  CHKERRQ(PetscFERegister(PETSCFEOPENCL, PetscFECreate_OpenCL));
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

/*@C
  PetscLimiterRegisterAll - Registers all of the PetscLimiter components in the PetscFV package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.seealso:  PetscLimiterRegister(), PetscLimiterRegisterDestroy()
@*/
PetscErrorCode PetscLimiterRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscLimiterRegisterAllCalled) PetscFunctionReturn(0);
  PetscLimiterRegisterAllCalled = PETSC_TRUE;

  CHKERRQ(PetscLimiterRegister(PETSCLIMITERSIN,       PetscLimiterCreate_Sin));
  CHKERRQ(PetscLimiterRegister(PETSCLIMITERZERO,      PetscLimiterCreate_Zero));
  CHKERRQ(PetscLimiterRegister(PETSCLIMITERNONE,      PetscLimiterCreate_None));
  CHKERRQ(PetscLimiterRegister(PETSCLIMITERMINMOD,    PetscLimiterCreate_Minmod));
  CHKERRQ(PetscLimiterRegister(PETSCLIMITERVANLEER,   PetscLimiterCreate_VanLeer));
  CHKERRQ(PetscLimiterRegister(PETSCLIMITERVANALBADA, PetscLimiterCreate_VanAlbada));
  CHKERRQ(PetscLimiterRegister(PETSCLIMITERSUPERBEE,  PetscLimiterCreate_Superbee));
  CHKERRQ(PetscLimiterRegister(PETSCLIMITERMC,        PetscLimiterCreate_MC));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscFVCreate_Upwind(PetscFV);
PETSC_EXTERN PetscErrorCode PetscFVCreate_LeastSquares(PetscFV);

/*@C
  PetscFVRegisterAll - Registers all of the PetscFV components in the PetscFV package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.seealso:  PetscFVRegister(), PetscFVRegisterDestroy()
@*/
PetscErrorCode PetscFVRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscFVRegisterAllCalled) PetscFunctionReturn(0);
  PetscFVRegisterAllCalled = PETSC_TRUE;

  CHKERRQ(PetscFVRegister(PETSCFVUPWIND,       PetscFVCreate_Upwind));
  CHKERRQ(PetscFVRegister(PETSCFVLEASTSQUARES, PetscFVCreate_LeastSquares));
  PetscFunctionReturn(0);
}
#include <petscds.h>     /*I  "petscds.h"  I*/

PETSC_EXTERN PetscErrorCode PetscDSCreate_Basic(PetscDS);

/*@C
  PetscDSRegisterAll - Registers all of the PetscDS components in the PetscDS package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.seealso:  PetscDSRegister(), PetscDSRegisterDestroy()
@*/
PetscErrorCode PetscDSRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscDSRegisterAllCalled) PetscFunctionReturn(0);
  PetscDSRegisterAllCalled = PETSC_TRUE;

  CHKERRQ(PetscDSRegister(PETSCDSBASIC, PetscDSCreate_Basic));
  PetscFunctionReturn(0);
}
