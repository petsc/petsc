
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

.seealso: `DMRegister()`, `DMRegisterDestroy()`
@*/
PetscErrorCode DMRegisterAll(void)
{
  PetscFunctionBegin;
  if (DMRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  DMRegisterAllCalled = PETSC_TRUE;

  PetscCall(DMRegister(DMDA, DMCreate_DA));
  PetscCall(DMRegister(DMCOMPOSITE, DMCreate_Composite));
  PetscCall(DMRegister(DMSLICED, DMCreate_Sliced));
  PetscCall(DMRegister(DMSHELL, DMCreate_Shell));
  PetscCall(DMRegister(DMREDUNDANT, DMCreate_Redundant));
  PetscCall(DMRegister(DMPLEX, DMCreate_Plex));
  PetscCall(DMRegister(DMPATCH, DMCreate_Patch));
  PetscCall(DMRegister(DMSWARM, DMCreate_Swarm));
#if defined(PETSC_HAVE_MOAB)
  PetscCall(DMRegister(DMMOAB, DMCreate_Moab));
#endif
  PetscCall(DMRegister(DMNETWORK, DMCreate_Network));
  PetscCall(DMRegister(DMFOREST, DMCreate_Forest));
#if defined(PETSC_HAVE_P4EST)
  PetscCall(DMRegister(DMP4EST, DMCreate_p4est));
  PetscCall(DMRegister(DMP8EST, DMCreate_p8est));
#endif
  PetscCall(DMRegister(DMPRODUCT, DMCreate_Product));
  PetscCall(DMRegister(DMSTAG, DMCreate_Stag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscfe.h> /*I  "petscfe.h"  I*/

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

.seealso: `PetscSpaceRegister()`, `PetscSpaceRegisterDestroy()`
@*/
PetscErrorCode PetscSpaceRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscSpaceRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscSpaceRegisterAllCalled = PETSC_TRUE;

  PetscCall(PetscSpaceRegister(PETSCSPACEPOLYNOMIAL, PetscSpaceCreate_Polynomial));
  PetscCall(PetscSpaceRegister(PETSCSPACEPTRIMMED, PetscSpaceCreate_Ptrimmed));
  PetscCall(PetscSpaceRegister(PETSCSPACETENSOR, PetscSpaceCreate_Tensor));
  PetscCall(PetscSpaceRegister(PETSCSPACESUM, PetscSpaceCreate_Sum));
  PetscCall(PetscSpaceRegister(PETSCSPACEPOINT, PetscSpaceCreate_Point));
  PetscCall(PetscSpaceRegister(PETSCSPACESUBSPACE, PetscSpaceCreate_Subspace));
  PetscCall(PetscSpaceRegister(PETSCSPACEWXY, PetscSpaceCreate_WXY));
  PetscFunctionReturn(PETSC_SUCCESS);
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

.seealso: `PetscDualSpaceRegister()`, `PetscDualSpaceRegisterDestroy()`
@*/
PetscErrorCode PetscDualSpaceRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscDualSpaceRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscDualSpaceRegisterAllCalled = PETSC_TRUE;

  PetscCall(PetscDualSpaceRegister(PETSCDUALSPACELAGRANGE, PetscDualSpaceCreate_Lagrange));
  PetscCall(PetscDualSpaceRegister(PETSCDUALSPACEBDM, PetscDualSpaceCreate_Lagrange));
  PetscCall(PetscDualSpaceRegister(PETSCDUALSPACESIMPLE, PetscDualSpaceCreate_Simple));
  PetscCall(PetscDualSpaceRegister(PETSCDUALSPACEREFINED, PetscDualSpaceCreate_Refined));
  PetscFunctionReturn(PETSC_SUCCESS);
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

.seealso: `PetscFERegister()`, `PetscFERegisterDestroy()`
@*/
PetscErrorCode PetscFERegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscFERegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscFERegisterAllCalled = PETSC_TRUE;

  PetscCall(PetscFERegister(PETSCFEBASIC, PetscFECreate_Basic));
  PetscCall(PetscFERegister(PETSCFECOMPOSITE, PetscFECreate_Composite));
#if defined(PETSC_HAVE_OPENCL)
  PetscCall(PetscFERegister(PETSCFEOPENCL, PetscFECreate_OpenCL));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
#include <petscfv.h> /*I  "petscfv.h"  I*/

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

.seealso: `PetscLimiterRegister()`, `PetscLimiterRegisterDestroy()`
@*/
PetscErrorCode PetscLimiterRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscLimiterRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscLimiterRegisterAllCalled = PETSC_TRUE;

  PetscCall(PetscLimiterRegister(PETSCLIMITERSIN, PetscLimiterCreate_Sin));
  PetscCall(PetscLimiterRegister(PETSCLIMITERZERO, PetscLimiterCreate_Zero));
  PetscCall(PetscLimiterRegister(PETSCLIMITERNONE, PetscLimiterCreate_None));
  PetscCall(PetscLimiterRegister(PETSCLIMITERMINMOD, PetscLimiterCreate_Minmod));
  PetscCall(PetscLimiterRegister(PETSCLIMITERVANLEER, PetscLimiterCreate_VanLeer));
  PetscCall(PetscLimiterRegister(PETSCLIMITERVANALBADA, PetscLimiterCreate_VanAlbada));
  PetscCall(PetscLimiterRegister(PETSCLIMITERSUPERBEE, PetscLimiterCreate_Superbee));
  PetscCall(PetscLimiterRegister(PETSCLIMITERMC, PetscLimiterCreate_MC));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PetscFVCreate_Upwind(PetscFV);
PETSC_EXTERN PetscErrorCode PetscFVCreate_LeastSquares(PetscFV);

/*@C
  PetscFVRegisterAll - Registers all of the PetscFV components in the PetscFV package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.seealso: `PetscFVRegister()`, `PetscFVRegisterDestroy()`
@*/
PetscErrorCode PetscFVRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscFVRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscFVRegisterAllCalled = PETSC_TRUE;

  PetscCall(PetscFVRegister(PETSCFVUPWIND, PetscFVCreate_Upwind));
  PetscCall(PetscFVRegister(PETSCFVLEASTSQUARES, PetscFVCreate_LeastSquares));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#include <petscds.h> /*I  "petscds.h"  I*/

PETSC_EXTERN PetscErrorCode PetscDSCreate_Basic(PetscDS);

/*@C
  PetscDSRegisterAll - Registers all of the PetscDS components in the PetscDS package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.seealso: `PetscDSRegister()`, `PetscDSRegisterDestroy()`
@*/
PetscErrorCode PetscDSRegisterAll(void)
{
  PetscFunctionBegin;
  if (PetscDSRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscDSRegisterAllCalled = PETSC_TRUE;

  PetscCall(PetscDSRegister(PETSCDSBASIC, PetscDSCreate_Basic));
  PetscFunctionReturn(PETSC_SUCCESS);
}
