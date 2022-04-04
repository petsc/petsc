/* Additional functions in the DMStag API, which are not part of the general DM API. */
#include <petsc/private/dmstagimpl.h>
#include <petscdmproduct.h>
/*@C
  DMStagGetBoundaryTypes - get boundary types

  Not Collective

  Input Parameter:
. dm - the DMStag object

  Output Parameters:
. boundaryTypeX,boundaryTypeY,boundaryTypeZ - boundary types

  Level: intermediate

.seealso: DMSTAG
@*/
PetscErrorCode DMStagGetBoundaryTypes(DM dm,DMBoundaryType *boundaryTypeX,DMBoundaryType *boundaryTypeY,DMBoundaryType *boundaryTypeZ)
{
  const DM_Stag * const stag  = (DM_Stag*)dm->data;
  PetscInt              dim;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscCall(DMGetDimension(dm,&dim));
  if (boundaryTypeX) *boundaryTypeX = stag->boundaryType[0];
  if (boundaryTypeY && dim > 1) *boundaryTypeY = stag->boundaryType[1];
  if (boundaryTypeZ && dim > 2) *boundaryTypeZ = stag->boundaryType[2];
  PetscFunctionReturn(0);
}

static PetscErrorCode DMStagGetProductCoordinateArrays_Private(DM dm,void* arrX,void* arrY,void* arrZ,PetscBool read)
{
  PetscInt       dim,d,dofCheck[DMSTAG_MAX_STRATA],s;
  DM             dmCoord;
  void*          arr[DMSTAG_MAX_DIM];
  PetscBool      checkDof;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDimension(dm,&dim));
  PetscCheck(dim <= DMSTAG_MAX_DIM,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for %D dimensions",dim);
  arr[0] = arrX; arr[1] = arrY; arr[2] = arrZ;
  PetscCall(DMGetCoordinateDM(dm,&dmCoord));
  PetscCheck(dmCoord,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"DM does not have a coordinate DM");
  {
    PetscBool isProduct;
    DMType    dmType;
    PetscCall(DMGetType(dmCoord,&dmType));
    PetscCall(PetscStrcmp(DMPRODUCT,dmType,&isProduct));
    PetscCheck(isProduct,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate DM is not of type DMPRODUCT");
  }
  for (s=0; s<DMSTAG_MAX_STRATA; ++s) dofCheck[s] = 0;
  checkDof = PETSC_FALSE;
  for (d=0; d<dim; ++d) {
    DM        subDM;
    DMType    dmType;
    PetscBool isStag;
    PetscInt  dof[DMSTAG_MAX_STRATA],subDim;
    Vec       coord1d_local;

    /* Ignore unrequested arrays */
    if (!arr[d]) continue;

    PetscCall(DMProductGetDM(dmCoord,d,&subDM));
    PetscCheck(subDM,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate DM is missing sub DM %D",d);
    PetscCall(DMGetDimension(subDM,&subDim));
    PetscCheck(subDim == 1,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate sub-DM is not of dimension 1");
    PetscCall(DMGetType(subDM,&dmType));
    PetscCall(PetscStrcmp(DMSTAG,dmType,&isStag));
    PetscCheck(isStag,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate sub-DM is not of type DMSTAG");
    PetscCall(DMStagGetDOF(subDM,&dof[0],&dof[1],&dof[2],&dof[3]));
    if (!checkDof) {
      for (s=0; s<DMSTAG_MAX_STRATA; ++s) dofCheck[s] = dof[s];
      checkDof = PETSC_TRUE;
    } else {
      for (s=0; s<DMSTAG_MAX_STRATA; ++s) {
        PetscCheck(dofCheck[s] == dof[s],PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate sub-DMs have different dofs");
      }
    }
    PetscCall(DMGetCoordinatesLocal(subDM,&coord1d_local));
    if (read) {
      PetscCall(DMStagVecGetArrayRead(subDM,coord1d_local,arr[d]));
    } else {
      PetscCall(DMStagVecGetArray(subDM,coord1d_local,arr[d]));
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  DMStagGetProductCoordinateArrays - extract local product coordinate arrays, one per dimension

  Logically Collective

  A high-level helper function to quickly extract local coordinate arrays.

  Note that 2-dimensional arrays are returned. See
  DMStagVecGetArray(), which is called internally to produce these arrays
  representing coordinates on elements and vertices (element boundaries)
  for a 1-dimensional DMStag in each coordinate direction.

  One should use DMStagGetProductCoordinateLocationSlot() to determine appropriate
  indices for the second dimension in these returned arrays. This function
  checks that the coordinate array is a suitable product of 1-dimensional
  DMStag objects.

  Input Parameter:
. dm - the DMStag object

  Output Parameters:
. arrX,arrY,arrZ - local 1D coordinate arrays

  Level: intermediate

.seealso: DMSTAG, DMPRODUCT, DMStagGetProductCoordinateArraysRead(), DMStagSetUniformCoordinates(), DMStagSetUniformCoordinatesProduct(), DMStagGetProductCoordinateLocationSlot()
@*/
PetscErrorCode DMStagGetProductCoordinateArrays(DM dm,void* arrX,void* arrY,void* arrZ)
{
  PetscFunctionBegin;
  PetscCall(DMStagGetProductCoordinateArrays_Private(dm,arrX,arrY,arrZ,PETSC_FALSE));
  PetscFunctionReturn(0);
}

/*@C
  DMStagGetProductCoordinateArraysRead - extract product coordinate arrays, read-only

  Logically Collective

  See the man page for DMStagGetProductCoordinateArrays() for more information.

  Input Parameter:
. dm - the DMStag object

  Output Parameters:
. arrX,arrY,arrZ - local 1D coordinate arrays

  Level: intermediate

.seealso: DMSTAG, DMPRODUCT, DMStagGetProductCoordinateArrays(), DMStagSetUniformCoordinates(), DMStagSetUniformCoordinatesProduct(), DMStagGetProductCoordinateLocationSlot()
@*/
PetscErrorCode DMStagGetProductCoordinateArraysRead(DM dm,void* arrX,void* arrY,void* arrZ)
{
  PetscFunctionBegin;
  PetscCall(DMStagGetProductCoordinateArrays_Private(dm,arrX,arrY,arrZ,PETSC_TRUE));
  PetscFunctionReturn(0);
}

/*@C
  DMStagGetProductCoordinateLocationSlot - get slot for use with local product coordinate arrays

  Not Collective

  High-level helper function to get slot indices for 1D coordinate DMs,
  for use with DMStagGetProductCoordinateArrays() and related functions.

  Input Parameters:
+ dm - the DMStag object
- loc - the grid location

  Output Parameter:
. slot - the index to use in local arrays

  Notes:
  For loc, one should use DMSTAG_LEFT, DMSTAG_ELEMENT, or DMSTAG_RIGHT for "previous", "center" and "next"
  locations, respectively, in each dimension.
  One can equivalently use DMSTAG_DOWN or DMSTAG_BACK in place of DMSTAG_LEFT,
  and DMSTAG_UP or DMSTACK_FRONT in place of DMSTAG_RIGHT;

  This function checks that the coordinates are actually set up so that using the
  slots from any of the 1D coordinate sub-DMs are valid for all the 1D coordinate sub-DMs.

  Level: intermediate

.seealso: DMSTAG, DMPRODUCT, DMStagGetProductCoordinateArrays(), DMStagGetProductCoordinateArraysRead(), DMStagSetUniformCoordinates()
@*/
PETSC_EXTERN PetscErrorCode DMStagGetProductCoordinateLocationSlot(DM dm,DMStagStencilLocation loc,PetscInt *slot)
{
  DM             dmCoord;
  PetscInt       dim,dofCheck[DMSTAG_MAX_STRATA],s,d;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDimension(dm,&dim));
  PetscCall(DMGetCoordinateDM(dm,&dmCoord));
  PetscCheck(dmCoord,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"DM does not have a coordinate DM");
  {
    PetscBool isProduct;
    DMType    dmType;
    PetscCall(DMGetType(dmCoord,&dmType));
    PetscCall(PetscStrcmp(DMPRODUCT,dmType,&isProduct));
    PetscCheck(isProduct,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate DM is not of type DMPRODUCT");
  }
  for (s=0; s<DMSTAG_MAX_STRATA; ++s) dofCheck[s] = 0;
  for (d=0; d<dim; ++d) {
    DM        subDM;
    DMType    dmType;
    PetscBool isStag;
    PetscInt  dof[DMSTAG_MAX_STRATA],subDim;
    PetscCall(DMProductGetDM(dmCoord,d,&subDM));
    PetscCheck(subDM,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate DM is missing sub DM %D",d);
    PetscCall(DMGetDimension(subDM,&subDim));
    PetscCheck(subDim == 1,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate sub-DM is not of dimension 1");
    PetscCall(DMGetType(subDM,&dmType));
    PetscCall(PetscStrcmp(DMSTAG,dmType,&isStag));
    PetscCheck(isStag,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate sub-DM is not of type DMSTAG");
    PetscCall(DMStagGetDOF(subDM,&dof[0],&dof[1],&dof[2],&dof[3]));
    if (d == 0) {
      const PetscInt component = 0;
      for (s=0; s<DMSTAG_MAX_STRATA; ++s) dofCheck[s] = dof[s];
      PetscCall(DMStagGetLocationSlot(subDM,loc,component,slot));
    } else {
      for (s=0; s<DMSTAG_MAX_STRATA; ++s) {
        PetscCheck(dofCheck[s] == dof[s],PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Coordinate sub-DMs have different dofs");
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  DMStagGetCorners - return global element indices of the local region (excluding ghost points)

  Not Collective

  Input Parameter:
. dm - the DMStag object

  Output Parameters:
+ x     - starting element index in first direction
. y     - starting element index in second direction
. z     - starting element index in third direction
. m     - element width in first direction
. n     - element width in second direction
. p     - element width in third direction
. nExtrax - number of extra partial elements in first direction
. nExtray - number of extra partial elements in second direction
- nExtraz - number of extra partial elements in third direction

  Notes:
  Arguments corresponding to higher dimensions are ignored for 1D and 2D grids. These arguments may be set to NULL in this case.

  The number of extra partial elements is either 1 or 0.
  The value is 1 on right, top, and front non-periodic domain ("physical") boundaries,
  in the x, y, and z directions respectively, and otherwise 0.

  Level: beginner

.seealso: DMSTAG, DMStagGetGhostCorners(), DMDAGetCorners()
@*/
PetscErrorCode DMStagGetCorners(DM dm,PetscInt *x,PetscInt *y,PetscInt *z,PetscInt *m,PetscInt *n,PetscInt *p,PetscInt *nExtrax,PetscInt *nExtray,PetscInt *nExtraz)
{
  const DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  if (x) *x = stag->start[0];
  if (y) *y = stag->start[1];
  if (z) *z = stag->start[2];
  if (m) *m = stag->n[0];
  if (n) *n = stag->n[1];
  if (p) *p = stag->n[2];
  if (nExtrax) *nExtrax = stag->boundaryType[0] != DM_BOUNDARY_PERIODIC && stag->lastRank[0] ? 1 : 0;
  if (nExtray) *nExtray = stag->boundaryType[1] != DM_BOUNDARY_PERIODIC && stag->lastRank[1] ? 1 : 0;
  if (nExtraz) *nExtraz = stag->boundaryType[2] != DM_BOUNDARY_PERIODIC && stag->lastRank[2] ? 1 : 0;
  PetscFunctionReturn(0);
}

/*@C
  DMStagGetDOF - get number of DOF associated with each stratum of the grid

  Not Collective

  Input Parameter:
. dm - the DMStag object

  Output Parameters:
+ dof0 - the number of points per 0-cell (vertex/node)
. dof1 - the number of points per 1-cell (element in 1D, edge in 2D and 3D)
. dof2 - the number of points per 2-cell (element in 2D, face in 3D)
- dof3 - the number of points per 3-cell (element in 3D)

  Level: beginner

.seealso: DMSTAG, DMStagGetCorners(), DMStagGetGhostCorners(), DMStagGetGlobalSizes(), DMStagGetStencilWidth(), DMStagGetBoundaryTypes(), DMStagGetLocationDOF(), DMDAGetDof()
@*/
PetscErrorCode DMStagGetDOF(DM dm,PetscInt *dof0,PetscInt *dof1,PetscInt *dof2,PetscInt *dof3)
{
  const DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  if (dof0) *dof0 = stag->dof[0];
  if (dof1) *dof1 = stag->dof[1];
  if (dof2) *dof2 = stag->dof[2];
  if (dof3) *dof3 = stag->dof[3];
  PetscFunctionReturn(0);
}

/*@C
  DMStagGetGhostCorners - return global element indices of the local region, including ghost points

  Not Collective

  Input Parameter:
. dm - the DMStag object

  Output Parameters:
+ x - the starting element index in the first direction
. y - the starting element index in the second direction
. z - the starting element index in the third direction
. m - the element width in the first direction
. n - the element width in the second direction
- p - the element width in the third direction

  Notes:
  Arguments corresponding to higher dimensions are ignored for 1D and 2D grids. These arguments may be set to NULL in this case.

  Level: beginner

.seealso: DMSTAG, DMStagGetCorners(), DMDAGetGhostCorners()
@*/
PetscErrorCode DMStagGetGhostCorners(DM dm,PetscInt *x,PetscInt *y,PetscInt *z,PetscInt *m,PetscInt *n,PetscInt *p)
{
  const DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  if (x) *x = stag->startGhost[0];
  if (y) *y = stag->startGhost[1];
  if (z) *z = stag->startGhost[2];
  if (m) *m = stag->nGhost[0];
  if (n) *n = stag->nGhost[1];
  if (p) *p = stag->nGhost[2];
  PetscFunctionReturn(0);
}

/*@C
  DMStagGetGlobalSizes - get global element counts

  Not Collective

  Input Parameter:
. dm - the DMStag object

  Output Parameters:
. M,N,P - global element counts in each direction

  Notes:
  Arguments corresponding to higher dimensions are ignored for 1D and 2D grids. These arguments may be set to NULL in this case.

  Level: beginner

.seealso: DMSTAG, DMStagGetLocalSizes(), DMDAGetInfo()
@*/
PetscErrorCode DMStagGetGlobalSizes(DM dm,PetscInt* M,PetscInt* N,PetscInt* P)
{
  const DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  if (M) *M = stag->N[0];
  if (N) *N = stag->N[1];
  if (P) *P = stag->N[2];
  PetscFunctionReturn(0);
}

/*@C
  DMStagGetIsFirstRank - get boolean value for whether this rank is first in each direction in the rank grid

  Not Collective

  Input Parameter:
. dm - the DMStag object

  Output Parameters:
. isFirstRank0,isFirstRank1,isFirstRank2 - whether this rank is first in each direction

  Level: intermediate

  Notes:
  Arguments corresponding to higher dimensions are ignored for 1D and 2D grids. These arguments may be set to NULL in this case.

.seealso: DMSTAG, DMStagGetIsLastRank()
@*/
PetscErrorCode DMStagGetIsFirstRank(DM dm,PetscBool *isFirstRank0,PetscBool *isFirstRank1,PetscBool *isFirstRank2)
{
  const DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  if (isFirstRank0) *isFirstRank0 = stag->firstRank[0];
  if (isFirstRank1) *isFirstRank1 = stag->firstRank[1];
  if (isFirstRank2) *isFirstRank2 = stag->firstRank[2];
  PetscFunctionReturn(0);
}

/*@C
  DMStagGetIsLastRank - get boolean value for whether this rank is last in each direction in the rank grid

  Not Collective

  Input Parameter:
. dm - the DMStag object

  Output Parameters:
. isLastRank0,isLastRank1,isLastRank2 - whether this rank is last in each direction

  Level: intermediate

  Notes:
  Arguments corresponding to higher dimensions are ignored for 1D and 2D grids. These arguments may be set to NULL in this case.
  Level: intermediate

.seealso: DMSTAG, DMStagGetIsFirstRank()
@*/
PetscErrorCode DMStagGetIsLastRank(DM dm,PetscBool *isLastRank0,PetscBool *isLastRank1,PetscBool *isLastRank2)
{
  const DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  if (isLastRank0) *isLastRank0 = stag->lastRank[0];
  if (isLastRank1) *isLastRank1 = stag->lastRank[1];
  if (isLastRank2) *isLastRank2 = stag->lastRank[2];
  PetscFunctionReturn(0);
}

/*@C
  DMStagGetLocalSizes - get local elementwise sizes

  Not Collective

  Input Parameter:
. dm - the DMStag object

  Output Parameters:
. m,n,p - local element counts (excluding ghosts) in each direction

  Notes:
  Arguments corresponding to higher dimensions are ignored for 1D and 2D grids. These arguments may be set to NULL in this case.
  Level: intermediate

  Level: beginner

.seealso: DMSTAG, DMStagGetGlobalSizes(), DMStagGetDOF(), DMStagGetNumRanks(), DMDAGetLocalInfo()
@*/
PetscErrorCode DMStagGetLocalSizes(DM dm,PetscInt* m,PetscInt* n,PetscInt* p)
{
  const DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  if (m) *m = stag->n[0];
  if (n) *n = stag->n[1];
  if (p) *p = stag->n[2];
  PetscFunctionReturn(0);
}

/*@C
  DMStagGetNumRanks - get number of ranks in each direction in the global grid decomposition

  Not Collective

  Input Parameter:
. dm - the DMStag object

  Output Parameters:
. nRanks0,nRanks1,nRanks2 - number of ranks in each direction in the grid decomposition

  Notes:
  Arguments corresponding to higher dimensions are ignored for 1D and 2D grids. These arguments may be set to NULL in this case.
  Level: intermediate

  Level: beginner

.seealso: DMSTAG, DMStagGetGlobalSizes(), DMStagGetLocalSize(), DMStagSetNumRanks(), DMDAGetInfo()
@*/
PetscErrorCode DMStagGetNumRanks(DM dm,PetscInt *nRanks0,PetscInt *nRanks1,PetscInt *nRanks2)
{
  const DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  if (nRanks0) *nRanks0 = stag->nRanks[0];
  if (nRanks1) *nRanks1 = stag->nRanks[1];
  if (nRanks2) *nRanks2 = stag->nRanks[2];
  PetscFunctionReturn(0);
}

/*@C
  DMStagGetEntries - get number of native entries in the global representation

  Not Collective

  Input Parameter:
. dm - the DMStag object

  Output Parameters:
. entries - number of rank-native entries in the global representation

  Note:
  This is the number of entries on this rank for a global vector associated with dm.

  Level: developer

.seealso: DMSTAG, DMStagGetDOF(), DMStagGetEntriesPerElement(), DMCreateLocalVector()
@*/
PetscErrorCode DMStagGetEntries(DM dm,PetscInt *entries)
{
  const DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  if (entries) *entries = stag->entries;
  PetscFunctionReturn(0);
}

/*@C
  DMStagGetEntriesPerElement - get number of entries per element in the local representation

  Not Collective

  Input Parameter:
. dm - the DMStag object

  Output Parameters:
. entriesPerElement - number of entries associated with each element in the local representation

  Notes:
  This is the natural block size for most local operations. In 1D it is equal to dof0 + dof1,
  in 2D it is equal to dof0 + 2*dof1 + dof2, and in 3D it is equal to dof0 + 3*dof1 + 3*dof2 + dof3

  Level: developer

.seealso: DMSTAG, DMStagGetDOF()
@*/
PetscErrorCode DMStagGetEntriesPerElement(DM dm,PetscInt *entriesPerElement)
{
  const DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  if (entriesPerElement) *entriesPerElement = stag->entriesPerElement;
  PetscFunctionReturn(0);
}

/*@C
  DMStagGetStencilType - get elementwise ghost/halo stencil type

  Not Collective

  Input Parameter:
. dm - the DMStag object

  Output Parameter:
. stencilType - the elementwise ghost stencil type: DMSTAG_STENCIL_BOX, DMSTAG_STENCIL_STAR, or DMSTAG_STENCIL_NONE

  Level: beginner

.seealso: DMSTAG, DMStagSetStencilType(), DMStagGetStencilWidth, DMStagStencilType
@*/
PetscErrorCode DMStagGetStencilType(DM dm,DMStagStencilType *stencilType)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  *stencilType = stag->stencilType;
  PetscFunctionReturn(0);
}

/*@C
  DMStagGetStencilWidth - get elementwise stencil width

  Not Collective

  Input Parameter:
. dm - the DMStag object

  Output Parameters:
. stencilWidth - stencil/halo/ghost width in elements

  Level: beginner

.seealso: DMSTAG, DMStagSetStencilWidth(), DMStagGetStencilType(), DMDAGetStencilType()
@*/
PetscErrorCode DMStagGetStencilWidth(DM dm,PetscInt *stencilWidth)
{
  const DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  if (stencilWidth) *stencilWidth = stag->stencilWidth;
  PetscFunctionReturn(0);
}

/*@C
  DMStagGetOwnershipRanges - get elements per rank in each direction

  Not Collective

  Input Parameter:
.     dm - the DMStag object

  Output Parameters:
+     lx - ownership along x direction (optional)
.     ly - ownership along y direction (optional)
-     lz - ownership along z direction (optional)

  Notes:
  These correspond to the optional final arguments passed to DMStagCreate1d(), DMStagCreate2d(), and DMStagCreate3d().

  Arguments corresponding to higher dimensions are ignored for 1D and 2D grids. These arguments may be set to NULL in this case.

  In C you should not free these arrays, nor change the values in them.
  They will only have valid values while the DMStag they came from still exists (has not been destroyed).

  Level: intermediate

.seealso: DMSTAG, DMStagSetGlobalSizes(), DMStagSetOwnershipRanges(), DMStagCreate1d(), DMStagCreate2d(), DMStagCreate3d(), DMDAGetOwnershipRanges()
@*/
PetscErrorCode DMStagGetOwnershipRanges(DM dm,const PetscInt *lx[],const PetscInt *ly[],const PetscInt *lz[])
{
  const DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  if (lx) *lx = stag->l[0];
  if (ly) *ly = stag->l[1];
  if (lz) *lz = stag->l[2];
  PetscFunctionReturn(0);
}

/*@C
  DMStagCreateCompatibleDMStag - create a compatible DMStag with different dof/stratum

  Collective

  Input Parameters:
+ dm - the DMStag object
- dof0,dof1,dof2,dof3 - number of dof on each stratum in the new DMStag

  Output Parameters:
. newdm - the new, compatible DMStag

  Notes:
  Dof supplied for strata too big for the dimension are ignored; these may be set to 0.
  For example, for a 2-dimensional DMStag, dof2 sets the number of dof per element,
  and dof3 is unused. For a 3-dimensional DMStag, dof3 sets the number of dof per element.

  In contrast to DMDACreateCompatibleDMDA(), coordinates are not reused.

  Level: intermediate

.seealso: DMSTAG, DMDACreateCompatibleDMDA(), DMGetCompatibility(), DMStagMigrateVec()
@*/
PetscErrorCode DMStagCreateCompatibleDMStag(DM dm,PetscInt dof0,PetscInt dof1,PetscInt dof2,PetscInt dof3,DM *newdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscCall(DMStagDuplicateWithoutSetup(dm,PetscObjectComm((PetscObject)dm),newdm));
  PetscCall(DMStagSetDOF(*newdm,dof0,dof1,dof2,dof3));
  PetscCall(DMSetUp(*newdm));
  PetscFunctionReturn(0);
}

/*@C
  DMStagGetLocationSlot - get index to use in accessing raw local arrays

  Not Collective

  Input Parameters:
+ dm - the DMStag object
. loc - location relative to an element
- c - component

  Output Parameter:
. slot - index to use

  Notes:
  Provides an appropriate index to use with DMStagVecGetArray() and friends.
  This is required so that the user doesn't need to know about the ordering of
  dof associated with each local element.

  Level: beginner

.seealso: DMSTAG, DMStagVecGetArray(), DMStagVecGetArrayRead(), DMStagGetDOF(), DMStagGetEntriesPerElement()
@*/
PetscErrorCode DMStagGetLocationSlot(DM dm,DMStagStencilLocation loc,PetscInt c,PetscInt *slot)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  if (PetscDefined(USE_DEBUG)) {
    PetscInt       dof;
    PetscCall(DMStagGetLocationDOF(dm,loc,&dof));
    PetscCheck(dof >= 1,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Location %s has no dof attached",DMStagStencilLocations[loc]);
    PetscCheck(c <= dof-1,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Supplied component number (%D) for location %s is too big (maximum %D)",c,DMStagStencilLocations[loc],dof-1);
  }
  *slot = stag->locationOffsets[loc] + c;
  PetscFunctionReturn(0);
}

/*@C
  DMStagMigrateVec - transfer a vector associated with a DMStag to a vector associated with a compatible DMStag

  Collective

  Input Parameters:
+ dm - the source DMStag object
. vec - the source vector, compatible with dm
. dmTo - the compatible destination DMStag object
- vecTo - the destination vector, compatible with dmTo

  Notes:
  Extra dof are ignored, and unfilled dof are zeroed.
  Currently only implemented to migrate global vectors to global vectors.

  Level: advanced

.seealso: DMSTAG, DMStagCreateCompatibleDMStag(), DMGetCompatibility(), DMStagVecSplitToDMDA()
@*/
PetscErrorCode DMStagMigrateVec(DM dm,Vec vec,DM dmTo,Vec vecTo)
{
  DM_Stag * const   stag = (DM_Stag*)dm->data;
  DM_Stag * const   stagTo = (DM_Stag*)dmTo->data;
  PetscInt          nLocalTo,nLocal,dim,i,j,k;
  PetscInt          start[DMSTAG_MAX_DIM],startGhost[DMSTAG_MAX_DIM],n[DMSTAG_MAX_DIM],nExtra[DMSTAG_MAX_DIM],offset[DMSTAG_MAX_DIM];
  Vec               vecToLocal,vecLocal;
  PetscBool         compatible,compatibleSet;
  const PetscScalar *arr;
  PetscScalar       *arrTo;
  const PetscInt    epe   = stag->entriesPerElement;
  const PetscInt    epeTo = stagTo->entriesPerElement;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscValidHeaderSpecificType(dmTo,DM_CLASSID,3,DMSTAG);
  PetscValidHeaderSpecific(vecTo,VEC_CLASSID,4);
  PetscCall(DMGetCompatibility(dm,dmTo,&compatible,&compatibleSet));
  PetscCheck(compatibleSet && compatible,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_INCOMP,"DMStag objects must be shown to be compatible");
  PetscCall(DMGetDimension(dm,&dim));
  PetscCall(VecGetLocalSize(vec,&nLocal));
  PetscCall(VecGetLocalSize(vecTo,&nLocalTo));
  PetscCheckFalse(nLocal != stag->entries|| nLocalTo !=stagTo->entries,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Vector migration only implemented for global vector to global vector.");
  PetscCall(DMStagGetCorners(dm,&start[0],&start[1],&start[2],&n[0],&n[1],&n[2],&nExtra[0],&nExtra[1],&nExtra[2]));
  PetscCall(DMStagGetGhostCorners(dm,&startGhost[0],&startGhost[1],&startGhost[2],NULL,NULL,NULL));
  for (i=0; i<DMSTAG_MAX_DIM; ++i) offset[i] = start[i]-startGhost[i];

  /* Proceed by transferring to a local vector, copying, and transferring back to a global vector */
  PetscCall(DMGetLocalVector(dm,&vecLocal));
  PetscCall(DMGetLocalVector(dmTo,&vecToLocal));
  PetscCall(DMGlobalToLocalBegin(dm,vec,INSERT_VALUES,vecLocal));
  PetscCall(DMGlobalToLocalEnd(dm,vec,INSERT_VALUES,vecLocal));
  PetscCall(VecGetArrayRead(vecLocal,&arr));
  PetscCall(VecGetArray(vecToLocal,&arrTo));
  /* Note that some superfluous copying of entries on partial dummy elements is done */
  if (dim == 1) {
    for (i=offset[0]; i<offset[0] + n[0] + nExtra[0]; ++i) {
      PetscInt d = 0,dTo = 0,b = 0,bTo = 0;
      PetscInt si;
      for (si=0; si<2; ++si) {
        b   += stag->dof[si];
        bTo += stagTo->dof[si];
        for (; d < b && dTo < bTo; ++d,++dTo) arrTo[i*epeTo + dTo] = arr[i*epe + d];
        for (; dTo < bTo         ;     ++dTo) arrTo[i*epeTo + dTo] = 0.0;
        d = b;
      }
    }
  } else if (dim == 2) {
    const PetscInt epr   = stag->nGhost[0] * epe;
    const PetscInt eprTo = stagTo->nGhost[0] * epeTo;
    for (j=offset[1]; j<offset[1] + n[1] + nExtra[1]; ++j) {
      for (i=offset[0]; i<offset[0] + n[0] + nExtra[0]; ++i) {
        const PetscInt base   = j*epr   + i*epe;
        const PetscInt baseTo = j*eprTo + i*epeTo;
        PetscInt d = 0,dTo = 0,b = 0,bTo = 0;
        const PetscInt s[4] = {0,1,1,2}; /* Dimensions of points, in order */
        PetscInt si;
        for (si=0; si<4; ++si) {
            b   += stag->dof[s[si]];
            bTo += stagTo->dof[s[si]];
            for (; d < b && dTo < bTo; ++d,++dTo) arrTo[baseTo + dTo] = arr[base + d];
            for (;          dTo < bTo;     ++dTo) arrTo[baseTo + dTo] = 0.0;
            d = b;
        }
      }
    }
  } else if (dim == 3) {
    const PetscInt epr   = stag->nGhost[0]   * epe;
    const PetscInt eprTo = stagTo->nGhost[0] * epeTo;
    const PetscInt epl   = stag->nGhost[1]   * epr;
    const PetscInt eplTo = stagTo->nGhost[1] * eprTo;
    for (k=offset[2]; k<offset[2] + n[2] + nExtra[2]; ++k) {
      for (j=offset[1]; j<offset[1] + n[1] + nExtra[1]; ++j) {
        for (i=offset[0]; i<offset[0] + n[0] + nExtra[0]; ++i) {
          PetscInt d = 0,dTo = 0,b = 0,bTo = 0;
          const PetscInt base   = k*epl   + j*epr   + i*epe;
          const PetscInt baseTo = k*eplTo + j*eprTo + i*epeTo;
          const PetscInt s[8] = {0,1,1,2,1,2,2,3}; /* dimensions of points, in order */
          PetscInt is;
          for (is=0; is<8; ++is) {
            b   += stag->dof[s[is]];
            bTo += stagTo->dof[s[is]];
            for (; d < b && dTo < bTo; ++d,++dTo) arrTo[baseTo + dTo] = arr[base + d];
            for (;          dTo < bTo;     ++dTo) arrTo[baseTo + dTo] = 0.0;
            d = b;
          }
        }
      }
    }
  } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %D",dim);
  PetscCall(VecRestoreArrayRead(vecLocal,&arr));
  PetscCall(VecRestoreArray(vecToLocal,&arrTo));
  PetscCall(DMRestoreLocalVector(dm,&vecLocal));
  PetscCall(DMLocalToGlobalBegin(dmTo,vecToLocal,INSERT_VALUES,vecTo));
  PetscCall(DMLocalToGlobalEnd(dmTo,vecToLocal,INSERT_VALUES,vecTo));
  PetscCall(DMRestoreLocalVector(dmTo,&vecToLocal));
  PetscFunctionReturn(0);
}

/*@C
  DMStagPopulateLocalToGlobalInjective - populate an internal 1-to-1 local-to-global map

  Collective

  Creates an internal object which explicitly maps a single local degree of
  freedom to each global degree of freedom. This is used, if populated,
  instead of SCATTER_REVERSE_LOCAL with the (1-to-many, in general)
  global-to-local map, when DMLocalToGlobal() is called with INSERT_VALUES.
  This allows usage, for example, even in the periodic, 1-rank case, where
  the inverse of the global-to-local map, even when restricted to on-rank
  communication, is non-injective. This is at the cost of storing an additional
  VecScatter object inside each DMStag object.

  Input Parameter:
. dm - the DMStag object

  Notes:
  In normal usage, library users shouldn't be concerned with this function,
  as it is called during DMSetUp(), when required.

  Returns immediately if the internal map is already populated.

  Developer Notes:
  This could, if desired, be moved up to a general DM routine. It would allow,
  for example, DMDA to support DMLocalToGlobal() with INSERT_VALUES,
  even in the single-rank periodic case.

  Level: developer

.seealso: DMSTAG, DMLocalToGlobal(), VecScatter
@*/
PetscErrorCode DMStagPopulateLocalToGlobalInjective(DM dm)
{
  PetscInt        dim;
  DM_Stag * const stag  = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  if (stag->ltog_injective) PetscFunctionReturn(0); /* Don't re-populate */
  PetscCall(DMGetDimension(dm,&dim));
  switch (dim) {
    case 1: PetscCall(DMStagPopulateLocalToGlobalInjective_1d(dm)); break;
    case 2: PetscCall(DMStagPopulateLocalToGlobalInjective_2d(dm)); break;
    case 3: PetscCall(DMStagPopulateLocalToGlobalInjective_3d(dm)); break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %D",dim);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMStagRestoreProductCoordinateArrays_Private(DM dm,void *arrX,void *arrY,void *arrZ,PetscBool read)
{
  PetscInt        dim,d;
  void*           arr[DMSTAG_MAX_DIM];
  DM              dmCoord;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMGetDimension(dm,&dim));
  PetscCheck(dim <= DMSTAG_MAX_DIM,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Not implemented for %D dimensions",dim);
  arr[0] = arrX; arr[1] = arrY; arr[2] = arrZ;
  PetscCall(DMGetCoordinateDM(dm,&dmCoord));
  for (d=0; d<dim; ++d) {
    DM  subDM;
    Vec coord1d_local;

    /* Ignore unrequested arrays */
    if (!arr[d]) continue;

    PetscCall(DMProductGetDM(dmCoord,d,&subDM));
    PetscCall(DMGetCoordinatesLocal(subDM,&coord1d_local));
    if (read) {
      PetscCall(DMStagVecRestoreArrayRead(subDM,coord1d_local,arr[d]));
    } else {
      PetscCall(DMStagVecRestoreArray(subDM,coord1d_local,arr[d]));
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  DMStagRestoreProductCoordinateArrays - restore local array access

  Logically Collective

  Input Parameter:
. dm - the DMStag object

  Output Parameters:
. arrX,arrY,arrZ - local 1D coordinate arrays

  Level: intermediate

  Notes:
  This function does not automatically perform a local->global scatter to populate global coordinates from the local coordinates. Thus, it may be required to explicitly perform these operations in some situations, as in the following partial example:

$   PetscCall(DMGetCoordinateDM(dm,&cdm));
$   for (d=0; d<3; ++d) {
$     DM  subdm;
$     Vec coor,coor_local;

$     PetscCall(DMProductGetDM(cdm,d,&subdm));
$     PetscCall(DMGetCoordinates(subdm,&coor));
$     PetscCall(DMGetCoordinatesLocal(subdm,&coor_local));
$     PetscCall(DMLocalToGlobal(subdm,coor_local,INSERT_VALUES,coor));
$     PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Coordinates dim %D:\n",d));
$     PetscCall(VecView(coor,PETSC_VIEWER_STDOUT_WORLD));
$   }

.seealso: DMSTAG, DMStagGetProductCoordinateArrays(), DMStagGetProductCoordinateArraysRead()
@*/
PetscErrorCode DMStagRestoreProductCoordinateArrays(DM dm,void *arrX,void *arrY,void *arrZ)
{
  PetscFunctionBegin;
  PetscCall(DMStagRestoreProductCoordinateArrays_Private(dm,arrX,arrY,arrZ,PETSC_FALSE));
  PetscFunctionReturn(0);
}

/*@C
  DMStagRestoreProductCoordinateArraysRead - restore local product array access, read-only

  Logically Collective

  Input Parameter:
. dm - the DMStag object

  Output Parameters:
. arrX,arrY,arrZ - local 1D coordinate arrays

  Level: intermediate

.seealso: DMSTAG, DMStagGetProductCoordinateArrays(), DMStagGetProductCoordinateArraysRead()
@*/
PetscErrorCode DMStagRestoreProductCoordinateArraysRead(DM dm,void *arrX,void *arrY,void *arrZ)
{
  PetscFunctionBegin;
  PetscCall(DMStagRestoreProductCoordinateArrays_Private(dm,arrX,arrY,arrZ,PETSC_TRUE));
  PetscFunctionReturn(0);
}

/*@C
  DMStagSetBoundaryTypes - set DMStag boundary types

  Logically Collective; boundaryType0, boundaryType1, and boundaryType2 must contain common values

  Input Parameters:
+ dm - the DMStag object
- boundaryType0,boundaryType1,boundaryType2 - boundary types in each direction

  Notes:
  Arguments corresponding to higher dimensions are ignored for 1D and 2D grids.

  Level: advanced

.seealso: DMSTAG, DMBoundaryType, DMStagCreate1d(), DMStagCreate2d(), DMStagCreate3d(), DMDASetBoundaryType()
@*/
PetscErrorCode DMStagSetBoundaryTypes(DM dm,DMBoundaryType boundaryType0,DMBoundaryType boundaryType1,DMBoundaryType boundaryType2)
{
  DM_Stag * const stag  = (DM_Stag*)dm->data;
  PetscInt        dim;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm,&dim));
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidLogicalCollectiveEnum(dm,boundaryType0,2);
  if (dim > 1) PetscValidLogicalCollectiveEnum(dm,boundaryType1,3);
  if (dim > 2) PetscValidLogicalCollectiveEnum(dm,boundaryType2,4);
  PetscCheck(!dm->setupcalled,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"This function must be called before DMSetUp()");
               stag->boundaryType[0] = boundaryType0;
  if (dim > 1) stag->boundaryType[1] = boundaryType1;
  if (dim > 2) stag->boundaryType[2] = boundaryType2;
  PetscFunctionReturn(0);
}

/*@C
  DMStagSetCoordinateDMType - set DM type to store coordinates

  Logically Collective; dmtype must contain common value

  Input Parameters:
+ dm - the DMStag object
- dmtype - DMtype for coordinates, either DMSTAG or DMPRODUCT

  Level: advanced

.seealso: DMSTAG, DMPRODUCT, DMGetCoordinateDM(), DMStagSetUniformCoordinates(), DMStagSetUniformCoordinatesExplicit(), DMStagSetUniformCoordinatesProduct(), DMType
@*/
PetscErrorCode DMStagSetCoordinateDMType(DM dm,DMType dmtype)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscCall(PetscFree(stag->coordinateDMType));
  PetscCall(PetscStrallocpy(dmtype,(char**)&stag->coordinateDMType));
  PetscFunctionReturn(0);
}

/*@C
  DMStagSetDOF - set dof/stratum

  Logically Collective; dof0, dof1, dof2, and dof3 must contain common values

  Input Parameters:
+ dm - the DMStag object
- dof0,dof1,dof2,dof3 - dof per stratum

  Notes:
  Arguments corresponding to higher dimensions are ignored for 1D and 2D grids.

  Level: advanced

.seealso: DMSTAG, DMDASetDof()
@*/
PetscErrorCode DMStagSetDOF(DM dm,PetscInt dof0, PetscInt dof1,PetscInt dof2,PetscInt dof3)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        dim;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidLogicalCollectiveInt(dm,dof0,2);
  PetscValidLogicalCollectiveInt(dm,dof1,3);
  PetscValidLogicalCollectiveInt(dm,dof2,4);
  PetscValidLogicalCollectiveInt(dm,dof3,5);
  PetscCheck(!dm->setupcalled,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"This function must be called before DMSetUp()");
  PetscCall(DMGetDimension(dm,&dim));
  PetscCheck(dof0 >= 0,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"dof0 cannot be negative");
  PetscCheck(dof1 >= 0,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"dof1 cannot be negative");
  PetscCheckFalse(dim > 1 && dof2 < 0,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"dof2 cannot be negative");
  PetscCheckFalse(dim > 2 && dof3 < 0,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"dof3 cannot be negative");
               stag->dof[0] = dof0;
               stag->dof[1] = dof1;
  if (dim > 1) stag->dof[2] = dof2;
  if (dim > 2) stag->dof[3] = dof3;
  PetscFunctionReturn(0);
}

/*@C
  DMStagSetNumRanks - set ranks in each direction in the global rank grid

  Logically Collective; nRanks0, nRanks1, and nRanks2 must contain common values

  Input Parameters:
+ dm - the DMStag object
- nRanks0,nRanks1,nRanks2 - number of ranks in each direction

  Notes:
  Arguments corresponding to higher dimensions are ignored for 1D and 2D grids.

  Level: developer

.seealso: DMSTAG, DMDASetNumProcs()
@*/
PetscErrorCode DMStagSetNumRanks(DM dm,PetscInt nRanks0,PetscInt nRanks1,PetscInt nRanks2)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        dim;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidLogicalCollectiveInt(dm,nRanks0,2);
  PetscValidLogicalCollectiveInt(dm,nRanks1,3);
  PetscValidLogicalCollectiveInt(dm,nRanks2,4);
  PetscCheck(!dm->setupcalled,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"This function must be called before DMSetUp()");
  PetscCall(DMGetDimension(dm,&dim));
  PetscCheckFalse(nRanks0 != PETSC_DECIDE && nRanks0 < 1,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"number of ranks in X direction cannot be less than 1");
  PetscCheckFalse(dim > 1 && nRanks1 != PETSC_DECIDE && nRanks1 < 1,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"number of ranks in Y direction cannot be less than 1");
  PetscCheckFalse(dim > 2 && nRanks2 != PETSC_DECIDE && nRanks2 < 1,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"number of ranks in Z direction cannot be less than 1");
  if (nRanks0) stag->nRanks[0] = nRanks0;
  if (dim > 1 && nRanks1) stag->nRanks[1] = nRanks1;
  if (dim > 2 && nRanks2) stag->nRanks[2] = nRanks2;
  PetscFunctionReturn(0);
}

/*@C
  DMStagSetStencilType - set elementwise ghost/halo stencil type

  Logically Collective; stencilType must contain common value

  Input Parameters:
+ dm - the DMStag object
- stencilType - the elementwise ghost stencil type: DMSTAG_STENCIL_BOX, DMSTAG_STENCIL_STAR, or DMSTAG_STENCIL_NONE

  Level: beginner

.seealso: DMSTAG, DMStagGetStencilType(), DMStagSetStencilWidth(), DMStagStencilType
@*/
PetscErrorCode DMStagSetStencilType(DM dm,DMStagStencilType stencilType)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidLogicalCollectiveEnum(dm,stencilType,2);
  PetscCheck(!dm->setupcalled,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"This function must be called before DMSetUp()");
  stag->stencilType = stencilType;
  PetscFunctionReturn(0);
}

/*@C
  DMStagSetStencilWidth - set elementwise stencil width

  Logically Collective; stencilWidth must contain common value

  Input Parameters:
+ dm - the DMStag object
- stencilWidth - stencil/halo/ghost width in elements

  Notes:
  The width value is not used when DMSTAG_STENCIL_NONE is specified.

  Level: beginner

.seealso: DMSTAG, DMStagGetStencilWidth(), DMStagGetStencilType(), DMStagStencilType
@*/
PetscErrorCode DMStagSetStencilWidth(DM dm,PetscInt stencilWidth)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidLogicalCollectiveInt(dm,stencilWidth,2);
  PetscCheck(!dm->setupcalled,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"This function must be called before DMSetUp()");
  PetscCheck(stencilWidth >= 0,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Stencil width must be non-negative");
  stag->stencilWidth = stencilWidth;
  PetscFunctionReturn(0);
}

/*@C
  DMStagSetGlobalSizes - set global element counts in each direction

  Logically Collective; N0, N1, and N2 must contain common values

  Input Parameters:
+ dm - the DMStag object
- N0,N1,N2 - global elementwise sizes

  Notes:
  Arguments corresponding to higher dimensions are ignored for 1D and 2D grids.

  Level: advanced

.seealso: DMSTAG, DMStagGetGlobalSizes(), DMDASetSizes()
@*/
PetscErrorCode DMStagSetGlobalSizes(DM dm,PetscInt N0,PetscInt N1,PetscInt N2)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        dim;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscCheck(!dm->setupcalled,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"This function must be called before DMSetUp()");
  PetscCall(DMGetDimension(dm,&dim));
  PetscCheck(N0 >= 1,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_SIZ,"Number of elements in X direction must be positive");
  PetscCheckFalse(dim > 1 && N1 < 0,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_SIZ,"Number of elements in Y direction must be positive");
  PetscCheckFalse(dim > 2 && N2 < 0,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_SIZ,"Number of elements in Z direction must be positive");
  if (N0) stag->N[0] = N0;
  if (N1) stag->N[1] = N1;
  if (N2) stag->N[2] = N2;
  PetscFunctionReturn(0);
}

/*@C
  DMStagSetOwnershipRanges - set elements per rank in each direction

  Logically Collective; lx, ly, and lz must contain common values

  Input Parameters:
+ dm - the DMStag object
- lx,ly,lz - element counts for each rank in each direction

  Notes:
  Arguments corresponding to higher dimensions are ignored for 1D and 2D grids. These arguments may be set to NULL in this case.

  Level: developer

.seealso: DMSTAG, DMStagSetGlobalSizes(), DMStagGetOwnershipRanges(), DMDASetOwnershipRanges()
@*/
PetscErrorCode DMStagSetOwnershipRanges(DM dm,PetscInt const *lx,PetscInt const *ly,PetscInt const *lz)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;
  const PetscInt  *lin[3];
  PetscInt        d,dim;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscCheck(!dm->setupcalled,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"This function must be called before DMSetUp()");
  lin[0] = lx; lin[1] = ly; lin[2] = lz;
  PetscCall(DMGetDimension(dm,&dim));
  for (d=0; d<dim; ++d) {
    if (lin[d]) {
      PetscCheck(stag->nRanks[d] >= 0,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot set ownership ranges before setting number of ranks");
      if (!stag->l[d]) {
        PetscCall(PetscMalloc1(stag->nRanks[d], &stag->l[d]));
      }
      PetscCall(PetscArraycpy(stag->l[d], lin[d], stag->nRanks[d]));
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  DMStagSetUniformCoordinates - set DMStag coordinates to be a uniform grid

  Collective

  Input Parameters:
+ dm - the DMStag object
- xmin,xmax,ymin,ymax,zmin,zmax - maximum and minimum global coordinate values

  Notes:
  DMStag supports 2 different types of coordinate DM: DMSTAG and DMPRODUCT.
  Arguments corresponding to higher dimensions are ignored for 1D and 2D grids.

  Local coordinates are populated (using DMSetCoordinatesLocal()), linearly
  extrapolated to ghost cells, including those outside the physical domain.
  This is also done in case of periodic boundaries, meaning that the same
  global point may have different coordinates in different local representations,
  which are equivalent assuming a periodicity implied by the arguments to this function,
  i.e. two points are equivalent if their difference is a multiple of (xmax - xmin)
  in the x direction, (ymax - ymin) in the y direction, and (zmax - zmin) in the z direction.

  Level: advanced

.seealso: DMSTAG, DMPRODUCT, DMStagSetUniformCoordinatesExplicit(), DMStagSetUniformCoordinatesProduct(), DMStagSetCoordinateDMType(), DMGetCoordinateDM(), DMGetCoordinates(), DMDASetUniformCoordinates(), DMBoundaryType
@*/
PetscErrorCode DMStagSetUniformCoordinates(DM dm,PetscReal xmin,PetscReal xmax,PetscReal ymin,PetscReal ymax,PetscReal zmin,PetscReal zmax)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscBool       flg_stag,flg_product;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscCheck(dm->setupcalled,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"This function must be called after DMSetUp()");
  PetscCheck(stag->coordinateDMType,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"You must first call DMStagSetCoordinateDMType()");
  PetscCall(PetscStrcmp(stag->coordinateDMType,DMSTAG,&flg_stag));
  PetscCall(PetscStrcmp(stag->coordinateDMType,DMPRODUCT,&flg_product));
  if (flg_stag) {
    PetscCall(DMStagSetUniformCoordinatesExplicit(dm,xmin,xmax,ymin,ymax,zmin,zmax));
  } else if (flg_product) {
    PetscCall(DMStagSetUniformCoordinatesProduct(dm,xmin,xmax,ymin,ymax,zmin,zmax));
  } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Unsupported DM Type %s",stag->coordinateDMType);
  PetscFunctionReturn(0);
}

/*@C
  DMStagSetUniformCoordinatesExplicit - set DMStag coordinates to be a uniform grid, storing all values

  Collective

  Input Parameters:
+ dm - the DMStag object
- xmin,xmax,ymin,ymax,zmin,zmax - maximum and minimum global coordinate values

  Notes:
  DMStag supports 2 different types of coordinate DM: either another DMStag, or a DMProduct.
  If the grid is orthogonal, using DMProduct should be more efficient.

  Arguments corresponding to higher dimensions are ignored for 1D and 2D grids.

  See the manual page for DMStagSetUniformCoordinates() for information on how
  coordinates for dummy cells outside the physical domain boundary are populated.

  Level: beginner

.seealso: DMSTAG, DMStagSetUniformCoordinates(), DMStagSetUniformCoordinatesProduct(), DMStagSetCoordinateDMType()
@*/
PetscErrorCode DMStagSetUniformCoordinatesExplicit(DM dm,PetscReal xmin,PetscReal xmax,PetscReal ymin,PetscReal ymax,PetscReal zmin,PetscReal zmax)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        dim;
  PetscBool       flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscCheck(dm->setupcalled,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"This function must be called after DMSetUp()");
  PetscCall(PetscStrcmp(stag->coordinateDMType,DMSTAG,&flg));
  PetscCheck(!stag->coordinateDMType || flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Refusing to change an already-set DM coordinate type");
  PetscCall(DMStagSetCoordinateDMType(dm,DMSTAG));
  PetscCall(DMGetDimension(dm,&dim));
  switch (dim) {
  case 1: PetscCall(DMStagSetUniformCoordinatesExplicit_1d(dm,xmin,xmax));                     break;
  case 2: PetscCall(DMStagSetUniformCoordinatesExplicit_2d(dm,xmin,xmax,ymin,ymax));           break;
  case 3: PetscCall(DMStagSetUniformCoordinatesExplicit_3d(dm,xmin,xmax,ymin,ymax,zmin,zmax)); break;
  default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %D",dim);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMStagSetUniformCoordinatesProduct - create uniform coordinates, as a product of 1D arrays

  Set the coordinate DM to be a DMProduct of 1D DMStag objects, each of which have a coordinate DM (also a 1d DMStag) holding uniform coordinates.

  Collective

  Input Parameters:
+ dm - the DMStag object
- xmin,xmax,ymin,ymax,zmin,zmax - maximum and minimum global coordinate values

  Notes:
  Arguments corresponding to higher dimensions are ignored for 1D and 2D grids.

  The per-dimension 1-dimensional DMStag objects that comprise the product
  always have active 0-cells (vertices, element boundaries) and 1-cells
  (element centers).

  See the manual page for DMStagSetUniformCoordinates() for information on how
  coordinates for dummy cells outside the physical domain boundary are populated.

  Level: intermediate

.seealso: DMSTAG, DMPRODUCT, DMStagSetUniformCoordinates(), DMStagSetUniformCoordinatesExplicit(), DMStagSetCoordinateDMType()
@*/
PetscErrorCode DMStagSetUniformCoordinatesProduct(DM dm,PetscReal xmin,PetscReal xmax,PetscReal ymin,PetscReal ymax,PetscReal zmin,PetscReal zmax)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;
  DM              dmc;
  PetscInt        dim,d,dof0,dof1;
  PetscBool       flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscCheck(dm->setupcalled,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"This function must be called after DMSetUp()");
  PetscCall(PetscStrcmp(stag->coordinateDMType,DMPRODUCT,&flg));
  PetscCheck(!stag->coordinateDMType || flg,PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Refusing to change an already-set DM coordinate type");
  PetscCall(DMStagSetCoordinateDMType(dm,DMPRODUCT));
  PetscCall(DMGetCoordinateDM(dm,&dmc));
  PetscCall(DMGetDimension(dm,&dim));

  /* Create 1D sub-DMs, living on subcommunicators.
     Always include both vertex and element dof, regardless of the active strata of the DMStag */
  dof0 = 1;
  dof1 = 1;

  for (d=0; d<dim; ++d) {
    DM                subdm;
    MPI_Comm          subcomm;
    PetscMPIInt       color;
    const PetscMPIInt key = 0; /* let existing rank break ties */

    /* Choose colors based on position in the plane orthogonal to this dim, and split */
    switch (d) {
      case 0: color = (dim > 1 ? stag->rank[1] : 0)  + (dim > 2 ? stag->nRanks[1]*stag->rank[2] : 0); break;
      case 1: color =            stag->rank[0]       + (dim > 2 ? stag->nRanks[0]*stag->rank[2] : 0); break;
      case 2: color =            stag->rank[0]       +            stag->nRanks[0]*stag->rank[1]     ; break;
      default: SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP,"Unsupported dimension index %D",d);
    }
    PetscCallMPI(MPI_Comm_split(PetscObjectComm((PetscObject)dm),color,key,&subcomm));

    /* Create sub-DMs living on these new communicators (which are destroyed by DMProduct) */
    PetscCall(DMStagCreate1d(subcomm,stag->boundaryType[d],stag->N[d],dof0,dof1,stag->stencilType,stag->stencilWidth,stag->l[d],&subdm));
    PetscCall(DMSetUp(subdm));
    switch (d) {
      case 0:
        PetscCall(DMStagSetUniformCoordinatesExplicit(subdm,xmin,xmax,0,0,0,0));
        break;
      case 1:
        PetscCall(DMStagSetUniformCoordinatesExplicit(subdm,ymin,ymax,0,0,0,0));
        break;
      case 2:
        PetscCall(DMStagSetUniformCoordinatesExplicit(subdm,zmin,zmax,0,0,0,0));
        break;
      default: SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP,"Unsupported dimension index %D",d);
    }
    PetscCall(DMProductSetDM(dmc,d,subdm));
    PetscCall(DMProductSetDimensionIndex(dmc,d,0));
    PetscCall(DMDestroy(&subdm));
    PetscCallMPI(MPI_Comm_free(&subcomm));
  }
  PetscFunctionReturn(0);
}

/*@C
  DMStagVecGetArray - get access to local array

  Logically Collective

  This function returns a (dim+1)-dimensional array for a dim-dimensional
  DMStag.

  The first 1-3 dimensions indicate an element in the global
  numbering, using the standard C ordering.

  The final dimension in this array corresponds to a degree
  of freedom with respect to this element, for example corresponding to
  the element or one of its neighboring faces, edges, or vertices.

  For example, for a 3D DMStag, indexing is array[k][j][i][idx], where k is the
  index in the z-direction, j is the index in the y-direction, and i is the
  index in the x-direction.

  "idx" is obtained with DMStagGetLocationSlot(), since the correct offset
  into the (dim+1)-dimensional C array depends on the grid size and the number
  of dof stored at each location.

  Input Parameters:
+ dm - the DMStag object
- vec - the Vec object

  Output Parameters:
. array - the array

  Notes:
  DMStagVecRestoreArray() must be called, once finished with the array

  Level: beginner

.seealso: DMSTAG, DMStagVecGetArrayRead(), DMStagGetLocationSlot(), DMGetLocalVector(), DMCreateLocalVector(), DMGetGlobalVector(), DMCreateGlobalVector(), DMDAVecGetArray(), DMDAVecGetArrayDOF()
@*/
PetscErrorCode DMStagVecGetArray(DM dm,Vec vec,void *array)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        dim;
  PetscInt        nLocal;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscCall(DMGetDimension(dm,&dim));
  PetscCall(VecGetLocalSize(vec,&nLocal));
  PetscCheck(nLocal == stag->entriesGhost,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Vector local size %D is not compatible with DMStag local size %D",nLocal,stag->entriesGhost);
  switch (dim) {
    case 1:
      PetscCall(VecGetArray2d(vec,stag->nGhost[0],stag->entriesPerElement,stag->startGhost[0],0,(PetscScalar***)array));
      break;
    case 2:
      PetscCall(VecGetArray3d(vec,stag->nGhost[1],stag->nGhost[0],stag->entriesPerElement,stag->startGhost[1],stag->startGhost[0],0,(PetscScalar****)array));
      break;
    case 3:
      PetscCall(VecGetArray4d(vec,stag->nGhost[2],stag->nGhost[1],stag->nGhost[0],stag->entriesPerElement,stag->startGhost[2],stag->startGhost[1],stag->startGhost[0],0,(PetscScalar*****)array));
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %D",dim);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMStagVecGetArrayRead - get read-only access to a local array

  Logically Collective

  See the man page for DMStagVecGetArray() for more information.

  Input Parameters:
+ dm - the DMStag object
- vec - the Vec object

  Output Parameters:
. array - the read-only array

  Notes:
  DMStagVecRestoreArrayRead() must be called, once finished with the array

  Level: beginner

.seealso: DMSTAG, DMStagVecGetArrayRead(), DMStagGetLocationSlot(), DMGetLocalVector(), DMCreateLocalVector(), DMGetGlobalVector(), DMCreateGlobalVector(), DMDAVecGetArrayRead(), DMDAVecGetArrayDOFRead()
@*/
PetscErrorCode DMStagVecGetArrayRead(DM dm,Vec vec,void *array)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        dim;
  PetscInt        nLocal;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscCall(DMGetDimension(dm,&dim));
  PetscCall(VecGetLocalSize(vec,&nLocal));
  PetscCheck(nLocal == stag->entriesGhost,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Vector local size %D is not compatible with DMStag local size %D",nLocal,stag->entriesGhost);
  switch (dim) {
    case 1:
      PetscCall(VecGetArray2dRead(vec,stag->nGhost[0],stag->entriesPerElement,stag->startGhost[0],0,(PetscScalar***)array));
      break;
    case 2:
      PetscCall(VecGetArray3dRead(vec,stag->nGhost[1],stag->nGhost[0],stag->entriesPerElement,stag->startGhost[1],stag->startGhost[0],0,(PetscScalar****)array));
      break;
    case 3:
      PetscCall(VecGetArray4dRead(vec,stag->nGhost[2],stag->nGhost[1],stag->nGhost[0],stag->entriesPerElement,stag->startGhost[2],stag->startGhost[1],stag->startGhost[0],0,(PetscScalar*****)array));
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %D",dim);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMStagVecRestoreArray - restore access to a raw array

  Logically Collective

  Input Parameters:
+ dm - the DMStag object
- vec - the Vec object

  Output Parameters:
. array - the array

  Level: beginner

.seealso: DMSTAG, DMStagVecGetArray(), DMDAVecRestoreArray(), DMDAVecRestoreArrayDOF()
@*/
PetscErrorCode DMStagVecRestoreArray(DM dm,Vec vec,void *array)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        dim;
  PetscInt        nLocal;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscCall(DMGetDimension(dm,&dim));
  PetscCall(VecGetLocalSize(vec,&nLocal));
  PetscCheck(nLocal == stag->entriesGhost,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Vector local size %D is not compatible with DMStag local size %D",nLocal,stag->entriesGhost);
  switch (dim) {
    case 1:
      PetscCall(VecRestoreArray2d(vec,stag->nGhost[0],stag->entriesPerElement,stag->startGhost[0],0,(PetscScalar***)array));
      break;
    case 2:
      PetscCall(VecRestoreArray3d(vec,stag->nGhost[1],stag->nGhost[0],stag->entriesPerElement,stag->startGhost[1],stag->startGhost[0],0,(PetscScalar****)array));
      break;
    case 3:
      PetscCall(VecRestoreArray4d(vec,stag->nGhost[2],stag->nGhost[1],stag->nGhost[0],stag->entriesPerElement,stag->startGhost[2],stag->startGhost[1],stag->startGhost[0],0,(PetscScalar*****)array));
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %D",dim);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMStagVecRestoreArrayRead - restore read-only access to a raw array

  Logically Collective

  Input Parameters:
+ dm - the DMStag object
- vec - the Vec object

  Output Parameters:
. array - the read-only array

  Level: beginner

.seealso: DMSTAG, DMStagVecGetArrayRead(), DMDAVecRestoreArrayRead(), DMDAVecRestoreArrayDOFRead()
@*/
PetscErrorCode DMStagVecRestoreArrayRead(DM dm,Vec vec,void *array)
{
  DM_Stag * const stag = (DM_Stag*)dm->data;
  PetscInt        dim;
  PetscInt        nLocal;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm,DM_CLASSID,1,DMSTAG);
  PetscValidHeaderSpecific(vec,VEC_CLASSID,2);
  PetscCall(DMGetDimension(dm,&dim));
  PetscCall(VecGetLocalSize(vec,&nLocal));
  PetscCheck(nLocal == stag->entriesGhost,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Vector local size %D is not compatible with DMStag local size %D",nLocal,stag->entriesGhost);
  switch (dim) {
    case 1:
      PetscCall(VecRestoreArray2dRead(vec,stag->nGhost[0],stag->entriesPerElement,stag->startGhost[0],0,(PetscScalar***)array));
      break;
    case 2:
      PetscCall(VecRestoreArray3dRead(vec,stag->nGhost[1],stag->nGhost[0],stag->entriesPerElement,stag->startGhost[1],stag->startGhost[0],0,(PetscScalar****)array));
      break;
    case 3:
      PetscCall(VecRestoreArray4dRead(vec,stag->nGhost[2],stag->nGhost[1],stag->nGhost[0],stag->entriesPerElement,stag->startGhost[2],stag->startGhost[1],stag->startGhost[0],0,(PetscScalar*****)array));
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Unsupported dimension %D",dim);
  }
  PetscFunctionReturn(0);
}
