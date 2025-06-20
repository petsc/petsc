#include <petsc/private/dmimpl.h> /*I      "petscdm.h"          I*/

#include <petscdmplex.h> /* For DMCreateAffineCoordinates_Internal() */
#include <petscsf.h>     /* For DMLocatePoints() */

PetscErrorCode DMRestrictHook_Coordinates(DM dm, DM dmc, void *ctx)
{
  DM  dm_coord, dmc_coord;
  Vec coords, ccoords;
  Mat inject;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(dm, &dm_coord));
  PetscCall(DMGetCoordinateDM(dmc, &dmc_coord));
  PetscCall(DMGetCoordinates(dm, &coords));
  PetscCall(DMGetCoordinates(dmc, &ccoords));
  if (coords && !ccoords) {
    PetscCall(DMCreateGlobalVector(dmc_coord, &ccoords));
    PetscCall(PetscObjectSetName((PetscObject)ccoords, "coordinates"));
    PetscCall(DMCreateInjection(dmc_coord, dm_coord, &inject));
    PetscCall(MatRestrict(inject, coords, ccoords));
    PetscCall(MatDestroy(&inject));
    PetscCall(DMSetCoordinates(dmc, ccoords));
    PetscCall(VecDestroy(&ccoords));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSubDomainHook_Coordinates(DM dm, DM subdm, void *ctx)
{
  DM          dm_coord, subdm_coord;
  Vec         coords, ccoords, clcoords;
  VecScatter *scat_i, *scat_g;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDM(dm, &dm_coord));
  PetscCall(DMGetCoordinateDM(subdm, &subdm_coord));
  PetscCall(DMGetCoordinates(dm, &coords));
  PetscCall(DMGetCoordinates(subdm, &ccoords));
  if (coords && !ccoords) {
    PetscCall(DMCreateGlobalVector(subdm_coord, &ccoords));
    PetscCall(PetscObjectSetName((PetscObject)ccoords, "coordinates"));
    PetscCall(DMCreateLocalVector(subdm_coord, &clcoords));
    PetscCall(PetscObjectSetName((PetscObject)clcoords, "coordinates"));
    PetscCall(DMCreateDomainDecompositionScatters(dm_coord, 1, &subdm_coord, NULL, &scat_i, &scat_g));
    PetscCall(VecScatterBegin(scat_i[0], coords, ccoords, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scat_i[0], coords, ccoords, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterBegin(scat_g[0], coords, clcoords, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(scat_g[0], coords, clcoords, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(DMSetCoordinates(subdm, ccoords));
    PetscCall(DMSetCoordinatesLocal(subdm, clcoords));
    PetscCall(VecScatterDestroy(&scat_i[0]));
    PetscCall(VecScatterDestroy(&scat_g[0]));
    PetscCall(VecDestroy(&ccoords));
    PetscCall(VecDestroy(&clcoords));
    PetscCall(PetscFree(scat_i));
    PetscCall(PetscFree(scat_g));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetCoordinateDM - Gets the `DM` that prescribes coordinate layout and scatters between global and local coordinates

  Collective

  Input Parameter:
. dm - the `DM`

  Output Parameter:
. cdm - coordinate `DM`

  Level: intermediate

.seealso: `DM`, `DMSetCoordinateDM()`, `DMSetCoordinates()`, `DMSetCoordinatesLocal()`, `DMGetCoordinates()`, `DMGetCoordinatesLocal()`, `DMGSetCellCoordinateDM()`,

@*/
PetscErrorCode DMGetCoordinateDM(DM dm, DM *cdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(cdm, 2);
  if (!dm->coordinates[0].dm) {
    DM cdm;

    PetscUseTypeMethod(dm, createcoordinatedm, &cdm);
    PetscCall(PetscObjectSetName((PetscObject)cdm, "coordinateDM"));
    /* Just in case the DM sets the coordinate DM when creating it (DMP4est can do this, because it may not setup
     * until the call to CreateCoordinateDM) */
    PetscCall(DMDestroy(&dm->coordinates[0].dm));
    dm->coordinates[0].dm = cdm;
  }
  *cdm = dm->coordinates[0].dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetCoordinateDM - Sets the `DM` that prescribes coordinate layout and scatters between global and local coordinates

  Logically Collective

  Input Parameters:
+ dm  - the `DM`
- cdm - coordinate `DM`

  Level: intermediate

.seealso: `DM`, `DMGetCoordinateDM()`, `DMSetCoordinates()`, `DMGetCellCoordinateDM()`, `DMSetCoordinatesLocal()`, `DMGetCoordinates()`, `DMGetCoordinatesLocal()`,
          `DMGSetCellCoordinateDM()`
@*/
PetscErrorCode DMSetCoordinateDM(DM dm, DM cdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (cdm) PetscValidHeaderSpecific(cdm, DM_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)cdm));
  PetscCall(DMDestroy(&dm->coordinates[0].dm));
  dm->coordinates[0].dm = cdm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetCellCoordinateDM - Gets the `DM` that prescribes cellwise coordinate layout and scatters between global and local cellwise coordinates

  Collective

  Input Parameter:
. dm - the `DM`

  Output Parameter:
. cdm - cellwise coordinate `DM`, or `NULL` if they are not defined

  Level: intermediate

  Note:
  Call `DMLocalizeCoordinates()` to automatically create cellwise coordinates for periodic geometries.

.seealso: `DM`, `DMSetCellCoordinateDM()`, `DMSetCellCoordinates()`, `DMSetCellCoordinatesLocal()`, `DMGetCellCoordinates()`, `DMGetCellCoordinatesLocal()`,
          `DMLocalizeCoordinates()`, `DMSetCoordinateDM()`, `DMGetCoordinateDM()`
@*/
PetscErrorCode DMGetCellCoordinateDM(DM dm, DM *cdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(cdm, 2);
  *cdm = dm->coordinates[1].dm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetCellCoordinateDM - Sets the `DM` that prescribes cellwise coordinate layout and scatters between global and local cellwise coordinates

  Logically Collective

  Input Parameters:
+ dm  - the `DM`
- cdm - cellwise coordinate `DM`

  Level: intermediate

  Note:
  As opposed to `DMSetCoordinateDM()` these coordinates are useful for discontinuous Galerkin methods since they support coordinate fields that are discontinuous at cell boundaries.

.seealso: `DMGetCellCoordinateDM()`, `DMSetCellCoordinates()`, `DMSetCellCoordinatesLocal()`, `DMGetCellCoordinates()`, `DMGetCellCoordinatesLocal()`,
          `DMSetCoordinateDM()`, `DMGetCoordinateDM()`
@*/
PetscErrorCode DMSetCellCoordinateDM(DM dm, DM cdm)
{
  PetscInt dim;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (cdm) {
    PetscValidHeaderSpecific(cdm, DM_CLASSID, 2);
    PetscCall(DMGetCoordinateDim(dm, &dim));
    dm->coordinates[1].dim = dim;
  }
  PetscCall(PetscObjectReference((PetscObject)cdm));
  PetscCall(DMDestroy(&dm->coordinates[1].dm));
  dm->coordinates[1].dm = cdm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetCoordinateDim - Retrieve the dimension of the embedding space for coordinate values. For example a mesh on the surface of a sphere would have a 3 dimensional embedding space

  Not Collective

  Input Parameter:
. dm - The `DM` object

  Output Parameter:
. dim - The embedding dimension

  Level: intermediate

.seealso: `DM`, `DMSetCoordinateDim()`, `DMGetCoordinateSection()`, `DMGetCoordinateDM()`, `DMGetLocalSection()`, `DMSetLocalSection()`
@*/
PetscErrorCode DMGetCoordinateDim(DM dm, PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(dim, 2);
  if (dm->coordinates[0].dim == PETSC_DEFAULT) dm->coordinates[0].dim = dm->dim;
  *dim = dm->coordinates[0].dim;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetCoordinateDim - Set the dimension of the embedding space for coordinate values.

  Not Collective

  Input Parameters:
+ dm  - The `DM` object
- dim - The embedding dimension

  Level: intermediate

.seealso: `DM`, `DMGetCoordinateDim()`, `DMSetCoordinateSection()`, `DMGetCoordinateSection()`, `DMGetLocalSection()`, `DMSetLocalSection()`
@*/
PetscErrorCode DMSetCoordinateDim(DM dm, PetscInt dim)
{
  PetscDS  ds;
  PetscInt Nds, n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  dm->coordinates[0].dim = dim;
  if (dm->dim >= 0) {
    PetscCall(DMGetNumDS(dm, &Nds));
    for (n = 0; n < Nds; ++n) {
      PetscCall(DMGetRegionNumDS(dm, n, NULL, NULL, &ds, NULL));
      PetscCall(PetscDSSetCoordinateDimension(ds, dim));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetCoordinateSection - Retrieve the `PetscSection` of coordinate values over the mesh.

  Collective

  Input Parameter:
. dm - The `DM` object

  Output Parameter:
. section - The `PetscSection` object

  Level: intermediate

  Note:
  This just retrieves the local section from the coordinate `DM`. In other words,
.vb
  DMGetCoordinateDM(dm, &cdm);
  DMGetLocalSection(cdm, &section);
.ve

.seealso: `DM`, `DMGetCoordinateDM()`, `DMGetLocalSection()`, `DMSetLocalSection()`
@*/
PetscErrorCode DMGetCoordinateSection(DM dm, PetscSection *section)
{
  DM cdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(section, 2);
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetLocalSection(cdm, section));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetCoordinateSection - Set the `PetscSection` of coordinate values over the mesh.

  Not Collective

  Input Parameters:
+ dm      - The `DM` object
. dim     - The embedding dimension, or `PETSC_DETERMINE`
- section - The `PetscSection` object

  Level: intermediate

.seealso: `DM`, `DMGetCoordinateDim()`, `DMGetCoordinateSection()`, `DMGetLocalSection()`, `DMSetLocalSection()`
@*/
PetscErrorCode DMSetCoordinateSection(DM dm, PetscInt dim, PetscSection section)
{
  DM cdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 3);
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMSetLocalSection(cdm, section));
  if (dim == PETSC_DETERMINE) {
    PetscInt d = PETSC_DEFAULT;
    PetscInt pStart, pEnd, vStart, vEnd, v, dd;

    PetscCall(PetscSectionGetChart(section, &pStart, &pEnd));
    PetscCall(DMGetDimPoints(dm, 0, &vStart, &vEnd));
    pStart = PetscMax(vStart, pStart);
    pEnd   = PetscMin(vEnd, pEnd);
    for (v = pStart; v < pEnd; ++v) {
      PetscCall(PetscSectionGetDof(section, v, &dd));
      if (dd) {
        d = dd;
        break;
      }
    }
    if (d >= 0) PetscCall(DMSetCoordinateDim(dm, d));
  } else {
    PetscCall(DMSetCoordinateDim(dm, dim));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetCellCoordinateSection - Retrieve the `PetscSection` of cellwise coordinate values over the mesh.

  Collective

  Input Parameter:
. dm - The `DM` object

  Output Parameter:
. section - The `PetscSection` object, or `NULL` if no cellwise coordinates are defined

  Level: intermediate

  Note:
  This just retrieves the local section from the cell coordinate `DM`. In other words,
.vb
  DMGetCellCoordinateDM(dm, &cdm);
  DMGetLocalSection(cdm, &section);
.ve

.seealso: `DM`, `DMGetCoordinateSection()`, `DMSetCellCoordinateSection()`, `DMGetCellCoordinateDM()`, `DMGetCoordinateDM()`, `DMGetLocalSection()`, `DMSetLocalSection()`
@*/
PetscErrorCode DMGetCellCoordinateSection(DM dm, PetscSection *section)
{
  DM cdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(section, 2);
  *section = NULL;
  PetscCall(DMGetCellCoordinateDM(dm, &cdm));
  if (cdm) PetscCall(DMGetLocalSection(cdm, section));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetCellCoordinateSection - Set the `PetscSection` of cellwise coordinate values over the mesh.

  Not Collective

  Input Parameters:
+ dm      - The `DM` object
. dim     - The embedding dimension, or `PETSC_DETERMINE`
- section - The `PetscSection` object for a cellwise layout

  Level: intermediate

.seealso: `DM`, `DMGetCoordinateDim()`, `DMSetCoordinateSection()`, `DMGetCellCoordinateSection()`, `DMGetCoordinateSection()`, `DMGetCellCoordinateDM()`, `DMGetLocalSection()`, `DMSetLocalSection()`
@*/
PetscErrorCode DMSetCellCoordinateSection(DM dm, PetscInt dim, PetscSection section)
{
  DM cdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(section, PETSC_SECTION_CLASSID, 3);
  PetscCall(DMGetCellCoordinateDM(dm, &cdm));
  PetscCheck(cdm, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "No DM defined for cellwise coordinates");
  PetscCall(DMSetLocalSection(cdm, section));
  if (dim == PETSC_DETERMINE) {
    PetscInt d = PETSC_DEFAULT;
    PetscInt pStart, pEnd, vStart, vEnd, v, dd;

    PetscCall(PetscSectionGetChart(section, &pStart, &pEnd));
    PetscCall(DMGetDimPoints(dm, 0, &vStart, &vEnd));
    pStart = PetscMax(vStart, pStart);
    pEnd   = PetscMin(vEnd, pEnd);
    for (v = pStart; v < pEnd; ++v) {
      PetscCall(PetscSectionGetDof(section, v, &dd));
      if (dd) {
        d = dd;
        break;
      }
    }
    if (d >= 0) PetscCall(DMSetCoordinateDim(dm, d));
  } else {
    PetscCall(DMSetCoordinateDim(dm, dim));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetCoordinates - Gets a global vector with the coordinates associated with the `DM`.

  Collective if the global vector with coordinates has not been set yet but the local vector with coordinates has been set

  Input Parameter:
. dm - the `DM`

  Output Parameter:
. c - global coordinate vector

  Level: intermediate

  Notes:
  This is a borrowed reference, so the user should NOT destroy this vector. When the `DM` is
  destroyed `c` will no longer be valid.

  Each process has only the locally-owned portion of the global coordinates (does NOT have the ghost coordinates), see `DMGetCoordinatesLocal()`.

  For `DMDA`, in two and three dimensions coordinates are interlaced (x_0,y_0,x_1,y_1,...)
  and (x_0,y_0,z_0,x_1,y_1,z_1...)

  Does not work for `DMSTAG`

.seealso: `DM`, `DMDA`, `DMSetCoordinates()`, `DMGetCoordinatesLocal()`, `DMGetCoordinateDM()`, `DMDASetUniformCoordinates()`
@*/
PetscErrorCode DMGetCoordinates(DM dm, Vec *c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(c, 2);
  if (!dm->coordinates[0].x && dm->coordinates[0].xl) {
    DM cdm = NULL;

    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMCreateGlobalVector(cdm, &dm->coordinates[0].x));
    PetscCall(PetscObjectSetName((PetscObject)dm->coordinates[0].x, "coordinates"));
    PetscCall(DMLocalToGlobalBegin(cdm, dm->coordinates[0].xl, INSERT_VALUES, dm->coordinates[0].x));
    PetscCall(DMLocalToGlobalEnd(cdm, dm->coordinates[0].xl, INSERT_VALUES, dm->coordinates[0].x));
  }
  *c = dm->coordinates[0].x;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetCoordinates - Sets into the `DM` a global vector that holds the coordinates

  Logically Collective

  Input Parameters:
+ dm - the `DM`
- c  - coordinate vector

  Level: intermediate

  Notes:
  The coordinates do not include those for ghost points, which are in the local vector.

  The vector `c` can be destroyed after the call

.seealso: `DM`, `DMSetCoordinatesLocal()`, `DMGetCoordinates()`, `DMGetCoordinatesLocal()`, `DMGetCoordinateDM()`, `DMDASetUniformCoordinates()`
@*/
PetscErrorCode DMSetCoordinates(DM dm, Vec c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (c) PetscValidHeaderSpecific(c, VEC_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)c));
  PetscCall(VecDestroy(&dm->coordinates[0].x));
  dm->coordinates[0].x = c;
  PetscCall(VecDestroy(&dm->coordinates[0].xl));
  PetscCall(DMCoarsenHookAdd(dm, DMRestrictHook_Coordinates, NULL, NULL));
  PetscCall(DMSubDomainHookAdd(dm, DMSubDomainHook_Coordinates, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetCellCoordinates - Gets a global vector with the cellwise coordinates associated with the `DM`.

  Collective

  Input Parameter:
. dm - the `DM`

  Output Parameter:
. c - global coordinate vector

  Level: intermediate

  Notes:
  This is a borrowed reference, so the user should NOT destroy this vector. When the `DM` is
  destroyed `c` will no longer be valid.

  Each process has only the locally-owned portion of the global coordinates (does NOT have the ghost coordinates).

.seealso: `DM`, `DMGetCoordinates()`, `DMSetCellCoordinates()`, `DMGetCellCoordinatesLocal()`, `DMGetCellCoordinateDM()`
@*/
PetscErrorCode DMGetCellCoordinates(DM dm, Vec *c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(c, 2);
  if (!dm->coordinates[1].x && dm->coordinates[1].xl) {
    DM cdm = NULL;

    PetscCall(DMGetCellCoordinateDM(dm, &cdm));
    PetscCall(DMCreateGlobalVector(cdm, &dm->coordinates[1].x));
    PetscCall(PetscObjectSetName((PetscObject)dm->coordinates[1].x, "DG coordinates"));
    PetscCall(DMLocalToGlobalBegin(cdm, dm->coordinates[1].xl, INSERT_VALUES, dm->coordinates[1].x));
    PetscCall(DMLocalToGlobalEnd(cdm, dm->coordinates[1].xl, INSERT_VALUES, dm->coordinates[1].x));
  }
  *c = dm->coordinates[1].x;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetCellCoordinates - Sets into the `DM` a global vector that holds the cellwise coordinates

  Collective

  Input Parameters:
+ dm - the `DM`
- c  - cellwise coordinate vector

  Level: intermediate

  Notes:
  The coordinates do not include those for ghost points, which are in the local vector.

  The vector `c` should be destroyed by the caller.

.seealso: `DM`, `DMGetCoordinates()`, `DMSetCellCoordinatesLocal()`, `DMGetCellCoordinates()`, `DMGetCellCoordinatesLocal()`, `DMGetCellCoordinateDM()`
@*/
PetscErrorCode DMSetCellCoordinates(DM dm, Vec c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (c) PetscValidHeaderSpecific(c, VEC_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)c));
  PetscCall(VecDestroy(&dm->coordinates[1].x));
  dm->coordinates[1].x = c;
  PetscCall(VecDestroy(&dm->coordinates[1].xl));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetCoordinatesLocalSetUp - Prepares a local vector of coordinates, so that `DMGetCoordinatesLocalNoncollective()` can be used as non-collective afterwards.

  Collective

  Input Parameter:
. dm - the `DM`

  Level: advanced

.seealso: `DM`, `DMSetCoordinates()`, `DMGetCoordinatesLocalNoncollective()`
@*/
PetscErrorCode DMGetCoordinatesLocalSetUp(DM dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!dm->coordinates[0].xl && dm->coordinates[0].x) {
    DM       cdm = NULL;
    PetscInt bs;

    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMCreateLocalVector(cdm, &dm->coordinates[0].xl));
    PetscCall(PetscObjectSetName((PetscObject)dm->coordinates[0].xl, "Local Coordinates"));
    // If the size of the vector is 0, it will not get the right block size
    PetscCall(VecGetBlockSize(dm->coordinates[0].x, &bs));
    PetscCall(VecSetBlockSize(dm->coordinates[0].xl, bs));
    PetscCall(PetscObjectSetName((PetscObject)dm->coordinates[0].xl, "coordinates"));
    PetscCall(DMGlobalToLocalBegin(cdm, dm->coordinates[0].x, INSERT_VALUES, dm->coordinates[0].xl));
    PetscCall(DMGlobalToLocalEnd(cdm, dm->coordinates[0].x, INSERT_VALUES, dm->coordinates[0].xl));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetCoordinatesLocal - Gets a local vector with the coordinates associated with the `DM`.

  Collective the first time it is called

  Input Parameter:
. dm - the `DM`

  Output Parameter:
. c - coordinate vector

  Level: intermediate

  Notes:
  This is a borrowed reference, so the user should NOT destroy `c`

  Each process has the local and ghost coordinates

  For `DMDA`, in two and three dimensions coordinates are interlaced (x_0,y_0,x_1,y_1,...)
  and (x_0,y_0,z_0,x_1,y_1,z_1...)

.seealso: `DM`, `DMSetCoordinatesLocal()`, `DMGetCoordinates()`, `DMSetCoordinates()`, `DMGetCoordinateDM()`, `DMGetCoordinatesLocalNoncollective()`
@*/
PetscErrorCode DMGetCoordinatesLocal(DM dm, Vec *c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(c, 2);
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  *c = dm->coordinates[0].xl;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetCoordinatesLocalNoncollective - Non-collective version of `DMGetCoordinatesLocal()`. Fails if global coordinates have been set and `DMGetCoordinatesLocalSetUp()` not called.

  Not Collective

  Input Parameter:
. dm - the `DM`

  Output Parameter:
. c - coordinate vector

  Level: advanced

  Note:
  A previous call to  `DMGetCoordinatesLocal()` or `DMGetCoordinatesLocalSetUp()` ensures that a call to this function will not error.

.seealso: `DM`, `DMGetCoordinatesLocalSetUp()`, `DMGetCoordinatesLocal()`, `DMSetCoordinatesLocal()`, `DMGetCoordinates()`, `DMSetCoordinates()`, `DMGetCoordinateDM()`
@*/
PetscErrorCode DMGetCoordinatesLocalNoncollective(DM dm, Vec *c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(c, 2);
  PetscCheck(dm->coordinates[0].xl || !dm->coordinates[0].x, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DMGetCoordinatesLocalSetUp() has not been called");
  *c = dm->coordinates[0].xl;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetCoordinatesLocalTuple - Gets a local vector with the coordinates of specified points and the section describing its layout.

  Not Collective

  Input Parameters:
+ dm - the `DM`
- p  - the `IS` of points whose coordinates will be returned

  Output Parameters:
+ pCoordSection - the `PetscSection` describing the layout of pCoord, i.e. each point corresponds to one point in `p`, and DOFs correspond to coordinates
- pCoord        - the `Vec` with coordinates of points in `p`

  Level: advanced

  Notes:
  `DMGetCoordinatesLocalSetUp()` must be called first. This function employs `DMGetCoordinatesLocalNoncollective()` so it is not collective.

  This creates a new vector, so the user SHOULD destroy this vector

  Each process has the local and ghost coordinates

  For `DMDA`, in two and three dimensions coordinates are interlaced (x_0,y_0,x_1,y_1,...)
  and (x_0,y_0,z_0,x_1,y_1,z_1...)

.seealso: `DM`, `DMDA`, `DMSetCoordinatesLocal()`, `DMGetCoordinatesLocal()`, `DMGetCoordinatesLocalNoncollective()`, `DMGetCoordinatesLocalSetUp()`, `DMGetCoordinates()`, `DMSetCoordinates()`, `DMGetCoordinateDM()`
@*/
PetscErrorCode DMGetCoordinatesLocalTuple(DM dm, IS p, PetscSection *pCoordSection, Vec *pCoord)
{
  DM                 cdm;
  PetscSection       cs, newcs;
  Vec                coords;
  const PetscScalar *arr;
  PetscScalar       *newarr = NULL;
  PetscInt           n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(p, IS_CLASSID, 2);
  if (pCoordSection) PetscAssertPointer(pCoordSection, 3);
  if (pCoord) PetscAssertPointer(pCoord, 4);
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetLocalSection(cdm, &cs));
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  PetscCheck(coords, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DMGetCoordinatesLocalSetUp() has not been called or coordinates not set");
  PetscCheck(cdm && cs, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DM not supported");
  PetscCall(VecGetArrayRead(coords, &arr));
  PetscCall(PetscSectionExtractDofsFromArray(cs, MPIU_SCALAR, arr, p, &newcs, pCoord ? ((void **)&newarr) : NULL));
  PetscCall(VecRestoreArrayRead(coords, &arr));
  if (pCoord) {
    PetscCall(PetscSectionGetStorageSize(newcs, &n));
    /* set array in two steps to mimic PETSC_OWN_POINTER */
    PetscCall(VecCreateSeqWithArray(PetscObjectComm((PetscObject)p), 1, n, NULL, pCoord));
    PetscCall(VecReplaceArray(*pCoord, newarr));
  } else {
    PetscCall(PetscFree(newarr));
  }
  if (pCoordSection) {
    *pCoordSection = newcs;
  } else PetscCall(PetscSectionDestroy(&newcs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetCoordinatesLocal - Sets into the `DM` a local vector, including ghost points, that holds the coordinates

  Not Collective

  Input Parameters:
+ dm - the `DM`
- c  - coordinate vector

  Level: intermediate

  Notes:
  The coordinates of ghost points can be set using `DMSetCoordinates()`
  followed by `DMGetCoordinatesLocal()`. This is intended to enable the
  setting of ghost coordinates outside of the domain.

  The vector `c` should be destroyed by the caller.

.seealso: `DM`, `DMGetCoordinatesLocal()`, `DMSetCoordinates()`, `DMGetCoordinates()`, `DMGetCoordinateDM()`
@*/
PetscErrorCode DMSetCoordinatesLocal(DM dm, Vec c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (c) PetscValidHeaderSpecific(c, VEC_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)c));
  PetscCall(VecDestroy(&dm->coordinates[0].xl));
  dm->coordinates[0].xl = c;
  PetscCall(VecDestroy(&dm->coordinates[0].x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetCellCoordinatesLocalSetUp - Prepares a local vector of cellwise coordinates, so that `DMGetCellCoordinatesLocalNoncollective()` can be used as non-collective afterwards.

  Collective

  Input Parameter:
. dm - the `DM`

  Level: advanced

.seealso: `DM`, `DMGetCellCoordinatesLocalNoncollective()`
@*/
PetscErrorCode DMGetCellCoordinatesLocalSetUp(DM dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!dm->coordinates[1].xl && dm->coordinates[1].x) {
    DM cdm = NULL;

    PetscCall(DMGetCellCoordinateDM(dm, &cdm));
    PetscCall(DMCreateLocalVector(cdm, &dm->coordinates[1].xl));
    PetscCall(PetscObjectSetName((PetscObject)dm->coordinates[1].xl, "DG coordinates"));
    PetscCall(DMGlobalToLocalBegin(cdm, dm->coordinates[1].x, INSERT_VALUES, dm->coordinates[1].xl));
    PetscCall(DMGlobalToLocalEnd(cdm, dm->coordinates[1].x, INSERT_VALUES, dm->coordinates[1].xl));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetCellCoordinatesLocal - Gets a local vector with the cellwise coordinates associated with the `DM`.

  Collective

  Input Parameter:
. dm - the `DM`

  Output Parameter:
. c - coordinate vector

  Level: intermediate

  Notes:
  This is a borrowed reference, so the user should NOT destroy this vector

  Each process has the local and ghost coordinates

.seealso: `DM`, `DMSetCellCoordinatesLocal()`, `DMGetCellCoordinates()`, `DMSetCellCoordinates()`, `DMGetCellCoordinateDM()`, `DMGetCellCoordinatesLocalNoncollective()`
@*/
PetscErrorCode DMGetCellCoordinatesLocal(DM dm, Vec *c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(c, 2);
  PetscCall(DMGetCellCoordinatesLocalSetUp(dm));
  *c = dm->coordinates[1].xl;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetCellCoordinatesLocalNoncollective - Non-collective version of `DMGetCellCoordinatesLocal()`. Fails if global cellwise coordinates have been set and `DMGetCellCoordinatesLocalSetUp()` not called.

  Not Collective

  Input Parameter:
. dm - the `DM`

  Output Parameter:
. c - cellwise coordinate vector

  Level: advanced

.seealso: `DM`, `DMGetCellCoordinatesLocalSetUp()`, `DMGetCellCoordinatesLocal()`, `DMSetCellCoordinatesLocal()`, `DMGetCellCoordinates()`, `DMSetCellCoordinates()`, `DMGetCellCoordinateDM()`
@*/
PetscErrorCode DMGetCellCoordinatesLocalNoncollective(DM dm, Vec *c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(c, 2);
  PetscCheck(dm->coordinates[1].xl || !dm->coordinates[1].x, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DMGetCellCoordinatesLocalSetUp() has not been called");
  *c = dm->coordinates[1].xl;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSetCellCoordinatesLocal - Sets into the `DM` a local vector including ghost points that holds the cellwise coordinates

  Not Collective

  Input Parameters:
+ dm - the `DM`
- c  - cellwise coordinate vector

  Level: intermediate

  Notes:
  The coordinates of ghost points can be set using `DMSetCoordinates()`
  followed by `DMGetCoordinatesLocal()`. This is intended to enable the
  setting of ghost coordinates outside of the domain.

  The vector `c` should be destroyed by the caller.

.seealso: `DM`, `DMGetCellCoordinatesLocal()`, `DMSetCellCoordinates()`, `DMGetCellCoordinates()`, `DMGetCellCoordinateDM()`
@*/
PetscErrorCode DMSetCellCoordinatesLocal(DM dm, Vec c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (c) PetscValidHeaderSpecific(c, VEC_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)c));
  PetscCall(VecDestroy(&dm->coordinates[1].xl));
  dm->coordinates[1].xl = c;
  PetscCall(VecDestroy(&dm->coordinates[1].x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMGetCoordinateField(DM dm, DMField *field)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(field, 2);
  if (!dm->coordinates[0].field) PetscTryTypeMethod(dm, createcoordinatefield, &dm->coordinates[0].field);
  *field = dm->coordinates[0].field;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSetCoordinateField(DM dm, DMField field)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (field) PetscValidHeaderSpecific(field, DMFIELD_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)field));
  PetscCall(DMFieldDestroy(&dm->coordinates[0].field));
  dm->coordinates[0].field = field;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSetCellCoordinateField(DM dm, DMField field)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (field) PetscValidHeaderSpecific(field, DMFIELD_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)field));
  PetscCall(DMFieldDestroy(&dm->coordinates[1].field));
  dm->coordinates[1].field = field;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMGetLocalBoundingBox_Coordinates(DM dm, PetscReal lmin[], PetscReal lmax[], PetscInt cs[], PetscInt ce[])
{
  Vec         coords = NULL;
  PetscReal   min[3] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL};
  PetscReal   max[3] = {PETSC_MIN_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};
  PetscInt    cdim, i, j;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  if (size == 1) {
    const PetscReal *L, *Lstart;

    PetscCall(DMGetPeriodicity(dm, NULL, &Lstart, &L));
    if (L) {
      for (PetscInt d = 0; d < cdim; ++d)
        if (L[d] > 0.0) {
          min[d] = Lstart[d];
          max[d] = Lstart[d] + L[d];
        }
    }
  }
  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  if (coords) {
    const PetscScalar *local_coords;
    PetscInt           N, Ni;

    for (j = cdim; j < 3; ++j) {
      min[j] = 0;
      max[j] = 0;
    }
    PetscCall(VecGetArrayRead(coords, &local_coords));
    PetscCall(VecGetLocalSize(coords, &N));
    Ni = N / cdim;
    for (i = 0; i < Ni; ++i) {
      for (j = 0; j < cdim; ++j) {
        min[j] = PetscMin(min[j], PetscRealPart(local_coords[i * cdim + j]));
        max[j] = PetscMax(max[j], PetscRealPart(local_coords[i * cdim + j]));
      }
    }
    PetscCall(VecRestoreArrayRead(coords, &local_coords));
    PetscCall(DMGetCellCoordinatesLocal(dm, &coords));
    if (coords) {
      PetscCall(VecGetArrayRead(coords, &local_coords));
      PetscCall(VecGetLocalSize(coords, &N));
      Ni = N / cdim;
      for (i = 0; i < Ni; ++i) {
        for (j = 0; j < cdim; ++j) {
          min[j] = PetscMin(min[j], PetscRealPart(local_coords[i * cdim + j]));
          max[j] = PetscMax(max[j], PetscRealPart(local_coords[i * cdim + j]));
        }
      }
      PetscCall(VecRestoreArrayRead(coords, &local_coords));
    }
    if (lmin) PetscCall(PetscArraycpy(lmin, min, cdim));
    if (lmax) PetscCall(PetscArraycpy(lmax, max, cdim));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetLocalBoundingBox - Returns the bounding box for the piece of the `DM` on this process.

  Not Collective

  Input Parameter:
. dm - the `DM`

  Output Parameters:
+ lmin - local minimum coordinates (length coord dim, optional)
- lmax - local maximum coordinates (length coord dim, optional)

  Level: beginner

  Note:
  If the `DM` is a `DMDA` and has no coordinates, the index bounds are returned instead.

.seealso: `DM`, `DMGetCoordinates()`, `DMGetCoordinatesLocal()`, `DMGetBoundingBox()`
@*/
PetscErrorCode DMGetLocalBoundingBox(DM dm, PetscReal lmin[], PetscReal lmax[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscUseTypeMethod(dm, getlocalboundingbox, lmin, lmax, NULL, NULL);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGetBoundingBox - Returns the global bounding box for the `DM`.

  Collective

  Input Parameter:
. dm - the `DM`

  Output Parameters:
+ gmin - global minimum coordinates (length coord dim, optional)
- gmax - global maximum coordinates (length coord dim, optional)

  Level: beginner

.seealso: `DM`, `DMGetLocalBoundingBox()`, `DMGetCoordinates()`, `DMGetCoordinatesLocal()`
@*/
PetscErrorCode DMGetBoundingBox(DM dm, PetscReal gmin[], PetscReal gmax[])
{
  PetscReal        lmin[3], lmax[3];
  const PetscReal *L, *Lstart;
  PetscInt         cdim;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMGetLocalBoundingBox(dm, lmin, lmax));
  if (gmin) PetscCallMPI(MPIU_Allreduce(lmin, gmin, cdim, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)dm)));
  if (gmax) PetscCallMPI(MPIU_Allreduce(lmax, gmax, cdim, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)dm)));
  PetscCall(DMGetPeriodicity(dm, NULL, &Lstart, &L));
  if (L) {
    for (PetscInt d = 0; d < cdim; ++d)
      if (L[d] > 0.0) {
        gmin[d] = Lstart[d];
        gmax[d] = Lstart[d] + L[d];
      }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMCreateAffineCoordinates_Internal(DM dm, PetscBool localized)
{
  DM             cdm;
  PetscFE        feLinear;
  DMPolytopeType ct;
  PetscInt       dim, dE, height, cStart, cEnd, gct;

  PetscFunctionBegin;
  if (!localized) {
    PetscCall(DMGetCoordinateDM(dm, &cdm));
  } else {
    PetscCall(DMGetCellCoordinateDM(dm, &cdm));
  }
  PetscCheck(cdm, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "No coordinateDM defined");
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &dE));
  PetscCall(DMPlexGetVTKCellHeight(dm, &height));
  PetscCall(DMPlexGetHeightStratum(dm, height, &cStart, &cEnd));
  if (cEnd > cStart) PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  else ct = DM_POLYTOPE_UNKNOWN;
  gct = (PetscInt)ct;
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &gct, 1, MPIU_INT, MPI_MIN, PetscObjectComm((PetscObject)dm)));
  ct = (DMPolytopeType)gct;
  // Work around current bug in PetscDualSpaceSetUp_Lagrange()
  //   Can be seen in plex_tutorials-ex10_1
  if (ct != DM_POLYTOPE_SEG_PRISM_TENSOR && ct != DM_POLYTOPE_TRI_PRISM_TENSOR && ct != DM_POLYTOPE_QUAD_PRISM_TENSOR) {
    PetscCall(PetscFECreateLagrangeByCell(PETSC_COMM_SELF, dim, dE, ct, 1, -1, &feLinear));
    if (localized) {
      PetscFE dgfe = NULL;

      PetscCall(PetscFECreateBrokenElement(feLinear, &dgfe));
      PetscCall(PetscFEDestroy(&feLinear));
      feLinear = dgfe;
    }
    PetscCall(DMSetField(cdm, 0, NULL, (PetscObject)feLinear));
    PetscCall(PetscFEDestroy(&feLinear));
    PetscCall(DMCreateDS(cdm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMGetCoordinateDegree_Internal(DM dm, PetscInt *degree)
{
  DM           cdm;
  PetscFE      fe;
  PetscSpace   sp;
  PetscClassId id;

  PetscFunctionBegin;
  *degree = 1;
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetField(cdm, 0, NULL, (PetscObject *)&fe));
  PetscCall(PetscObjectGetClassId((PetscObject)fe, &id));
  if (id != PETSCFE_CLASSID) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFEGetBasisSpace(fe, &sp));
  PetscCall(PetscSpaceGetDegree(sp, degree, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void evaluate_coordinates(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar xnew[])
{
  for (PetscInt i = 0; i < dim; i++) xnew[i] = x[i];
}

/*@
  DMSetCoordinateDisc - Set a coordinate space

  Input Parameters:
+ dm        - The `DM` object
. disc      - The new coordinate discretization or `NULL` to ensure a coordinate discretization exists
. localized - Set a localized (DG) coordinate space
- project   - Project coordinates to new discretization

  Level: intermediate

  Notes:
  A `PetscFE` defines an approximation space using a `PetscSpace`, which represents the basis functions, and a `PetscDualSpace`, which defines the interpolation operation in the space.

  This function takes the current mesh coordinates, which are discretized using some `PetscFE` space, and projects this function into a new `PetscFE` space.
  The coordinate projection is done on the continuous coordinates, but the discontinuous coordinates are not updated.

  Developer Note:
  With more effort, we could directly project the discontinuous coordinates also.

.seealso: `DM`, `PetscFE`, `DMGetCoordinateField()`
@*/
PetscErrorCode DMSetCoordinateDisc(DM dm, PetscFE disc, PetscBool localized, PetscBool project)
{
  DM           cdmOld, cdmNew;
  PetscFE      discOld;
  PetscClassId classid;
  PetscBool    same_space = PETSC_TRUE;
  const char  *prefix;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (disc) PetscValidHeaderSpecific(disc, PETSCFE_CLASSID, 2);

  /* Note that plexgmsh.c can pass DG element with localized = PETSC_FALSE. */
  if (!localized) {
    PetscCall(DMGetCoordinateDM(dm, &cdmOld));
  } else {
    PetscCall(DMGetCellCoordinateDM(dm, &cdmOld));
    if (!cdmOld) {
      PetscUseTypeMethod(dm, createcellcoordinatedm, &cdmOld);
      PetscCall(DMSetCellCoordinateDM(dm, cdmOld));
      PetscCall(DMDestroy(&cdmOld));
      PetscCall(DMGetCellCoordinateDM(dm, &cdmOld));
    }
  }
  /* Check current discretization is compatible */
  PetscCall(DMGetField(cdmOld, 0, NULL, (PetscObject *)&discOld));
  PetscCall(PetscObjectGetClassId((PetscObject)discOld, &classid));
  if (classid != PETSCFE_CLASSID) {
    if (classid == PETSC_CONTAINER_CLASSID) {
      PetscCall(DMCreateAffineCoordinates_Internal(dm, localized));
      PetscCall(DMGetField(cdmOld, 0, NULL, (PetscObject *)&discOld));
    } else {
      const char *discname;

      PetscCall(PetscObjectGetType((PetscObject)discOld, &discname));
      SETERRQ(PetscObjectComm((PetscObject)discOld), PETSC_ERR_SUP, "Discretization type %s not supported", discname);
    }
  }
  // Linear space has been created by now
  if (!disc) PetscFunctionReturn(PETSC_SUCCESS);
  // Check if the new space is the same as the old modulo quadrature
  {
    PetscDualSpace dsOld, ds;
    PetscCall(PetscFEGetDualSpace(discOld, &dsOld));
    PetscCall(PetscFEGetDualSpace(disc, &ds));
    PetscCall(PetscDualSpaceEqual(dsOld, ds, &same_space));
  }
  // Make a fresh clone of the coordinate DM
  PetscCall(DMClone(cdmOld, &cdmNew));
  cdmNew->cloneOpts = PETSC_TRUE;
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)cdmOld, &prefix));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)cdmNew, prefix));
  PetscCall(DMSetField(cdmNew, 0, NULL, (PetscObject)disc));
  PetscCall(DMCreateDS(cdmNew));
  {
    PetscDS ds, nds;

    PetscCall(DMGetDS(cdmOld, &ds));
    PetscCall(DMGetDS(cdmNew, &nds));
    PetscCall(PetscDSCopyConstants(ds, nds));
  }
  if (cdmOld->periodic.setup) {
    PetscSF dummy;
    // Force IsoperiodicPointSF to be built, required for periodic coordinate setup
    PetscCall(DMGetIsoperiodicPointSF_Internal(dm, &dummy));
    cdmNew->periodic.setup = cdmOld->periodic.setup;
    PetscCall(cdmNew->periodic.setup(cdmNew));
  }
  if (dm->setfromoptionscalled) PetscCall(DMSetFromOptions(cdmNew));
  if (project) {
    Vec      coordsOld, coordsNew;
    PetscInt num_face_sfs = 0;

    PetscCall(DMPlexGetIsoperiodicFaceSF(dm, &num_face_sfs, NULL));
    if (num_face_sfs) { // Isoperiodicity requires projecting the local coordinates
      PetscCall(DMGetCoordinatesLocal(dm, &coordsOld));
      PetscCall(DMCreateLocalVector(cdmNew, &coordsNew));
      PetscCall(PetscObjectSetName((PetscObject)coordsNew, "coordinates"));
      if (same_space) {
        // Need to copy so that the new vector has the right dm
        PetscCall(VecCopy(coordsOld, coordsNew));
      } else {
        void (*funcs[])(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]) = {evaluate_coordinates};

        // We can't call DMProjectField directly because it depends on KSP for DMGlobalToLocalSolve(), but we can use the core strategy
        PetscCall(DMSetCoordinateDM(cdmNew, cdmOld));
        // See DMPlexRemapGeometry() for a similar pattern handling the coordinate field
        DMField cf;
        PetscCall(DMGetCoordinateField(dm, &cf));
        cdmNew->coordinates[0].field = cf;
        PetscCall(DMProjectFieldLocal(cdmNew, 0.0, NULL, funcs, INSERT_VALUES, coordsNew));
        cdmNew->coordinates[0].field = NULL;
        PetscCall(DMSetCoordinateDM(cdmNew, NULL));
      }
      PetscCall(DMSetCoordinatesLocal(dm, coordsNew));
      PetscCall(VecDestroy(&coordsNew));
    } else {
      PetscCall(DMGetCoordinates(dm, &coordsOld));
      PetscCall(DMCreateGlobalVector(cdmNew, &coordsNew));
      if (same_space) {
        // Need to copy so that the new vector has the right dm
        PetscCall(VecCopy(coordsOld, coordsNew));
      } else {
        Mat In;

        PetscCall(DMCreateInterpolation(cdmOld, cdmNew, &In, NULL));
        PetscCall(MatMult(In, coordsOld, coordsNew));
        PetscCall(MatDestroy(&In));
      }
      PetscCall(DMSetCoordinates(dm, coordsNew));
      PetscCall(VecDestroy(&coordsNew));
    }
  }
  /* Set new coordinate structures */
  if (!localized) {
    PetscCall(DMSetCoordinateField(dm, NULL));
    PetscCall(DMSetCoordinateDM(dm, cdmNew));
  } else {
    PetscCall(DMSetCellCoordinateField(dm, NULL));
    PetscCall(DMSetCellCoordinateDM(dm, cdmNew));
  }
  PetscCall(DMDestroy(&cdmNew));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMLocatePoints - Locate the points in `v` in the mesh and return a `PetscSF` of the containing cells

  Collective

  Input Parameters:
+ dm    - The `DM`
- ltype - The type of point location, e.g. `DM_POINTLOCATION_NONE` or `DM_POINTLOCATION_NEAREST`

  Input/Output Parameters:
+ v      - The `Vec` of points, on output contains the nearest mesh points to the given points if `DM_POINTLOCATION_NEAREST` is used
- cellSF - Points to either `NULL`, or a `PetscSF` with guesses for which cells contain each point;
           on output, the `PetscSF` containing the MPI ranks and local indices of the containing points

  Level: developer

  Notes:
  To do a search of the local cells of the mesh, `v` should have `PETSC_COMM_SELF` as its communicator.
  To do a search of all the cells in the distributed mesh, `v` should have the same MPI communicator as `dm`.

  Points will only be located in owned cells, not overlap cells arising from `DMPlexDistribute()` or other overlapping distributions.

  If *cellSF is `NULL` on input, a `PetscSF` will be created.
  If *cellSF is not `NULL` on input, it should point to an existing `PetscSF`, whose graph will be used as initial guesses.

  An array that maps each point to its containing cell can be obtained with
.vb
    const PetscSFNode *cells;
    PetscInt           nFound;
    const PetscInt    *found;

    PetscSFGetGraph(cellSF,NULL,&nFound,&found,&cells);
.ve

  Where cells[i].rank is the MPI rank of the process owning the cell containing point found[i] (or i if found == NULL), and cells[i].index is
  the index of the cell in its MPI process' local numbering. This rank is in the communicator for `v`, so if `v` is on `PETSC_COMM_SELF` then the rank will always be 0.

.seealso: `DM`, `DMSetCoordinates()`, `DMSetCoordinatesLocal()`, `DMGetCoordinates()`, `DMGetCoordinatesLocal()`, `DMPointLocationType`
@*/
PetscErrorCode DMLocatePoints(DM dm, Vec v, DMPointLocationType ltype, PetscSF *cellSF)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 2);
  PetscAssertPointer(cellSF, 4);
  if (*cellSF) {
    PetscMPIInt result;

    PetscValidHeaderSpecific(*cellSF, PETSCSF_CLASSID, 4);
    PetscCallMPI(MPI_Comm_compare(PetscObjectComm((PetscObject)v), PetscObjectComm((PetscObject)*cellSF), &result));
    PetscCheck(result == MPI_IDENT || result == MPI_CONGRUENT, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "cellSF must have a communicator congruent to v's");
  } else {
    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)v), cellSF));
  }
  PetscCall(PetscLogEventBegin(DM_LocatePoints, dm, 0, 0, 0));
  PetscUseTypeMethod(dm, locatepoints, v, ltype, *cellSF);
  PetscCall(PetscLogEventEnd(DM_LocatePoints, dm, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
