#include <petsc/private/dmimpl.h> /*I      "petscdm.h"          I*/

#include <petscdmplex.h> /* For DMProjectCoordinates() */
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
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinateDM - Gets the `DM` that prescribes coordinate layout and scatters between global and local coordinates

  Collective on dm

  Input Parameter:
. dm - the `DM`

  Output Parameter:
. cdm - coordinate `DM`

  Level: intermediate

.seealso: `DM`, `DMSetCoordinateDM()`, `DMSetCoordinates()`, `DMSetCoordinatesLocal()`, `DMGetCoordinates()`, `DMGetCoordinatesLocal()`, `DMGSetCellCoordinateDM()`,
          `DMGSetCellCoordinateDM()`
@*/
PetscErrorCode DMGetCoordinateDM(DM dm, DM *cdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(cdm, 2);
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
  PetscFunctionReturn(0);
}

/*@
  DMSetCoordinateDM - Sets the `DM` that prescribes coordinate layout and scatters between global and local coordinates

  Logically Collective on dm

  Input Parameters:
+ dm - the `DM`
- cdm - coordinate `DM`

  Level: intermediate

.seealso: `DM`, `DMGetCoordinateDM()`, `DMSetCoordinates()`, `DMGetCellCoordinateDM()`, `DMSetCoordinatesLocal()`, `DMGetCoordinates()`, `DMGetCoordinatesLocal()`,
          `DMGSetCellCoordinateDM()`
@*/
PetscErrorCode DMSetCoordinateDM(DM dm, DM cdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(cdm, DM_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)cdm));
  PetscCall(DMDestroy(&dm->coordinates[0].dm));
  dm->coordinates[0].dm = cdm;
  PetscFunctionReturn(0);
}

/*@
  DMGetCellCoordinateDM - Gets the `DM` that prescribes cellwise coordinate layout and scatters between global and local cellwise coordinates

  Collective on dm

  Input Parameter:
. dm - the `DM`

  Output Parameter:
. cdm - cellwise coordinate `DM`, or NULL if they are not defined

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
  PetscValidPointer(cdm, 2);
  *cdm = dm->coordinates[1].dm;
  PetscFunctionReturn(0);
}

/*@
  DMSetCellCoordinateDM - Sets the `DM` that prescribes cellwise coordinate layout and scatters between global and local cellwise coordinates

  Logically Collective on dm

  Input Parameters:
+ dm - the `DM`
- cdm - cellwise coordinate `DM`

  Level: intermediate

  Note:
  As opposed to `DMSetCoordinateDM()` these coordinates are useful for discontinous Galerkin methods since they support coordinate fields that are discontinuous at cell boundaries.

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
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinateDim - Retrieve the dimension of embedding space for coordinate values. For example a mesh on the surface of a sphere would have a 3 dimensional embedding space

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
  PetscValidIntPointer(dim, 2);
  if (dm->coordinates[0].dim == PETSC_DEFAULT) dm->coordinates[0].dim = dm->dim;
  *dim = dm->coordinates[0].dim;
  PetscFunctionReturn(0);
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
      PetscCall(DMGetRegionNumDS(dm, n, NULL, NULL, &ds));
      PetscCall(PetscDSSetCoordinateDimension(ds, dim));
    }
  }
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinateSection - Retrieve the layout of coordinate values over the mesh.

  Collective on dm

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

.seealso: `DMGetCoordinateDM()`, `DMGetLocalSection()`, `DMSetLocalSection()`
@*/
PetscErrorCode DMGetCoordinateSection(DM dm, PetscSection *section)
{
  DM cdm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(section, 2);
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetLocalSection(cdm, section));
  PetscFunctionReturn(0);
}

/*@
  DMSetCoordinateSection - Set the layout of coordinate values over the mesh.

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
  }
  PetscFunctionReturn(0);
}

/*@
  DMGetCellCoordinateSection - Retrieve the layout of cellwise coordinate values over the mesh.

  Collective on dm

  Input Parameter:
. dm - The `DM` object

  Output Parameter:
. section - The `PetscSection` object, or NULL if no cellwise coordinates are defined

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
  PetscValidPointer(section, 2);
  *section = NULL;
  PetscCall(DMGetCellCoordinateDM(dm, &cdm));
  if (cdm) PetscCall(DMGetLocalSection(cdm, section));
  PetscFunctionReturn(0);
}

/*@
  DMSetCellCoordinateSection - Set the layout of cellwise coordinate values over the mesh.

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
  }
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinates - Gets a global vector with the coordinates associated with the `DM`.

  Collective on dm

  Input Parameter:
. dm - the `DM`

  Output Parameter:
. c - global coordinate vector

  Level: intermediate

  Notes:
  This is a borrowed reference, so the user should NOT destroy this vector. When the `DM` is
  destroyed the array will no longer be valid.

  Each process has only the locally-owned portion of the global coordinates (does NOT have the ghost coordinates).

  For `DMDA`, in two and three dimensions coordinates are interlaced (x_0,y_0,x_1,y_1,...)
  and (x_0,y_0,z_0,x_1,y_1,z_1...)

.seealso: `DM`, `DMDA`, `DMSetCoordinates()`, `DMGetCoordinatesLocal()`, `DMGetCoordinateDM()`, `DMDASetUniformCoordinates()`
@*/
PetscErrorCode DMGetCoordinates(DM dm, Vec *c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(c, 2);
  if (!dm->coordinates[0].x && dm->coordinates[0].xl) {
    DM cdm = NULL;

    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMCreateGlobalVector(cdm, &dm->coordinates[0].x));
    PetscCall(PetscObjectSetName((PetscObject)dm->coordinates[0].x, "coordinates"));
    PetscCall(DMLocalToGlobalBegin(cdm, dm->coordinates[0].xl, INSERT_VALUES, dm->coordinates[0].x));
    PetscCall(DMLocalToGlobalEnd(cdm, dm->coordinates[0].xl, INSERT_VALUES, dm->coordinates[0].x));
  }
  *c = dm->coordinates[0].x;
  PetscFunctionReturn(0);
}

/*@
  DMSetCoordinates - Sets into the `DM` a global vector that holds the coordinates

  Collective on dm

  Input Parameters:
+ dm - the `DM`
- c - coordinate vector

  Level: intermediate

  Notes:
  The coordinates do not include those for ghost points, which are in the local vector.

  The vector c can be destroyed after the call

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
  PetscFunctionReturn(0);
}

/*@
  DMGetCellCoordinates - Gets a global vector with the cellwise coordinates associated with the `DM`.

  Collective on dm

  Input Parameter:
. dm - the `DM`

  Output Parameter:
. c - global coordinate vector

  Level: intermediate

  Notes:
  This is a borrowed reference, so the user should NOT destroy this vector. When the `DM` is
  destroyed the array will no longer be valid.

  Each process has only the locally-owned portion of the global coordinates (does NOT have the ghost coordinates).

.seealso: `DM`, `DMGetCoordinates()`, `DMSetCellCoordinates()`, `DMGetCellCoordinatesLocal()`, `DMGetCellCoordinateDM()`
@*/
PetscErrorCode DMGetCellCoordinates(DM dm, Vec *c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(c, 2);
  if (!dm->coordinates[1].x && dm->coordinates[1].xl) {
    DM cdm = NULL;

    PetscCall(DMGetCellCoordinateDM(dm, &cdm));
    PetscCall(DMCreateGlobalVector(cdm, &dm->coordinates[1].x));
    PetscCall(PetscObjectSetName((PetscObject)dm->coordinates[1].x, "DG coordinates"));
    PetscCall(DMLocalToGlobalBegin(cdm, dm->coordinates[1].xl, INSERT_VALUES, dm->coordinates[1].x));
    PetscCall(DMLocalToGlobalEnd(cdm, dm->coordinates[1].xl, INSERT_VALUES, dm->coordinates[1].x));
  }
  *c = dm->coordinates[1].x;
  PetscFunctionReturn(0);
}

/*@
  DMSetCellCoordinates - Sets into the `DM` a global vector that holds the cellwise coordinates

  Collective on dm

  Input Parameters:
+ dm - the `DM`
- c - cellwise coordinate vector

  Level: intermediate

  Notes:
  The coordinates do not include those for ghost points, which are in the local vector.

  The vector c should be destroyed by the caller.

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
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinatesLocalSetUp - Prepares a local vector of coordinates, so that `DMGetCoordinatesLocalNoncollective()` can be used as non-collective afterwards.

  Collective on dm

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
    DM cdm = NULL;

    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMCreateLocalVector(cdm, &dm->coordinates[0].xl));
    PetscCall(PetscObjectSetName((PetscObject)dm->coordinates[0].xl, "coordinates"));
    PetscCall(DMGlobalToLocalBegin(cdm, dm->coordinates[0].x, INSERT_VALUES, dm->coordinates[0].xl));
    PetscCall(DMGlobalToLocalEnd(cdm, dm->coordinates[0].x, INSERT_VALUES, dm->coordinates[0].xl));
  }
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinatesLocal - Gets a local vector with the coordinates associated with the `DM`.

  Collective on dm the first time it is called

  Input Parameter:
. dm - the `DM`

  Output Parameter:
. c - coordinate vector

  Level: intermediate

  Notes:
  This is a borrowed reference, so the user should NOT destroy this vector

  Each process has the local and ghost coordinates

  For `DMDA`, in two and three dimensions coordinates are interlaced (x_0,y_0,x_1,y_1,...)
  and (x_0,y_0,z_0,x_1,y_1,z_1...)

.seealso: `DM`, `DMSetCoordinatesLocal()`, `DMGetCoordinates()`, `DMSetCoordinates()`, `DMGetCoordinateDM()`, `DMGetCoordinatesLocalNoncollective()`
@*/
PetscErrorCode DMGetCoordinatesLocal(DM dm, Vec *c)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(c, 2);
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  *c = dm->coordinates[0].xl;
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinatesLocalNoncollective - Non-collective version of `DMGetCoordinatesLocal()`. Fails if global coordinates have been set and `DMGetCoordinatesLocalSetUp()` not called.

  Not collective

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
  PetscValidPointer(c, 2);
  PetscCheck(dm->coordinates[0].xl || !dm->coordinates[0].x, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DMGetCoordinatesLocalSetUp() has not been called");
  *c = dm->coordinates[0].xl;
  PetscFunctionReturn(0);
}

/*@
  DMGetCoordinatesLocalTuple - Gets a local vector with the coordinates of specified points and the section describing its layout.

  Not collective

  Input Parameters:
+ dm - the `DM`
- p - the `IS` of points whose coordinates will be returned

  Output Parameters:
+ pCoordSection - the `PetscSection` describing the layout of pCoord, i.e. each point corresponds to one point in p, and DOFs correspond to coordinates
- pCoord - the `Vec` with coordinates of points in p

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
  if (pCoordSection) PetscValidPointer(pCoordSection, 3);
  if (pCoord) PetscValidPointer(pCoord, 4);
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
  PetscFunctionReturn(0);
}

/*@
  DMSetCoordinatesLocal - Sets into the `DM` a local vector, including ghost points, that holds the coordinates

  Not collective

   Input Parameters:
+  dm - the `DM`
-  c - coordinate vector

  Level: intermediate

  Notes:
  The coordinates of ghost points can be set using `DMSetCoordinates()`
  followed by `DMGetCoordinatesLocal()`. This is intended to enable the
  setting of ghost coordinates outside of the domain.

  The vector c should be destroyed by the caller.

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
  PetscFunctionReturn(0);
}

/*@
  DMGetCellCoordinatesLocalSetUp - Prepares a local vector of cellwise coordinates, so that `DMGetCellCoordinatesLocalNoncollective()` can be used as non-collective afterwards.

  Collective on dm

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
  PetscFunctionReturn(0);
}

/*@
  DMGetCellCoordinatesLocal - Gets a local vector with the cellwise coordinates associated with the `DM`.

  Collective on dm

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
  PetscValidPointer(c, 2);
  PetscCall(DMGetCellCoordinatesLocalSetUp(dm));
  *c = dm->coordinates[1].xl;
  PetscFunctionReturn(0);
}

/*@
  DMGetCellCoordinatesLocalNoncollective - Non-collective version of `DMGetCellCoordinatesLocal()`. Fails if global cellwise coordinates have been set and `DMGetCellCoordinatesLocalSetUp()` not called.

  Not collective

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
  PetscValidPointer(c, 2);
  PetscCheck(dm->coordinates[1].xl || !dm->coordinates[1].x, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "DMGetCellCoordinatesLocalSetUp() has not been called");
  *c = dm->coordinates[1].xl;
  PetscFunctionReturn(0);
}

/*@
  DMSetCellCoordinatesLocal - Sets into the `DM` a local vector including ghost points that holds the cellwise coordinates

  Not collective

   Input Parameters:
+  dm - the `DM`
-  c - cellwise coordinate vector

  Level: intermediate

  Notes:
  The coordinates of ghost points can be set using `DMSetCoordinates()`
  followed by `DMGetCoordinatesLocal()`. This is intended to enable the
  setting of ghost coordinates outside of the domain.

  The vector c should be destroyed by the caller.

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
  PetscFunctionReturn(0);
}

PetscErrorCode DMGetCoordinateField(DM dm, DMField *field)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(field, 2);
  if (!dm->coordinates[0].field) {
    if (dm->ops->createcoordinatefield) PetscCall((*dm->ops->createcoordinatefield)(dm, &dm->coordinates[0].field));
  }
  *field = dm->coordinates[0].field;
  PetscFunctionReturn(0);
}

PetscErrorCode DMSetCoordinateField(DM dm, DMField field)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (field) PetscValidHeaderSpecific(field, DMFIELD_CLASSID, 2);
  PetscCall(PetscObjectReference((PetscObject)field));
  PetscCall(DMFieldDestroy(&dm->coordinates[0].field));
  dm->coordinates[0].field = field;
  PetscFunctionReturn(0);
}

/*@
  DMGetLocalBoundingBox - Returns the bounding box for the piece of the `DM` on this process.

  Not collective

  Input Parameter:
. dm - the `DM`

  Output Parameters:
+ lmin - local minimum coordinates (length coord dim, optional)
- lmax - local maximim coordinates (length coord dim, optional)

  Level: beginner

  Note:
  If the `DM` is a `DMDA` and has no coordinates, the index bounds are returned instead.

.seealso: `DM`, `DMGetCoordinates()`, `DMGetCoordinatesLocal()`, `DMGetBoundingBox()`
@*/
PetscErrorCode DMGetLocalBoundingBox(DM dm, PetscReal lmin[], PetscReal lmax[])
{
  Vec       coords = NULL;
  PetscReal min[3] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL};
  PetscReal max[3] = {PETSC_MIN_REAL, PETSC_MIN_REAL, PETSC_MIN_REAL};
  PetscInt  cdim, i, j;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetCoordinateDim(dm, &cdim));
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
  } else {
    PetscBool isda;

    PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMDA, &isda));
    if (isda) PetscCall(DMGetLocalBoundingIndices_DMDA(dm, min, max));
  }
  if (lmin) PetscCall(PetscArraycpy(lmin, min, cdim));
  if (lmax) PetscCall(PetscArraycpy(lmax, max, cdim));
  PetscFunctionReturn(0);
}

/*@
  DMGetBoundingBox - Returns the global bounding box for the `DM`.

  Collective

  Input Parameter:
. dm - the `DM`

  Output Parameters:
+ gmin - global minimum coordinates (length coord dim, optional)
- gmax - global maximim coordinates (length coord dim, optional)

  Level: beginner

.seealso: `DM`, `DMGetLocalBoundingBox()`, `DMGetCoordinates()`, `DMGetCoordinatesLocal()`
@*/
PetscErrorCode DMGetBoundingBox(DM dm, PetscReal gmin[], PetscReal gmax[])
{
  PetscReal   lmin[3], lmax[3];
  PetscInt    cdim;
  PetscMPIInt count;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(PetscMPIIntCast(cdim, &count));
  PetscCall(DMGetLocalBoundingBox(dm, lmin, lmax));
  if (gmin) PetscCall(MPIU_Allreduce(lmin, gmin, count, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)dm)));
  if (gmax) PetscCall(MPIU_Allreduce(lmax, gmax, count, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)dm)));
  PetscFunctionReturn(0);
}

/*@
  DMProjectCoordinates - Project coordinates to a different space

  Input Parameters:
+ dm      - The `DM` object
- disc    - The new coordinate discretization or NULL to ensure a coordinate discretization exists

  Level: intermediate

  Notes:
  A `PetscFE` defines an approximation space using a `PetscSpace`, which represents the basis functions, and a `PetscDualSpace`, which defines the interpolation operation
  in the space.

  This function takes the current mesh coordinates, which are discretized using some `PetscFE` space, and projects this function into a new `PetscFE` space.
  The coordinate projection is done on the continuous coordinates, and if possible, the discontinuous coordinates are also updated.

  Developer Note:
  With more effort, we could directly project the discontinuous coordinates also.

.seealso: `DM`, `PetscFE`, `DMGetCoordinateField()`
@*/
PetscErrorCode DMProjectCoordinates(DM dm, PetscFE disc)
{
  PetscFE      discOld;
  PetscClassId classid;
  DM           cdmOld, cdmNew;
  Vec          coordsOld, coordsNew;
  Mat          matInterp;
  PetscBool    same_space = PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (disc) PetscValidHeaderSpecific(disc, PETSCFE_CLASSID, 2);

  PetscCall(DMGetCoordinateDM(dm, &cdmOld));
  /* Check current discretization is compatible */
  PetscCall(DMGetField(cdmOld, 0, NULL, (PetscObject *)&discOld));
  PetscCall(PetscObjectGetClassId((PetscObject)discOld, &classid));
  if (classid != PETSCFE_CLASSID) {
    if (classid == PETSC_CONTAINER_CLASSID) {
      PetscFE        feLinear;
      DMPolytopeType ct;
      PetscInt       dim, dE, cStart, cEnd;
      PetscBool      simplex;

      /* Assume linear vertex coordinates */
      PetscCall(DMGetDimension(dm, &dim));
      PetscCall(DMGetCoordinateDim(dm, &dE));
      PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
      if (cEnd > cStart) {
        PetscCall(DMPlexGetCellType(dm, cStart, &ct));
        switch (ct) {
        case DM_POLYTOPE_TRI_PRISM:
        case DM_POLYTOPE_TRI_PRISM_TENSOR:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot autoamtically create coordinate space for prisms");
        default:
          break;
        }
      }
      PetscCall(DMPlexIsSimplex(dm, &simplex));
      PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, dE, simplex, 1, -1, &feLinear));
      PetscCall(DMSetField(cdmOld, 0, NULL, (PetscObject)feLinear));
      PetscCall(PetscFEDestroy(&feLinear));
      PetscCall(DMCreateDS(cdmOld));
      PetscCall(DMGetField(cdmOld, 0, NULL, (PetscObject *)&discOld));
    } else {
      const char *discname;

      PetscCall(PetscObjectGetType((PetscObject)discOld, &discname));
      SETERRQ(PetscObjectComm((PetscObject)discOld), PETSC_ERR_SUP, "Discretization type %s not supported", discname);
    }
  }
  if (!disc) PetscFunctionReturn(0);
  { // Check if the new space is the same as the old modulo quadrature
    PetscDualSpace dsOld, ds;
    PetscCall(PetscFEGetDualSpace(discOld, &dsOld));
    PetscCall(PetscFEGetDualSpace(disc, &ds));
    PetscCall(PetscDualSpaceEqual(dsOld, ds, &same_space));
  }
  /* Make a fresh clone of the coordinate DM */
  PetscCall(DMClone(cdmOld, &cdmNew));
  PetscCall(DMSetField(cdmNew, 0, NULL, (PetscObject)disc));
  PetscCall(DMCreateDS(cdmNew));
  PetscCall(DMGetCoordinates(dm, &coordsOld));
  if (same_space) {
    PetscCall(PetscObjectReference((PetscObject)coordsOld));
    coordsNew = coordsOld;
  } else { // Project the coordinate vector from old to new space
    PetscCall(DMCreateGlobalVector(cdmNew, &coordsNew));
    PetscCall(DMCreateInterpolation(cdmOld, cdmNew, &matInterp, NULL));
    PetscCall(MatInterpolate(matInterp, coordsOld, coordsNew));
    PetscCall(MatDestroy(&matInterp));
  }
  /* Set new coordinate structures */
  PetscCall(DMSetCoordinateField(dm, NULL));
  PetscCall(DMSetCoordinateDM(dm, cdmNew));
  PetscCall(DMSetCoordinates(dm, coordsNew));
  PetscCall(VecDestroy(&coordsNew));
  PetscCall(DMDestroy(&cdmNew));
  PetscFunctionReturn(0);
}

/*@
  DMLocatePoints - Locate the points in v in the mesh and return a `PetscSF` of the containing cells

  Collective on v (see explanation below)

  Input Parameters:
+ dm - The `DM`
- ltype - The type of point location, e.g. `DM_POINTLOCATION_NONE` or `DM_POINTLOCATION_NEAREST`

  Input/Output Parameters:
+ v - The `Vec` of points, on output contains the nearest mesh points to the given points if `DM_POINTLOCATION_NEAREST` is used
- cellSF - Points to either NULL, or a `PetscSF` with guesses for which cells contain each point;
           on output, the `PetscSF` containing the ranks and local indices of the containing points

  Level: developer

  Notes:
  To do a search of the local cells of the mesh, v should have `PETSC_COMM_SELF` as its communicator.
  To do a search of all the cells in the distributed mesh, v should have the same communicator as dm.

  Points will only be located in owned cells, not overlap cells arising from `DMPlexDistribute()` or other overlapping distributions.

  If *cellSF is NULL on input, a `PetscSF` will be created.
  If *cellSF is not NULL on input, it should point to an existing `PetscSF`, whose graph will be used as initial guesses.

  An array that maps each point to its containing cell can be obtained with
.vb
    const PetscSFNode *cells;
    PetscInt           nFound;
    const PetscInt    *found;

    PetscSFGetGraph(cellSF,NULL,&nFound,&found,&cells);
.ve

  Where cells[i].rank is the rank of the cell containing point found[i] (or i if found == NULL), and cells[i].index is
  the index of the cell in its rank's local numbering.

.seealso: `DM`, `DMSetCoordinates()`, `DMSetCoordinatesLocal()`, `DMGetCoordinates()`, `DMGetCoordinatesLocal()`, `DMPointLocationType`
@*/
PetscErrorCode DMLocatePoints(DM dm, Vec v, DMPointLocationType ltype, PetscSF *cellSF)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 2);
  PetscValidPointer(cellSF, 4);
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
  PetscFunctionReturn(0);
}
