#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

/*@C
  PetscFEGeomCreate - Create a `PetscFEGeom` object to manage geometry for a group of cells

  Input Parameters:
+ quad     - A `PetscQuadrature` determining the tabulation
. numCells - The number of cells in the group
. dimEmbed - The coordinate dimension
- mode     - Type of geometry data to store

  Output Parameter:
. geom - The `PetscFEGeom` object, which is a struct not a `PetscObject`

  Level: beginner

.seealso: `PetscFEGeom`, `PetscQuadrature`, `PetscFEGeomDestroy()`, `PetscFEGeomComplete()`
@*/
PetscErrorCode PetscFEGeomCreate(PetscQuadrature quad, PetscInt numCells, PetscInt dimEmbed, PetscFEGeomMode mode, PetscFEGeom **geom)
{
  PetscFEGeom     *g;
  PetscInt         dim, Nq, N;
  const PetscReal *p;

  PetscFunctionBegin;
  PetscCall(PetscQuadratureGetData(quad, &dim, NULL, &Nq, &p, NULL));
  PetscCall(PetscNew(&g));
  g->mode      = mode;
  g->xi        = p;
  g->numCells  = numCells;
  g->numPoints = Nq;
  g->dim       = dim;
  g->dimEmbed  = dimEmbed;
  N            = numCells * Nq;
  PetscCall(PetscCalloc3(N * dimEmbed, &g->v, N * dimEmbed * dimEmbed, &g->J, N, &g->detJ));
  if (mode == PETSC_FEGEOM_BOUNDARY || mode == PETSC_FEGEOM_COHESIVE) {
    PetscCall(PetscCalloc2(numCells, &g->face, N * dimEmbed, &g->n));
    PetscCall(PetscCalloc6(N * dimEmbed * dimEmbed, &g->suppJ[0], N * dimEmbed * dimEmbed, &g->suppJ[1], N * dimEmbed * dimEmbed, &g->suppInvJ[0], N * dimEmbed * dimEmbed, &g->suppInvJ[1], N, &g->suppDetJ[0], N, &g->suppDetJ[1]));
  }
  PetscCall(PetscCalloc1(N * dimEmbed * dimEmbed, &g->invJ));
  *geom = g;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFEGeomDestroy - Destroy a `PetscFEGeom` object

  Input Parameter:
. geom - `PetscFEGeom` object

  Level: beginner

.seealso: `PetscFEGeom`, `PetscFEGeomCreate()`
@*/
PetscErrorCode PetscFEGeomDestroy(PetscFEGeom **geom)
{
  PetscFunctionBegin;
  if (!*geom) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFree3((*geom)->v, (*geom)->J, (*geom)->detJ));
  PetscCall(PetscFree((*geom)->invJ));
  PetscCall(PetscFree2((*geom)->face, (*geom)->n));
  PetscCall(PetscFree6((*geom)->suppJ[0], (*geom)->suppJ[1], (*geom)->suppInvJ[0], (*geom)->suppInvJ[1], (*geom)->suppDetJ[0], (*geom)->suppDetJ[1]));
  PetscCall(PetscFree(*geom));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFEGeomGetChunk - Get a chunk of cells in the group as a `PetscFEGeom`

  Input Parameters:
+ geom   - `PetscFEGeom` object
. cStart - The first cell in the chunk
- cEnd   - The first cell not in the chunk

  Output Parameter:
. chunkGeom - an array of cells of length `cEnd` - `cStart`

  Level: intermediate

  Note:
  Use `PetscFEGeomRestoreChunk()` to return the result

.seealso: `PetscFEGeom`, `PetscFEGeomRestoreChunk()`, `PetscFEGeomCreate()`
@*/
PetscErrorCode PetscFEGeomGetChunk(PetscFEGeom *geom, PetscInt cStart, PetscInt cEnd, PetscFEGeom *chunkGeom[])
{
  PetscInt Nq;
  PetscInt dE;

  PetscFunctionBegin;
  PetscAssertPointer(geom, 1);
  PetscAssertPointer(chunkGeom, 4);
  if (!*chunkGeom) PetscCall(PetscNew(chunkGeom));
  Nq                        = geom->numPoints;
  dE                        = geom->dimEmbed;
  (*chunkGeom)->mode        = geom->mode;
  (*chunkGeom)->dim         = geom->dim;
  (*chunkGeom)->dimEmbed    = geom->dimEmbed;
  (*chunkGeom)->numPoints   = geom->numPoints;
  (*chunkGeom)->numCells    = cEnd - cStart;
  (*chunkGeom)->xi          = geom->xi;
  (*chunkGeom)->v           = PetscSafePointerPlusOffset(geom->v, Nq * dE * cStart);
  (*chunkGeom)->J           = PetscSafePointerPlusOffset(geom->J, Nq * dE * dE * cStart);
  (*chunkGeom)->invJ        = PetscSafePointerPlusOffset(geom->invJ, Nq * dE * dE * cStart);
  (*chunkGeom)->detJ        = PetscSafePointerPlusOffset(geom->detJ, Nq * cStart);
  (*chunkGeom)->n           = PetscSafePointerPlusOffset(geom->n, Nq * dE * cStart);
  (*chunkGeom)->face        = PetscSafePointerPlusOffset(geom->face, cStart);
  (*chunkGeom)->suppJ[0]    = PetscSafePointerPlusOffset(geom->suppJ[0], Nq * dE * dE * cStart);
  (*chunkGeom)->suppJ[1]    = PetscSafePointerPlusOffset(geom->suppJ[1], Nq * dE * dE * cStart);
  (*chunkGeom)->suppInvJ[0] = PetscSafePointerPlusOffset(geom->suppInvJ[0], Nq * dE * dE * cStart);
  (*chunkGeom)->suppInvJ[1] = PetscSafePointerPlusOffset(geom->suppInvJ[1], Nq * dE * dE * cStart);
  (*chunkGeom)->suppDetJ[0] = PetscSafePointerPlusOffset(geom->suppDetJ[0], Nq * cStart);
  (*chunkGeom)->suppDetJ[1] = PetscSafePointerPlusOffset(geom->suppDetJ[1], Nq * cStart);
  (*chunkGeom)->isAffine    = geom->isAffine;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFEGeomRestoreChunk - Restore the chunk obtained with `PetscFEGeomCreateChunk()`

  Input Parameters:
+ geom      - `PetscFEGeom` object
. cStart    - The first cell in the chunk
. cEnd      - The first cell not in the chunk
- chunkGeom - The chunk of cells

  Level: intermediate

.seealso: `PetscFEGeom`, `PetscFEGeomGetChunk()`, `PetscFEGeomCreate()`
@*/
PetscErrorCode PetscFEGeomRestoreChunk(PetscFEGeom *geom, PetscInt cStart, PetscInt cEnd, PetscFEGeom **chunkGeom)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(*chunkGeom));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFEGeomGetPoint - Get the geometry for cell `c` at point `p` as a `PetscFEGeom`

  Input Parameters:
+ geom    - `PetscFEGeom` object
. c       - The cell
. p       - The point
- pcoords - The reference coordinates of point `p`, or `NULL`

  Output Parameter:
. pgeom - The geometry of cell `c` at point `p`

  Level: intermediate

  Notes:
  For affine geometries, this only copies to `pgeom` at point 0. Since we copy pointers into `pgeom`,
  nothing needs to be done with it afterwards.

  In the affine case, `pgeom` must have storage for the integration point coordinates in pgeom->v if `pcoords` is passed in.

.seealso: `PetscFEGeom`, `PetscFEGeomRestoreChunk()`, `PetscFEGeomCreate()`
@*/
PetscErrorCode PetscFEGeomGetPoint(PetscFEGeom *geom, PetscInt c, PetscInt p, const PetscReal pcoords[], PetscFEGeom *pgeom)
{
  const PetscInt dim = geom->dim;
  const PetscInt dE  = geom->dimEmbed;
  const PetscInt Np  = geom->numPoints;

  PetscFunctionBeginHot;
  pgeom->mode     = geom->mode;
  pgeom->dim      = dim;
  pgeom->dimEmbed = dE;
  //pgeom->isAffine = geom->isAffine;
  if (geom->isAffine) {
    if (!p) {
      pgeom->xi   = geom->xi;
      pgeom->J    = &geom->J[c * Np * dE * dE];
      pgeom->invJ = &geom->invJ[c * Np * dE * dE];
      pgeom->detJ = &geom->detJ[c * Np];
      pgeom->n    = PetscSafePointerPlusOffset(geom->n, c * Np * dE);
    }
    if (pcoords) CoordinatesRefToReal(dE, dim, pgeom->xi, &geom->v[c * Np * dE], pgeom->J, pcoords, pgeom->v);
  } else {
    pgeom->v    = &geom->v[(c * Np + p) * dE];
    pgeom->J    = &geom->J[(c * Np + p) * dE * dE];
    pgeom->invJ = &geom->invJ[(c * Np + p) * dE * dE];
    pgeom->detJ = &geom->detJ[c * Np + p];
    pgeom->n    = PetscSafePointerPlusOffset(geom->n, (c * Np + p) * dE);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFEGeomGetCellPoint - Get the cell geometry for cell `c` at point `p` as a `PetscFEGeom`

  Input Parameters:
+ geom - `PetscFEGeom` object
. c    - The cell
- p    - The point

  Output Parameter:
. pgeom - The cell geometry of cell `c` at point `p`

  Level: intermediate

  Notes:
  For PETSC_FEGEOM_BOUNDARY mode, this gives the geometry for supporting cell 0. For PETSC_FEGEOM_COHESIVE mode,
  this gives the bulk geometry for that internal face.

  For affine geometries, this only copies to pgeom at point 0. Since we copy pointers into `pgeom`,
  nothing needs to be done with it afterwards.

.seealso: `PetscFEGeom`, `PetscFEGeomMode`, `PetscFEGeomRestoreChunk()`, `PetscFEGeomCreate()`
@*/
PetscErrorCode PetscFEGeomGetCellPoint(PetscFEGeom *geom, PetscInt c, PetscInt p, PetscFEGeom *pgeom)
{
  const PetscBool bd  = geom->mode == PETSC_FEGEOM_BOUNDARY ? PETSC_TRUE : PETSC_FALSE;
  const PetscInt  dim = bd ? geom->dimEmbed : geom->dim;
  const PetscInt  dE  = geom->dimEmbed;
  const PetscInt  Np  = geom->numPoints;

  PetscFunctionBeginHot;
  pgeom->mode     = geom->mode;
  pgeom->dim      = dim;
  pgeom->dimEmbed = dE;
  //pgeom->isAffine = geom->isAffine;
  if (geom->isAffine) {
    if (!p) {
      if (bd) {
        pgeom->J    = &geom->suppJ[0][c * Np * dE * dE];
        pgeom->invJ = &geom->suppInvJ[0][c * Np * dE * dE];
        pgeom->detJ = &geom->suppDetJ[0][c * Np];
      } else {
        pgeom->J    = &geom->J[c * Np * dE * dE];
        pgeom->invJ = &geom->invJ[c * Np * dE * dE];
        pgeom->detJ = &geom->detJ[c * Np];
      }
    }
  } else {
    if (bd) {
      pgeom->J    = &geom->suppJ[0][(c * Np + p) * dE * dE];
      pgeom->invJ = &geom->suppInvJ[0][(c * Np + p) * dE * dE];
      pgeom->detJ = &geom->suppDetJ[0][c * Np + p];
    } else {
      pgeom->J    = &geom->J[(c * Np + p) * dE * dE];
      pgeom->invJ = &geom->invJ[(c * Np + p) * dE * dE];
      pgeom->detJ = &geom->detJ[c * Np + p];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFEGeomComplete - Calculate derived quantities from a base geometry specification

  Input Parameter:
. geom - `PetscFEGeom` object

  Level: intermediate

.seealso: `PetscFEGeom`, `PetscFEGeomCreate()`
@*/
PetscErrorCode PetscFEGeomComplete(PetscFEGeom *geom)
{
  PetscInt i, j, N, dE;

  PetscFunctionBeginHot;
  N  = geom->numPoints * geom->numCells;
  dE = geom->dimEmbed;
  switch (dE) {
  case 3:
    for (i = 0; i < N; i++) {
      DMPlex_Det3D_Internal(&geom->detJ[i], &geom->J[dE * dE * i]);
      if (geom->invJ) DMPlex_Invert3D_Internal(&geom->invJ[dE * dE * i], &geom->J[dE * dE * i], geom->detJ[i]);
    }
    break;
  case 2:
    for (i = 0; i < N; i++) {
      DMPlex_Det2D_Internal(&geom->detJ[i], &geom->J[dE * dE * i]);
      if (geom->invJ) DMPlex_Invert2D_Internal(&geom->invJ[dE * dE * i], &geom->J[dE * dE * i], geom->detJ[i]);
    }
    break;
  case 1:
    for (i = 0; i < N; i++) {
      geom->detJ[i] = PetscAbsReal(geom->J[i]);
      if (geom->invJ) geom->invJ[i] = 1. / geom->J[i];
    }
    break;
  }
  if (geom->n) {
    for (i = 0; i < N; i++) {
      for (j = 0; j < dE; j++) geom->n[dE * i + j] = geom->J[dE * dE * i + dE * j + dE - 1] * ((dE == 2) ? -1. : 1.);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
