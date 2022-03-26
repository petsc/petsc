#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

/*@C
  PetscFEGeomCreate - Create a PetscFEGeom object to manage geometry for a group of cells

  Input Parameters:
+ quad     - A PetscQuadrature determining the tabulation
. numCells - The number of cells in the group
. dimEmbed - The coordinate dimension
- faceData - Flag to construct geometry data for the faces

  Output Parameter:
. geom     - The PetscFEGeom object

  Level: beginner

.seealso: PetscFEGeomDestroy(), PetscFEGeomComplete()
@*/
PetscErrorCode PetscFEGeomCreate(PetscQuadrature quad, PetscInt numCells, PetscInt dimEmbed, PetscBool faceData, PetscFEGeom **geom)
{
  PetscFEGeom     *g;
  PetscInt        dim, Nq, N;
  const PetscReal *p;

  PetscFunctionBegin;
  PetscCall(PetscQuadratureGetData(quad,&dim,NULL,&Nq,&p,NULL));
  PetscCall(PetscNew(&g));
  g->xi         = p;
  g->numCells   = numCells;
  g->numPoints  = Nq;
  g->dim        = dim;
  g->dimEmbed   = dimEmbed;
  g->isCohesive = PETSC_FALSE;
  N = numCells * Nq;
  PetscCall(PetscCalloc3(N * dimEmbed, &g->v, N * dimEmbed * dimEmbed, &g->J, N, &g->detJ));
  if (faceData) {
    PetscCall(PetscCalloc2(numCells, &g->face, N * dimEmbed, &g->n));
    PetscCall(PetscCalloc6(N * dimEmbed * dimEmbed, &(g->suppJ[0]),    N * dimEmbed * dimEmbed, &(g->suppJ[1]),
                         N * dimEmbed * dimEmbed, &(g->suppInvJ[0]), N * dimEmbed * dimEmbed, &(g->suppInvJ[1]),
                         N,                       &(g->suppDetJ[0]), N,                       &(g->suppDetJ[1])));
  }
  PetscCall(PetscCalloc1(N * dimEmbed * dimEmbed, &g->invJ));
  *geom = g;
  PetscFunctionReturn(0);
}

/*@C
  PetscFEGeomDestroy - Destroy a PetscFEGeom object

  Input Parameter:
. geom - PetscFEGeom object

  Level: beginner

.seealso: PetscFEGeomCreate()
@*/
PetscErrorCode PetscFEGeomDestroy(PetscFEGeom **geom)
{
  PetscFunctionBegin;
  if (!*geom) PetscFunctionReturn(0);
  PetscCall(PetscFree3((*geom)->v,(*geom)->J,(*geom)->detJ));
  PetscCall(PetscFree((*geom)->invJ));
  PetscCall(PetscFree2((*geom)->face,(*geom)->n));
  PetscCall(PetscFree6((*geom)->suppJ[0],(*geom)->suppJ[1],(*geom)->suppInvJ[0],(*geom)->suppInvJ[1],(*geom)->suppDetJ[0],(*geom)->suppDetJ[1]));
  PetscCall(PetscFree(*geom));
  PetscFunctionReturn(0);
}

/*@C
  PetscFEGeomGetChunk - Get a chunk of cells in the group as a PetscFEGeom

  Input Parameters:
+ geom   - PetscFEGeom object
. cStart - The first cell in the chunk
- cEnd   - The first cell not in the chunk

  Output Parameter:
. chunkGeom - The chunk of cells

  Level: intermediate

.seealso: PetscFEGeomRestoreChunk(), PetscFEGeomCreate()
@*/
PetscErrorCode PetscFEGeomGetChunk(PetscFEGeom *geom, PetscInt cStart, PetscInt cEnd, PetscFEGeom **chunkGeom)
{
  PetscInt       Nq;
  PetscInt       dE;

  PetscFunctionBegin;
  PetscValidPointer(geom,1);
  PetscValidPointer(chunkGeom,4);
  if (!(*chunkGeom)) {
    PetscCall(PetscNew(chunkGeom));
  }
  Nq = geom->numPoints;
  dE= geom->dimEmbed;
  (*chunkGeom)->dim = geom->dim;
  (*chunkGeom)->dimEmbed = geom->dimEmbed;
  (*chunkGeom)->numPoints = geom->numPoints;
  (*chunkGeom)->numCells = cEnd - cStart;
  (*chunkGeom)->xi = geom->xi;
  (*chunkGeom)->v = &geom->v[Nq*dE*cStart];
  (*chunkGeom)->J = &geom->J[Nq*dE*dE*cStart];
  (*chunkGeom)->invJ = (geom->invJ) ? &geom->invJ[Nq*dE*dE*cStart] : NULL;
  (*chunkGeom)->detJ = &geom->detJ[Nq*cStart];
  (*chunkGeom)->n = geom->n ? &geom->n[Nq*dE*cStart] : NULL;
  (*chunkGeom)->face = geom->face ? &geom->face[cStart] : NULL;
  (*chunkGeom)->suppJ[0]    = geom->suppJ[0]    ? &geom->suppJ[0][Nq*dE*dE*cStart]    : NULL;
  (*chunkGeom)->suppJ[1]    = geom->suppJ[1]    ? &geom->suppJ[1][Nq*dE*dE*cStart]    : NULL;
  (*chunkGeom)->suppInvJ[0] = geom->suppInvJ[0] ? &geom->suppInvJ[0][Nq*dE*dE*cStart] : NULL;
  (*chunkGeom)->suppInvJ[1] = geom->suppInvJ[1] ? &geom->suppInvJ[1][Nq*dE*dE*cStart] : NULL;
  (*chunkGeom)->suppDetJ[0] = geom->suppDetJ[0] ? &geom->suppDetJ[0][Nq*cStart]       : NULL;
  (*chunkGeom)->suppDetJ[1] = geom->suppDetJ[1] ? &geom->suppDetJ[1][Nq*cStart]       : NULL;
  (*chunkGeom)->isAffine = geom->isAffine;
  PetscFunctionReturn(0);
}

/*@C
  PetscFEGeomRestoreChunk - Restore the chunk

  Input Parameters:
+ geom      - PetscFEGeom object
. cStart    - The first cell in the chunk
. cEnd      - The first cell not in the chunk
- chunkGeom - The chunk of cells

  Level: intermediate

.seealso: PetscFEGeomGetChunk(), PetscFEGeomCreate()
@*/
PetscErrorCode PetscFEGeomRestoreChunk(PetscFEGeom *geom, PetscInt cStart, PetscInt cEnd, PetscFEGeom **chunkGeom)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(*chunkGeom));
  PetscFunctionReturn(0);
}

/*@C
  PetscFEGeomGetPoint - Get the geometry for cell c at point p as a PetscFEGeom

  Input Parameters:
+ geom    - PetscFEGeom object
. c       - The cell
. p       - The point
- pcoords - The reference coordinates of point p, or NULL

  Output Parameter:
. pgeom - The geometry of cell c at point p

  Note: For affine geometries, this only copies to pgeom at point 0. Since we copy pointers into pgeom,
  nothing needs to be done with it afterwards.

  In the affine case, pgeom must have storage for the integration point coordinates in pgeom->v if pcoords is passed in.

  Level: intermediate

.seealso: PetscFEGeomRestoreChunk(), PetscFEGeomCreate()
@*/
PetscErrorCode PetscFEGeomGetPoint(PetscFEGeom *geom, PetscInt c, PetscInt p, const PetscReal pcoords[], PetscFEGeom *pgeom)
{
  const PetscInt dim = geom->dim;
  const PetscInt dE  = geom->dimEmbed;
  const PetscInt Np  = geom->numPoints;

  PetscFunctionBeginHot;
  pgeom->dim      = dim;
  pgeom->dimEmbed = dE;
  //pgeom->isAffine = geom->isAffine;
  if (geom->isAffine) {
    if (!p) {
      pgeom->xi   = geom->xi;
      pgeom->J    = &geom->J[c*Np*dE*dE];
      pgeom->invJ = &geom->invJ[c*Np*dE*dE];
      pgeom->detJ = &geom->detJ[c*Np];
      pgeom->n    = geom->n ? &geom->n[c*Np*dE] : NULL;
    }
    if (pcoords) {CoordinatesRefToReal(dE, dim, pgeom->xi, &geom->v[c*Np*dE], pgeom->J, pcoords, pgeom->v);}
  } else {
    pgeom->v    = &geom->v[(c*Np+p)*dE];
    pgeom->J    = &geom->J[(c*Np+p)*dE*dE];
    pgeom->invJ = &geom->invJ[(c*Np+p)*dE*dE];
    pgeom->detJ = &geom->detJ[c*Np+p];
    pgeom->n    = geom->n ? &geom->n[(c*Np+p)*dE] : NULL;
  }
  PetscFunctionReturn(0);
}

/*@C
  PetscFEGeomGetCellPoint - Get the cell geometry for face f at point p as a PetscFEGeom

  Input Parameters:
+ geom    - PetscFEGeom object
. f       - The face
- p       - The point

  Output Parameter:
. pgeom - The cell geometry of face f at point p

  Note: For affine geometries, this only copies to pgeom at point 0. Since we copy pointers into pgeom,
  nothing needs to be done with it afterwards.

  Level: intermediate

.seealso: PetscFEGeomRestoreChunk(), PetscFEGeomCreate()
@*/
PetscErrorCode PetscFEGeomGetCellPoint(PetscFEGeom *geom, PetscInt c, PetscInt p, PetscFEGeom *pgeom)
{
  const PetscBool bd  = geom->dimEmbed > geom->dim && !geom->isCohesive ? PETSC_TRUE : PETSC_FALSE;
  const PetscInt  dim = bd ? geom->dimEmbed : geom->dim;
  const PetscInt  dE  = geom->dimEmbed;
  const PetscInt  Np  = geom->numPoints;

  PetscFunctionBeginHot;
  pgeom->dim      = dim;
  pgeom->dimEmbed = dE;
  //pgeom->isAffine = geom->isAffine;
  if (geom->isAffine) {
    if (!p) {
      if (bd) {
        pgeom->J     = &geom->suppJ[0][c*Np*dE*dE];
        pgeom->invJ  = &geom->suppInvJ[0][c*Np*dE*dE];
        pgeom->detJ  = &geom->suppDetJ[0][c*Np];
      } else {
        pgeom->J    = &geom->J[c*Np*dE*dE];
        pgeom->invJ = &geom->invJ[c*Np*dE*dE];
        pgeom->detJ = &geom->detJ[c*Np];
      }
    }
  } else {
    if (bd) {
      pgeom->J     = &geom->suppJ[0][(c*Np+p)*dE*dE];
      pgeom->invJ  = &geom->suppInvJ[0][(c*Np+p)*dE*dE];
      pgeom->detJ  = &geom->suppDetJ[0][c*Np+p];
    } else {
      pgeom->J    = &geom->J[(c*Np+p)*dE*dE];
      pgeom->invJ = &geom->invJ[(c*Np+p)*dE*dE];
      pgeom->detJ = &geom->detJ[c*Np+p];
    }
  }
  PetscFunctionReturn(0);
}

/*@
  PetscFEGeomComplete - Calculate derived quntites from base geometry specification

  Input Parameter:
. geom - PetscFEGeom object

  Level: intermediate

.seealso: PetscFEGeomCreate()
@*/
PetscErrorCode PetscFEGeomComplete(PetscFEGeom *geom)
{
  PetscInt i, j, N, dE;

  PetscFunctionBeginHot;
  N = geom->numPoints * geom->numCells;
  dE = geom->dimEmbed;
  switch (dE) {
  case 3:
    for (i = 0; i < N; i++) {
      DMPlex_Det3D_Internal(&geom->detJ[i], &geom->J[dE*dE*i]);
      if (geom->invJ) DMPlex_Invert3D_Internal(&geom->invJ[dE*dE*i], &geom->J[dE*dE*i], geom->detJ[i]);
    }
    break;
  case 2:
    for (i = 0; i < N; i++) {
      DMPlex_Det2D_Internal(&geom->detJ[i], &geom->J[dE*dE*i]);
      if (geom->invJ) DMPlex_Invert2D_Internal(&geom->invJ[dE*dE*i], &geom->J[dE*dE*i], geom->detJ[i]);
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
      for (j = 0; j < dE; j++) {
        geom->n[dE*i + j] = geom->J[dE*dE*i + dE*j + dE-1] * ((dE == 2) ? -1. : 1.);
      }
    }
  }
  PetscFunctionReturn(0);
}
