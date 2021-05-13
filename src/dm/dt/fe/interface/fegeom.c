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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscQuadratureGetData(quad,&dim,NULL,&Nq,&p,NULL);CHKERRQ(ierr);
  ierr = PetscNew(&g);CHKERRQ(ierr);
  g->xi        = p;
  g->numCells  = numCells;
  g->numPoints = Nq;
  g->dim       = dim;
  g->dimEmbed  = dimEmbed;
  N = numCells * Nq;
  ierr = PetscCalloc3(N * dimEmbed, &g->v, N * dimEmbed * dimEmbed, &g->J, N, &g->detJ);CHKERRQ(ierr);
  if (faceData) {
    ierr = PetscCalloc2(numCells, &g->face, N * dimEmbed, &g->n);CHKERRQ(ierr);
    ierr = PetscCalloc6(N * dimEmbed * dimEmbed, &(g->suppJ[0]),    N * dimEmbed * dimEmbed, &(g->suppJ[1]),
                        N * dimEmbed * dimEmbed, &(g->suppInvJ[0]), N * dimEmbed * dimEmbed, &(g->suppInvJ[1]),
                        N,                       &(g->suppDetJ[0]), N,                       &(g->suppDetJ[1]));CHKERRQ(ierr);
  }
  ierr = PetscCalloc1(N * dimEmbed * dimEmbed, &g->invJ);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*geom) PetscFunctionReturn(0);
  ierr = PetscFree3((*geom)->v,(*geom)->J,(*geom)->detJ);CHKERRQ(ierr);
  ierr = PetscFree((*geom)->invJ);CHKERRQ(ierr);
  ierr = PetscFree2((*geom)->face,(*geom)->n);CHKERRQ(ierr);
  ierr = PetscFree6((*geom)->suppJ[0],(*geom)->suppJ[1],(*geom)->suppInvJ[0],(*geom)->suppInvJ[1],(*geom)->suppDetJ[0],(*geom)->suppDetJ[1]);CHKERRQ(ierr);
  ierr = PetscFree(*geom);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(geom,1);
  PetscValidPointer(chunkGeom,4);
  if (!(*chunkGeom)) {
    ierr = PetscNew(chunkGeom);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(*chunkGeom);CHKERRQ(ierr);
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
