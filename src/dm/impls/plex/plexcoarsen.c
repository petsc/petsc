#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#ifdef PETSC_HAVE_PRAGMATIC
#include <pragmatic/cpragmatic.h>
#endif

#undef __FUNCT__
#define __FUNCT__ "DMCoarsen_Plex"
PetscErrorCode DMCoarsen_Plex(DM dm, MPI_Comm comm, DM *dmCoarsened)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  DM             udm, coordDM;
  DMLabel        bd;
  Vec            coordinates;
  PetscSection   coordSection;
  const PetscScalar *coords;
  double        *coarseCoords;
  IS             bdIS;
  PetscReal     *x, *y, *z, *metric;
  const PetscInt *faces;
  PetscInt      *cells, *bdFaces, *bdFaceIds;
  PetscInt       dim, numCorners, cStart, cEnd, numCells, numCoarseCells, c, vStart, vEnd, numVertices, numCoarseVertices, v, numBdFaces, numCoarseBdFaces, f, maxConeSize, bdSize, coff;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#ifdef PETSC_HAVE_PRAGMATIC
  if (!mesh->coarseMesh) {
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(dm, &coordDM);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(coordDM, &coordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = DMPlexUninterpolate(dm, &udm);CHKERRQ(ierr);
    ierr = DMPlexGetMaxSizes(udm, &maxConeSize, NULL);CHKERRQ(ierr);
    numCells    = cEnd - cStart;
    numVertices = vEnd - vStart;
    ierr = PetscCalloc5(numVertices, &x, numVertices, &y, numVertices, &z, numVertices, &metric, numCells*maxConeSize, &cells);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
    for (v = 0; v < numVertices; ++v) {
      PetscInt off;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      x[v] = coords[off+0];
      y[v] = coords[off+1];
      if (dim > 2) z[v] = coords[off+2];
    }
    ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
    for (c = 0, coff = 0; c < numCells; ++c) {
      const PetscInt *cone;
      PetscInt        coneSize, cl;

      ierr = DMPlexGetConeSize(udm, c, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(udm, c, &cone);CHKERRQ(ierr);
      for (cl = 0; cl < coneSize; ++cl) cells[coff++] = cone[cl];
    }
    switch (dim) {
    case 2:
      pragmatic_2d_init(&numVertices, &numCells, cells, x, y);
      break;
    case 3:
      pragmatic_3d_init(&numVertices, &numCells, cells, x, y, z);
      break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No Pragmatic coarsening defined for dimension %d", dim);
    }
    ierr = DMDestroy(&udm);CHKERRQ(ierr);
    /* Create boundary mesh */
    ierr = DMLabelCreate("boundary", &bd);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(dm, bd);CHKERRQ(ierr);
    ierr = DMLabelGetStratumIS(bd, 1, &bdIS);CHKERRQ(ierr);
    ierr = DMLabelGetStratumSize(bd, 1, &numBdFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(bdIS, &faces);CHKERRQ(ierr);
    for (f = 0, bdSize = 0; f < numBdFaces; ++f) {
      PetscInt *closure = NULL;
      PetscInt  closureSize, cl;

      ierr = DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (cl = 0; cl < closureSize*2; cl += 2) {
        if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) ++bdSize;
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = PetscMalloc2(bdSize, &bdFaces, numBdFaces, &bdFaceIds);CHKERRQ(ierr);
    for (f = 0, bdSize = 0; f < numBdFaces; ++f) {
      PetscInt *closure = NULL;
      PetscInt  closureSize, cl;

      ierr = DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (cl = 0; cl < closureSize*2; cl += 2) {
        if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) bdFaces[bdSize++] = closure[cl];
      }
      /* TODO Fix */
      bdFaceIds[f] = 1;
      ierr = DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&bdIS);CHKERRQ(ierr);
    pragmatic_set_boundary(&numBdFaces, bdFaces, bdFaceIds);
    /* Create metric */
    if (dim == 2) {for (v = 0; v < numVertices; ++v) metric[v*4+0] = metric[v*4+3] = 1.0;}
    else          {for (v = 0; v < numVertices; ++v) metric[v*9+0] = metric[v*9+4] = metric[v*9+8] = 1.0;}
    pragmatic_set_metric(metric);
    pragmatic_adapt();
    /* Read out mesh */
    pragmatic_get_info(&numCoarseVertices, &numCoarseCells);
    switch (dim) {
    case 2:
      pragmatic_get_coords_2d(x, y);
      numCorners = 3;
      for (v = 0; v < numCoarseVertices; ++v) {coarseCoords[v*2+0] = x[v]; coarseCoords[v*2+1] = y[v];}
      break;
    case 3:
      pragmatic_get_coords_3d(x, y, z);
      numCorners = 4;
      for (v = 0; v < numCoarseVertices; ++v) {coarseCoords[v*3+0] = x[v]; coarseCoords[v*3+1] = y[v]; coarseCoords[v*3+2] = z[v];}
      break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No Pragmatic coarsening defined for dimension %d", dim);
    }
    pragmatic_get_elements(cells);
    /* TODO Read out markers for boundary */
    ierr = PetscMalloc1(numCoarseVertices*dim, &coarseCoords);CHKERRQ(ierr);
    ierr = DMPlexCreateFromCellList(PetscObjectComm((PetscObject) dm), dim, numCoarseCells, numCoarseVertices, numCorners, PETSC_TRUE, cells, dim, coarseCoords, &mesh->coarseMesh);CHKERRQ(ierr);
    pragmatic_finalize();
    ierr = PetscFree4(x, y, z, metric);CHKERRQ(ierr);
    ierr = PetscFree2(bdFaces, bdFaceIds);CHKERRQ(ierr);
    ierr = PetscFree(coarseCoords);CHKERRQ(ierr);
  }
#endif
  ierr = PetscObjectReference((PetscObject) mesh->coarseMesh);CHKERRQ(ierr);
  *dmCoarsened = mesh->coarseMesh;
  PetscFunctionReturn(0);
}
