#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#ifdef PETSC_HAVE_PRAGMATIC
#include <pragmatic/cpragmatic.h>
#endif



#undef __FUNCT__
#define __FUNCT__ "DMPlexAdaptdm->coarseMesh"
PetscErrorCode DMPlexAdapt(DM dm, Vec metric, DM *dmCoarsened)
{
#ifdef PETSC_HAVE_PRAGMATIC
  DM             udm, coordDM;
  DMLabel        bd;
  Vec            coordinates;
  PetscSection   coordSection;
  const PetscScalar *coords;
  double        *coarseCoords;
  IS             bdIS;
  PetscReal     *x, *y, *z;
  const PetscInt *faces;
  PetscInt      *cells, *ccells, *bdFaces, *bdFaceIds;
  PetscInt       dim, numCorners, cStart, cEnd, numCells, numCoarseCells, c, vStart, vEnd, numVertices, numCoarseVertices, v, numBdFaces, f, maxConeSize, bdSize, coff;
  const PetscScalar   *metricArray;
  PetscReal     *met;
#endif
  PetscErrorCode ierr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
#ifdef PETSC_HAVE_PRAGMATIC
  if (!dm->coarseMesh) {
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
    ierr = PetscCalloc5(numVertices, &x, numVertices, &y, numVertices, &z, numCells*maxConeSize, &cells, dim*dim*numVertices, &met);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
    ierr = VecGetArrayRead(metric, &metricArray);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      PetscInt off;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      x[v-vStart] = coords[off+0];
      y[v-vStart] = coords[off+1];
      if (dim > 2) z[v-vStart] = coords[off+2];
      
      ierr = PetscMemcpy(&met[dim*dim*(v-vStart)], &metricArray[dim*off], dim*dim*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(metric, &metricArray);CHKERRQ(ierr);
    for (c = 0, coff = 0; c < numCells; ++c) {
      const PetscInt *cone;
      PetscInt        coneSize, cl;

      ierr = DMPlexGetConeSize(udm, c, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(udm, c, &cone);CHKERRQ(ierr);
      for (cl = 0; cl < coneSize; ++cl) cells[coff++] = cone[cl] - vStart;
    }
    switch (dim) {
    case 2:
      pragmatic_2d_init(&numVertices, &numCells, cells, x, y);
      break;
    case 3:
      pragmatic_3d_init(&numVertices, &numCells, cells, x, y, z);
      break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No Pragmatic adaptation defined for dimension %d", dim);
    }
    /* Create boundary mesh */
    ierr = DMLabelCreate("boundary", &bd);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(dm, bd);CHKERRQ(ierr);        
    ierr = DMLabelGetStratumIS(bd, 1, &bdIS);CHKERRQ(ierr);
    ierr = DMLabelGetStratumSize(bd, 1, &numBdFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(bdIS, &faces);CHKERRQ(ierr);
    for (f = 0, bdSize = 0; f < numBdFaces; ++f) {
      PetscInt *closure = NULL;
      PetscInt  closureSize, cl;

      ierr = DMPlexGetTransitiveClosure(dm, faces[f], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (cl = 0; cl < closureSize*2; cl += 2) {
        if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) ++bdSize;
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = PetscMalloc2(bdSize, &bdFaces, numBdFaces, &bdFaceIds);CHKERRQ(ierr);
    for (f = 0, bdSize = 0; f < numBdFaces; ++f) {
      PetscInt *closure = NULL;
      PetscInt  closureSize, cl;

      ierr = DMPlexGetTransitiveClosure(dm, faces[f], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (cl = 0; cl < closureSize*2; cl += 2) {
        if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) bdFaces[bdSize++] = closure[cl] - vStart;
      }
      /* TODO Fix */
      bdFaceIds[f] = 1;
      ierr = DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&bdIS);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&bd);CHKERRQ(ierr);
    pragmatic_set_boundary(&numBdFaces, bdFaces, bdFaceIds);
    pragmatic_set_metric(met);
    pragmatic_adapt();
    /* Read out mesh */
    pragmatic_get_info(&numCoarseVertices, &numCoarseCells);
    ierr = PetscMalloc1(numCoarseVertices*dim, &coarseCoords);CHKERRQ(ierr);
    switch (dim) {
    case 2:
      ierr = PetscMalloc2(numCoarseVertices, &x, numCoarseVertices, &y);CHKERRQ(ierr);
      pragmatic_get_coords_2d(x, y);
      numCorners = 3;
      for (v = 0; v < numCoarseVertices; ++v) {coarseCoords[v*2+0] = x[v]; coarseCoords[v*2+1] = y[v];}
      break;
    case 3:
      ierr = PetscMalloc3(numCoarseVertices, &x, numCoarseVertices, &y, numCoarseVertices, &z);CHKERRQ(ierr);
      pragmatic_get_coords_3d(x, y, z);
      numCorners = 4;
      for (v = 0; v < numCoarseVertices; ++v) {coarseCoords[v*3+0] = x[v]; coarseCoords[v*3+1] = y[v]; coarseCoords[v*3+2] = z[v];}
      break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No Pragmatic coarsening defined for dimension %d", dim);
    }
    ierr = PetscMalloc1(numCoarseCells*(dim+1), &ccells);CHKERRQ(ierr); // only for simplicial meshes
    pragmatic_get_elements(ccells);
    /* TODO Read out markers for boundary */
    ierr = DMPlexCreateFromCellList(PetscObjectComm((PetscObject) dm), dim, numCoarseCells, numCoarseVertices, numCorners, PETSC_TRUE, ccells, dim, coarseCoords, &dm->coarseMesh);CHKERRQ(ierr);  
    pragmatic_finalize();
    ierr = PetscFree4(x, y, z, cells);CHKERRQ(ierr);
    ierr = PetscFree2(bdFaces, bdFaceIds);CHKERRQ(ierr);
    ierr = PetscFree(coarseCoords);CHKERRQ(ierr);
  }
#endif
  ierr = PetscObjectReference((PetscObject) dm->coarseMesh);CHKERRQ(ierr);
  *dmCoarsened = dm->coarseMesh;
  PetscFunctionReturn(0);
}
