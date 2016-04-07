#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#ifdef PETSC_HAVE_PRAGMATIC
#include <pragmatic/cpragmatic.h>
#endif


#undef __FUNCT__
#define __FUNCT__ "DMCoarsen_Plex"
PetscErrorCode DMCoarsen_Plex(DM dm, MPI_Comm comm, DM *dmCoarsened)
{
#ifdef PETSC_HAVE_PRAGMATIC
  DM             udm, coordDM;
  DMLabel        bd;
  Mat            A;
  Vec            coordinates, mb, mx;
  PetscSection   coordSection;
  const PetscScalar *coords;
  double        *coarseCoords;
  IS             bdIS;
  PetscReal     *x, *y, *z, *eqns, *metric;
  PetscReal      coarseRatio = PetscSqr(0.5);
  const PetscInt *faces;
  PetscInt      *cells, *bdFaces, *bdFaceIds;
  PetscInt       dim, numCorners, cStart, cEnd, numCells, numCoarseCells, c, vStart, vEnd, numVertices, numCoarseVertices, v, numBdFaces, f, maxConeSize, size, bdSize, coff;
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
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
    ierr = PetscCalloc5(numVertices, &x, numVertices, &y, numVertices, &z, numVertices*PetscSqr(dim), &metric, numCells*maxConeSize, &cells);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      PetscInt off;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      x[v-vStart] = coords[off+0];
      y[v-vStart] = coords[off+1];
      if (dim > 2) z[v-vStart] = coords[off+2];
    }
    ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
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
      SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No Pragmatic coarsening defined for dimension %d", dim);
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
    /* Create metric */
    size = (dim*(dim+1))/2;
    ierr = PetscMalloc1(PetscSqr(size), &eqns);CHKERRQ(ierr);
    ierr = MatCreateSeqDense(PETSC_COMM_SELF, size, size, eqns, &A);CHKERRQ(ierr);
    ierr = MatCreateVecs(A, &mx, &mb);CHKERRQ(ierr);
    ierr = VecSet(mb, 1.0);CHKERRQ(ierr);
    for (c = 0; c < numCells; ++c) {
      const PetscScalar *sol;
      PetscScalar       *cellCoords = NULL;
      PetscReal          e[3], vol;
      const PetscInt    *cone;
      PetscInt           coneSize, cl, i, j, d, r;

      ierr = DMPlexVecGetClosure(dm, coordSection, coordinates, c, NULL, &cellCoords);CHKERRQ(ierr);
      /* Only works for simplices */
      for (i = 0, r = 0; i < dim+1; ++i) {
        for (j = 0; j < i; ++j, ++r) {
          for (d = 0; d < dim; ++d) e[d] = cellCoords[i*dim+d] - cellCoords[j*dim+d];
          /* FORTRAN ORDERING */
          if (dim == 2) {
            eqns[0*size+r] = PetscSqr(e[0]);
            eqns[1*size+r] = 2.0*e[0]*e[1];
            eqns[2*size+r] = PetscSqr(e[1]);
          } else {
            eqns[0*size+r] = PetscSqr(e[0]);
            eqns[1*size+r] = 2.0*e[0]*e[1];
            eqns[2*size+r] = 2.0*e[0]*e[2];
            eqns[3*size+r] = PetscSqr(e[1]);
            eqns[4*size+r] = 2.0*e[1]*e[2];
            eqns[5*size+r] = PetscSqr(e[2]);
          }
        }
      }
      ierr = MatSetUnfactored(A);CHKERRQ(ierr);
      ierr = DMPlexVecRestoreClosure(dm, coordSection, coordinates, c, NULL, &cellCoords);CHKERRQ(ierr);
      ierr = MatLUFactor(A, NULL, NULL, NULL);CHKERRQ(ierr);
      ierr = MatSolve(A, mb, mx);CHKERRQ(ierr);
      ierr = VecGetArrayRead(mx, &sol);CHKERRQ(ierr);
      ierr = DMPlexComputeCellGeometryFVM(dm, c, &vol, NULL, NULL);CHKERRQ(ierr);
      ierr = DMPlexGetCone(udm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeSize(udm, c, &coneSize);CHKERRQ(ierr);
      for (cl = 0; cl < coneSize; ++cl) {
        const PetscInt v = cone[cl] - vStart;

        if (dim == 2) {
          metric[v*4+0] += vol*coarseRatio*sol[0];
          metric[v*4+1] += vol*coarseRatio*sol[1];
          metric[v*4+2] += vol*coarseRatio*sol[1];
          metric[v*4+3] += vol*coarseRatio*sol[2];
        } else {
          metric[v*9+0] += vol*coarseRatio*sol[0];
          metric[v*9+1] += vol*coarseRatio*sol[1];
          metric[v*9+3] += vol*coarseRatio*sol[1];
          metric[v*9+2] += vol*coarseRatio*sol[2];
          metric[v*9+6] += vol*coarseRatio*sol[2];
          metric[v*9+4] += vol*coarseRatio*sol[3];
          metric[v*9+5] += vol*coarseRatio*sol[4];
          metric[v*9+7] += vol*coarseRatio*sol[4];
          metric[v*9+8] += vol*coarseRatio*sol[5];
        }
      }
      ierr = VecRestoreArrayRead(mx, &sol);CHKERRQ(ierr);
    }
    for (v = 0; v < numVertices; ++v) {
      const PetscInt *support;
      PetscInt        supportSize, s;
      PetscReal       vol, totVol = 0.0;

      ierr = DMPlexGetSupport(udm, v+vStart, &support);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(udm, v+vStart, &supportSize);CHKERRQ(ierr);
      for (s = 0; s < supportSize; ++s) {ierr = DMPlexComputeCellGeometryFVM(dm, support[s], &vol, NULL, NULL);CHKERRQ(ierr); totVol += vol;}
      for (s = 0; s < PetscSqr(dim); ++s) metric[v*PetscSqr(dim)+s] /= totVol;
    }
    ierr = VecDestroy(&mx);CHKERRQ(ierr);
    ierr = VecDestroy(&mb);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = DMDestroy(&udm);CHKERRQ(ierr);
    ierr = PetscFree(eqns);CHKERRQ(ierr);
    pragmatic_set_metric(metric);
    pragmatic_adapt();
    /* Read out mesh */
    pragmatic_get_info(&numCoarseVertices, &numCoarseCells);
    ierr = PetscMalloc1(numCoarseVertices*dim, &coarseCoords);CHKERRQ(ierr);
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
    ierr = DMPlexCreateFromCellList(PetscObjectComm((PetscObject) dm), dim, numCoarseCells, numCoarseVertices, numCorners, PETSC_TRUE, cells, dim, coarseCoords, &dm->coarseMesh);CHKERRQ(ierr);
    pragmatic_finalize();
    ierr = PetscFree5(x, y, z, metric, cells);CHKERRQ(ierr);
    ierr = PetscFree2(bdFaces, bdFaceIds);CHKERRQ(ierr);
    ierr = PetscFree(coarseCoords);CHKERRQ(ierr);
  }
#endif
  ierr = PetscObjectReference((PetscObject) dm->coarseMesh);CHKERRQ(ierr);
  *dmCoarsened = dm->coarseMesh;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCoarsenHierarchy_Plex"
PetscErrorCode DMCoarsenHierarchy_Plex(DM dm, PetscInt nlevels, DM dmCoarsened[])
{
  DM             rdm = dm;
  PetscInt       c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (c = nlevels-1; c >= 0; --c) {
    ierr = DMCoarsen(rdm, PetscObjectComm((PetscObject) dm), &dmCoarsened[c]);CHKERRQ(ierr);
    ierr = DMCopyBoundary(rdm, dmCoarsened[c]);CHKERRQ(ierr);
    ierr = DMSetCoarseDM(rdm, dmCoarsened[c]);CHKERRQ(ierr);
    rdm  = dmCoarsened[c];
  }
  PetscFunctionReturn(0);
}
