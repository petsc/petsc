#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <pragmatic/cpragmatic.h>

PETSC_EXTERN PetscErrorCode DMAdaptMetric_Pragmatic_Plex(DM dm, Vec vertexMetric, DMLabel bdLabel, DMLabel rgLabel, DM *dmNew)
{
  MPI_Comm           comm;
  const char        *bdName = "_boundary_";
#if 0
  DM                 odm = dm;
#endif
  DM                 udm, cdm;
  DMLabel            bdLabelFull;
  const char        *bdLabelName;
  IS                 bdIS, globalVertexNum;
  PetscSection       coordSection;
  Vec                coordinates;
  const PetscScalar *coords, *met;
  const PetscInt    *bdFacesFull, *gV;
  PetscInt          *bdFaces, *bdFaceIds, *l2gv;
  PetscReal         *x, *y, *z, *metric;
  PetscInt          *cells;
  PetscInt           dim, cStart, cEnd, numCells, c, coff, vStart, vEnd, numVertices, numLocVertices, v;
  PetscInt           off, maxConeSize, numBdFaces, f, bdSize, i, j, Nd;
  PetscBool          flg, isotropic, uniform;
  DMLabel            bdLabelNew;
  PetscReal         *coordsNew;
  PetscInt          *bdTags;
  PetscReal         *xNew[3] = {NULL, NULL, NULL};
  PetscInt          *cellsNew;
  PetscInt           d, numCellsNew, numVerticesNew;
  PetscInt           numCornersNew, fStart, fEnd;
  PetscMPIInt        numProcs;
  PetscErrorCode     ierr;

  PetscFunctionBegin;

  /* Check for FEM adjacency flags */
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRMPI(ierr);
  if (bdLabel) {
    ierr = PetscObjectGetName((PetscObject) bdLabel, &bdLabelName);CHKERRQ(ierr);
    ierr = PetscStrcmp(bdLabelName, bdName, &flg);CHKERRQ(ierr);
    PetscAssertFalse(flg,comm, PETSC_ERR_ARG_WRONG, "\"%s\" cannot be used as label for boundary facets", bdLabelName);
  }
  PetscAssertFalse(rgLabel,comm, PETSC_ERR_ARG_WRONG, "Cannot currently preserve cell tags with Pragmatic");
#if 0
  /* Check for overlap by looking for cell in the SF */
  if (!overlapped) {
    ierr = DMPlexDistributeOverlap(odm, 1, NULL, &dm);CHKERRQ(ierr);
    if (!dm) {dm = odm; ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);}
  }
#endif

  /* Get mesh information */
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexUninterpolate(dm, &udm);CHKERRQ(ierr);
  ierr = DMPlexGetMaxSizes(udm, &maxConeSize, NULL);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  if (numCells == 0) {
    PetscMPIInt rank;

    ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot perform mesh adaptation because process %d does not own any cells.", rank);
  }
  numVertices = vEnd - vStart;
  ierr = PetscCalloc5(numVertices, &x, numVertices, &y, numVertices, &z, numVertices*PetscSqr(dim), &metric, numCells*maxConeSize, &cells);CHKERRQ(ierr);

  /* Get cell offsets */
  for (c = 0, coff = 0; c < numCells; ++c) {
    const PetscInt *cone;
    PetscInt        coneSize, cl;

    ierr = DMPlexGetConeSize(udm, c, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(udm, c, &cone);CHKERRQ(ierr);
    for (cl = 0; cl < coneSize; ++cl) cells[coff++] = cone[cl] - vStart;
  }

  /* Get local-to-global vertex map */
  ierr = PetscCalloc1(numVertices, &l2gv);CHKERRQ(ierr);
  ierr = DMPlexGetVertexNumbering(udm, &globalVertexNum);CHKERRQ(ierr);
  ierr = ISGetIndices(globalVertexNum, &gV);CHKERRQ(ierr);
  for (v = 0, numLocVertices = 0; v < numVertices; ++v) {
    if (gV[v] >= 0) ++numLocVertices;
    l2gv[v] = gV[v] < 0 ? -(gV[v]+1) : gV[v];
  }
  ierr = ISRestoreIndices(globalVertexNum, &gV);CHKERRQ(ierr);
  ierr = DMDestroy(&udm);CHKERRQ(ierr);

  /* Get vertex coordinate arrays */
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(cdm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
    x[v-vStart] = PetscRealPart(coords[off+0]);
    if (dim > 1) y[v-vStart] = PetscRealPart(coords[off+1]);
    if (dim > 2) z[v-vStart] = PetscRealPart(coords[off+2]);
  }
  ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);

  /* Get boundary mesh */
  ierr = DMLabelCreate(PETSC_COMM_SELF, bdName, &bdLabelFull);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(dm, 1, bdLabelFull);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(bdLabelFull, 1, &bdIS);CHKERRQ(ierr);
  ierr = DMLabelGetStratumSize(bdLabelFull, 1, &numBdFaces);CHKERRQ(ierr);
  ierr = ISGetIndices(bdIS, &bdFacesFull);CHKERRQ(ierr);
  for (f = 0, bdSize = 0; f < numBdFaces; ++f) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, cl;

    ierr = DMPlexGetTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) ++bdSize;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
  }
  ierr = PetscMalloc2(bdSize, &bdFaces, numBdFaces, &bdFaceIds);CHKERRQ(ierr);
  for (f = 0, bdSize = 0; f < numBdFaces; ++f) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, cl;

    ierr = DMPlexGetTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) bdFaces[bdSize++] = closure[cl] - vStart;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    if (bdLabel) {ierr = DMLabelGetValue(bdLabel, bdFacesFull[f], &bdFaceIds[f]);CHKERRQ(ierr);}
    else         {bdFaceIds[f] = 1;}
  }
  ierr = ISDestroy(&bdIS);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&bdLabelFull);CHKERRQ(ierr);

  /* Get metric */
  ierr = VecViewFromOptions(vertexMetric, NULL, "-adapt_metric_view");CHKERRQ(ierr);
  ierr = VecGetArrayRead(vertexMetric, &met);CHKERRQ(ierr);
  ierr = DMPlexMetricIsIsotropic(dm, &isotropic);CHKERRQ(ierr);
  ierr = DMPlexMetricIsUniform(dm, &uniform);CHKERRQ(ierr);
  Nd = PetscSqr(dim);
  for (v = 0; v < vEnd-vStart; ++v) {
    for (i = 0; i < dim; ++i) {
      for (j = 0; j < dim; ++j) {
        if (isotropic) {
          if (i == j) {
            if (uniform) metric[Nd*v+dim*i+j] = PetscRealPart(met[0]);
            else metric[Nd*v+dim*i+j] = PetscRealPart(met[v]);
          } else metric[Nd*v+dim*i+j] = 0.0;
        } else metric[Nd*v+dim*i+j] = PetscRealPart(met[Nd*v+dim*i+j]);
      }
    }
  }
  ierr = VecRestoreArrayRead(vertexMetric, &met);CHKERRQ(ierr);

#if 0
  /* Destroy overlap mesh */
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
#endif
  /* Send to Pragmatic and remesh */
  switch (dim) {
  case 2:
    pragmatic_2d_mpi_init(&numVertices, &numCells, cells, x, y, l2gv, numLocVertices, comm);
    break;
  case 3:
    pragmatic_3d_mpi_init(&numVertices, &numCells, cells, x, y, z, l2gv, numLocVertices, comm);
    break;
  default: SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Pragmatic adaptation defined for dimension %D", dim);
  }
  pragmatic_set_boundary(&numBdFaces, bdFaces, bdFaceIds);
  pragmatic_set_metric(metric);
  pragmatic_adapt(((DM_Plex *) dm->data)->remeshBd ? 1 : 0);
  ierr = PetscFree(l2gv);CHKERRQ(ierr);

  /* Retrieve mesh from Pragmatic and create new plex */
  pragmatic_get_info_mpi(&numVerticesNew, &numCellsNew);
  ierr = PetscMalloc1(numVerticesNew*dim, &coordsNew);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    numCornersNew = 3;
    ierr = PetscMalloc2(numVerticesNew, &xNew[0], numVerticesNew, &xNew[1]);CHKERRQ(ierr);
    pragmatic_get_coords_2d_mpi(xNew[0], xNew[1]);
    break;
  case 3:
    numCornersNew = 4;
    ierr = PetscMalloc3(numVerticesNew, &xNew[0], numVerticesNew, &xNew[1], numVerticesNew, &xNew[2]);CHKERRQ(ierr);
    pragmatic_get_coords_3d_mpi(xNew[0], xNew[1], xNew[2]);
    break;
  default:
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Pragmatic adaptation defined for dimension %D", dim);
  }
  for (v = 0; v < numVerticesNew; ++v) {for (d = 0; d < dim; ++d) coordsNew[v*dim+d] = xNew[d][v];}
  ierr = PetscMalloc1(numCellsNew*(dim+1), &cellsNew);CHKERRQ(ierr);
  pragmatic_get_elements(cellsNew);
  ierr = DMPlexCreateFromCellListParallelPetsc(comm, dim, numCellsNew, numVerticesNew, PETSC_DECIDE, numCornersNew, PETSC_TRUE, cellsNew, dim, coordsNew, NULL, NULL, dmNew);CHKERRQ(ierr);

  /* Rebuild boundary label */
  pragmatic_get_boundaryTags(&bdTags);
  ierr = DMCreateLabel(*dmNew, bdLabel ? bdLabelName : bdName);CHKERRQ(ierr);
  ierr = DMGetLabel(*dmNew, bdLabel ? bdLabelName : bdName, &bdLabelNew);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dmNew, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dmNew, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(*dmNew, 0, &vStart, &vEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {

    /* Only for simplicial meshes */
    coff = (c-cStart)*(dim+1);

    /* d is the local cell number of the vertex opposite to the face we are marking */
    for (d = 0; d < dim+1; ++d) {
      if (bdTags[coff+d]) {
        const PetscInt  perm[4][4] = {{-1, -1, -1, -1}, {-1, -1, -1, -1}, {1, 2, 0, -1}, {3, 2, 1, 0}}; /* perm[d] = face opposite */
        const PetscInt *cone;

        /* Mark face opposite to this vertex: This pattern is specified in DMPlexGetRawFaces_Internal() */
        ierr = DMPlexGetCone(*dmNew, c, &cone);CHKERRQ(ierr);
        ierr = DMLabelSetValue(bdLabelNew, cone[perm[dim][d]], bdTags[coff+d]);CHKERRQ(ierr);
      }
    }
  }

  /* Clean up */
  switch (dim) {
  case 2: ierr = PetscFree2(xNew[0], xNew[1]);CHKERRQ(ierr);
  break;
  case 3: ierr = PetscFree3(xNew[0], xNew[1], xNew[2]);CHKERRQ(ierr);
  break;
  }
  ierr = PetscFree(cellsNew);CHKERRQ(ierr);
  ierr = PetscFree5(x, y, z, metric, cells);CHKERRQ(ierr);
  ierr = PetscFree2(bdFaces, bdFaceIds);CHKERRQ(ierr);
  ierr = PetscFree(coordsNew);CHKERRQ(ierr);
  pragmatic_finalize();
  PetscFunctionReturn(0);
}
