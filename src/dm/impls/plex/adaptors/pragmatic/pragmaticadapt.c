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

  PetscFunctionBegin;

  /* Check for FEM adjacency flags */
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRMPI(MPI_Comm_size(comm, &numProcs));
  if (bdLabel) {
    CHKERRQ(PetscObjectGetName((PetscObject) bdLabel, &bdLabelName));
    CHKERRQ(PetscStrcmp(bdLabelName, bdName, &flg));
    PetscCheck(!flg,comm, PETSC_ERR_ARG_WRONG, "\"%s\" cannot be used as label for boundary facets", bdLabelName);
  }
  PetscCheck(!rgLabel,comm, PETSC_ERR_ARG_WRONG, "Cannot currently preserve cell tags with Pragmatic");
#if 0
  /* Check for overlap by looking for cell in the SF */
  if (!overlapped) {
    CHKERRQ(DMPlexDistributeOverlap(odm, 1, NULL, &dm));
    if (!dm) {dm = odm; CHKERRQ(PetscObjectReference((PetscObject) dm));}
  }
#endif

  /* Get mesh information */
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(DMPlexUninterpolate(dm, &udm));
  CHKERRQ(DMPlexGetMaxSizes(udm, &maxConeSize, NULL));
  numCells = cEnd - cStart;
  if (numCells == 0) {
    PetscMPIInt rank;

    CHKERRMPI(MPI_Comm_rank(comm, &rank));
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot perform mesh adaptation because process %d does not own any cells.", rank);
  }
  numVertices = vEnd - vStart;
  CHKERRQ(PetscCalloc5(numVertices, &x, numVertices, &y, numVertices, &z, numVertices*PetscSqr(dim), &metric, numCells*maxConeSize, &cells));

  /* Get cell offsets */
  for (c = 0, coff = 0; c < numCells; ++c) {
    const PetscInt *cone;
    PetscInt        coneSize, cl;

    CHKERRQ(DMPlexGetConeSize(udm, c, &coneSize));
    CHKERRQ(DMPlexGetCone(udm, c, &cone));
    for (cl = 0; cl < coneSize; ++cl) cells[coff++] = cone[cl] - vStart;
  }

  /* Get local-to-global vertex map */
  CHKERRQ(PetscCalloc1(numVertices, &l2gv));
  CHKERRQ(DMPlexGetVertexNumbering(udm, &globalVertexNum));
  CHKERRQ(ISGetIndices(globalVertexNum, &gV));
  for (v = 0, numLocVertices = 0; v < numVertices; ++v) {
    if (gV[v] >= 0) ++numLocVertices;
    l2gv[v] = gV[v] < 0 ? -(gV[v]+1) : gV[v];
  }
  CHKERRQ(ISRestoreIndices(globalVertexNum, &gV));
  CHKERRQ(DMDestroy(&udm));

  /* Get vertex coordinate arrays */
  CHKERRQ(DMGetCoordinateDM(dm, &cdm));
  CHKERRQ(DMGetLocalSection(cdm, &coordSection));
  CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
  CHKERRQ(VecGetArrayRead(coordinates, &coords));
  for (v = vStart; v < vEnd; ++v) {
    CHKERRQ(PetscSectionGetOffset(coordSection, v, &off));
    x[v-vStart] = PetscRealPart(coords[off+0]);
    if (dim > 1) y[v-vStart] = PetscRealPart(coords[off+1]);
    if (dim > 2) z[v-vStart] = PetscRealPart(coords[off+2]);
  }
  CHKERRQ(VecRestoreArrayRead(coordinates, &coords));

  /* Get boundary mesh */
  CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, bdName, &bdLabelFull));
  CHKERRQ(DMPlexMarkBoundaryFaces(dm, 1, bdLabelFull));
  CHKERRQ(DMLabelGetStratumIS(bdLabelFull, 1, &bdIS));
  CHKERRQ(DMLabelGetStratumSize(bdLabelFull, 1, &numBdFaces));
  CHKERRQ(ISGetIndices(bdIS, &bdFacesFull));
  for (f = 0, bdSize = 0; f < numBdFaces; ++f) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, cl;

    CHKERRQ(DMPlexGetTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize*2; cl += 2) {
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) ++bdSize;
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure));
  }
  CHKERRQ(PetscMalloc2(bdSize, &bdFaces, numBdFaces, &bdFaceIds));
  for (f = 0, bdSize = 0; f < numBdFaces; ++f) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, cl;

    CHKERRQ(DMPlexGetTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize*2; cl += 2) {
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) bdFaces[bdSize++] = closure[cl] - vStart;
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure));
    if (bdLabel) CHKERRQ(DMLabelGetValue(bdLabel, bdFacesFull[f], &bdFaceIds[f]));
    else         {bdFaceIds[f] = 1;}
  }
  CHKERRQ(ISDestroy(&bdIS));
  CHKERRQ(DMLabelDestroy(&bdLabelFull));

  /* Get metric */
  CHKERRQ(VecViewFromOptions(vertexMetric, NULL, "-adapt_metric_view"));
  CHKERRQ(VecGetArrayRead(vertexMetric, &met));
  CHKERRQ(DMPlexMetricIsIsotropic(dm, &isotropic));
  CHKERRQ(DMPlexMetricIsUniform(dm, &uniform));
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
  CHKERRQ(VecRestoreArrayRead(vertexMetric, &met));

#if 0
  /* Destroy overlap mesh */
  CHKERRQ(DMDestroy(&dm));
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
  CHKERRQ(PetscFree(l2gv));

  /* Retrieve mesh from Pragmatic and create new plex */
  pragmatic_get_info_mpi(&numVerticesNew, &numCellsNew);
  CHKERRQ(PetscMalloc1(numVerticesNew*dim, &coordsNew));
  switch (dim) {
  case 2:
    numCornersNew = 3;
    CHKERRQ(PetscMalloc2(numVerticesNew, &xNew[0], numVerticesNew, &xNew[1]));
    pragmatic_get_coords_2d_mpi(xNew[0], xNew[1]);
    break;
  case 3:
    numCornersNew = 4;
    CHKERRQ(PetscMalloc3(numVerticesNew, &xNew[0], numVerticesNew, &xNew[1], numVerticesNew, &xNew[2]));
    pragmatic_get_coords_3d_mpi(xNew[0], xNew[1], xNew[2]);
    break;
  default:
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Pragmatic adaptation defined for dimension %D", dim);
  }
  for (v = 0; v < numVerticesNew; ++v) {for (d = 0; d < dim; ++d) coordsNew[v*dim+d] = xNew[d][v];}
  CHKERRQ(PetscMalloc1(numCellsNew*(dim+1), &cellsNew));
  pragmatic_get_elements(cellsNew);
  CHKERRQ(DMPlexCreateFromCellListParallelPetsc(comm, dim, numCellsNew, numVerticesNew, PETSC_DECIDE, numCornersNew, PETSC_TRUE, cellsNew, dim, coordsNew, NULL, NULL, dmNew));

  /* Rebuild boundary label */
  pragmatic_get_boundaryTags(&bdTags);
  CHKERRQ(DMCreateLabel(*dmNew, bdLabel ? bdLabelName : bdName));
  CHKERRQ(DMGetLabel(*dmNew, bdLabel ? bdLabelName : bdName, &bdLabelNew));
  CHKERRQ(DMPlexGetHeightStratum(*dmNew, 0, &cStart, &cEnd));
  CHKERRQ(DMPlexGetHeightStratum(*dmNew, 1, &fStart, &fEnd));
  CHKERRQ(DMPlexGetDepthStratum(*dmNew, 0, &vStart, &vEnd));
  for (c = cStart; c < cEnd; ++c) {

    /* Only for simplicial meshes */
    coff = (c-cStart)*(dim+1);

    /* d is the local cell number of the vertex opposite to the face we are marking */
    for (d = 0; d < dim+1; ++d) {
      if (bdTags[coff+d]) {
        const PetscInt  perm[4][4] = {{-1, -1, -1, -1}, {-1, -1, -1, -1}, {1, 2, 0, -1}, {3, 2, 1, 0}}; /* perm[d] = face opposite */
        const PetscInt *cone;

        /* Mark face opposite to this vertex: This pattern is specified in DMPlexGetRawFaces_Internal() */
        CHKERRQ(DMPlexGetCone(*dmNew, c, &cone));
        CHKERRQ(DMLabelSetValue(bdLabelNew, cone[perm[dim][d]], bdTags[coff+d]));
      }
    }
  }

  /* Clean up */
  switch (dim) {
  case 2: CHKERRQ(PetscFree2(xNew[0], xNew[1]));
  break;
  case 3: CHKERRQ(PetscFree3(xNew[0], xNew[1], xNew[2]));
  break;
  }
  CHKERRQ(PetscFree(cellsNew));
  CHKERRQ(PetscFree5(x, y, z, metric, cells));
  CHKERRQ(PetscFree2(bdFaces, bdFaceIds));
  CHKERRQ(PetscFree(coordsNew));
  pragmatic_finalize();
  PetscFunctionReturn(0);
}
