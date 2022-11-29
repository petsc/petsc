#include <petsc/private/dmpleximpl.h> /*I      "petscdmplex.h"   I*/
#include <pragmatic/cpragmatic.h>

PETSC_EXTERN PetscErrorCode DMAdaptMetric_Pragmatic_Plex(DM dm, Vec vertexMetric, DMLabel bdLabel, DMLabel rgLabel, DM *dmNew)
{
  MPI_Comm    comm;
  const char *bdName = "_boundary_";
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
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &numProcs));
  if (bdLabel) {
    PetscCall(PetscObjectGetName((PetscObject)bdLabel, &bdLabelName));
    PetscCall(PetscStrcmp(bdLabelName, bdName, &flg));
    PetscCheck(!flg, comm, PETSC_ERR_ARG_WRONG, "\"%s\" cannot be used as label for boundary facets", bdLabelName);
  }
  PetscCheck(!rgLabel, comm, PETSC_ERR_ARG_WRONG, "Cannot currently preserve cell tags with Pragmatic");
#if 0
  /* Check for overlap by looking for cell in the SF */
  if (!overlapped) {
    PetscCall(DMPlexDistributeOverlap(odm, 1, NULL, &dm));
    if (!dm) {dm = odm; PetscCall(PetscObjectReference((PetscObject) dm));}
  }
#endif

  /* Get mesh information */
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMPlexUninterpolate(dm, &udm));
  PetscCall(DMPlexGetMaxSizes(udm, &maxConeSize, NULL));
  numCells = cEnd - cStart;
  if (numCells == 0) {
    PetscMPIInt rank;

    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot perform mesh adaptation because process %d does not own any cells.", rank);
  }
  numVertices = vEnd - vStart;
  PetscCall(PetscCalloc5(numVertices, &x, numVertices, &y, numVertices, &z, numVertices * PetscSqr(dim), &metric, numCells * maxConeSize, &cells));

  /* Get cell offsets */
  for (c = 0, coff = 0; c < numCells; ++c) {
    const PetscInt *cone;
    PetscInt        coneSize, cl;

    PetscCall(DMPlexGetConeSize(udm, c, &coneSize));
    PetscCall(DMPlexGetCone(udm, c, &cone));
    for (cl = 0; cl < coneSize; ++cl) cells[coff++] = cone[cl] - vStart;
  }

  /* Get local-to-global vertex map */
  PetscCall(PetscCalloc1(numVertices, &l2gv));
  PetscCall(DMPlexGetVertexNumbering(udm, &globalVertexNum));
  PetscCall(ISGetIndices(globalVertexNum, &gV));
  for (v = 0, numLocVertices = 0; v < numVertices; ++v) {
    if (gV[v] >= 0) ++numLocVertices;
    l2gv[v] = gV[v] < 0 ? -(gV[v] + 1) : gV[v];
  }
  PetscCall(ISRestoreIndices(globalVertexNum, &gV));
  PetscCall(DMDestroy(&udm));

  /* Get vertex coordinate arrays */
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetLocalSection(cdm, &coordSection));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(VecGetArrayRead(coordinates, &coords));
  for (v = vStart; v < vEnd; ++v) {
    PetscCall(PetscSectionGetOffset(coordSection, v, &off));
    x[v - vStart] = PetscRealPart(coords[off + 0]);
    if (dim > 1) y[v - vStart] = PetscRealPart(coords[off + 1]);
    if (dim > 2) z[v - vStart] = PetscRealPart(coords[off + 2]);
  }
  PetscCall(VecRestoreArrayRead(coordinates, &coords));

  /* Get boundary mesh */
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, bdName, &bdLabelFull));
  PetscCall(DMPlexMarkBoundaryFaces(dm, 1, bdLabelFull));
  PetscCall(DMLabelGetStratumIS(bdLabelFull, 1, &bdIS));
  PetscCall(DMLabelGetStratumSize(bdLabelFull, 1, &numBdFaces));
  PetscCall(ISGetIndices(bdIS, &bdFacesFull));
  for (f = 0, bdSize = 0; f < numBdFaces; ++f) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, cl;

    PetscCall(DMPlexGetTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize * 2; cl += 2) {
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) ++bdSize;
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure));
  }
  PetscCall(PetscMalloc2(bdSize, &bdFaces, numBdFaces, &bdFaceIds));
  for (f = 0, bdSize = 0; f < numBdFaces; ++f) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, cl;

    PetscCall(DMPlexGetTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize * 2; cl += 2) {
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) bdFaces[bdSize++] = closure[cl] - vStart;
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure));
    if (bdLabel) PetscCall(DMLabelGetValue(bdLabel, bdFacesFull[f], &bdFaceIds[f]));
    else bdFaceIds[f] = 1;
  }
  PetscCall(ISDestroy(&bdIS));
  PetscCall(DMLabelDestroy(&bdLabelFull));

  /* Get metric */
  PetscCall(VecViewFromOptions(vertexMetric, NULL, "-adapt_metric_view"));
  PetscCall(VecGetArrayRead(vertexMetric, &met));
  PetscCall(DMPlexMetricIsIsotropic(dm, &isotropic));
  PetscCall(DMPlexMetricIsUniform(dm, &uniform));
  Nd = PetscSqr(dim);
  for (v = 0; v < vEnd - vStart; ++v) {
    for (i = 0; i < dim; ++i) {
      for (j = 0; j < dim; ++j) {
        if (isotropic) {
          if (i == j) {
            if (uniform) metric[Nd * v + dim * i + j] = PetscRealPart(met[0]);
            else metric[Nd * v + dim * i + j] = PetscRealPart(met[v]);
          } else metric[Nd * v + dim * i + j] = 0.0;
        } else metric[Nd * v + dim * i + j] = PetscRealPart(met[Nd * v + dim * i + j]);
      }
    }
  }
  PetscCall(VecRestoreArrayRead(vertexMetric, &met));

#if 0
  /* Destroy overlap mesh */
  PetscCall(DMDestroy(&dm));
#endif
  /* Send to Pragmatic and remesh */
  switch (dim) {
  case 2:
    pragmatic_2d_mpi_init(&numVertices, &numCells, cells, x, y, l2gv, numLocVertices, comm);
    break;
  case 3:
    pragmatic_3d_mpi_init(&numVertices, &numCells, cells, x, y, z, l2gv, numLocVertices, comm);
    break;
  default:
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Pragmatic adaptation defined for dimension %" PetscInt_FMT, dim);
  }
  pragmatic_set_boundary(&numBdFaces, bdFaces, bdFaceIds);
  pragmatic_set_metric(metric);
  pragmatic_adapt(((DM_Plex *)dm->data)->remeshBd ? 1 : 0);
  PetscCall(PetscFree(l2gv));

  /* Retrieve mesh from Pragmatic and create new plex */
  pragmatic_get_info_mpi(&numVerticesNew, &numCellsNew);
  PetscCall(PetscMalloc1(numVerticesNew * dim, &coordsNew));
  switch (dim) {
  case 2:
    numCornersNew = 3;
    PetscCall(PetscMalloc2(numVerticesNew, &xNew[0], numVerticesNew, &xNew[1]));
    pragmatic_get_coords_2d_mpi(xNew[0], xNew[1]);
    break;
  case 3:
    numCornersNew = 4;
    PetscCall(PetscMalloc3(numVerticesNew, &xNew[0], numVerticesNew, &xNew[1], numVerticesNew, &xNew[2]));
    pragmatic_get_coords_3d_mpi(xNew[0], xNew[1], xNew[2]);
    break;
  default:
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Pragmatic adaptation defined for dimension %" PetscInt_FMT, dim);
  }
  for (v = 0; v < numVerticesNew; ++v) {
    for (d = 0; d < dim; ++d) coordsNew[v * dim + d] = xNew[d][v];
  }
  PetscCall(PetscMalloc1(numCellsNew * (dim + 1), &cellsNew));
  pragmatic_get_elements(cellsNew);
  PetscCall(DMPlexCreateFromCellListParallelPetsc(comm, dim, numCellsNew, numVerticesNew, PETSC_DECIDE, numCornersNew, PETSC_TRUE, cellsNew, dim, coordsNew, NULL, NULL, dmNew));

  /* Rebuild boundary label */
  pragmatic_get_boundaryTags(&bdTags);
  PetscCall(DMCreateLabel(*dmNew, bdLabel ? bdLabelName : bdName));
  PetscCall(DMGetLabel(*dmNew, bdLabel ? bdLabelName : bdName, &bdLabelNew));
  PetscCall(DMPlexGetHeightStratum(*dmNew, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetHeightStratum(*dmNew, 1, &fStart, &fEnd));
  PetscCall(DMPlexGetDepthStratum(*dmNew, 0, &vStart, &vEnd));
  for (c = cStart; c < cEnd; ++c) {
    /* Only for simplicial meshes */
    coff = (c - cStart) * (dim + 1);

    /* d is the local cell number of the vertex opposite to the face we are marking */
    for (d = 0; d < dim + 1; ++d) {
      if (bdTags[coff + d]) {
        const PetscInt perm[4][4] = {
          {-1, -1, -1, -1},
          {-1, -1, -1, -1},
          {1,  2,  0,  -1},
          {3,  2,  1,  0 }
        }; /* perm[d] = face opposite */
        const PetscInt *cone;

        /* Mark face opposite to this vertex: This pattern is specified in DMPlexGetRawFaces_Internal() */
        PetscCall(DMPlexGetCone(*dmNew, c, &cone));
        PetscCall(DMLabelSetValue(bdLabelNew, cone[perm[dim][d]], bdTags[coff + d]));
      }
    }
  }

  /* Clean up */
  switch (dim) {
  case 2:
    PetscCall(PetscFree2(xNew[0], xNew[1]));
    break;
  case 3:
    PetscCall(PetscFree3(xNew[0], xNew[1], xNew[2]));
    break;
  }
  PetscCall(PetscFree(cellsNew));
  PetscCall(PetscFree5(x, y, z, metric, cells));
  PetscCall(PetscFree2(bdFaces, bdFaceIds));
  PetscCall(PetscFree(coordsNew));
  pragmatic_finalize();
  PetscFunctionReturn(0);
}
