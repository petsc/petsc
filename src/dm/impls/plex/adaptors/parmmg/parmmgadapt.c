#include "../mmgcommon.h" /*I      "petscdmplex.h"   I*/
#include <parmmg/libparmmg.h>

PetscBool  ParMmgCite       = PETSC_FALSE;
const char ParMmgCitation[] = "@techreport{cirrottola:hal-02386837,\n"
                              "  title       = {Parallel unstructured mesh adaptation using iterative remeshing and repartitioning},\n"
                              "  institution = {Inria Bordeaux},\n"
                              "  author      = {L. Cirrottola and A. Froehly},\n"
                              "  number      = {9307},\n"
                              "  note        = {\\url{https://hal.inria.fr/hal-02386837}},\n"
                              "  year        = {2019}\n}\n";

PETSC_EXTERN PetscErrorCode DMAdaptMetric_ParMmg_Plex(DM dm, Vec vertexMetric, DMLabel bdLabel, DMLabel rgLabel, DM *dmNew)
{
  MPI_Comm           comm;
  const char        *bdName = "_boundary_";
  const char        *rgName = "_regions_";
  DM                 udm, cdm;
  DMLabel            bdLabelNew, rgLabelNew;
  const char        *bdLabelName, *rgLabelName;
  IS                 globalVertexNum;
  PetscSection       coordSection;
  Vec                coordinates;
  PetscSF            sf;
  const PetscScalar *coords, *met;
  PetscReal         *vertices, *metric, *verticesNew, *verticesNewLoc, gradationFactor, hausdorffNumber;
  PetscInt          *cells, *cellsNew, *cellTags, *cellTagsNew, *verTags, *verTagsNew;
  PetscInt          *bdFaces, *faceTags, *facesNew, *faceTagsNew;
  PetscInt          *corners, *requiredCells, *requiredVer, *ridges, *requiredFaces;
  PetscInt           cStart, cEnd, c, numCells, fStart, fEnd, f, numFaceTags, vStart, vEnd, v, numVertices;
  PetscInt           dim, off, coff, maxConeSize, bdSize, i, j, k, Neq, verbosity, numIter;
  PetscInt          *numVerInterfaces, *ngbRanks, *verNgbRank, *interfaces_lv, *interfaces_gv, *intOffset;
  PetscInt           niranks, nrranks, numNgbRanks, numVerNgbRanksTotal, count, sliceSize, p, r, n, lv, gv;
  PetscInt          *gv_new, *owners, *verticesNewSorted, pStart, pEnd;
  PetscInt           numCellsNew, numVerticesNew, numCornersNew, numFacesNew, numVerticesNewLoc;
  const PetscInt    *gV, *ioffset, *irootloc, *roffset, *rmine, *rremote;
  PetscBool          flg = PETSC_FALSE, noInsert, noSwap, noMove, noSurf, isotropic, uniform;
  const PetscMPIInt *iranks, *rranks;
  PetscMPIInt        numProcs, rank;
  PMMG_pParMesh      parmesh = NULL;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(ParMmgCitation, &ParMmgCite));
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &numProcs));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (bdLabel) {
    PetscCall(PetscObjectGetName((PetscObject)bdLabel, &bdLabelName));
    PetscCall(PetscStrcmp(bdLabelName, bdName, &flg));
    PetscCheck(!flg, comm, PETSC_ERR_ARG_WRONG, "\"%s\" cannot be used as label for boundary facets", bdLabelName);
  }
  if (rgLabel) {
    PetscCall(PetscObjectGetName((PetscObject)rgLabel, &rgLabelName));
    PetscCall(PetscStrcmp(rgLabelName, rgName, &flg));
    PetscCheck(!flg, comm, PETSC_ERR_ARG_WRONG, "\"%s\" cannot be used as label for element tags", rgLabelName);
  }

  /* Get mesh information */
  PetscCall(DMGetDimension(dm, &dim));
  PetscCheck(dim == 3, comm, PETSC_ERR_ARG_OUTOFRANGE, "ParMmg only works in 3D.");
  Neq = (dim * (dim + 1)) / 2;
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMPlexUninterpolate(dm, &udm));
  PetscCall(DMPlexGetMaxSizes(udm, &maxConeSize, NULL));
  numCells    = cEnd - cStart;
  numVertices = vEnd - vStart;

  /* Get cell offsets */
  PetscCall(PetscMalloc1(numCells * maxConeSize, &cells));
  for (c = 0, coff = 0; c < numCells; ++c) {
    const PetscInt *cone;
    PetscInt        coneSize, cl;

    PetscCall(DMPlexGetConeSize(udm, c, &coneSize));
    PetscCall(DMPlexGetCone(udm, c, &cone));
    for (cl = 0; cl < coneSize; ++cl) cells[coff++] = cone[cl] - vStart + 1;
  }

  /* Get vertex coordinate array */
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetLocalSection(cdm, &coordSection));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(VecGetArrayRead(coordinates, &coords));
  PetscCall(PetscMalloc2(numVertices * Neq, &metric, dim * numVertices, &vertices));
  for (v = 0; v < vEnd - vStart; ++v) {
    PetscCall(PetscSectionGetOffset(coordSection, v + vStart, &off));
    for (i = 0; i < dim; ++i) vertices[dim * v + i] = PetscRealPart(coords[off + i]);
  }
  PetscCall(VecRestoreArrayRead(coordinates, &coords));

  /* Get face tags */
  if (!bdLabel) {
    flg = PETSC_TRUE;
    PetscCall(DMLabelCreate(PETSC_COMM_SELF, bdName, &bdLabel));
    PetscCall(DMPlexMarkBoundaryFaces(dm, 1, bdLabel));
  }
  PetscCall(DMLabelGetBounds(bdLabel, &pStart, &pEnd));
  for (f = pStart, bdSize = 0, numFaceTags = 0; f < pEnd; ++f) {
    PetscBool hasPoint;
    PetscInt *closure = NULL, closureSize, cl;

    PetscCall(DMLabelHasPoint(bdLabel, f, &hasPoint));
    if ((!hasPoint) || (f < fStart) || (f >= fEnd)) continue;
    numFaceTags++;

    PetscCall(DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize * 2; cl += 2) {
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) ++bdSize;
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure));
  }
  PetscCall(PetscMalloc2(bdSize, &bdFaces, numFaceTags, &faceTags));
  for (f = pStart, bdSize = 0, numFaceTags = 0; f < pEnd; ++f) {
    PetscBool hasPoint;
    PetscInt *closure = NULL, closureSize, cl;

    PetscCall(DMLabelHasPoint(bdLabel, f, &hasPoint));
    if ((!hasPoint) || (f < fStart) || (f >= fEnd)) continue;

    PetscCall(DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize * 2; cl += 2) {
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) bdFaces[bdSize++] = closure[cl] - vStart + 1;
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure));
    PetscCall(DMLabelGetValue(bdLabel, f, &faceTags[numFaceTags++]));
  }

  /* Get cell tags */
  PetscCall(PetscCalloc2(numVertices, &verTags, numCells, &cellTags));
  if (rgLabel) {
    for (c = cStart; c < cEnd; ++c) PetscCall(DMLabelGetValue(rgLabel, c, &cellTags[c]));
  }

  /* Get metric */
  PetscCall(VecViewFromOptions(vertexMetric, NULL, "-adapt_metric_view"));
  PetscCall(VecGetArrayRead(vertexMetric, &met));
  PetscCall(DMPlexMetricIsIsotropic(dm, &isotropic));
  PetscCall(DMPlexMetricIsUniform(dm, &uniform));
  for (v = 0; v < (vEnd - vStart); ++v) {
    for (i = 0, k = 0; i < dim; ++i) {
      for (j = i; j < dim; ++j, ++k) {
        if (isotropic) {
          if (i == j) {
            if (uniform) metric[Neq * v + k] = PetscRealPart(met[0]);
            else metric[Neq * v + k] = PetscRealPart(met[v]);
          } else metric[Neq * v + k] = 0.0;
        } else metric[Neq * v + k] = PetscRealPart(met[dim * dim * v + dim * i + j]);
      }
    }
  }
  PetscCall(VecRestoreArrayRead(vertexMetric, &met));

  /* Build ParMMG communicators: the list of vertices between two partitions  */
  niranks = nrranks = 0;
  numNgbRanks       = 0;
  if (numProcs > 1) {
    PetscCall(DMGetPointSF(dm, &sf));
    PetscCall(PetscSFSetUp(sf));
    PetscCall(PetscSFGetLeafRanks(sf, &niranks, &iranks, &ioffset, &irootloc));
    PetscCall(PetscSFGetRootRanks(sf, &nrranks, &rranks, &roffset, &rmine, &rremote));
    PetscCall(PetscCalloc1(numProcs, &numVerInterfaces));

    /* Count number of roots associated with each leaf */
    for (r = 0; r < niranks; ++r) {
      for (i = ioffset[r], count = 0; i < ioffset[r + 1]; ++i) {
        if (irootloc[i] >= vStart && irootloc[i] < vEnd) count++;
      }
      numVerInterfaces[iranks[r]] += count;
    }

    /* Count number of leaves associated with each root */
    for (r = 0; r < nrranks; ++r) {
      for (i = roffset[r], count = 0; i < roffset[r + 1]; ++i) {
        if (rmine[i] >= vStart && rmine[i] < vEnd) count++;
      }
      numVerInterfaces[rranks[r]] += count;
    }

    /* Count global number of ranks */
    for (p = 0; p < numProcs; ++p) {
      if (numVerInterfaces[p]) numNgbRanks++;
    }

    /* Provide numbers of vertex interfaces */
    PetscCall(PetscMalloc2(numNgbRanks, &ngbRanks, numNgbRanks, &verNgbRank));
    for (p = 0, n = 0; p < numProcs; ++p) {
      if (numVerInterfaces[p]) {
        ngbRanks[n]   = p;
        verNgbRank[n] = numVerInterfaces[p];
        n++;
      }
    }
    numVerNgbRanksTotal = 0;
    for (p = 0; p < numNgbRanks; ++p) numVerNgbRanksTotal += verNgbRank[p];

    /* For each neighbor, fill in interface arrays */
    PetscCall(PetscMalloc3(numVerNgbRanksTotal, &interfaces_lv, numVerNgbRanksTotal, &interfaces_gv, numNgbRanks + 1, &intOffset));
    intOffset[0] = 0;
    for (p = 0, r = 0, i = 0; p < numNgbRanks; ++p) {
      intOffset[p + 1] = intOffset[p];

      /* Leaf case */
      if (iranks && iranks[i] == ngbRanks[p]) {
        /* Add the right slice of irootloc at the right place */
        sliceSize = ioffset[i + 1] - ioffset[i];
        for (j = 0, count = 0; j < sliceSize; ++j) {
          PetscCheck(ioffset[i] + j < ioffset[niranks], comm, PETSC_ERR_ARG_OUTOFRANGE, "Leaf index %" PetscInt_FMT " out of range (expected < %" PetscInt_FMT ")", ioffset[i] + j, ioffset[niranks]);
          v = irootloc[ioffset[i] + j];
          if (v >= vStart && v < vEnd) {
            PetscCheck(intOffset[p + 1] + count < numVerNgbRanksTotal, comm, PETSC_ERR_ARG_OUTOFRANGE, "Leaf interface index %" PetscInt_FMT " out of range (expected < %" PetscInt_FMT ")", intOffset[p + 1] + count, numVerNgbRanksTotal);
            interfaces_lv[intOffset[p + 1] + count] = v - vStart;
            count++;
          }
        }
        intOffset[p + 1] += count;
        i++;
      }

      /* Root case */
      if (rranks && rranks[r] == ngbRanks[p]) {
        /* Add the right slice of rmine at the right place */
        sliceSize = roffset[r + 1] - roffset[r];
        for (j = 0, count = 0; j < sliceSize; ++j) {
          PetscCheck(roffset[r] + j < roffset[nrranks], comm, PETSC_ERR_ARG_OUTOFRANGE, "Root index %" PetscInt_FMT " out of range (expected < %" PetscInt_FMT ")", roffset[r] + j, roffset[nrranks]);
          v = rmine[roffset[r] + j];
          if (v >= vStart && v < vEnd) {
            PetscCheck(intOffset[p + 1] + count < numVerNgbRanksTotal, comm, PETSC_ERR_ARG_OUTOFRANGE, "Root interface index %" PetscInt_FMT " out of range (expected < %" PetscInt_FMT ")", intOffset[p + 1] + count, numVerNgbRanksTotal);
            interfaces_lv[intOffset[p + 1] + count] = v - vStart;
            count++;
          }
        }
        intOffset[p + 1] += count;
        r++;
      }

      /* Check validity of offsets */
      PetscCheck(intOffset[p + 1] == intOffset[p] + verNgbRank[p], comm, PETSC_ERR_ARG_OUTOFRANGE, "Missing offsets (expected %" PetscInt_FMT ", got %" PetscInt_FMT ")", intOffset[p] + verNgbRank[p], intOffset[p + 1]);
    }
    PetscCall(DMPlexGetVertexNumbering(udm, &globalVertexNum));
    PetscCall(ISGetIndices(globalVertexNum, &gV));
    for (i = 0; i < numVerNgbRanksTotal; ++i) {
      v                = gV[interfaces_lv[i]];
      interfaces_gv[i] = v < 0 ? -v - 1 : v;
      interfaces_lv[i] += 1;
      interfaces_gv[i] += 1;
    }
    PetscCall(ISRestoreIndices(globalVertexNum, &gV));
    PetscCall(PetscFree(numVerInterfaces));
  }
  PetscCall(DMDestroy(&udm));

  /* Send the data to ParMmg and remesh */
  PetscCall(DMPlexMetricNoInsertion(dm, &noInsert));
  PetscCall(DMPlexMetricNoSwapping(dm, &noSwap));
  PetscCall(DMPlexMetricNoMovement(dm, &noMove));
  PetscCall(DMPlexMetricNoSurf(dm, &noSurf));
  PetscCall(DMPlexMetricGetVerbosity(dm, &verbosity));
  PetscCall(DMPlexMetricGetNumIterations(dm, &numIter));
  PetscCall(DMPlexMetricGetGradationFactor(dm, &gradationFactor));
  PetscCall(DMPlexMetricGetHausdorffNumber(dm, &hausdorffNumber));
  PetscCallMMG_NONSTANDARD(PMMG_Init_parMesh(PMMG_ARG_start, PMMG_ARG_ppParMesh, &parmesh, PMMG_ARG_pMesh, PMMG_ARG_pMet, PMMG_ARG_dim, 3, PMMG_ARG_MPIComm, comm, PMMG_ARG_end));
  PetscCallMMG_NONSTANDARD(PMMG_Set_meshSize(parmesh, numVertices, numCells, 0, numFaceTags, 0, 0));
  PetscCallMMG_NONSTANDARD(PMMG_Set_iparameter(parmesh, PMMG_IPARAM_APImode, PMMG_APIDISTRIB_nodes));
  PetscCallMMG_NONSTANDARD(PMMG_Set_iparameter(parmesh, PMMG_IPARAM_noinsert, noInsert));
  PetscCallMMG_NONSTANDARD(PMMG_Set_iparameter(parmesh, PMMG_IPARAM_noswap, noSwap));
  PetscCallMMG_NONSTANDARD(PMMG_Set_iparameter(parmesh, PMMG_IPARAM_nomove, noMove));
  PetscCallMMG_NONSTANDARD(PMMG_Set_iparameter(parmesh, PMMG_IPARAM_nosurf, noSurf));
  PetscCallMMG_NONSTANDARD(PMMG_Set_iparameter(parmesh, PMMG_IPARAM_verbose, verbosity));
  PetscCallMMG_NONSTANDARD(PMMG_Set_iparameter(parmesh, PMMG_IPARAM_globalNum, 1));
  PetscCallMMG_NONSTANDARD(PMMG_Set_iparameter(parmesh, PMMG_IPARAM_niter, numIter));
  PetscCallMMG_NONSTANDARD(PMMG_Set_dparameter(parmesh, PMMG_DPARAM_hgrad, gradationFactor));
  PetscCallMMG_NONSTANDARD(PMMG_Set_dparameter(parmesh, PMMG_DPARAM_hausd, hausdorffNumber));
  PetscCallMMG_NONSTANDARD(PMMG_Set_vertices(parmesh, vertices, verTags));
  PetscCallMMG_NONSTANDARD(PMMG_Set_tetrahedra(parmesh, cells, cellTags));
  PetscCallMMG_NONSTANDARD(PMMG_Set_triangles(parmesh, bdFaces, faceTags));
  PetscCallMMG_NONSTANDARD(PMMG_Set_metSize(parmesh, MMG5_Vertex, numVertices, MMG5_Tensor));
  PetscCallMMG_NONSTANDARD(PMMG_Set_tensorMets(parmesh, metric));
  PetscCallMMG_NONSTANDARD(PMMG_Set_numberOfNodeCommunicators(parmesh, numNgbRanks));
  for (c = 0; c < numNgbRanks; ++c) {
    PetscCallMMG_NONSTANDARD(PMMG_Set_ithNodeCommunicatorSize(parmesh, c, ngbRanks[c], intOffset[c + 1] - intOffset[c]));
    PetscCallMMG_NONSTANDARD(PMMG_Set_ithNodeCommunicator_nodes(parmesh, c, &interfaces_lv[intOffset[c]], &interfaces_gv[intOffset[c]], 1));
  }
  PetscCallMMG(PMMG_parmmglib_distributed(parmesh));
  PetscCall(PetscFree(cells));
  PetscCall(PetscFree2(metric, vertices));
  PetscCall(PetscFree2(bdFaces, faceTags));
  PetscCall(PetscFree2(verTags, cellTags));
  if (numProcs > 1) {
    PetscCall(PetscFree2(ngbRanks, verNgbRank));
    PetscCall(PetscFree3(interfaces_lv, interfaces_gv, intOffset));
  }

  /* Retrieve mesh from Mmg */
  numCornersNew = 4;
  PetscCallMMG_NONSTANDARD(PMMG_Get_meshSize(parmesh, &numVerticesNew, &numCellsNew, 0, &numFacesNew, 0, 0));
  PetscCall(PetscMalloc4(dim * numVerticesNew, &verticesNew, numVerticesNew, &verTagsNew, numVerticesNew, &corners, numVerticesNew, &requiredVer));
  PetscCall(PetscMalloc3((dim + 1) * numCellsNew, &cellsNew, numCellsNew, &cellTagsNew, numCellsNew, &requiredCells));
  PetscCall(PetscMalloc4(dim * numFacesNew, &facesNew, numFacesNew, &faceTagsNew, numFacesNew, &ridges, numFacesNew, &requiredFaces));
  PetscCallMMG_NONSTANDARD(PMMG_Get_vertices(parmesh, verticesNew, verTagsNew, corners, requiredVer));
  PetscCallMMG_NONSTANDARD(PMMG_Get_tetrahedra(parmesh, cellsNew, cellTagsNew, requiredCells));
  PetscCallMMG_NONSTANDARD(PMMG_Get_triangles(parmesh, facesNew, faceTagsNew, requiredFaces));
  PetscCall(PetscMalloc2(numVerticesNew, &owners, numVerticesNew, &gv_new));
  PetscCallMMG_NONSTANDARD(PMMG_Set_iparameter(parmesh, PMMG_IPARAM_globalNum, 1));
  PetscCallMMG_NONSTANDARD(PMMG_Get_verticesGloNum(parmesh, gv_new, owners));
  for (i = 0; i < dim * numFacesNew; ++i) facesNew[i] -= 1;
  for (i = 0; i < (dim + 1) * numCellsNew; ++i) cellsNew[i] = gv_new[cellsNew[i] - 1] - 1;
  for (i = 0, numVerticesNewLoc = 0; i < numVerticesNew; ++i) {
    if (owners[i] == rank) numVerticesNewLoc++;
  }
  PetscCall(PetscMalloc2(numVerticesNewLoc * dim, &verticesNewLoc, numVerticesNew, &verticesNewSorted));
  for (i = 0, c = 0; i < numVerticesNew; i++) {
    if (owners[i] == rank) {
      for (j = 0; j < dim; ++j) verticesNewLoc[dim * c + j] = verticesNew[dim * i + j];
      c++;
    }
  }

  /* Reorder for consistency with DMPlex */
  for (i = 0; i < numCellsNew; ++i) PetscCall(DMPlexInvertCell(DM_POLYTOPE_TETRAHEDRON, &cellsNew[4 * i]));

  /* Create new plex */
  PetscCall(DMPlexCreateFromCellListParallelPetsc(comm, dim, numCellsNew, numVerticesNewLoc, PETSC_DECIDE, numCornersNew, PETSC_TRUE, cellsNew, dim, verticesNewLoc, NULL, &verticesNewSorted, dmNew));
  PetscCallMMG_NONSTANDARD(PMMG_Free_all(PMMG_ARG_start, PMMG_ARG_ppParMesh, &parmesh, PMMG_ARG_end));
  PetscCall(PetscFree4(verticesNew, verTagsNew, corners, requiredVer));

  /* Get adapted mesh information */
  PetscCall(DMPlexGetHeightStratum(*dmNew, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetHeightStratum(*dmNew, 1, &fStart, &fEnd));
  PetscCall(DMPlexGetDepthStratum(*dmNew, 0, &vStart, &vEnd));

  /* Rebuild boundary label */
  PetscCall(DMCreateLabel(*dmNew, flg ? bdName : bdLabelName));
  PetscCall(DMGetLabel(*dmNew, flg ? bdName : bdLabelName, &bdLabelNew));
  for (i = 0; i < numFacesNew; i++) {
    PetscBool       hasTag = PETSC_FALSE;
    PetscInt        numCoveredPoints, numFaces = 0, facePoints[3];
    const PetscInt *coveredPoints = NULL;

    for (j = 0; j < dim; ++j) {
      lv = facesNew[i * dim + j];
      gv = gv_new[lv] - 1;
      PetscCall(PetscFindInt(gv, numVerticesNew, verticesNewSorted, &lv));
      facePoints[j] = lv + vStart;
    }
    PetscCall(DMPlexGetFullJoin(*dmNew, dim, facePoints, &numCoveredPoints, &coveredPoints));
    for (j = 0; j < numCoveredPoints; ++j) {
      if (coveredPoints[j] >= fStart && coveredPoints[j] < fEnd) {
        numFaces++;
        f = j;
      }
    }
    PetscCheck(numFaces == 1, comm, PETSC_ERR_ARG_OUTOFRANGE, "%" PetscInt_FMT " vertices cannot define more than 1 facet (%" PetscInt_FMT ")", dim, numFaces);
    PetscCall(DMLabelHasStratum(bdLabel, faceTagsNew[i], &hasTag));
    if (hasTag) PetscCall(DMLabelSetValue(bdLabelNew, coveredPoints[f], faceTagsNew[i]));
    PetscCall(DMPlexRestoreJoin(*dmNew, dim, facePoints, &numCoveredPoints, &coveredPoints));
  }
  PetscCall(PetscFree4(facesNew, faceTagsNew, ridges, requiredFaces));
  PetscCall(PetscFree2(owners, gv_new));
  PetscCall(PetscFree2(verticesNewLoc, verticesNewSorted));
  if (flg) PetscCall(DMLabelDestroy(&bdLabel));

  /* Rebuild cell labels */
  PetscCall(DMCreateLabel(*dmNew, rgLabel ? rgLabelName : rgName));
  PetscCall(DMGetLabel(*dmNew, rgLabel ? rgLabelName : rgName, &rgLabelNew));
  for (c = cStart; c < cEnd; ++c) PetscCall(DMLabelSetValue(rgLabelNew, c, cellTagsNew[c - cStart]));
  PetscCall(PetscFree3(cellsNew, cellTagsNew, requiredCells));

  PetscFunctionReturn(PETSC_SUCCESS);
}
