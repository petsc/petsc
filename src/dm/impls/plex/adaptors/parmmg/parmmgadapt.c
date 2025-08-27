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

/*
 Coupling code for the ParMmg metric-based mesh adaptation package.
*/
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
  PetscInt           numCellsNotShared, *cIsLeaf, numUsedVertices, *vertexNumber, *fIsIncluded;
  PetscInt           dim, off, coff, maxConeSize, bdSize, i, j, k, Neq, verbosity, numIter;
  PetscInt          *interfaces_lv, *interfaces_gv, *interfacesOffset;
  PetscMPIInt        niranks, nrranks, numNgbRanks;
  PetscInt           r, lv, gv;
  PetscInt          *gv_new, *owners, *verticesNewSorted, pStart, pEnd;
  PetscInt           numCellsNew, numVerticesNew, numCornersNew, numFacesNew, numVerticesNewLoc;
  const PetscInt    *gV, *ioffset, *irootloc, *roffset, *rmine, *rremote;
  PetscBool          flg = PETSC_FALSE, noInsert, noSwap, noMove, noSurf, isotropic, uniform;
  const PetscMPIInt *iranks, *rranks;
  PetscMPIInt        numProcs, rank;
  PMMG_pParMesh      parmesh = NULL;

  // DEVELOPER NOTE: ParMmg wants to know the rank of every process which is sharing a given point and
  //                 for this information to be conveyed to every process that is sharing that point.

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

  /* Get parallel data; work out which cells are owned and which are leaves */
  PetscCall(PetscCalloc1(numCells, &cIsLeaf));
  numCellsNotShared = numCells;
  niranks = nrranks = 0;
  if (numProcs > 1) {
    PetscCall(DMGetPointSF(dm, &sf));
    PetscCall(PetscSFSetUp(sf));
    PetscCall(PetscSFGetLeafRanks(sf, &niranks, &iranks, &ioffset, &irootloc));
    PetscCall(PetscSFGetRootRanks(sf, &nrranks, &rranks, &roffset, &rmine, &rremote));
    for (r = 0; r < nrranks; ++r) {
      for (i = roffset[r]; i < roffset[r + 1]; ++i) {
        if (rmine[i] >= cStart && rmine[i] < cEnd) {
          cIsLeaf[rmine[i] - cStart] = 1;
          numCellsNotShared--;
        }
      }
    }
  }

  /*
    Create a vertex numbering for ParMmg starting at 1. Vertices not included in any
    owned cell remain 0 and will be removed. Using this numbering, create cells.
  */
  numUsedVertices = 0;
  PetscCall(PetscCalloc1(numVertices, &vertexNumber));
  PetscCall(PetscMalloc1(numCellsNotShared * maxConeSize, &cells));
  for (c = 0, coff = 0; c < numCells; ++c) {
    const PetscInt *cone;
    PetscInt        coneSize, cl;

    if (!cIsLeaf[c]) {
      PetscCall(DMPlexGetConeSize(udm, cStart + c, &coneSize));
      PetscCall(DMPlexGetCone(udm, cStart + c, &cone));
      for (cl = 0; cl < coneSize; ++cl) {
        if (!vertexNumber[cone[cl] - vStart]) vertexNumber[cone[cl] - vStart] = ++numUsedVertices;
        cells[coff++] = vertexNumber[cone[cl] - vStart];
      }
    }
  }

  /* Get array of vertex coordinates */
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetLocalSection(cdm, &coordSection));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(VecGetArrayRead(coordinates, &coords));
  PetscCall(PetscMalloc2(numUsedVertices * Neq, &metric, dim * numUsedVertices, &vertices));
  for (v = 0; v < vEnd - vStart; ++v) {
    PetscCall(PetscSectionGetOffset(coordSection, v + vStart, &off));
    if (vertexNumber[v]) {
      for (i = 0; i < dim; ++i) vertices[dim * (vertexNumber[v] - 1) + i] = PetscRealPart(coords[off + i]);
    }
  }
  PetscCall(VecRestoreArrayRead(coordinates, &coords));

  /* Get face tags */
  if (!bdLabel) {
    flg = PETSC_TRUE;
    PetscCall(DMLabelCreate(PETSC_COMM_SELF, bdName, &bdLabel));
    PetscCall(DMPlexMarkBoundaryFaces(dm, 1, bdLabel));
  }
  PetscCall(DMLabelGetBounds(bdLabel, &pStart, &pEnd));
  PetscCall(PetscCalloc1(pEnd - pStart, &fIsIncluded));
  for (f = pStart, bdSize = 0, numFaceTags = 0; f < pEnd; ++f) {
    PetscBool hasPoint;
    PetscInt *closure = NULL, closureSize, cl;

    PetscCall(DMLabelHasPoint(bdLabel, f, &hasPoint));
    if ((!hasPoint) || (f < fStart) || (f >= fEnd)) continue;

    /* Only faces adjacent to an owned (non-leaf) cell are included */
    PetscInt        nnbrs;
    const PetscInt *nbrs;
    PetscCall(DMPlexGetSupportSize(dm, f, &nnbrs));
    PetscCall(DMPlexGetSupport(dm, f, &nbrs));
    for (c = 0; c < nnbrs; ++c) fIsIncluded[f - pStart] = fIsIncluded[f - pStart] || !cIsLeaf[nbrs[c]];
    if (!fIsIncluded[f - pStart]) continue;

    numFaceTags++;

    PetscCall(DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize * 2; cl += 2) {
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) ++bdSize;
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure));
  }
  PetscCall(PetscMalloc2(bdSize, &bdFaces, numFaceTags, &faceTags));
  for (f = pStart, bdSize = 0, numFaceTags = 0; f < pEnd; ++f) {
    PetscInt *closure = NULL, closureSize, cl;

    if (!fIsIncluded[f - pStart]) continue;

    PetscCall(DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize * 2; cl += 2) {
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) bdFaces[bdSize++] = vertexNumber[closure[cl] - vStart];
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure));
    PetscCall(DMLabelGetValue(bdLabel, f, &faceTags[numFaceTags++]));
  }
  PetscCall(PetscFree(fIsIncluded));

  /* Get cell tags */
  PetscCall(PetscCalloc2(numUsedVertices, &verTags, numCellsNotShared, &cellTags));
  if (rgLabel) {
    for (c = cStart, coff = 0; c < cEnd; ++c) {
      if (!cIsLeaf[c - cStart]) PetscCall(DMLabelGetValue(rgLabel, c, &cellTags[coff++]));
    }
  }
  PetscCall(PetscFree(cIsLeaf));

  /* Get metric, using only the upper triangular part */
  PetscCall(VecViewFromOptions(vertexMetric, NULL, "-adapt_metric_view"));
  PetscCall(VecGetArrayRead(vertexMetric, &met));
  PetscCall(DMPlexMetricIsIsotropic(dm, &isotropic));
  PetscCall(DMPlexMetricIsUniform(dm, &uniform));
  for (v = 0; v < (vEnd - vStart); ++v) {
    PetscInt vv = vertexNumber[v];
    if (!vv--) continue;
    for (i = 0, k = 0; i < dim; ++i) {
      for (j = i; j < dim; ++j, ++k) {
        if (isotropic) {
          if (i == j) {
            if (uniform) metric[Neq * vv + k] = PetscRealPart(met[0]);
            else metric[Neq * vv + k] = PetscRealPart(met[v]);
          } else metric[Neq * vv + k] = 0.0;
        } else metric[Neq * vv + k] = PetscRealPart(met[dim * dim * v + dim * i + j]);
      }
    }
  }
  PetscCall(VecRestoreArrayRead(vertexMetric, &met));

  /* Build ParMmg communicators: the list of vertices between two partitions  */
  numNgbRanks = 0;
  if (numProcs > 1) {
    DM              rankdm;
    PetscSection    rankSection, rankGlobalSection;
    PetscSF         rankSF;
    const PetscInt *degree;
    PetscInt       *rankOfUsedVertices, *rankOfUsedMultiRootLeaves, *usedCopies;
    PetscInt       *rankArray, *rankGlobalArray, *interfacesPerRank;
    PetscInt        offset, mrl, rootDegreeCnt, s, shareCnt, gv;

    PetscCall(PetscSFComputeDegreeBegin(sf, &degree));
    PetscCall(PetscSFComputeDegreeEnd(sf, &degree));
    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
    for (i = 0, rootDegreeCnt = 0; i < pEnd - pStart; ++i) rootDegreeCnt += degree[i];

    /* rankOfUsedVertices, point-array: rank+1 if vertex and in use */
    PetscCall(PetscCalloc1(pEnd - pStart, &rankOfUsedVertices));
    for (i = 0; i < pEnd - pStart; ++i) rankOfUsedVertices[i] = -1;
    for (i = vStart; i < vEnd; ++i) {
      if (vertexNumber[i - vStart]) rankOfUsedVertices[i] = rank;
    }

    /* rankOfUsedMultiRootLeaves, multiroot-array: rank if vertex and in use, else -1 */
    PetscCall(PetscMalloc1(rootDegreeCnt, &rankOfUsedMultiRootLeaves));
    PetscCall(PetscSFGatherBegin(sf, MPIU_INT, rankOfUsedVertices, rankOfUsedMultiRootLeaves));
    PetscCall(PetscSFGatherEnd(sf, MPIU_INT, rankOfUsedVertices, rankOfUsedMultiRootLeaves));
    PetscCall(PetscFree(rankOfUsedVertices));

    /* usedCopies, point-array: if vertex, shared by how many processes */
    PetscCall(PetscCalloc1(pEnd - pStart, &usedCopies));
    for (i = 0, mrl = 0; i < vStart - pStart; i++) mrl += degree[i];
    for (i = vStart - pStart; i < vEnd - pStart; ++i) {
      for (j = 0; j < degree[i]; j++, mrl++) {
        if (rankOfUsedMultiRootLeaves[mrl] != -1) usedCopies[i]++;
      }
      if (vertexNumber[i - vStart + pStart]) usedCopies[i]++;
    }
    PetscCall(PetscSFBcastBegin(sf, MPIU_INT, usedCopies, usedCopies, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_INT, usedCopies, usedCopies, MPI_REPLACE));

    /* Create a section to store ranks of vertices shared by more than one process */
    PetscCall(PetscSectionCreate(comm, &rankSection));
    PetscCall(PetscSectionSetNumFields(rankSection, 1));
    PetscCall(PetscSectionSetChart(rankSection, pStart, pEnd));
    for (i = vStart - pStart; i < vEnd - pStart; ++i) {
      if (usedCopies[i] > 1) PetscCall(PetscSectionSetDof(rankSection, i + pStart, usedCopies[i]));
    }
    PetscCall(PetscSectionSetUp(rankSection));
    PetscCall(PetscSectionCreateGlobalSection(rankSection, sf, PETSC_FALSE, PETSC_FALSE, PETSC_TRUE, &rankGlobalSection));

    PetscCall(PetscSectionGetStorageSize(rankGlobalSection, &s));
    PetscCall(PetscMalloc1(s, &rankGlobalArray));
    for (i = 0, mrl = 0; i < vStart - pStart; i++) mrl += degree[i];
    for (i = vStart - pStart, k = 0; i < vEnd - pStart; ++i) {
      if (usedCopies[i] > 1 && degree[i]) {
        PetscCall(PetscSectionGetOffset(rankSection, k, &offset));
        if (vertexNumber[i - vStart + pStart]) rankGlobalArray[k++] = rank;
        for (j = 0; j < degree[i]; j++, mrl++) {
          if (rankOfUsedMultiRootLeaves[mrl] != -1) rankGlobalArray[k++] = rankOfUsedMultiRootLeaves[mrl];
        }
      } else mrl += degree[i];
    }
    PetscCall(PetscFree(rankOfUsedMultiRootLeaves));
    PetscCall(PetscFree(usedCopies));
    PetscCall(PetscSectionDestroy(&rankGlobalSection));

    /*
      Broadcast the array of ranks.
        (We want all processes to know all the ranks that are looking at each point.
        Above, we tell the roots. Here, the roots tell the leaves.)
    */
    PetscCall(DMClone(dm, &rankdm));
    PetscCall(DMSetLocalSection(rankdm, rankSection));
    PetscCall(DMGetSectionSF(rankdm, &rankSF));
    PetscCall(PetscSectionGetStorageSize(rankSection, &s));
    PetscCall(PetscMalloc1(s, &rankArray));
    PetscCall(PetscSFBcastBegin(rankSF, MPI_INT, rankGlobalArray, rankArray, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(rankSF, MPI_INT, rankGlobalArray, rankArray, MPI_REPLACE));
    PetscCall(PetscFree(rankGlobalArray));
    PetscCall(DMDestroy(&rankdm));

    /* Count the number of interfaces per rank, not including those on the root */
    PetscCall(PetscCalloc1(numProcs, &interfacesPerRank));
    for (v = vStart; v < vEnd; v++) {
      if (vertexNumber[v - vStart]) {
        PetscCall(PetscSectionGetDof(rankSection, v, &shareCnt));
        if (shareCnt) {
          PetscCall(PetscSectionGetOffset(rankSection, v, &offset));
          for (j = 0; j < shareCnt; j++) interfacesPerRank[rankArray[offset + j]]++;
        }
      }
    }
    for (r = 0, k = 0, interfacesPerRank[rank] = 0; r < numProcs; r++) k += interfacesPerRank[r];

    /* Get the degree of the vertex */
    PetscCall(PetscMalloc3(k, &interfaces_lv, k, &interfaces_gv, numProcs + 1, &interfacesOffset));
    interfacesOffset[0] = 0;
    for (r = 0; r < numProcs; r++) {
      interfacesOffset[r + 1] = interfacesOffset[r] + interfacesPerRank[r];
      if (interfacesPerRank[r]) numNgbRanks++;
      interfacesPerRank[r] = 0;
    }

    /* Get the local and global vertex numbers at interfaces */
    PetscCall(DMPlexGetVertexNumbering(dm, &globalVertexNum));
    PetscCall(ISGetIndices(globalVertexNum, &gV));
    for (v = vStart; v < vEnd; v++) {
      if (vertexNumber[v - vStart]) {
        PetscCall(PetscSectionGetDof(rankSection, v, &shareCnt));
        if (shareCnt) {
          PetscCall(PetscSectionGetOffset(rankSection, v, &offset));
          for (j = 0; j < shareCnt; j++) {
            r = rankArray[offset + j];
            if (r == rank) continue;
            k                = interfacesOffset[r] + interfacesPerRank[r]++;
            interfaces_lv[k] = vertexNumber[v - vStart];
            gv               = gV[v - vStart];
            interfaces_gv[k] = gv < 0 ? -gv : gv + 1;
          }
        }
      }
    }
    PetscCall(PetscFree(interfacesPerRank));
    PetscCall(PetscFree(rankArray));
    PetscCall(ISRestoreIndices(globalVertexNum, &gV));
    PetscCall(PetscSectionDestroy(&rankSection));
  }
  PetscCall(DMDestroy(&udm));
  PetscCall(PetscFree(vertexNumber));

  /* Send the data to ParMmg and remesh */
  PetscCall(DMPlexMetricNoInsertion(dm, &noInsert));
  PetscCall(DMPlexMetricNoSwapping(dm, &noSwap));
  PetscCall(DMPlexMetricNoMovement(dm, &noMove));
  PetscCall(DMPlexMetricNoSurf(dm, &noSurf));
  PetscCall(DMPlexMetricGetVerbosity(dm, &verbosity));
  PetscCall(DMPlexMetricGetNumIterations(dm, &numIter));
  PetscCall(DMPlexMetricGetGradationFactor(dm, &gradationFactor));
  PetscCall(DMPlexMetricGetHausdorffNumber(dm, &hausdorffNumber));
  PetscCallMMG_NONSTANDARD(PMMG_Init_parMesh, PMMG_ARG_start, PMMG_ARG_ppParMesh, &parmesh, PMMG_ARG_pMesh, PMMG_ARG_pMet, PMMG_ARG_dim, 3, PMMG_ARG_MPIComm, comm, PMMG_ARG_end);
  PetscCallMMG_NONSTANDARD(PMMG_Set_meshSize, parmesh, numUsedVertices, numCellsNotShared, 0, numFaceTags, 0, 0);
  PetscCallMMG_NONSTANDARD(PMMG_Set_iparameter, parmesh, PMMG_IPARAM_APImode, PMMG_APIDISTRIB_nodes);
  PetscCallMMG_NONSTANDARD(PMMG_Set_iparameter, parmesh, PMMG_IPARAM_noinsert, noInsert);
  PetscCallMMG_NONSTANDARD(PMMG_Set_iparameter, parmesh, PMMG_IPARAM_noswap, noSwap);
  PetscCallMMG_NONSTANDARD(PMMG_Set_iparameter, parmesh, PMMG_IPARAM_nomove, noMove);
  PetscCallMMG_NONSTANDARD(PMMG_Set_iparameter, parmesh, PMMG_IPARAM_nosurf, noSurf);
  PetscCallMMG_NONSTANDARD(PMMG_Set_iparameter, parmesh, PMMG_IPARAM_verbose, verbosity);
  PetscCallMMG_NONSTANDARD(PMMG_Set_iparameter, parmesh, PMMG_IPARAM_globalNum, 1);
  PetscCallMMG_NONSTANDARD(PMMG_Set_iparameter, parmesh, PMMG_IPARAM_niter, numIter);
  PetscCallMMG_NONSTANDARD(PMMG_Set_dparameter, parmesh, PMMG_DPARAM_hgrad, gradationFactor);
  PetscCallMMG_NONSTANDARD(PMMG_Set_dparameter, parmesh, PMMG_DPARAM_hausd, hausdorffNumber);
  PetscCallMMG_NONSTANDARD(PMMG_Set_vertices, parmesh, vertices, verTags);
  PetscCallMMG_NONSTANDARD(PMMG_Set_tetrahedra, parmesh, cells, cellTags);
  PetscCallMMG_NONSTANDARD(PMMG_Set_triangles, parmesh, bdFaces, faceTags);
  PetscCallMMG_NONSTANDARD(PMMG_Set_metSize, parmesh, MMG5_Vertex, numUsedVertices, MMG5_Tensor);
  PetscCallMMG_NONSTANDARD(PMMG_Set_tensorMets, parmesh, metric);
  PetscCallMMG_NONSTANDARD(PMMG_Set_numberOfNodeCommunicators, parmesh, numNgbRanks);
  for (r = 0, c = 0; r < numProcs; ++r) {
    if (interfacesOffset[r + 1] > interfacesOffset[r]) {
      PetscCallMMG_NONSTANDARD(PMMG_Set_ithNodeCommunicatorSize, parmesh, c, r, interfacesOffset[r + 1] - interfacesOffset[r]);
      PetscCallMMG_NONSTANDARD(PMMG_Set_ithNodeCommunicator_nodes, parmesh, c++, &interfaces_lv[interfacesOffset[r]], &interfaces_gv[interfacesOffset[r]], 1);
    }
  }
  PetscCallMMG(PMMG_parmmglib_distributed, parmesh);
  PetscCall(PetscFree(cells));
  PetscCall(PetscFree2(metric, vertices));
  PetscCall(PetscFree2(bdFaces, faceTags));
  PetscCall(PetscFree2(verTags, cellTags));
  if (numProcs > 1) PetscCall(PetscFree3(interfaces_lv, interfaces_gv, interfacesOffset));

  /* Retrieve mesh from Mmg */
  numCornersNew = 4;
  PetscCallMMG_NONSTANDARD(PMMG_Get_meshSize, parmesh, &numVerticesNew, &numCellsNew, 0, &numFacesNew, 0, 0);
  PetscCall(PetscMalloc4(dim * numVerticesNew, &verticesNew, numVerticesNew, &verTagsNew, numVerticesNew, &corners, numVerticesNew, &requiredVer));
  PetscCall(PetscMalloc3((dim + 1) * numCellsNew, &cellsNew, numCellsNew, &cellTagsNew, numCellsNew, &requiredCells));
  PetscCall(PetscMalloc4(dim * numFacesNew, &facesNew, numFacesNew, &faceTagsNew, numFacesNew, &ridges, numFacesNew, &requiredFaces));
  PetscCallMMG_NONSTANDARD(PMMG_Get_vertices, parmesh, verticesNew, verTagsNew, corners, requiredVer);
  PetscCallMMG_NONSTANDARD(PMMG_Get_tetrahedra, parmesh, cellsNew, cellTagsNew, requiredCells);
  PetscCallMMG_NONSTANDARD(PMMG_Get_triangles, parmesh, facesNew, faceTagsNew, requiredFaces);
  PetscCall(PetscMalloc2(numVerticesNew, &owners, numVerticesNew, &gv_new));
  PetscCallMMG_NONSTANDARD(PMMG_Set_iparameter, parmesh, PMMG_IPARAM_globalNum, 1);
  PetscCallMMG_NONSTANDARD(PMMG_Get_verticesGloNum, parmesh, gv_new, owners);
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
  PetscCallMMG_NONSTANDARD(PMMG_Free_all, PMMG_ARG_start, PMMG_ARG_ppParMesh, &parmesh, PMMG_ARG_end);
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
