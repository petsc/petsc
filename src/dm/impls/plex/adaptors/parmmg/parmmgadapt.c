#include "../mmgcommon.h"   /*I      "petscdmplex.h"   I*/
#include <parmmg/libparmmg.h>

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
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRMPI(MPI_Comm_size(comm, &numProcs));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  if (bdLabel) {
    CHKERRQ(PetscObjectGetName((PetscObject) bdLabel, &bdLabelName));
    CHKERRQ(PetscStrcmp(bdLabelName, bdName, &flg));
    PetscCheckFalse(flg,comm, PETSC_ERR_ARG_WRONG, "\"%s\" cannot be used as label for boundary facets", bdLabelName);
  }
  if (rgLabel) {
    CHKERRQ(PetscObjectGetName((PetscObject) rgLabel, &rgLabelName));
    CHKERRQ(PetscStrcmp(rgLabelName, rgName, &flg));
    PetscCheckFalse(flg,comm, PETSC_ERR_ARG_WRONG, "\"%s\" cannot be used as label for element tags", rgLabelName);
  }

  /* Get mesh information */
  CHKERRQ(DMGetDimension(dm, &dim));
  PetscCheck(dim == 3,comm, PETSC_ERR_ARG_OUTOFRANGE, "ParMmg only works in 3D.");
  Neq  = (dim*(dim+1))/2;
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(DMPlexUninterpolate(dm, &udm));
  CHKERRQ(DMPlexGetMaxSizes(udm, &maxConeSize, NULL));
  numCells    = cEnd - cStart;
  numVertices = vEnd - vStart;

  /* Get cell offsets */
  CHKERRQ(PetscMalloc1(numCells*maxConeSize, &cells));
  for (c = 0, coff = 0; c < numCells; ++c) {
    const PetscInt *cone;
    PetscInt        coneSize, cl;

    CHKERRQ(DMPlexGetConeSize(udm, c, &coneSize));
    CHKERRQ(DMPlexGetCone(udm, c, &cone));
    for (cl = 0; cl < coneSize; ++cl) cells[coff++] = cone[cl] - vStart+1;
  }

  /* Get vertex coordinate array */
  CHKERRQ(DMGetCoordinateDM(dm, &cdm));
  CHKERRQ(DMGetLocalSection(cdm, &coordSection));
  CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
  CHKERRQ(VecGetArrayRead(coordinates, &coords));
  CHKERRQ(PetscMalloc2(numVertices*Neq, &metric, dim*numVertices, &vertices));
  for (v = 0; v < vEnd-vStart; ++v) {
    CHKERRQ(PetscSectionGetOffset(coordSection, v+vStart, &off));
    for (i = 0; i < dim; ++i) vertices[dim*v+i] = PetscRealPart(coords[off+i]);
  }
  CHKERRQ(VecRestoreArrayRead(coordinates, &coords));

  /* Get face tags */
  if (!bdLabel) {
    flg = PETSC_TRUE;
    CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, bdName, &bdLabel));
    CHKERRQ(DMPlexMarkBoundaryFaces(dm, 1, bdLabel));
  }
  CHKERRQ(DMLabelGetBounds(bdLabel, &pStart, &pEnd));
  for (f = pStart, bdSize = 0, numFaceTags = 0; f < pEnd; ++f) {
    PetscBool hasPoint;
    PetscInt *closure = NULL, closureSize, cl;

    CHKERRQ(DMLabelHasPoint(bdLabel, f, &hasPoint));
    if ((!hasPoint) || (f < fStart) || (f >= fEnd)) continue;
    numFaceTags++;

    CHKERRQ(DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize*2; cl += 2) {
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) ++bdSize;
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure));
  }
  CHKERRQ(PetscMalloc2(bdSize, &bdFaces, numFaceTags, &faceTags));
  for (f = pStart, bdSize = 0, numFaceTags = 0; f < pEnd; ++f) {
    PetscBool hasPoint;
    PetscInt *closure = NULL, closureSize, cl;

    CHKERRQ(DMLabelHasPoint(bdLabel, f, &hasPoint));
    if ((!hasPoint) || (f < fStart) || (f >= fEnd)) continue;

    CHKERRQ(DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure));
    for (cl = 0; cl < closureSize*2; cl += 2) {
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) bdFaces[bdSize++] = closure[cl] - vStart+1;
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure));
    CHKERRQ(DMLabelGetValue(bdLabel, f, &faceTags[numFaceTags++]));
  }

  /* Get cell tags */
  CHKERRQ(PetscCalloc2(numVertices, &verTags, numCells, &cellTags));
  if (rgLabel) {
    for (c = cStart; c < cEnd; ++c) CHKERRQ(DMLabelGetValue(rgLabel, c, &cellTags[c]));
  }

  /* Get metric */
  CHKERRQ(VecViewFromOptions(vertexMetric, NULL, "-adapt_metric_view"));
  CHKERRQ(VecGetArrayRead(vertexMetric, &met));
  CHKERRQ(DMPlexMetricIsIsotropic(dm, &isotropic));
  CHKERRQ(DMPlexMetricIsUniform(dm, &uniform));
  for (v = 0; v < (vEnd-vStart); ++v) {
    for (i = 0, k = 0; i < dim; ++i) {
      for (j = i; j < dim; ++j, ++k) {
        if (isotropic) {
          if (i == j) {
            if (uniform) metric[Neq*v+k] = PetscRealPart(met[0]);
            else metric[Neq*v+k] = PetscRealPart(met[v]);
          } else metric[Neq*v+k] = 0.0;
        } else metric[Neq*v+k] = PetscRealPart(met[dim*dim*v+dim*i+j]);
      }
    }
  }
  CHKERRQ(VecRestoreArrayRead(vertexMetric, &met));

  /* Build ParMMG communicators: the list of vertices between two partitions  */
  niranks = nrranks = 0;
  numNgbRanks = 0;
  if (numProcs > 1) {
    CHKERRQ(DMGetPointSF(dm, &sf));
    CHKERRQ(PetscSFSetUp(sf));
    CHKERRQ(PetscSFGetLeafRanks(sf, &niranks, &iranks, &ioffset, &irootloc));
    CHKERRQ(PetscSFGetRootRanks(sf, &nrranks, &rranks, &roffset, &rmine, &rremote));
    CHKERRQ(PetscCalloc1(numProcs, &numVerInterfaces));

    /* Count number of roots associated with each leaf */
    for (r = 0; r < niranks; ++r) {
      for (i=ioffset[r], count=0; i<ioffset[r+1]; ++i) {
        if (irootloc[i] >= vStart && irootloc[i] < vEnd) count++;
      }
      numVerInterfaces[iranks[r]] += count;
    }

    /* Count number of leaves associated with each root */
    for (r = 0; r < nrranks; ++r) {
      for (i=roffset[r], count=0; i<roffset[r+1]; ++i) {
        if (rmine[i] >= vStart && rmine[i] < vEnd) count++;
      }
      numVerInterfaces[rranks[r]] += count;
    }

    /* Count global number of ranks */
    for (p = 0; p < numProcs; ++p) {
      if (numVerInterfaces[p]) numNgbRanks++;
    }

    /* Provide numbers of vertex interfaces */
    CHKERRQ(PetscMalloc2(numNgbRanks, &ngbRanks, numNgbRanks, &verNgbRank));
    for (p = 0, n = 0; p < numProcs; ++p) {
      if (numVerInterfaces[p]) {
        ngbRanks[n] = p;
        verNgbRank[n] = numVerInterfaces[p];
        n++;
      }
    }
    numVerNgbRanksTotal = 0;
    for (p = 0; p < numNgbRanks; ++p) numVerNgbRanksTotal += verNgbRank[p];

    /* For each neighbor, fill in interface arrays */
    CHKERRQ(PetscMalloc3(numVerNgbRanksTotal, &interfaces_lv, numVerNgbRanksTotal, &interfaces_gv,  numNgbRanks+1, &intOffset));
    intOffset[0] = 0;
    for (p = 0, r = 0, i = 0; p < numNgbRanks; ++p) {
      intOffset[p+1] = intOffset[p];

      /* Leaf case */
      if (iranks && iranks[i] == ngbRanks[p]) {

        /* Add the right slice of irootloc at the right place */
        sliceSize = ioffset[i+1]-ioffset[i];
        for (j = 0, count = 0; j < sliceSize; ++j) {
          PetscCheck(ioffset[i]+j < ioffset[niranks],comm, PETSC_ERR_ARG_OUTOFRANGE, "Leaf index %" PetscInt_FMT " out of range (expected < %" PetscInt_FMT ")", ioffset[i]+j, ioffset[niranks]);
          v = irootloc[ioffset[i]+j];
          if (v >= vStart && v < vEnd) {
            PetscCheck(intOffset[p+1]+count < numVerNgbRanksTotal,comm, PETSC_ERR_ARG_OUTOFRANGE, "Leaf interface index %" PetscInt_FMT " out of range (expected < %" PetscInt_FMT ")", intOffset[p+1]+count, numVerNgbRanksTotal);
            interfaces_lv[intOffset[p+1]+count] = v-vStart;
            count++;
          }
        }
        intOffset[p+1] += count;
        i++;
      }

      /* Root case */
      if (rranks && rranks[r] == ngbRanks[p]) {

        /* Add the right slice of rmine at the right place */
        sliceSize = roffset[r+1]-roffset[r];
        for (j = 0, count = 0; j < sliceSize; ++j) {
          PetscCheck(roffset[r]+j < roffset[nrranks],comm, PETSC_ERR_ARG_OUTOFRANGE, "Root index %" PetscInt_FMT " out of range (expected < %" PetscInt_FMT ")", roffset[r]+j, roffset[nrranks]);
          v = rmine[roffset[r]+j];
          if (v >= vStart && v < vEnd) {
            PetscCheck(intOffset[p+1]+count < numVerNgbRanksTotal,comm, PETSC_ERR_ARG_OUTOFRANGE, "Root interface index %" PetscInt_FMT " out of range (expected < %" PetscInt_FMT ")", intOffset[p+1]+count, numVerNgbRanksTotal);
            interfaces_lv[intOffset[p+1]+count] = v-vStart;
            count++;
          }
        }
        intOffset[p+1] += count;
        r++;
      }

      /* Check validity of offsets */
      PetscCheck(intOffset[p+1] == intOffset[p]+verNgbRank[p],comm, PETSC_ERR_ARG_OUTOFRANGE, "Missing offsets (expected %" PetscInt_FMT ", got %" PetscInt_FMT ")", intOffset[p]+verNgbRank[p], intOffset[p+1]);
    }
    CHKERRQ(DMPlexGetVertexNumbering(udm, &globalVertexNum));
    CHKERRQ(ISGetIndices(globalVertexNum, &gV));
    for (i = 0; i < numVerNgbRanksTotal; ++i) {
      v = gV[interfaces_lv[i]];
      interfaces_gv[i] = v < 0 ? -v-1 : v;
      interfaces_lv[i] += 1;
      interfaces_gv[i] += 1;
    }
    CHKERRQ(ISRestoreIndices(globalVertexNum, &gV));
    CHKERRQ(PetscFree(numVerInterfaces));
  }
  CHKERRQ(DMDestroy(&udm));

  /* Send the data to ParMmg and remesh */
  CHKERRQ(DMPlexMetricNoInsertion(dm, &noInsert));
  CHKERRQ(DMPlexMetricNoSwapping(dm, &noSwap));
  CHKERRQ(DMPlexMetricNoMovement(dm, &noMove));
  CHKERRQ(DMPlexMetricNoSurf(dm, &noSurf));
  CHKERRQ(DMPlexMetricGetVerbosity(dm, &verbosity));
  CHKERRQ(DMPlexMetricGetNumIterations(dm, &numIter));
  CHKERRQ(DMPlexMetricGetGradationFactor(dm, &gradationFactor));
  CHKERRQ(DMPlexMetricGetHausdorffNumber(dm, &hausdorffNumber));
  CHKERRMMG_NONSTANDARD(PMMG_Init_parMesh(PMMG_ARG_start, PMMG_ARG_ppParMesh, &parmesh, PMMG_ARG_pMesh, PMMG_ARG_pMet, PMMG_ARG_dim, 3, PMMG_ARG_MPIComm, comm, PMMG_ARG_end));
  CHKERRMMG_NONSTANDARD(PMMG_Set_meshSize(parmesh, numVertices, numCells, 0, numFaceTags, 0, 0));
  CHKERRMMG_NONSTANDARD(PMMG_Set_iparameter(parmesh, PMMG_IPARAM_APImode, PMMG_APIDISTRIB_nodes));
  CHKERRMMG_NONSTANDARD(PMMG_Set_iparameter(parmesh, PMMG_IPARAM_noinsert, noInsert));
  CHKERRMMG_NONSTANDARD(PMMG_Set_iparameter(parmesh, PMMG_IPARAM_noswap, noSwap));
  CHKERRMMG_NONSTANDARD(PMMG_Set_iparameter(parmesh, PMMG_IPARAM_nomove, noMove));
  CHKERRMMG_NONSTANDARD(PMMG_Set_iparameter(parmesh, PMMG_IPARAM_nosurf, noSurf));
  CHKERRMMG_NONSTANDARD(PMMG_Set_iparameter(parmesh, PMMG_IPARAM_verbose, verbosity));
  CHKERRMMG_NONSTANDARD(PMMG_Set_iparameter(parmesh, PMMG_IPARAM_globalNum, 1));
  CHKERRMMG_NONSTANDARD(PMMG_Set_iparameter(parmesh, PMMG_IPARAM_niter, numIter));
  CHKERRMMG_NONSTANDARD(PMMG_Set_dparameter(parmesh, PMMG_DPARAM_hgrad, gradationFactor));
  CHKERRMMG_NONSTANDARD(PMMG_Set_dparameter(parmesh, PMMG_DPARAM_hausd, hausdorffNumber));
  CHKERRMMG_NONSTANDARD(PMMG_Set_vertices(parmesh, vertices, verTags));
  CHKERRMMG_NONSTANDARD(PMMG_Set_tetrahedra(parmesh, cells, cellTags));
  CHKERRMMG_NONSTANDARD(PMMG_Set_triangles(parmesh, bdFaces, faceTags));
  CHKERRMMG_NONSTANDARD(PMMG_Set_metSize(parmesh, MMG5_Vertex, numVertices, MMG5_Tensor));
  CHKERRMMG_NONSTANDARD(PMMG_Set_tensorMets(parmesh, metric));
  CHKERRMMG_NONSTANDARD(PMMG_Set_numberOfNodeCommunicators(parmesh, numNgbRanks));
  for (c = 0; c < numNgbRanks; ++c) {
    CHKERRMMG_NONSTANDARD(PMMG_Set_ithNodeCommunicatorSize, parmesh, c, ngbRanks[c], intOffset[c+1]-intOffset[c]);
    CHKERRMMG_NONSTANDARD(PMMG_Set_ithNodeCommunicator_nodes, parmesh, c, &interfaces_lv[intOffset[c]], &interfaces_gv[intOffset[c]], 1);
  }
  CHKERRMMG(PMMG_parmmglib_distributed, parmesh);
  CHKERRQ(PetscFree(cells));
  CHKERRQ(PetscFree2(metric, vertices));
  CHKERRQ(PetscFree2(bdFaces, faceTags));
  CHKERRQ(PetscFree2(verTags, cellTags));
  if (numProcs > 1) {
    CHKERRQ(PetscFree2(ngbRanks, verNgbRank));
    CHKERRQ(PetscFree3(interfaces_lv, interfaces_gv, intOffset));
  }

  /* Retrieve mesh from Mmg */
  numCornersNew = 4;
  CHKERRMMG_NONSTANDARD(PMMG_Get_meshSize, parmesh, &numVerticesNew, &numCellsNew, 0, &numFacesNew, 0, 0);
  CHKERRQ(PetscMalloc4(dim*numVerticesNew, &verticesNew, numVerticesNew, &verTagsNew, numVerticesNew, &corners, numVerticesNew, &requiredVer));
  CHKERRQ(PetscMalloc3((dim+1)*numCellsNew, &cellsNew, numCellsNew, &cellTagsNew, numCellsNew, &requiredCells));
  CHKERRQ(PetscMalloc4(dim*numFacesNew, &facesNew, numFacesNew, &faceTagsNew, numFacesNew, &ridges, numFacesNew, &requiredFaces));
  CHKERRMMG_NONSTANDARD(PMMG_Get_vertices, parmesh, verticesNew, verTagsNew, corners, requiredVer);
  CHKERRMMG_NONSTANDARD(PMMG_Get_tetrahedra, parmesh, cellsNew, cellTagsNew, requiredCells);
  CHKERRMMG_NONSTANDARD(PMMG_Get_triangles, parmesh, facesNew, faceTagsNew, requiredFaces);
  CHKERRQ(PetscMalloc2(numVerticesNew, &owners, numVerticesNew, &gv_new));
  CHKERRMMG_NONSTANDARD(PMMG_Set_iparameter, parmesh, PMMG_IPARAM_globalNum, 1);
  CHKERRMMG_NONSTANDARD(PMMG_Get_verticesGloNum, parmesh, gv_new, owners);
  for (i = 0; i < dim*numFacesNew; ++i) facesNew[i] -= 1;
  for (i = 0; i < (dim+1)*numCellsNew; ++i) cellsNew[i] = gv_new[cellsNew[i]-1]-1;
  for (i = 0, numVerticesNewLoc = 0; i < numVerticesNew; ++i) {
    if (owners[i] == rank) numVerticesNewLoc++;
  }
  CHKERRQ(PetscMalloc2(numVerticesNewLoc*dim, &verticesNewLoc, numVerticesNew, &verticesNewSorted));
  for (i = 0, c = 0; i < numVerticesNew; i++) {
    if (owners[i] == rank) {
      for (j=0; j<dim; ++j) verticesNewLoc[dim*c+j] = verticesNew[dim*i+j];
      c++;
    }
  }

  /* Reorder for consistency with DMPlex */
  for (i = 0; i < numCellsNew; ++i) CHKERRQ(DMPlexInvertCell(DM_POLYTOPE_TETRAHEDRON, &cellsNew[4*i]));

  /* Create new plex */
  CHKERRQ(DMPlexCreateFromCellListParallelPetsc(comm, dim, numCellsNew, numVerticesNewLoc, PETSC_DECIDE, numCornersNew, PETSC_TRUE, cellsNew, dim, verticesNewLoc, NULL, &verticesNewSorted, dmNew));
  CHKERRMMG_NONSTANDARD(PMMG_Free_all, PMMG_ARG_start, PMMG_ARG_ppParMesh, &parmesh, PMMG_ARG_end);
  CHKERRQ(PetscFree4(verticesNew, verTagsNew, corners, requiredVer));

  /* Get adapted mesh information */
  CHKERRQ(DMPlexGetHeightStratum(*dmNew, 0, &cStart, &cEnd));
  CHKERRQ(DMPlexGetHeightStratum(*dmNew, 1, &fStart, &fEnd));
  CHKERRQ(DMPlexGetDepthStratum(*dmNew, 0, &vStart, &vEnd));

  /* Rebuild boundary label */
  CHKERRQ(DMCreateLabel(*dmNew, flg ? bdName : bdLabelName));
  CHKERRQ(DMGetLabel(*dmNew, flg ? bdName : bdLabelName, &bdLabelNew));
  for (i = 0; i < numFacesNew; i++) {
    PetscBool       hasTag = PETSC_FALSE;
    PetscInt        numCoveredPoints, numFaces = 0, facePoints[3];
    const PetscInt *coveredPoints = NULL;

    for (j = 0; j < dim; ++j) {
      lv = facesNew[i*dim+j];
      gv = gv_new[lv]-1;
      CHKERRQ(PetscFindInt(gv, numVerticesNew, verticesNewSorted, &lv));
      facePoints[j] = lv+vStart;
    }
    CHKERRQ(DMPlexGetFullJoin(*dmNew, dim, facePoints, &numCoveredPoints, &coveredPoints));
    for (j = 0; j < numCoveredPoints; ++j) {
      if (coveredPoints[j] >= fStart && coveredPoints[j] < fEnd) {
        numFaces++;
        f = j;
      }
    }
    PetscCheck(numFaces == 1,comm, PETSC_ERR_ARG_OUTOFRANGE, "%" PetscInt_FMT " vertices cannot define more than 1 facet (%" PetscInt_FMT ")", dim, numFaces);
    CHKERRQ(DMLabelHasStratum(bdLabel, faceTagsNew[i], &hasTag));
    if (hasTag) CHKERRQ(DMLabelSetValue(bdLabelNew, coveredPoints[f], faceTagsNew[i]));
    CHKERRQ(DMPlexRestoreJoin(*dmNew, dim, facePoints, &numCoveredPoints, &coveredPoints));
  }
  CHKERRQ(PetscFree4(facesNew, faceTagsNew, ridges, requiredFaces));
  CHKERRQ(PetscFree2(owners, gv_new));
  CHKERRQ(PetscFree2(verticesNewLoc, verticesNewSorted));
  if (flg) CHKERRQ(DMLabelDestroy(&bdLabel));

  /* Rebuild cell labels */
  CHKERRQ(DMCreateLabel(*dmNew, rgLabel ? rgLabelName : rgName));
  CHKERRQ(DMGetLabel(*dmNew, rgLabel ? rgLabelName : rgName, &rgLabelNew));
  for (c = cStart; c < cEnd; ++c) CHKERRQ(DMLabelSetValue(rgLabelNew, c, cellTagsNew[c-cStart]));
  CHKERRQ(PetscFree3(cellsNew, cellTagsNew, requiredCells));

  PetscFunctionReturn(0);
}
