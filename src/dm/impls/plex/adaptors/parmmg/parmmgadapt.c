#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <parmmg/libparmmg.h>

PETSC_EXTERN PetscErrorCode DMAdaptMetric_ParMmg_Plex(DM dm, Vec vertexMetric, DMLabel bdLabel, DM *dmNew)
{
  MPI_Comm           comm, tmpComm;
  const char        *bdName = "_boundary_";
  DM                 udm, cdm;
  DMLabel            bdLabelFull;
  const char        *bdLabelName;
  IS                 bdIS, globalVertexNum;
  PetscSection       coordSection;
  Vec                coordinates;
  PetscSF            sf;
  const PetscScalar *coords, *met;
  const PetscInt    *bdFacesFull;
  PetscInt          *bdFaces, *bdFaceIds;
  PetscReal         *vertices, *metric, *verticesNew;
  PetscInt          *cells;
  const PetscInt    *gV, *ioffset, *irootloc, *roffset, *rmine, *rremote;
  PetscInt          *numVerInterfaces, *ngbRanks, *verNgbRank, *interfaces_lv, *interfaces_gv, *intOffset;
  PetscInt           niranks, nrranks, numNgbRanks, numVerNgbRanksTotal, count, sliceSize;
  PetscInt          *verTags, *cellTags;
  PetscInt           dim, Neq, cStart, cEnd, numCells, coff, vStart, vEnd, numVertices;
  PetscInt           maxConeSize, numBdFaces, bdSize, fStart, fEnd;
  PetscBool          flg, noInsert, noSwap, noMove;
  DMLabel            bdLabelNew;
  PetscReal         *verticesNewLoc, gradationFactor;
  PetscInt          *verTagsNew, *cellsNew, *cellTagsNew, *corners;
  PetscInt          *requiredCells, *requiredVer, *facesNew, *faceTagsNew, *ridges, *requiredFaces;
  PetscInt          *gv_new, *owners, *verticesNewSorted;
  PetscInt           numCellsNew, numVerticesNew, numCornersNew, numFacesNew, numVerticesNewLoc;
  PetscInt           i, j, k, p, r, n, v, c, f, off, lv, gv;
  PetscInt           verbosity, numIter;
  const PetscMPIInt *iranks, *rranks;
  PetscMPIInt        numProcs, rank;
  PMMG_pParMesh      parmesh = NULL;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_dup(comm, &tmpComm);CHKERRMPI(ierr);
  if (bdLabel) {
    ierr = PetscObjectGetName((PetscObject) bdLabel, &bdLabelName);CHKERRQ(ierr);
    ierr = PetscStrcmp(bdLabelName, bdName, &flg);CHKERRQ(ierr);
    if (flg) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "\"%s\" cannot be used as label for boundary facets", bdLabelName);
  }

  /* Get mesh information */
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  if (dim != 3) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "ParMmg only works in 3D.\n");
  Neq  = (dim*(dim+1))/2;
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexUninterpolate(dm, &udm);CHKERRQ(ierr);
  ierr = DMPlexGetMaxSizes(udm, &maxConeSize, NULL);CHKERRQ(ierr);
  numCells    = cEnd - cStart;
  numVertices = vEnd - vStart;

  /* Get cell offsets */
  ierr = PetscMalloc1(numCells*maxConeSize, &cells);CHKERRQ(ierr);
  for (c = 0, coff = 0; c < numCells; ++c) {
    const PetscInt *cone;
    PetscInt        coneSize, cl;

    ierr = DMPlexGetConeSize(udm, c, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(udm, c, &cone);CHKERRQ(ierr);
    for (cl = 0; cl < coneSize; ++cl) cells[coff++] = cone[cl] - vStart+1;
  }

  /* Get vertex coordinate array */
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(cdm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
  ierr = PetscMalloc2(numVertices*Neq, &metric, dim*numVertices, &vertices);CHKERRQ(ierr);
  for (v = 0; v < vEnd-vStart; ++v) {
    ierr = PetscSectionGetOffset(coordSection, v+vStart, &off);CHKERRQ(ierr);
    for (i = 0; i < dim; ++i) vertices[dim*v+i] = PetscRealPart(coords[off+i]);
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
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) bdFaces[bdSize++] = closure[cl] - vStart+1;
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
  for (v = 0; v < (vEnd-vStart); ++v) {
    for (i = 0, k = 0; i < dim; ++i) {
      for (j = i; j < dim; ++j) {
        metric[Neq*v+k] = PetscRealPart(met[dim*dim*v+dim*i+j]);
        k++;
      }
    }
  }
  ierr = VecRestoreArrayRead(vertexMetric, &met);CHKERRQ(ierr);

  /* Build ParMMG communicators: the list of vertices between two partitions  */
  niranks = nrranks = 0;
  numNgbRanks = 0;
  if (numProcs > 1) {
    ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
    ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
    ierr = PetscSFGetLeafRanks(sf, &niranks, &iranks, &ioffset, &irootloc);CHKERRQ(ierr);
    ierr = PetscSFGetRootRanks(sf, &nrranks, &rranks, &roffset, &rmine, &rremote);CHKERRQ(ierr);
    ierr = PetscCalloc1(numProcs, &numVerInterfaces);CHKERRQ(ierr);

    /* Counting */
    for (r = 0; r < niranks; ++r) {
      for (i=ioffset[r], count=0; i<ioffset[r+1]; ++i) {
        if (irootloc[i] >= vStart && irootloc[i] < vEnd) count++;
      }
      numVerInterfaces[iranks[r]] += count;
    }
    for (r = 0; r < nrranks; ++r) {
      for (i=roffset[r], count=0; i<roffset[r+1]; ++i) {
        if (rmine[i] >= vStart && rmine[i] < vEnd) count++;
      }
      numVerInterfaces[rranks[r]] += count;
    }
    for (p = 0; p < numProcs; ++p) {
      if (numVerInterfaces[p]) numNgbRanks++;
    }
    ierr = PetscMalloc2(numNgbRanks, &ngbRanks, numNgbRanks, &verNgbRank);CHKERRQ(ierr);
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
    ierr = PetscMalloc3(numVerNgbRanksTotal, &interfaces_lv, numVerNgbRanksTotal, &interfaces_gv,  numNgbRanks+1, &intOffset);CHKERRQ(ierr);
    intOffset[0] = 0;
    for (p = 0, r = 0, i = 0; p < numNgbRanks; ++p) {
      intOffset[p+1] = intOffset[p];
      if (iranks && iranks[i] == ngbRanks[p]) {

        /* Add the right slice of irootloc at the right place */
        sliceSize = ioffset[i+1]-ioffset[i];
        for (j = 0, count = 0; j < sliceSize; ++j) {
          if (ioffset[i]+j >= ioffset[niranks]) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Offset out of range");
          v = irootloc[ioffset[i]+j];
          if (v >= vStart && v < vEnd) {
            if (intOffset[p+1]+count >= numVerNgbRanksTotal) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Offset out of range");
            interfaces_lv[intOffset[p+1]+count] = v-vStart;
            count++;
          }
        }
        intOffset[p+1] += count;
        i++;
      }
      if (rranks && rranks[r] == ngbRanks[p]) {

        /* Add the right slice of rmine at the right place */
        sliceSize = roffset[r+1]-roffset[r];
        for (j = 0, count = 0; j < sliceSize; ++j) {
          if (roffset[r]+j >= roffset[nrranks]) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Offset out of range");
          v = rmine[roffset[r]+j];
          if (v >= vStart && v < vEnd) {
            if (intOffset[p+1]+count >= numVerNgbRanksTotal) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Offset out of range");
            interfaces_lv[intOffset[p+1]+count] = v-vStart;
            count++;
          }
        }
        intOffset[p+1] += count;
        r++;
      }
      if (intOffset[p+1] != intOffset[p] + verNgbRank[p]) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Unequal offsets");
    }
    ierr = DMPlexGetVertexNumbering(udm, &globalVertexNum);CHKERRQ(ierr);
    ierr = ISGetIndices(globalVertexNum, &gV);CHKERRQ(ierr);
    for (i = 0; i < numVerNgbRanksTotal; ++i) {
      v = gV[interfaces_lv[i]];
      interfaces_gv[i] = v < 0 ? -v-1 : v;
      interfaces_lv[i] += 1;
      interfaces_gv[i] += 1;
    }
    ierr = ISRestoreIndices(globalVertexNum, &gV);CHKERRQ(ierr);
    ierr = PetscFree(numVerInterfaces);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&udm);CHKERRQ(ierr);

  /* Send the data to ParMmg and remesh */
  ierr = DMPlexMetricNoInsertion(dm, &noInsert);CHKERRQ(ierr);
  ierr = DMPlexMetricNoSwapping(dm, &noSwap);CHKERRQ(ierr);
  ierr = DMPlexMetricNoMovement(dm, &noMove);CHKERRQ(ierr);
  ierr = DMPlexMetricGetVerbosity(dm, &verbosity);CHKERRQ(ierr);
  ierr = DMPlexMetricGetNumIterations(dm, &numIter);CHKERRQ(ierr);
  ierr = DMPlexMetricGetGradationFactor(dm, &gradationFactor);CHKERRQ(ierr);
  ierr = PetscCalloc2(numVertices, &verTags, numCells, &cellTags);CHKERRQ(ierr);
  ierr = PMMG_Init_parMesh(PMMG_ARG_start, PMMG_ARG_ppParMesh, &parmesh, PMMG_ARG_pMesh, PMMG_ARG_pMet, PMMG_ARG_dim, 3, PMMG_ARG_MPIComm, tmpComm, PMMG_ARG_end);
  ierr = PMMG_Set_meshSize(parmesh, numVertices, numCells, 0, numBdFaces, 0, 0);
  ierr = PMMG_Set_iparameter(parmesh, PMMG_IPARAM_APImode, PMMG_APIDISTRIB_nodes);
  ierr = PMMG_Set_iparameter(parmesh, PMMG_IPARAM_noinsert, noInsert);
  ierr = PMMG_Set_iparameter(parmesh, PMMG_IPARAM_noswap, noSwap);
  ierr = PMMG_Set_iparameter(parmesh, PMMG_IPARAM_nomove, noMove);
  ierr = PMMG_Set_iparameter(parmesh, PMMG_IPARAM_verbose, verbosity);
  ierr = PMMG_Set_iparameter(parmesh, PMMG_IPARAM_globalNum, 1);
  ierr = PMMG_Set_iparameter(parmesh, PMMG_IPARAM_niter, numIter);
  ierr = PMMG_Set_dparameter(parmesh, PMMG_DPARAM_hgrad, gradationFactor);
  ierr = PMMG_Set_vertices(parmesh, vertices, verTags);
  for (i=0; i<numCells; i++) ierr = PMMG_Set_tetrahedron(parmesh, cells[4*i+0], cells[4*i+1], cells[4*i+2], cells[4*i+3], 0, i+1);
  ierr = PMMG_Set_triangles(parmesh, bdFaces, bdFaceIds);
  ierr = PMMG_Set_metSize(parmesh, MMG5_Vertex, numVertices, MMG5_Tensor);
  for (i=0; i<numVertices; i++) {
    PMMG_Set_tensorMet(parmesh, metric[6*i], metric[6*i+1], metric[6*i+2], metric[6*i+3], metric[6*i+4], metric[6*i+5], i+1);
  }
  ierr = PMMG_Set_numberOfNodeCommunicators(parmesh, numNgbRanks);
  for (c = 0; c < numNgbRanks; ++c) {
    ierr = PMMG_Set_ithNodeCommunicatorSize(parmesh, c, ngbRanks[c], intOffset[c+1]-intOffset[c]);
    ierr = PMMG_Set_ithNodeCommunicator_nodes(parmesh, c, &interfaces_lv[intOffset[c]], &interfaces_gv[intOffset[c]], 1);
  }
  ierr = PMMG_parmmglib_distributed(parmesh);
  ierr = PetscFree(cells);CHKERRQ(ierr);
  ierr = PetscFree2(metric, vertices);CHKERRQ(ierr);
  ierr = PetscFree2(bdFaces, bdFaceIds);CHKERRQ(ierr);
  ierr = PetscFree2(verTags, cellTags);CHKERRQ(ierr);
  if (numProcs > 1) {
    ierr = PetscFree2(ngbRanks, verNgbRank);CHKERRQ(ierr);
    ierr = PetscFree3(interfaces_lv, interfaces_gv, intOffset);CHKERRQ(ierr);
  }

  /* Retrieve mesh from Mmg and create new Plex */
  numCornersNew = 4;
  ierr = PMMG_Get_meshSize(parmesh, &numVerticesNew, &numCellsNew, 0, &numFacesNew, 0, 0);
  ierr = PetscMalloc4(dim*numVerticesNew, &verticesNew, numVerticesNew, &verTagsNew, numVerticesNew, &corners, numVerticesNew, &requiredVer);CHKERRQ(ierr);
  ierr = PetscMalloc3((dim+1)*numCellsNew, &cellsNew, numCellsNew, &cellTagsNew, numCellsNew, &requiredCells);CHKERRQ(ierr);
  ierr = PetscMalloc4(dim*numFacesNew, &facesNew, numFacesNew, &faceTagsNew, numFacesNew, &ridges, numFacesNew, &requiredFaces);CHKERRQ(ierr);
  ierr = PMMG_Get_vertices(parmesh, verticesNew, verTagsNew, corners, requiredVer);
  ierr = PMMG_Get_tetrahedra(parmesh, cellsNew, cellTagsNew, requiredCells);
  ierr = PMMG_Get_triangles(parmesh, facesNew, faceTagsNew, requiredFaces);
  ierr = PetscMalloc2(numVerticesNew, &owners, numVerticesNew, &gv_new);
  ierr = PMMG_Set_iparameter(parmesh, PMMG_IPARAM_globalNum, 1);
  ierr = PMMG_Get_verticesGloNum(parmesh, gv_new, owners);
  for (i = 0; i < dim*numFacesNew; ++i) facesNew[i] -= 1;
  for (i = 0; i < (dim+1)*numCellsNew; ++i) {
    cellsNew[i] = gv_new[cellsNew[i]-1]-1;
  }
  numVerticesNewLoc = 0;
  for (i = 0; i < numVerticesNew; ++i) {
    if (owners[i] == rank) numVerticesNewLoc++;
  }
  ierr = PetscMalloc2(numVerticesNewLoc*dim, &verticesNewLoc, numVerticesNew, &verticesNewSorted);CHKERRQ(ierr);
  for (i = 0, c = 0; i < numVerticesNew; i++) {
    if (owners[i] == rank) {
      for (j=0; j<dim; ++j) verticesNewLoc[dim*c+j] = verticesNew[dim*i+j];
        c++;
    }
  }
  ierr = DMPlexCreateFromCellListParallelPetsc(comm, dim, numCellsNew, numVerticesNewLoc, PETSC_DECIDE, numCornersNew, PETSC_TRUE, cellsNew, dim, verticesNewLoc, NULL, &verticesNewSorted, dmNew);CHKERRQ(ierr);
  ierr = PMMG_Free_all(PMMG_ARG_start, PMMG_ARG_ppParMesh, &parmesh, PMMG_ARG_end);

  /* Rebuild boundary label*/
  ierr = DMPlexGetHeightStratum(*dmNew, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dmNew, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(*dmNew, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMCreateLabel(*dmNew, bdLabel ? bdLabelName : bdName);CHKERRQ(ierr);
  ierr = DMGetLabel(*dmNew, bdLabel ? bdLabelName : bdName, &bdLabelNew);CHKERRQ(ierr);
  for (i = 0; i < numFacesNew; i++) {
    PetscInt        numCoveredPoints, numFaces = 0, facePoints[3];
    const PetscInt *coveredPoints = NULL;

    for (j = 0; j < dim; ++j) {
      lv = facesNew[i*dim+j];
      gv = gv_new[lv]-1;
      ierr = PetscFindInt(gv, numVerticesNew, verticesNewSorted, &lv);CHKERRQ(ierr);
      facePoints[j] = lv+vStart;
    }
    ierr = DMPlexGetFullJoin(*dmNew, dim, facePoints, &numCoveredPoints, &coveredPoints);CHKERRQ(ierr);
    for (j = 0; j < numCoveredPoints; ++j) {
      if (coveredPoints[j] >= fStart && coveredPoints[j] < fEnd) {
        numFaces++;
        f = j;
      }
    }
    if (numFaces != 1) SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "%d vertices cannot define more than 1 facet (%d)", dim, numFaces);
    ierr = DMLabelSetValue(bdLabelNew, coveredPoints[f], faceTagsNew[i]);CHKERRQ(ierr);
    ierr = DMPlexRestoreJoin(*dmNew, dim, facePoints, &numCoveredPoints, &coveredPoints);CHKERRQ(ierr);
  }

  /* Clean up */
  ierr = PetscFree4(verticesNew, verTagsNew, corners, requiredVer);CHKERRQ(ierr);
  ierr = PetscFree3(cellsNew, cellTagsNew, requiredCells);CHKERRQ(ierr);
  ierr = PetscFree4(facesNew, faceTagsNew, ridges, requiredFaces);CHKERRQ(ierr);
  ierr = PetscFree2(owners, gv_new);CHKERRQ(ierr);
  ierr = PetscFree2(verticesNewLoc, verticesNewSorted);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
