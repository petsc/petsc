#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <mmg/libmmg.h>

PETSC_EXTERN PetscErrorCode DMAdaptMetric_Mmg_Plex(DM dm, Vec vertexMetric, DMLabel bdLabel, DM *dmNew)
{
  MPI_Comm           comm;
  const char        *bdName = "_boundary_";
  DM                 udm, cdm;
  DMLabel            bdLabelFull;
  const char        *bdLabelName;
  IS                 bdIS;
  PetscSection       coordSection;
  Vec                coordinates;
  const PetscScalar *coords, *met;
  const PetscInt    *bdFacesFull;
  PetscInt          *bdFaces, *bdFaceIds;
  PetscReal         *vertices, *metric, *verticesNew, gradationFactor;
  PetscInt          *cells, *cellsNew;
  PetscInt          *facesNew, *faceTagsNew;
  PetscInt          *verTags, *cellTags, *verTagsNew, *cellTagsNew;
  PetscInt          *corners, *requiredCells, *requiredVer, *ridges, *requiredFaces;
  PetscInt           dim, cStart, cEnd, numCells, c, coff, vStart, vEnd, numVertices, v;
  PetscInt           off, maxConeSize, numBdFaces, f, bdSize, fStart, fEnd, i, j, k, Neq;
  PetscInt           numCellsNew, numVerticesNew, numCornersNew, numFacesNew;
  PetscInt           verbosity;
  PetscBool          flg, noInsert, noSwap, noMove;
  DMLabel            bdLabelNew;
  MMG5_pMesh         mmg_mesh = NULL;
  MMG5_pSol          mmg_metric = NULL;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  if (bdLabel) {
    ierr = PetscObjectGetName((PetscObject) bdLabel, &bdLabelName);CHKERRQ(ierr);
    ierr = PetscStrcmp(bdLabelName, bdName, &flg);CHKERRQ(ierr);
    if (flg) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "\"%s\" cannot be used as label for boundary facets", bdLabelName);
  }

  /* Get mesh information */
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
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
    for (cl = 0; cl < coneSize; ++cl) cells[coff++] = cone[cl] - vStart + 1;
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
  ierr = DMDestroy(&udm);CHKERRQ(ierr);

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
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) bdFaces[bdSize++] = closure[cl] - vStart + 1;
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

  /* Send mesh to Mmg and remesh */
  ierr = DMPlexMetricGetVerbosity(dm, &verbosity);CHKERRQ(ierr);
  ierr = DMPlexMetricGetGradationFactor(dm, &gradationFactor);CHKERRQ(ierr);
  ierr = DMPlexMetricNoInsertion(dm, &noInsert);CHKERRQ(ierr);
  ierr = DMPlexMetricNoSwapping(dm, &noSwap);CHKERRQ(ierr);
  ierr = DMPlexMetricNoMovement(dm, &noMove);CHKERRQ(ierr);
  ierr = PetscCalloc2(numVertices, &verTags, numCells, &cellTags);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    ierr = MMG2D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end);
    ierr = MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_noinsert, noInsert);
    ierr = MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_noswap, noSwap);
    ierr = MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_nomove, noMove);
    ierr = MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_verbose, verbosity);
    ierr = MMG2D_Set_dparameter(mmg_mesh, mmg_metric, MMG2D_DPARAM_hgrad, gradationFactor);
    ierr = MMG2D_Set_meshSize(mmg_mesh, numVertices, numCells, 0, numBdFaces);
    ierr = MMG2D_Set_vertices(mmg_mesh, vertices, verTags);
    ierr = MMG2D_Set_triangles(mmg_mesh, cells, cellTags);
    ierr = MMG2D_Set_edges(mmg_mesh, bdFaces, bdFaceIds);
    ierr = MMG2D_Set_solSize(mmg_mesh, mmg_metric, MMG5_Vertex, numVertices, MMG5_Tensor);
    ierr = MMG2D_Set_tensorSols(mmg_metric, metric);
    ierr = MMG2D_mmg2dlib(mmg_mesh, mmg_metric);
    break;
  case 3:
    ierr = MMG3D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end);
    ierr = MMG3D_Set_iparameter(mmg_mesh, mmg_metric, MMG3D_IPARAM_noinsert, noInsert);
    ierr = MMG3D_Set_iparameter(mmg_mesh, mmg_metric, MMG3D_IPARAM_noswap, noSwap);
    ierr = MMG3D_Set_iparameter(mmg_mesh, mmg_metric, MMG3D_IPARAM_nomove, noMove);
    ierr = MMG3D_Set_iparameter(mmg_mesh, mmg_metric, MMG3D_IPARAM_verbose, verbosity);
    ierr = MMG3D_Set_dparameter(mmg_mesh, mmg_metric, MMG3D_DPARAM_hgrad, gradationFactor);
    ierr = MMG3D_Set_meshSize(mmg_mesh, numVertices, numCells, 0, numBdFaces, 0, 0);
    ierr = MMG3D_Set_vertices(mmg_mesh, vertices, verTags);
    ierr = MMG3D_Set_tetrahedra(mmg_mesh, cells, cellTags);
    ierr = MMG3D_Set_triangles(mmg_mesh, bdFaces, bdFaceIds);
    ierr = MMG3D_Set_solSize(mmg_mesh, mmg_metric, MMG5_Vertex, numVertices, MMG5_Tensor);
    ierr = MMG3D_Set_tensorSols(mmg_metric, metric);
    ierr = MMG3D_mmg3dlib(mmg_mesh, mmg_metric);
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Mmg adaptation defined for dimension %D", dim);
  }

  /* Retrieve mesh from Mmg and create new Plex*/
  switch (dim) {
  case 2:
    numCornersNew = 3;
    ierr = MMG2D_Get_meshSize(mmg_mesh, &numVerticesNew, &numCellsNew, 0, &numFacesNew);
    ierr = PetscMalloc4(2*numVerticesNew, &verticesNew, numVerticesNew, &verTagsNew, numVerticesNew, &corners, numVerticesNew, &requiredVer);CHKERRQ(ierr);
    ierr = PetscMalloc3(3*numCellsNew, &cellsNew, numCellsNew, &cellTagsNew, numCellsNew, &requiredCells);CHKERRQ(ierr);
    ierr = PetscMalloc4(2*numFacesNew, &facesNew, numFacesNew, &faceTagsNew, numFacesNew, &ridges, numFacesNew, &requiredFaces);CHKERRQ(ierr);
    ierr = MMG2D_Get_vertices(mmg_mesh, verticesNew, verTagsNew, corners, requiredVer);
    ierr = MMG2D_Get_triangles(mmg_mesh, cellsNew, cellTagsNew, requiredCells);
    ierr = MMG2D_Get_edges(mmg_mesh, facesNew, faceTagsNew, ridges, requiredFaces);
    break;
  case 3:
    numCornersNew = 4;
    ierr = MMG3D_Get_meshSize(mmg_mesh, &numVerticesNew, &numCellsNew, 0, &numFacesNew, 0, 0);
    ierr = PetscMalloc4(3*numVerticesNew, &verticesNew, numVerticesNew, &verTagsNew, numVerticesNew, &corners, numVerticesNew, &requiredVer);CHKERRQ(ierr);
    ierr = PetscMalloc3(4*numCellsNew, &cellsNew, numCellsNew, &cellTagsNew, numCellsNew, &requiredCells);CHKERRQ(ierr);
    ierr = PetscMalloc4(3*numFacesNew, &facesNew, numFacesNew, &faceTagsNew, numFacesNew, &ridges, numFacesNew, &requiredFaces);CHKERRQ(ierr);
    ierr = MMG3D_Get_vertices(mmg_mesh, verticesNew, verTagsNew, corners, requiredVer);
    ierr = MMG3D_Get_tetrahedra(mmg_mesh, cellsNew, cellTagsNew, requiredCells);
    ierr = MMG3D_Get_triangles(mmg_mesh, facesNew, faceTagsNew, requiredFaces);
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Mmg adaptation defined for dimension %D", dim);
  }
  for (i = 0; i < (dim+1)*numCellsNew; i++) cellsNew[i] -= 1;
  for (i = 0; i < dim*numFacesNew; i++) facesNew[i] -= 1;
  ierr = DMPlexCreateFromCellListParallelPetsc(comm, dim, numCellsNew, numVerticesNew, PETSC_DECIDE, numCornersNew, PETSC_TRUE, cellsNew, dim, verticesNew, NULL, NULL, dmNew);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    ierr = MMG2D_Free_all(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end);
    break;
  case 3:
    ierr = MMG3D_Free_all(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end);
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Mmg adaptation defined for dimension %D", dim);
  }

  /* Rebuild boundary labels */
  ierr = DMCreateLabel(*dmNew, bdLabel ? bdLabelName : bdName);CHKERRQ(ierr);
  ierr = DMGetLabel(*dmNew, bdLabel ? bdLabelName : bdName, &bdLabelNew);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dmNew, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(*dmNew, 0, &vStart, &vEnd);CHKERRQ(ierr);
  for (i = 0; i < numFacesNew; i++) {
    PetscInt        numCoveredPoints, numFaces = 0, facePoints[3];
    const PetscInt *coveredPoints = NULL;

    for (j = 0; j < dim; ++j) facePoints[j] = facesNew[i*dim+j]+vStart;
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
  ierr = PetscFree(cells);CHKERRQ(ierr);
  ierr = PetscFree2(metric, vertices);CHKERRQ(ierr);
  ierr = PetscFree2(bdFaces, bdFaceIds);CHKERRQ(ierr);
  ierr = PetscFree2(verTags, cellTags);CHKERRQ(ierr);
  ierr = PetscFree4(verticesNew, verTagsNew, corners, requiredVer);CHKERRQ(ierr);
  ierr = PetscFree3(cellsNew, cellTagsNew, requiredCells);CHKERRQ(ierr);
  ierr = PetscFree4(facesNew, faceTagsNew, ridges, requiredFaces);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
