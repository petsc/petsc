#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <mmg/libmmg.h>

PETSC_EXTERN PetscErrorCode DMAdaptMetric_Mmg_Plex(DM dm, Vec vertexMetric, DMLabel bdLabel, DMLabel rgLabel, DM *dmNew)
{
  MPI_Comm           comm;
  const char        *bdName = "_boundary_";
  const char        *rgName = "_regions_";
  DM                 udm, cdm;
  DMLabel            bdLabelNew, rgLabelNew;
  const char        *bdLabelName, *rgLabelName;
  PetscSection       coordSection;
  Vec                coordinates;
  const PetscScalar *coords, *met;
  PetscReal         *vertices, *metric, *verticesNew, gradationFactor;
  PetscInt          *cells, *cellsNew, *cellTags, *cellTagsNew, *verTags, *verTagsNew;
  PetscInt          *bdFaces, *faceTags, *facesNew, *faceTagsNew;
  PetscInt          *corners, *requiredCells, *requiredVer, *ridges, *requiredFaces;
  PetscInt           cStart, cEnd, c, numCells, fStart, fEnd, numFaceTags, f, vStart, vEnd, v, numVertices;
  PetscInt           dim, off, coff, maxConeSize, bdSize, i, j, k, Neq, verbosity, pStart, pEnd;
  PetscInt           numCellsNew, numVerticesNew, numCornersNew, numFacesNew;
  PetscBool          flg = PETSC_FALSE, noInsert, noSwap, noMove;
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
  if (rgLabel) {
    ierr = PetscObjectGetName((PetscObject) rgLabel, &rgLabelName);CHKERRQ(ierr);
    ierr = PetscStrcmp(rgLabelName, rgName, &flg);CHKERRQ(ierr);
    if (flg) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "\"%s\" cannot be used as label for element tags", rgLabelName);
  }

  /* Get mesh information */
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  Neq  = (dim*(dim+1))/2;
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
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

  /* Get face tags */
  if (!bdLabel) {
    flg = PETSC_TRUE;
    ierr = DMLabelCreate(PETSC_COMM_SELF, bdName, &bdLabel);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(dm, 1, bdLabel);CHKERRQ(ierr);
  }
  ierr = DMLabelGetBounds(bdLabel, &pStart, &pEnd);CHKERRQ(ierr);
  for (f = pStart, bdSize = 0, numFaceTags = 0; f < pEnd; ++f) {
    PetscBool hasPoint;
    PetscInt *closure = NULL, closureSize, cl;

    ierr = DMLabelHasPoint(bdLabel, f, &hasPoint);CHKERRQ(ierr);
    if ((!hasPoint) || (f < fStart) || (f >= fEnd)) continue;
    numFaceTags++;

    ierr = DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) ++bdSize;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
  }
  ierr = PetscMalloc2(bdSize, &bdFaces, numFaceTags, &faceTags);CHKERRQ(ierr);
  for (f = pStart, bdSize = 0, numFaceTags = 0; f < pEnd; ++f) {
    PetscBool hasPoint;
    PetscInt *closure = NULL, closureSize, cl;

    ierr = DMLabelHasPoint(bdLabel, f, &hasPoint);CHKERRQ(ierr);
    if ((!hasPoint) || (f < fStart) || (f >= fEnd)) continue;

    ierr = DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) bdFaces[bdSize++] = closure[cl] - vStart + 1;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    ierr = DMLabelGetValue(bdLabel, f, &faceTags[numFaceTags++]);CHKERRQ(ierr);
  }
  if (flg) { ierr = DMLabelDestroy(&bdLabel);CHKERRQ(ierr); }

  /* Get cell tags */
  ierr = PetscCalloc2(numVertices, &verTags, numCells, &cellTags);CHKERRQ(ierr);
  if (rgLabel) {
    for (c = cStart; c < cEnd; ++c) { ierr = DMLabelGetValue(rgLabel, c, &cellTags[c]);CHKERRQ(ierr); }
  }

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
  switch (dim) {
  case 2:
    ierr = MMG2D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end);
    ierr = MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_noinsert, noInsert);
    ierr = MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_noswap, noSwap);
    ierr = MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_nomove, noMove);
    ierr = MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_verbose, verbosity);
    ierr = MMG2D_Set_dparameter(mmg_mesh, mmg_metric, MMG2D_DPARAM_hgrad, gradationFactor);
    ierr = MMG2D_Set_meshSize(mmg_mesh, numVertices, numCells, 0, numFaceTags);
    ierr = MMG2D_Set_vertices(mmg_mesh, vertices, verTags);
    ierr = MMG2D_Set_triangles(mmg_mesh, cells, cellTags);
    ierr = MMG2D_Set_edges(mmg_mesh, bdFaces, faceTags);
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
    ierr = MMG3D_Set_meshSize(mmg_mesh, numVertices, numCells, 0, numFaceTags, 0, 0);
    ierr = MMG3D_Set_vertices(mmg_mesh, vertices, verTags);
    ierr = MMG3D_Set_tetrahedra(mmg_mesh, cells, cellTags);
    ierr = MMG3D_Set_triangles(mmg_mesh, bdFaces, faceTags);
    ierr = MMG3D_Set_solSize(mmg_mesh, mmg_metric, MMG5_Vertex, numVertices, MMG5_Tensor);
    ierr = MMG3D_Set_tensorSols(mmg_metric, metric);
    ierr = MMG3D_mmg3dlib(mmg_mesh, mmg_metric);
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Mmg adaptation defined for dimension %D", dim);
  }
  ierr = PetscFree(cells);CHKERRQ(ierr);
  ierr = PetscFree2(metric, vertices);CHKERRQ(ierr);
  ierr = PetscFree2(bdFaces, faceTags);CHKERRQ(ierr);
  ierr = PetscFree2(verTags, cellTags);CHKERRQ(ierr);

  /* Retrieve mesh from Mmg */
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

    /* Reorder for consistency with DMPlex */
    for (i = 0; i < numCellsNew; ++i) { ierr = DMPlexInvertCell(DM_POLYTOPE_TETRAHEDRON, &cellsNew[4*i]);CHKERRQ(ierr); }
    break;

  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Mmg adaptation defined for dimension %D", dim);
  }

  /* Create new Plex */
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
  ierr = PetscFree4(verticesNew, verTagsNew, corners, requiredVer);CHKERRQ(ierr);

  /* Get adapted mesh information */
  ierr = DMPlexGetHeightStratum(*dmNew, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dmNew, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(*dmNew, 0, &vStart, &vEnd);CHKERRQ(ierr);

  /* Rebuild boundary labels */
  ierr = DMCreateLabel(*dmNew, bdLabel ? bdLabelName : bdName);CHKERRQ(ierr);
  ierr = DMGetLabel(*dmNew, bdLabel ? bdLabelName : bdName, &bdLabelNew);CHKERRQ(ierr);
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
  ierr = PetscFree4(facesNew, faceTagsNew, ridges, requiredFaces);CHKERRQ(ierr);

  /* Rebuild cell labels */
  ierr = DMCreateLabel(*dmNew, rgLabel ? rgLabelName : rgName);CHKERRQ(ierr);
  ierr = DMGetLabel(*dmNew, rgLabel ? rgLabelName : rgName, &rgLabelNew);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) { ierr = DMLabelSetValue(rgLabelNew, c, cellTagsNew[c-cStart]);CHKERRQ(ierr); }
  ierr = PetscFree3(cellsNew, cellTagsNew, requiredCells);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
