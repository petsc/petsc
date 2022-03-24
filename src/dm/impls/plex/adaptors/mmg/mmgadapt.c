#include "../mmgcommon.h"   /*I      "petscdmplex.h"   I*/
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
  PetscReal         *vertices, *metric, *verticesNew, gradationFactor, hausdorffNumber;
  PetscInt          *cells, *cellsNew, *cellTags, *cellTagsNew, *verTags, *verTagsNew;
  PetscInt          *bdFaces, *faceTags, *facesNew, *faceTagsNew;
  PetscInt          *corners, *requiredCells, *requiredVer, *ridges, *requiredFaces;
  PetscInt           cStart, cEnd, c, numCells, fStart, fEnd, numFaceTags, f, vStart, vEnd, v, numVertices;
  PetscInt           dim, off, coff, maxConeSize, bdSize, i, j, k, Neq, verbosity, pStart, pEnd;
  PetscInt           numCellsNew, numVerticesNew, numCornersNew, numFacesNew;
  PetscBool          flg = PETSC_FALSE, noInsert, noSwap, noMove, noSurf, isotropic, uniform;
  MMG5_pMesh         mmg_mesh = NULL;
  MMG5_pSol          mmg_metric = NULL;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  if (bdLabel) {
    CHKERRQ(PetscObjectGetName((PetscObject) bdLabel, &bdLabelName));
    CHKERRQ(PetscStrcmp(bdLabelName, bdName, &flg));
    PetscCheck(!flg,comm, PETSC_ERR_ARG_WRONG, "\"%s\" cannot be used as label for boundary facets", bdLabelName);
  }
  if (rgLabel) {
    CHKERRQ(PetscObjectGetName((PetscObject) rgLabel, &rgLabelName));
    CHKERRQ(PetscStrcmp(rgLabelName, rgName, &flg));
    PetscCheck(!flg,comm, PETSC_ERR_ARG_WRONG, "\"%s\" cannot be used as label for element tags", rgLabelName);
  }

  /* Get mesh information */
  CHKERRQ(DMGetDimension(dm, &dim));
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
    for (cl = 0; cl < coneSize; ++cl) cells[coff++] = cone[cl] - vStart + 1;
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
  CHKERRQ(DMDestroy(&udm));

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
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) bdFaces[bdSize++] = closure[cl] - vStart + 1;
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure));
    CHKERRQ(DMLabelGetValue(bdLabel, f, &faceTags[numFaceTags++]));
  }
  if (flg) CHKERRQ(DMLabelDestroy(&bdLabel));

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
      for (j = i; j < dim; ++j) {
        if (isotropic) {
          if (i == j) {
            if (uniform) metric[Neq*v+k] = PetscRealPart(met[0]);
            else metric[Neq*v+k] = PetscRealPart(met[v]);
          } else metric[Neq*v+k] = 0.0;
        } else {
          metric[Neq*v+k] = PetscRealPart(met[dim*dim*v+dim*i+j]);
        }
        k++;
      }
    }
  }
  CHKERRQ(VecRestoreArrayRead(vertexMetric, &met));

  /* Send mesh to Mmg and remesh */
  CHKERRQ(DMPlexMetricGetVerbosity(dm, &verbosity));
  CHKERRQ(DMPlexMetricGetGradationFactor(dm, &gradationFactor));
  CHKERRQ(DMPlexMetricGetHausdorffNumber(dm, &hausdorffNumber));
  CHKERRQ(DMPlexMetricNoInsertion(dm, &noInsert));
  CHKERRQ(DMPlexMetricNoSwapping(dm, &noSwap));
  CHKERRQ(DMPlexMetricNoMovement(dm, &noMove));
  CHKERRQ(DMPlexMetricNoSurf(dm, &noSurf));
  switch (dim) {
  case 2:
    CHKERRMMG_NONSTANDARD(MMG2D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end));
    CHKERRMMG_NONSTANDARD(MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_noinsert, noInsert));
    CHKERRMMG_NONSTANDARD(MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_noswap, noSwap));
    CHKERRMMG_NONSTANDARD(MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_nomove, noMove));
    CHKERRMMG_NONSTANDARD(MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_nosurf, noSurf));
    CHKERRMMG_NONSTANDARD(MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_verbose, verbosity));
    CHKERRMMG_NONSTANDARD(MMG2D_Set_dparameter(mmg_mesh, mmg_metric, MMG2D_DPARAM_hgrad, gradationFactor));
    CHKERRMMG_NONSTANDARD(MMG2D_Set_dparameter(mmg_mesh, mmg_metric, MMG2D_DPARAM_hausd, hausdorffNumber));
    CHKERRMMG_NONSTANDARD(MMG2D_Set_meshSize(mmg_mesh, numVertices, numCells, 0, numFaceTags));
    CHKERRMMG_NONSTANDARD(MMG2D_Set_vertices(mmg_mesh, vertices, verTags));
    CHKERRMMG_NONSTANDARD(MMG2D_Set_triangles(mmg_mesh, cells, cellTags));
    CHKERRMMG_NONSTANDARD(MMG2D_Set_edges(mmg_mesh, bdFaces, faceTags));
    CHKERRMMG_NONSTANDARD(MMG2D_Set_solSize(mmg_mesh, mmg_metric, MMG5_Vertex, numVertices, MMG5_Tensor));
    CHKERRMMG_NONSTANDARD(MMG2D_Set_tensorSols(mmg_metric, metric));
    CHKERRMMG(MMG2D_mmg2dlib(mmg_mesh, mmg_metric));
    break;
  case 3:
    CHKERRMMG_NONSTANDARD(MMG3D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end));
    CHKERRMMG_NONSTANDARD(MMG3D_Set_iparameter(mmg_mesh, mmg_metric, MMG3D_IPARAM_noinsert, noInsert));
    CHKERRMMG_NONSTANDARD(MMG3D_Set_iparameter(mmg_mesh, mmg_metric, MMG3D_IPARAM_noswap, noSwap));
    CHKERRMMG_NONSTANDARD(MMG3D_Set_iparameter(mmg_mesh, mmg_metric, MMG3D_IPARAM_nomove, noMove));
    CHKERRMMG_NONSTANDARD(MMG3D_Set_iparameter(mmg_mesh, mmg_metric, MMG3D_IPARAM_nosurf, noSurf));
    CHKERRMMG_NONSTANDARD(MMG3D_Set_iparameter(mmg_mesh, mmg_metric, MMG3D_IPARAM_verbose, verbosity));
    CHKERRMMG_NONSTANDARD(MMG3D_Set_dparameter(mmg_mesh, mmg_metric, MMG3D_DPARAM_hgrad, gradationFactor));
    CHKERRMMG_NONSTANDARD(MMG3D_Set_dparameter(mmg_mesh, mmg_metric, MMG2D_DPARAM_hausd, hausdorffNumber));
    CHKERRMMG_NONSTANDARD(MMG3D_Set_meshSize(mmg_mesh, numVertices, numCells, 0, numFaceTags, 0, 0));
    CHKERRMMG_NONSTANDARD(MMG3D_Set_vertices(mmg_mesh, vertices, verTags));
    CHKERRMMG_NONSTANDARD(MMG3D_Set_tetrahedra(mmg_mesh, cells, cellTags));
    CHKERRMMG_NONSTANDARD(MMG3D_Set_triangles(mmg_mesh, bdFaces, faceTags));
    CHKERRMMG_NONSTANDARD(MMG3D_Set_solSize(mmg_mesh, mmg_metric, MMG5_Vertex, numVertices, MMG5_Tensor));
    CHKERRMMG_NONSTANDARD(MMG3D_Set_tensorSols(mmg_metric, metric));
    CHKERRMMG(MMG3D_mmg3dlib(mmg_mesh, mmg_metric));
    break;
  default: SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Mmg adaptation defined for dimension %" PetscInt_FMT, dim);
  }
  CHKERRQ(PetscFree(cells));
  CHKERRQ(PetscFree2(metric, vertices));
  CHKERRQ(PetscFree2(bdFaces, faceTags));
  CHKERRQ(PetscFree2(verTags, cellTags));

  /* Retrieve mesh from Mmg */
  switch (dim) {
  case 2:
    numCornersNew = 3;
    CHKERRMMG_NONSTANDARD(MMG2D_Get_meshSize(mmg_mesh, &numVerticesNew, &numCellsNew, 0, &numFacesNew));
    CHKERRQ(PetscMalloc4(2*numVerticesNew, &verticesNew, numVerticesNew, &verTagsNew, numVerticesNew, &corners, numVerticesNew, &requiredVer));
    CHKERRQ(PetscMalloc3(3*numCellsNew, &cellsNew, numCellsNew, &cellTagsNew, numCellsNew, &requiredCells));
    CHKERRQ(PetscMalloc4(2*numFacesNew, &facesNew, numFacesNew, &faceTagsNew, numFacesNew, &ridges, numFacesNew, &requiredFaces));
    CHKERRMMG_NONSTANDARD(MMG2D_Get_vertices(mmg_mesh, verticesNew, verTagsNew, corners, requiredVer));
    CHKERRMMG_NONSTANDARD(MMG2D_Get_triangles(mmg_mesh, cellsNew, cellTagsNew, requiredCells));
    CHKERRMMG_NONSTANDARD(MMG2D_Get_edges(mmg_mesh, facesNew, faceTagsNew, ridges, requiredFaces));
    break;
  case 3:
    numCornersNew = 4;
    CHKERRMMG_NONSTANDARD(MMG3D_Get_meshSize(mmg_mesh, &numVerticesNew, &numCellsNew, 0, &numFacesNew, 0, 0));
    CHKERRQ(PetscMalloc4(3*numVerticesNew, &verticesNew, numVerticesNew, &verTagsNew, numVerticesNew, &corners, numVerticesNew, &requiredVer));
    CHKERRQ(PetscMalloc3(4*numCellsNew, &cellsNew, numCellsNew, &cellTagsNew, numCellsNew, &requiredCells));
    CHKERRQ(PetscMalloc4(3*numFacesNew, &facesNew, numFacesNew, &faceTagsNew, numFacesNew, &ridges, numFacesNew, &requiredFaces));
    CHKERRMMG_NONSTANDARD(MMG3D_Get_vertices(mmg_mesh, verticesNew, verTagsNew, corners, requiredVer));
    CHKERRMMG_NONSTANDARD(MMG3D_Get_tetrahedra(mmg_mesh, cellsNew, cellTagsNew, requiredCells));
    CHKERRMMG_NONSTANDARD(MMG3D_Get_triangles(mmg_mesh, facesNew, faceTagsNew, requiredFaces));

    /* Reorder for consistency with DMPlex */
    for (i = 0; i < numCellsNew; ++i) CHKERRQ(DMPlexInvertCell(DM_POLYTOPE_TETRAHEDRON, &cellsNew[4*i]));
    break;

  default: SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Mmg adaptation defined for dimension %" PetscInt_FMT, dim);
  }

  /* Create new Plex */
  for (i = 0; i < (dim+1)*numCellsNew; i++) cellsNew[i] -= 1;
  for (i = 0; i < dim*numFacesNew; i++) facesNew[i] -= 1;
  CHKERRQ(DMPlexCreateFromCellListParallelPetsc(comm, dim, numCellsNew, numVerticesNew, PETSC_DECIDE, numCornersNew, PETSC_TRUE, cellsNew, dim, verticesNew, NULL, NULL, dmNew));
  switch (dim) {
  case 2:
    CHKERRMMG_NONSTANDARD(MMG2D_Free_all(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end));
    break;
  case 3:
    CHKERRMMG_NONSTANDARD(MMG3D_Free_all(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end));
    break;
  default: SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Mmg adaptation defined for dimension %" PetscInt_FMT, dim);
  }
  CHKERRQ(PetscFree4(verticesNew, verTagsNew, corners, requiredVer));

  /* Get adapted mesh information */
  CHKERRQ(DMPlexGetHeightStratum(*dmNew, 0, &cStart, &cEnd));
  CHKERRQ(DMPlexGetHeightStratum(*dmNew, 1, &fStart, &fEnd));
  CHKERRQ(DMPlexGetDepthStratum(*dmNew, 0, &vStart, &vEnd));

  /* Rebuild boundary labels */
  CHKERRQ(DMCreateLabel(*dmNew, flg ? bdName : bdLabelName));
  CHKERRQ(DMGetLabel(*dmNew, flg ? bdName : bdLabelName, &bdLabelNew));
  for (i = 0; i < numFacesNew; i++) {
    PetscInt        numCoveredPoints, numFaces = 0, facePoints[3];
    const PetscInt *coveredPoints = NULL;

    for (j = 0; j < dim; ++j) facePoints[j] = facesNew[i*dim+j]+vStart;
    CHKERRQ(DMPlexGetFullJoin(*dmNew, dim, facePoints, &numCoveredPoints, &coveredPoints));
    for (j = 0; j < numCoveredPoints; ++j) {
      if (coveredPoints[j] >= fStart && coveredPoints[j] < fEnd) {
        numFaces++;
        f = j;
      }
    }
    PetscCheck(numFaces == 1,comm, PETSC_ERR_ARG_OUTOFRANGE, "%" PetscInt_FMT " vertices cannot define more than 1 facet (%" PetscInt_FMT ")", dim, numFaces);
    CHKERRQ(DMLabelSetValue(bdLabelNew, coveredPoints[f], faceTagsNew[i]));
    CHKERRQ(DMPlexRestoreJoin(*dmNew, dim, facePoints, &numCoveredPoints, &coveredPoints));
  }
  CHKERRQ(PetscFree4(facesNew, faceTagsNew, ridges, requiredFaces));

  /* Rebuild cell labels */
  CHKERRQ(DMCreateLabel(*dmNew, rgLabel ? rgLabelName : rgName));
  CHKERRQ(DMGetLabel(*dmNew, rgLabel ? rgLabelName : rgName, &rgLabelNew));
  for (c = cStart; c < cEnd; ++c) CHKERRQ(DMLabelSetValue(rgLabelNew, c, cellTagsNew[c-cStart]));
  CHKERRQ(PetscFree3(cellsNew, cellTagsNew, requiredCells));

  PetscFunctionReturn(0);
}
