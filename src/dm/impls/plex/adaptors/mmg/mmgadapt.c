#include "../mmgcommon.h" /*I      "petscdmplex.h"   I*/
#include <mmg/libmmg.h>

PetscBool  MmgCite       = PETSC_FALSE;
const char MmgCitation[] = "@article{DAPOGNY2014358,\n"
                           "  title   = {Three-dimensional adaptive domain remeshing, implicit domain meshing, and applications to free and moving boundary problems},\n"
                           "  journal = {Journal of Computational Physics},\n"
                           "  author  = {C. Dapogny and C. Dobrzynski and P. Frey},\n"
                           "  volume  = {262},\n"
                           "  pages   = {358--378},\n"
                           "  doi     = {10.1016/j.jcp.2014.01.005},\n"
                           "  year    = {2014}\n}\n";

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
  PetscBool          flg        = PETSC_FALSE, noInsert, noSwap, noMove, noSurf, isotropic, uniform;
  MMG5_pMesh         mmg_mesh   = NULL;
  MMG5_pSol          mmg_metric = NULL;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(MmgCitation, &MmgCite));
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
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
  PetscCall(DMDestroy(&udm));

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
  if (flg) PetscCall(DMLabelDestroy(&bdLabel));

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
      for (j = i; j < dim; ++j) {
        if (isotropic) {
          if (i == j) {
            if (uniform) metric[Neq * v + k] = PetscRealPart(met[0]);
            else metric[Neq * v + k] = PetscRealPart(met[v]);
          } else metric[Neq * v + k] = 0.0;
        } else {
          metric[Neq * v + k] = PetscRealPart(met[dim * dim * v + dim * i + j]);
        }
        k++;
      }
    }
  }
  PetscCall(VecRestoreArrayRead(vertexMetric, &met));

  /* Send mesh to Mmg and remesh */
  PetscCall(DMPlexMetricGetVerbosity(dm, &verbosity));
  PetscCall(DMPlexMetricGetGradationFactor(dm, &gradationFactor));
  PetscCall(DMPlexMetricGetHausdorffNumber(dm, &hausdorffNumber));
  PetscCall(DMPlexMetricNoInsertion(dm, &noInsert));
  PetscCall(DMPlexMetricNoSwapping(dm, &noSwap));
  PetscCall(DMPlexMetricNoMovement(dm, &noMove));
  PetscCall(DMPlexMetricNoSurf(dm, &noSurf));
  switch (dim) {
  case 2:
    PetscCallMMG_NONSTANDARD(MMG2D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end));
    PetscCallMMG_NONSTANDARD(MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_noinsert, noInsert));
    PetscCallMMG_NONSTANDARD(MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_noswap, noSwap));
    PetscCallMMG_NONSTANDARD(MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_nomove, noMove));
    PetscCallMMG_NONSTANDARD(MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_nosurf, noSurf));
    PetscCallMMG_NONSTANDARD(MMG2D_Set_iparameter(mmg_mesh, mmg_metric, MMG2D_IPARAM_verbose, verbosity));
    PetscCallMMG_NONSTANDARD(MMG2D_Set_dparameter(mmg_mesh, mmg_metric, MMG2D_DPARAM_hgrad, gradationFactor));
    PetscCallMMG_NONSTANDARD(MMG2D_Set_dparameter(mmg_mesh, mmg_metric, MMG2D_DPARAM_hausd, hausdorffNumber));
    PetscCallMMG_NONSTANDARD(MMG2D_Set_meshSize(mmg_mesh, numVertices, numCells, 0, numFaceTags));
    PetscCallMMG_NONSTANDARD(MMG2D_Set_vertices(mmg_mesh, vertices, verTags));
    PetscCallMMG_NONSTANDARD(MMG2D_Set_triangles(mmg_mesh, cells, cellTags));
    PetscCallMMG_NONSTANDARD(MMG2D_Set_edges(mmg_mesh, bdFaces, faceTags));
    PetscCallMMG_NONSTANDARD(MMG2D_Set_solSize(mmg_mesh, mmg_metric, MMG5_Vertex, numVertices, MMG5_Tensor));
    PetscCallMMG_NONSTANDARD(MMG2D_Set_tensorSols(mmg_metric, metric));
    PetscCallMMG(MMG2D_mmg2dlib(mmg_mesh, mmg_metric));
    break;
  case 3:
    PetscCallMMG_NONSTANDARD(MMG3D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end));
    PetscCallMMG_NONSTANDARD(MMG3D_Set_iparameter(mmg_mesh, mmg_metric, MMG3D_IPARAM_noinsert, noInsert));
    PetscCallMMG_NONSTANDARD(MMG3D_Set_iparameter(mmg_mesh, mmg_metric, MMG3D_IPARAM_noswap, noSwap));
    PetscCallMMG_NONSTANDARD(MMG3D_Set_iparameter(mmg_mesh, mmg_metric, MMG3D_IPARAM_nomove, noMove));
    PetscCallMMG_NONSTANDARD(MMG3D_Set_iparameter(mmg_mesh, mmg_metric, MMG3D_IPARAM_nosurf, noSurf));
    PetscCallMMG_NONSTANDARD(MMG3D_Set_iparameter(mmg_mesh, mmg_metric, MMG3D_IPARAM_verbose, verbosity));
    PetscCallMMG_NONSTANDARD(MMG3D_Set_dparameter(mmg_mesh, mmg_metric, MMG3D_DPARAM_hgrad, gradationFactor));
    PetscCallMMG_NONSTANDARD(MMG3D_Set_dparameter(mmg_mesh, mmg_metric, MMG3D_DPARAM_hausd, hausdorffNumber));
    PetscCallMMG_NONSTANDARD(MMG3D_Set_meshSize(mmg_mesh, numVertices, numCells, 0, numFaceTags, 0, 0));
    PetscCallMMG_NONSTANDARD(MMG3D_Set_vertices(mmg_mesh, vertices, verTags));
    PetscCallMMG_NONSTANDARD(MMG3D_Set_tetrahedra(mmg_mesh, cells, cellTags));
    PetscCallMMG_NONSTANDARD(MMG3D_Set_triangles(mmg_mesh, bdFaces, faceTags));
    PetscCallMMG_NONSTANDARD(MMG3D_Set_solSize(mmg_mesh, mmg_metric, MMG5_Vertex, numVertices, MMG5_Tensor));
    PetscCallMMG_NONSTANDARD(MMG3D_Set_tensorSols(mmg_metric, metric));
    PetscCallMMG(MMG3D_mmg3dlib(mmg_mesh, mmg_metric));
    break;
  default:
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Mmg adaptation defined for dimension %" PetscInt_FMT, dim);
  }
  PetscCall(PetscFree(cells));
  PetscCall(PetscFree2(metric, vertices));
  PetscCall(PetscFree2(bdFaces, faceTags));
  PetscCall(PetscFree2(verTags, cellTags));

  /* Retrieve mesh from Mmg */
  switch (dim) {
  case 2:
    numCornersNew = 3;
    PetscCallMMG_NONSTANDARD(MMG2D_Get_meshSize(mmg_mesh, &numVerticesNew, &numCellsNew, 0, &numFacesNew));
    PetscCall(PetscMalloc4(2 * numVerticesNew, &verticesNew, numVerticesNew, &verTagsNew, numVerticesNew, &corners, numVerticesNew, &requiredVer));
    PetscCall(PetscMalloc3(3 * numCellsNew, &cellsNew, numCellsNew, &cellTagsNew, numCellsNew, &requiredCells));
    PetscCall(PetscMalloc4(2 * numFacesNew, &facesNew, numFacesNew, &faceTagsNew, numFacesNew, &ridges, numFacesNew, &requiredFaces));
    PetscCallMMG_NONSTANDARD(MMG2D_Get_vertices(mmg_mesh, verticesNew, verTagsNew, corners, requiredVer));
    PetscCallMMG_NONSTANDARD(MMG2D_Get_triangles(mmg_mesh, cellsNew, cellTagsNew, requiredCells));
    PetscCallMMG_NONSTANDARD(MMG2D_Get_edges(mmg_mesh, facesNew, faceTagsNew, ridges, requiredFaces));
    break;
  case 3:
    numCornersNew = 4;
    PetscCallMMG_NONSTANDARD(MMG3D_Get_meshSize(mmg_mesh, &numVerticesNew, &numCellsNew, 0, &numFacesNew, 0, 0));
    PetscCall(PetscMalloc4(3 * numVerticesNew, &verticesNew, numVerticesNew, &verTagsNew, numVerticesNew, &corners, numVerticesNew, &requiredVer));
    PetscCall(PetscMalloc3(4 * numCellsNew, &cellsNew, numCellsNew, &cellTagsNew, numCellsNew, &requiredCells));
    PetscCall(PetscMalloc4(3 * numFacesNew, &facesNew, numFacesNew, &faceTagsNew, numFacesNew, &ridges, numFacesNew, &requiredFaces));
    PetscCallMMG_NONSTANDARD(MMG3D_Get_vertices(mmg_mesh, verticesNew, verTagsNew, corners, requiredVer));
    PetscCallMMG_NONSTANDARD(MMG3D_Get_tetrahedra(mmg_mesh, cellsNew, cellTagsNew, requiredCells));
    PetscCallMMG_NONSTANDARD(MMG3D_Get_triangles(mmg_mesh, facesNew, faceTagsNew, requiredFaces));

    /* Reorder for consistency with DMPlex */
    for (i = 0; i < numCellsNew; ++i) PetscCall(DMPlexInvertCell(DM_POLYTOPE_TETRAHEDRON, &cellsNew[4 * i]));
    break;

  default:
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Mmg adaptation defined for dimension %" PetscInt_FMT, dim);
  }

  /* Create new Plex */
  for (i = 0; i < (dim + 1) * numCellsNew; i++) cellsNew[i] -= 1;
  for (i = 0; i < dim * numFacesNew; i++) facesNew[i] -= 1;
  PetscCall(DMPlexCreateFromCellListParallelPetsc(comm, dim, numCellsNew, numVerticesNew, PETSC_DECIDE, numCornersNew, PETSC_TRUE, cellsNew, dim, verticesNew, NULL, NULL, dmNew));
  switch (dim) {
  case 2:
    PetscCallMMG_NONSTANDARD(MMG2D_Free_all(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end));
    break;
  case 3:
    PetscCallMMG_NONSTANDARD(MMG3D_Free_all(MMG5_ARG_start, MMG5_ARG_ppMesh, &mmg_mesh, MMG5_ARG_ppMet, &mmg_metric, MMG5_ARG_end));
    break;
  default:
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No Mmg adaptation defined for dimension %" PetscInt_FMT, dim);
  }
  PetscCall(PetscFree4(verticesNew, verTagsNew, corners, requiredVer));

  /* Get adapted mesh information */
  PetscCall(DMPlexGetHeightStratum(*dmNew, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetHeightStratum(*dmNew, 1, &fStart, &fEnd));
  PetscCall(DMPlexGetDepthStratum(*dmNew, 0, &vStart, &vEnd));

  /* Rebuild boundary labels */
  PetscCall(DMCreateLabel(*dmNew, flg ? bdName : bdLabelName));
  PetscCall(DMGetLabel(*dmNew, flg ? bdName : bdLabelName, &bdLabelNew));
  for (i = 0; i < numFacesNew; i++) {
    PetscInt        numCoveredPoints, numFaces = 0, facePoints[3];
    const PetscInt *coveredPoints = NULL;

    for (j = 0; j < dim; ++j) facePoints[j] = facesNew[i * dim + j] + vStart;
    PetscCall(DMPlexGetFullJoin(*dmNew, dim, facePoints, &numCoveredPoints, &coveredPoints));
    for (j = 0; j < numCoveredPoints; ++j) {
      if (coveredPoints[j] >= fStart && coveredPoints[j] < fEnd) {
        numFaces++;
        f = j;
      }
    }
    PetscCheck(numFaces == 1, comm, PETSC_ERR_ARG_OUTOFRANGE, "%" PetscInt_FMT " vertices cannot define more than 1 facet (%" PetscInt_FMT ")", dim, numFaces);
    PetscCall(DMLabelSetValue(bdLabelNew, coveredPoints[f], faceTagsNew[i]));
    PetscCall(DMPlexRestoreJoin(*dmNew, dim, facePoints, &numCoveredPoints, &coveredPoints));
  }
  PetscCall(PetscFree4(facesNew, faceTagsNew, ridges, requiredFaces));

  /* Rebuild cell labels */
  PetscCall(DMCreateLabel(*dmNew, rgLabel ? rgLabelName : rgName));
  PetscCall(DMGetLabel(*dmNew, rgLabel ? rgLabelName : rgName, &rgLabelNew));
  for (c = cStart; c < cEnd; ++c) PetscCall(DMLabelSetValue(rgLabelNew, c, cellTagsNew[c - cStart]));
  PetscCall(PetscFree3(cellsNew, cellTagsNew, requiredCells));

  PetscFunctionReturn(0);
}
