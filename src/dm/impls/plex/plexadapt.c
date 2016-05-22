#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#ifdef PETSC_HAVE_PRAGMATIC
#include <pragmatic/cpragmatic.h>
#endif

#undef __FUNCT__
#define __FUNCT__ "DMPlexRemesh_Internal"
/*
  DMPlexRemesh_Internal - Generates a new mesh conforming to a metric field.

  Input Parameters:
+ dm - The DM object
. vertexMetric - The metric to which the mesh is adapted, defined vertex-wise.
. remeshBd - Flag to allow boundary changes
- bdLabelName - Label name for boundary tags which are preserved in dmNew, or NULL. Should not be "_boundary_".

  Output Parameter:
. dmNew  - the new DM

  Level: advanced

.seealso: DMCoarsen(), DMRefine()
*/
PetscErrorCode DMPlexRemesh_Internal(DM dm, Vec vertexMetric, const char bdLabelName[], PetscBool remeshBd, DM *dmNew)
{
  const char        *bdName = "_boundary_";
  DM                 udm, cdm;
  DMLabel            bdLabel = NULL, bdLabelFull;
  IS                 bdIS;
  PetscSection       coordSection;
  Vec                coordinates;
  const PetscScalar *coords, *met;
  const PetscInt    *bdFacesFull;
  PetscInt          *bdFaces, *bdFaceIds;
  PetscReal         *x, *y, *z, *metric;
  PetscInt          *cells;
  PetscInt           dim, cStart, cEnd, numCells, c, coff, vStart, vEnd, numVertices, v;
  PetscInt           off, maxConeSize, numBdFaces, f, bdSize;
  PetscBool          flg;
#ifdef PETSC_HAVE_PRAGMATIC
  DMLabel            bdLabelNew;
  PetscScalar       *coordsNew;
  PetscInt          *bdTags;
  PetscReal         *xNew[3] = {NULL, NULL, NULL};
  PetscInt          *cellsNew;
  PetscInt           d, numCellsNew, numVerticesNew;
  PetscInt           numCornersNew, fStart, fEnd;
#endif
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(vertexMetric, VEC_CLASSID, 2);
  PetscValidCharPointer(bdLabelName, 3);
  PetscValidPointer(dmNew, 5);
  if (bdLabelName) {
    size_t len;

    ierr = PetscStrcmp(bdLabelName, bdName, &flg);CHKERRQ(ierr);
    if (flg) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "\"%s\" cannot be used as label for boundary facets", bdLabelName);
    ierr = PetscStrlen(bdLabelName, &len);CHKERRQ(ierr);
    if (len) {
      ierr = DMGetLabel(dm, bdLabelName, &bdLabel);CHKERRQ(ierr);
      if (!bdLabel) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Label \"%s\" does not exist in DM", bdLabelName);
    }
  }
  /* Get mesh information */
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexUninterpolate(dm, &udm);CHKERRQ(ierr);
  ierr = DMPlexGetMaxSizes(udm, &maxConeSize, NULL);CHKERRQ(ierr);
  numCells    = cEnd - cStart;
  numVertices = vEnd - vStart;
  ierr = PetscCalloc5(numVertices, &x, numVertices, &y, numVertices, &z, numVertices*PetscSqr(dim), &metric, numCells*maxConeSize, &cells);CHKERRQ(ierr);
  for (c = 0, coff = 0; c < numCells; ++c) {
    const PetscInt *cone;
    PetscInt        coneSize, cl;

    ierr = DMPlexGetConeSize(udm, c, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(udm, c, &cone);CHKERRQ(ierr);
    for (cl = 0; cl < coneSize; ++cl) cells[coff++] = cone[cl] - vStart;
  }
  ierr = DMDestroy(&udm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(cdm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
    x[v-vStart] = PetscRealPart(coords[off+0]);
    if (dim > 1) y[v-vStart] = PetscRealPart(coords[off+1]);
    if (dim > 2) z[v-vStart] = PetscRealPart(coords[off+2]);
  }
  ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
  /* Get boundary mesh */
  ierr = DMLabelCreate(bdName, &bdLabelFull);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(dm, bdLabelFull);CHKERRQ(ierr);
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
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) bdFaces[bdSize++] = closure[cl] - vStart;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, bdFacesFull[f], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    if (bdLabel) {ierr = DMLabelGetValue(bdLabel, bdFacesFull[f], &bdFaceIds[f]);CHKERRQ(ierr);}
    else         {bdFaceIds[f] = 1;}
  }
  ierr = ISDestroy(&bdIS);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&bdLabelFull);CHKERRQ(ierr);
  /* Get metric */
  ierr = VecGetArrayRead(vertexMetric, &met);CHKERRQ(ierr);
  for (v = 0; v < (vEnd-vStart)*PetscSqr(dim); ++v) metric[v] = PetscRealPart(met[v]);
  ierr = VecRestoreArrayRead(vertexMetric, &met);CHKERRQ(ierr);
  /* Create new mesh */
#ifdef PETSC_HAVE_PRAGMATIC
  switch (dim) {
  case 2:
    pragmatic_2d_init(&numVertices, &numCells, cells, x, y);break;
  case 3:
    pragmatic_3d_init(&numVertices, &numCells, cells, x, y, z);break;
  default: SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No Pragmatic adaptation defined for dimension %d", dim);
  }
  pragmatic_set_boundary(&numBdFaces, bdFaces, bdFaceIds);
  pragmatic_set_metric(metric);
  pragmatic_adapt(remeshBd ? 1 : 0);
  /* Read out mesh */
  pragmatic_get_info(&numVerticesNew, &numCellsNew);
  ierr = PetscMalloc1(numVerticesNew*dim, &coordsNew);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    numCornersNew = 3;
    ierr = PetscMalloc2(numVerticesNew, &xNew[0], numVerticesNew, &xNew[1]);CHKERRQ(ierr);
    pragmatic_get_coords_2d(xNew[0], xNew[1]);
    break;
  case 3:
    numCornersNew = 4;
    ierr = PetscMalloc3(numVerticesNew, &xNew[0], numVerticesNew, &xNew[1], numVerticesNew, &xNew[2]);CHKERRQ(ierr);
    pragmatic_get_coords_3d(xNew[0], xNew[1], xNew[2]);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No Pragmatic adaptation defined for dimension %d", dim);
  }
  for (v = 0; v < numVerticesNew; ++v) {for (d = 0; d < dim; ++d) coordsNew[v*dim+d] = xNew[d][v];}
  ierr = PetscMalloc1(numCellsNew*(dim+1), &cellsNew);CHKERRQ(ierr);
  pragmatic_get_elements(cellsNew);
  ierr = DMPlexCreateFromCellList(PetscObjectComm((PetscObject) dm), dim, numCellsNew, numVerticesNew, numCornersNew, PETSC_TRUE, cellsNew, dim, coordsNew, dmNew);CHKERRQ(ierr);
  /* Read out boundary label */
  pragmatic_get_boundaryTags(&bdTags);
  ierr = DMCreateLabel(*dmNew, bdLabel ? bdLabelName : bdName);CHKERRQ(ierr);
  ierr = DMGetLabel(*dmNew, bdLabel ? bdLabelName : bdName, &bdLabelNew);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dmNew, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dmNew, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(*dmNew, 0, &vStart, &vEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    /* Only for simplicial meshes */
    coff = (c-cStart)*(dim+1);
    for (d = 0; d < dim+1; ++d) {
      if (bdTags[coff+d]) {
        const PetscInt *faces;
        PetscInt        numFaces, vertices[3], p;

        /* Mark face opposite to this vertex */
        for (p = 0, v = 0; p < dim+1; ++p) if (p != d) {vertices[v] = cellsNew[coff+p] + vStart; ++v;}
        ierr = DMPlexGetFullJoin(*dmNew, dim, vertices, &numFaces, &faces);CHKERRQ(ierr);
        if (numFaces != 1) SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cannot find boundary face in new cell %D for vertex %D(%D) with tag %D", c, cellsNew[coff+d], d, bdTags[coff+d]);
        if (faces[0] < fStart || faces[0] >= fEnd) SETERRQ5(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Boundary point %D in new cell %D for vertex %D(%D) with tag %D is not a face", faces[0], c, cellsNew[coff+d], d, bdTags[coff+d]);
        ierr = DMLabelSetValue(bdLabelNew, faces[0], bdTags[coff+d]);CHKERRQ(ierr);
        ierr = DMPlexRestoreJoin(*dmNew, dim, vertices, &numFaces, &faces);CHKERRQ(ierr);
      }
    }
  }
  /* Cleanup */
  switch (dim) {
  case 2: ierr = PetscFree2(xNew[0], xNew[1]);CHKERRQ(ierr);break;
  case 3: ierr = PetscFree3(xNew[0], xNew[1], xNew[2]);CHKERRQ(ierr);break;
  }
  ierr = PetscFree(cellsNew);CHKERRQ(ierr);
  ierr = PetscFree5(x, y, z, metric, cells);CHKERRQ(ierr);
  ierr = PetscFree2(bdFaces, bdFaceIds);CHKERRQ(ierr);
  ierr = PetscFree(coordsNew);CHKERRQ(ierr);
  pragmatic_finalize();
#else
  SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Remeshing needs external package support.\nPlease reconfigure with --download-pragmatic.");
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexAdapt"
/*@
  DMPlexAdapt - Generates a mesh adapted to the specified metric field using the pragmatic library.

  Input Parameters:
+ dm - The DM object
. metric - The metric to which the mesh is adapted, defined vertex-wise.
- bdyLabelName - Label name for boundary tags. These will be preserved in the output mesh. bdyLabelName should be "" (empty string) if there is no such label, and should be different from "boundary".

  Output Parameter:
. dmAdapt  - Pointer to the DM object containing the adapted mesh

  Level: advanced

.seealso: DMCoarsen(), DMRefine()
@*/
PetscErrorCode DMPlexAdapt(DM dm, Vec metric, const char bdLabelName[], DM *dmAdapt)
{
  DM_Plex       *mesh = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(metric, VEC_CLASSID, 2);
  PetscValidCharPointer(bdLabelName, 3);
  PetscValidPointer(dmAdapt, 4);
  ierr = DMPlexRemesh_Internal(dm, metric, bdLabelName, mesh->remeshBd, dmAdapt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
