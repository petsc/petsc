#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#ifdef PETSC_HAVE_PRAGMATIC
#include <pragmatic/cpragmatic.h>
#endif



#undef __FUNCT__
#define __FUNCT__ "DMPlexAdapt"
/*@
  DMPlexAdapt - Generates a mesh adapted to the specified metric field using the pragmatic library.

  Input Parameters:
+ dm - The DM object
. metric - The metric to which the mesh is adapted, defined vertex-wise.
- bdyLabelName - Label name for boundary tags. These will be preserved in the output mesh. bdyLabelName should be "" (empty string) if there is no such label, and should be different from "boundary".

  Output Parameter:
. dmAdapted  - Pointer to the DM object containing the adapted mesh

  Level: advanced

.seealso: DMCoarsen(), DMRefine()
@*/
PetscErrorCode DMPlexAdapt(DM dm, Vec metric, const char bdyLabelName[], DM *dmAdapted)
{
#ifdef PETSC_HAVE_PRAGMATIC
  DM                   udm, coordDM;
  DMLabel              bd, bdTags, bdTagsAdp;
  Vec                  coordinates;
  PetscSection         coordSection;
  IS                   bdIS;
  double              *coordsAdp;  
  const PetscScalar   *coords;
  PetscReal           *x, *y, *z, *xAdp, *yAdp, *zAdp;
  const PetscScalar   *metricArray;
  PetscReal           *met;
  const PetscInt      *faces;
  PetscInt            *cells, *cellsAdp, *bdFaces, *bdFaceIds;
  PetscInt            *boundaryTags;
  PetscInt             dim, numCornersAdp, cStart, cEnd, numCells, numCellsAdp, c; 
  PetscInt             vStart, vEnd, numVertices, numVerticesAdp, v;
  PetscInt             numBdFaces, f, maxConeSize, bdSize, coff;
  PetscInt            *cell, iVer, cellOff, i, j, idx, facet; 
  PetscInt             perm[4], isInFacetClosure[4]; // 4 = max number of facets for an element in 2D & 3D. Only for simplicial meshes
  PetscBool            flag, bdyLabelExt, hasBdyFacet;
#endif
  PetscErrorCode       ierr;
  MPI_Comm             comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
#ifdef PETSC_HAVE_PRAGMATIC
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &coordDM);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(coordDM, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexUninterpolate(dm, &udm);CHKERRQ(ierr);
  ierr = DMPlexGetMaxSizes(udm, &maxConeSize, NULL);CHKERRQ(ierr);
  numCells    = cEnd - cStart;
  numVertices = vEnd - vStart;
  ierr = PetscCalloc5(numVertices, &x, numVertices, &y, numVertices, &z, numCells*maxConeSize, &cells, dim*dim*numVertices, &met);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecGetArrayRead(metric, &metricArray);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    PetscInt off;
  
    ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
    x[v-vStart] = coords[off+0];
    y[v-vStart] = coords[off+1];
    if (dim > 2) z[v-vStart] = coords[off+2];
    ierr = PetscMemcpy(&met[dim*dim*(v-vStart)], &metricArray[dim*off], dim*dim*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(metric, &metricArray);CHKERRQ(ierr);
  for (c = 0, coff = 0; c < numCells; ++c) {
    const PetscInt *cone;
    PetscInt        coneSize, cl;
  
    ierr = DMPlexGetConeSize(udm, c, &coneSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(udm, c, &cone);CHKERRQ(ierr);
    for (cl = 0; cl < coneSize; ++cl) cells[coff++] = cone[cl] - vStart;
  }
  switch (dim) {
  case 2:
    pragmatic_2d_init(&numVertices, &numCells, cells, x, y);
    break;
  case 3:
    pragmatic_3d_init(&numVertices, &numCells, cells, x, y, z);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No Pragmatic adaptation defined for dimension %d", dim);
  }
  /* Create boundary mesh */
  if (!bdyLabelName) {
    SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_NULL, "Argument bdyLabelName cannot be NULL", dim);
  } else {
    ierr = PetscStrcmp(bdyLabelName, "", &flag);CHKERRQ(ierr);
    if (flag) bdyLabelExt = PETSC_FALSE;
    else bdyLabelExt = PETSC_TRUE;
  }
  if (bdyLabelExt) {
    ierr = PetscStrcmp(bdyLabelName, "boundary", &flag);CHKERRQ(ierr);
    if (flag) {
      SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "\"%s\" cannot be used as label for boundary facets", bdyLabelName);
    }
    ierr = DMGetLabel(dm, bdyLabelName, &bdTags);CHKERRQ(ierr);
    if (!bdTags) {
      SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Label %s does not exist in DM", bdyLabelName);
    }
  }
  /* TODO To avoid marking bdy facets again if bdyLabelName exists, could/should we loop over bdyLabelName stratas ? */
  ierr = DMLabelCreate("boundary", &bd);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(dm, bd);CHKERRQ(ierr);        
  ierr = DMLabelGetStratumIS(bd, 1, &bdIS);CHKERRQ(ierr);
  ierr = DMLabelGetStratumSize(bd, 1, &numBdFaces);CHKERRQ(ierr);
  ierr = ISGetIndices(bdIS, &faces);CHKERRQ(ierr);
  /* TODO why not assume that we are considering simplicial meshes, in which case bdSize = dim*numBdFaces ? */
  for (f = 0, bdSize = 0; f < numBdFaces; ++f) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, cl;
  
    ierr = DMPlexGetTransitiveClosure(dm, faces[f], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) ++bdSize;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
  }
  ierr = PetscMalloc2(bdSize, &bdFaces, numBdFaces, &bdFaceIds);CHKERRQ(ierr);
  for (f = 0, bdSize = 0; f < numBdFaces; ++f) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, cl;
  
    ierr = DMPlexGetTransitiveClosure(dm, faces[f], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      if ((closure[cl] >= vStart) && (closure[cl] < vEnd)) bdFaces[bdSize++] = closure[cl] - vStart;
    }
    if (bdyLabelExt) {
      ierr = DMLabelGetValue(bdTags, faces[f], &bdFaceIds[f]);CHKERRQ(ierr);
    } else {
      bdFaceIds[f] = 1;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&bdIS);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&bd);CHKERRQ(ierr);
  pragmatic_set_boundary(&numBdFaces, bdFaces, bdFaceIds);
  pragmatic_set_metric(met);
  ierr = PetscFree5(x, y, z, cells, met);CHKERRQ(ierr);
  ierr = PetscFree2(bdFaces, bdFaceIds);CHKERRQ(ierr);
  ierr = PetscFree(coords);CHKERRQ(ierr);
  ierr = DMDestroy(&udm);CHKERRQ(ierr);
  pragmatic_adapt();
  /* Read out mesh */
  pragmatic_get_info(&numVerticesAdp, &numCellsAdp);
  ierr = PetscMalloc1(numVerticesAdp*dim, &coordsAdp);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    ierr = PetscMalloc2(numVerticesAdp, &xAdp, numVerticesAdp, &yAdp);CHKERRQ(ierr);
    zAdp = NULL;
    pragmatic_get_coords_2d(xAdp, yAdp);
    numCornersAdp = 3;
    for (v = 0; v < numVerticesAdp; ++v) {coordsAdp[v*2+0] = xAdp[v]; coordsAdp[v*2+1] = yAdp[v];}
    break;
  case 3:
    ierr = PetscMalloc3(numVerticesAdp, &xAdp, numVerticesAdp, &yAdp, numVerticesAdp, &zAdp);CHKERRQ(ierr);
    pragmatic_get_coords_3d(xAdp, yAdp, zAdp);
    numCornersAdp = 4;
    for (v = 0; v < numVerticesAdp; ++v) {coordsAdp[v*3+0] = xAdp[v]; coordsAdp[v*3+1] = yAdp[v]; coordsAdp[v*3+2] = zAdp[v];}
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No Pragmatic adaptation defined for dimension %d", dim);
  }
  ierr = PetscMalloc1(numCellsAdp*(dim+1), &cellsAdp);CHKERRQ(ierr); // only for simplicial meshes
  pragmatic_get_elements(cellsAdp);
  ierr = DMPlexCreateFromCellList(PetscObjectComm((PetscObject) dm), dim, numCellsAdp, numVerticesAdp, numCornersAdp, PETSC_TRUE, cellsAdp, dim, coordsAdp, dmAdapted);CHKERRQ(ierr);
  /* Read out boundary tags */
  pragmatic_get_boundaryTags(&boundaryTags);
  if (!bdyLabelExt) bdyLabelName = "boundary";
  ierr = DMCreateLabel(*dmAdapted, bdyLabelName);CHKERRQ(ierr);
  ierr = DMGetLabel(*dmAdapted, bdyLabelName, &bdTagsAdp);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(*dmAdapted, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(*dmAdapted, 0, &vStart, &vEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    PetscInt       *cellClosure = NULL;
    PetscInt        cellClosureSize, cl;
    PetscInt       *facetClosure = NULL;
    PetscInt        facetClosureSize, cl2;   
    const PetscInt *facetList;
    PetscInt        facetListSize, f;
    
    cellOff = (c-cStart)*(dim+1);  // gives the offset corresponding to the cell in pragmatic boundary arrays. Only for simplicial meshes
    hasBdyFacet = PETSC_FALSE;
    for (i = 0; i < dim+1; ++i) {   // only for simplicial meshes
      if (boundaryTags[cellOff+i]) {
        hasBdyFacet = PETSC_TRUE;
        break;
      }
    }
    if (!hasBdyFacet) continue; // The cell has no boundary edge/facet => no boundary tagging
    
    cell = &cellsAdp[cellOff]; // pointing to the current cell in the cellsAdp table
    ierr = DMPlexGetTransitiveClosure(*dmAdapted, c, PETSC_TRUE, &cellClosureSize, &cellClosure);CHKERRQ(ierr);
    /* first, encode the permutation of the vertices indices betwenn the cell closure and pragmatic ordering */
    j = 0;
    for (cl = 0; cl < cellClosureSize*2; cl += 2) {
      if ((cellClosure[cl] < vStart) || (cellClosure[cl] >= vEnd)) continue;
      iVer = cellClosure[cl] - vStart;
      for (i = 0; i < dim+1; ++i) {  // only for simplicial meshes
        if (iVer == cell[i]) {
          perm[j] = i;    // the cl-th element of the closure is the i-th vertex of the cell
          break;
        }
      }
      j++;
    }
    ierr = DMPlexGetCone(*dmAdapted, c, &facetList);CHKERRQ(ierr);     // list of edges/facets of the cell
    ierr = DMPlexGetConeSize(*dmAdapted, c, &facetListSize);CHKERRQ(ierr);
    /* then, for each edge/facet of the cell, find the opposite vertex (ie the one in the closure of the cell and not in the closure of the facet/edge) */
    for (f = 0; f < facetListSize; ++f){
      facet = facetList[f];
      ierr = DMPlexGetTransitiveClosure(*dmAdapted, facet, PETSC_TRUE, &facetClosureSize, &facetClosure);CHKERRQ(ierr);
      /* loop over the vertices of the cell closure, and check if each vertex belongs to the edge/facet closure */
      /* TODO should we use bitmaps to mark vertices instead of a static array ? */
      PetscMemzero(isInFacetClosure, sizeof(isInFacetClosure));
      j = 0;
      for (cl = 0; cl < cellClosureSize*2; cl += 2) {
        if ((cellClosure[cl] < vStart) || (cellClosure[cl] >= vEnd)) continue;
        for (cl2 = 0; cl2 < facetClosureSize*2; cl2 += 2){
          if ((facetClosure[cl2] < vStart) || (facetClosure[cl2] >= vEnd)) continue;
          if (cellClosure[cl] == facetClosure[cl2]) {
            isInFacetClosure[j] = 1;
          }
        }
        j++;
      }
      /* the vertex that was not marked is the vertex opposite to the edge/facet, ie the one giving the edge/facet boundary tag in pragmatic */
      j = 0;
      for (cl = 0; cl < cellClosureSize*2; cl += 2) {
        if ((cellClosure[cl] < vStart) || (cellClosure[cl] >= vEnd)) continue;
        if (!isInFacetClosure[j]) {
          idx = cellOff + perm[j];
          if (boundaryTags[idx]) {
            ierr = DMLabelSetValue(bdTagsAdp, facet, boundaryTags[idx]);CHKERRQ(ierr);
          }
          break;
        }
        j++;
      }
      ierr = DMPlexRestoreTransitiveClosure(*dmAdapted, facet, PETSC_TRUE, &facetClosureSize, &facetClosure);CHKERRQ(ierr);
    }
    ierr = DMPlexRestoreTransitiveClosure(*dmAdapted, c, PETSC_TRUE, &cellClosureSize, &cellClosure);CHKERRQ(ierr);
  }
  pragmatic_finalize();
  ierr = PetscFree5(xAdp, yAdp, zAdp, cellsAdp, coordsAdp);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
