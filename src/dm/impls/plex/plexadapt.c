#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#ifdef PETSC_HAVE_PRAGMATIC
#include <pragmatic/cpragmatic.h>
#endif



#undef __FUNCT__
#define __FUNCT__ "DMPlexAdapt"
PetscErrorCode DMPlexAdapt(DM dm, Vec metric, const char bdyLabelName[], DM *dmCoarsened)
{
#ifdef PETSC_HAVE_PRAGMATIC
  DM             udm, coordDM;
  DMLabel        bd, bdtags;
  Vec            coordinates;
  PetscSection   coordSection;
  const PetscScalar *coords;
  double        *coarseCoords;
  IS             bdIS;
  PetscReal     *x, *y, *z;
  const PetscInt *faces;
  PetscInt      *cells, *ccells, *bdFaces, *bdFaceIds;
  PetscInt       dim, numCorners, cStart, cEnd, numCells, numCoarseCells, c, vStart, vEnd, numVertices, numCoarseVertices, v, numBdFaces, f, maxConeSize, bdSize, coff;
  const PetscScalar   *metricArray;
  PetscReal     *met;
  PetscInt       *cell, perm[30], isInFacetClosure[30], iVer, i, idx, facet; // 30 is twice the max size of a cell closure in 3D for tet meshes
  PetscInt      *boundaryTags;
    
#endif
  PetscErrorCode ierr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
#ifdef PETSC_HAVE_PRAGMATIC
  if (!dm->coarseMesh) {
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
    if ( bdyLabelName ) {
      ierr = DMGetLabel(dm, bdyLabelName, &bdtags);CHKERRQ(ierr);
    }
    // TODO fix if bdyLabelName = "boundary"
    // TODO a way to use only bdyLabelName would be to loop over its strata, if I can't get all the strata at once ?
    ierr = DMLabelCreate("boundary", &bd);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(dm, bd);CHKERRQ(ierr);        
    ierr = DMLabelGetStratumIS(bd, 1, &bdIS);CHKERRQ(ierr);
    ierr = DMLabelGetStratumSize(bd, 1, &numBdFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(bdIS, &faces);CHKERRQ(ierr);
    // TODO why not assume that we are considering simplicial meshes, in which case bdSize = dim*numBdFaces ?
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
      if ( bdyLabelName ) 
        DMLabelGetValue(bdtags, faces[f], &bdFaceIds[f]);
      else  
        bdFaceIds[f] = 1;
      ierr = DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&bdIS);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&bd);CHKERRQ(ierr);
    pragmatic_set_boundary(&numBdFaces, bdFaces, bdFaceIds);
    pragmatic_set_metric(met);
    pragmatic_adapt();
    /* Read out mesh */
    pragmatic_get_info(&numCoarseVertices, &numCoarseCells);
    ierr = PetscMalloc1(numCoarseVertices*dim, &coarseCoords);CHKERRQ(ierr);
    switch (dim) {
    case 2:
      ierr = PetscMalloc2(numCoarseVertices, &x, numCoarseVertices, &y);CHKERRQ(ierr);
      pragmatic_get_coords_2d(x, y);
      numCorners = 3;
      for (v = 0; v < numCoarseVertices; ++v) {coarseCoords[v*2+0] = x[v]; coarseCoords[v*2+1] = y[v];}
      break;
    case 3:
      ierr = PetscMalloc3(numCoarseVertices, &x, numCoarseVertices, &y, numCoarseVertices, &z);CHKERRQ(ierr);
      pragmatic_get_coords_3d(x, y, z);
      numCorners = 4;
      for (v = 0; v < numCoarseVertices; ++v) {coarseCoords[v*3+0] = x[v]; coarseCoords[v*3+1] = y[v]; coarseCoords[v*3+2] = z[v];}
      break;
    default:
      SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No Pragmatic coarsening defined for dimension %d", dim);
    }
    ierr = PetscMalloc1(numCoarseCells*(dim+1), &ccells);CHKERRQ(ierr); // only for simplicial meshes
    pragmatic_get_elements(ccells);
    ierr = DMPlexCreateFromCellList(PetscObjectComm((PetscObject) dm), dim, numCoarseCells, numCoarseVertices, numCorners, PETSC_TRUE, ccells, dim, coarseCoords, &dm->coarseMesh);CHKERRQ(ierr);
    /* Read out boundary tags */
    pragmatic_get_boundaryTags(&boundaryTags);
    if ( !bdyLabelName ) bdyLabelName = "boundary";
    ierr = DMCreateLabel(dm->coarseMesh, bdyLabelName);CHKERRQ(ierr);
    ierr = DMGetLabel(dm->coarseMesh, bdyLabelName, &bd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm->coarseMesh, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm->coarseMesh, 0, &vStart, &vEnd);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      PetscInt       *cellClosure = NULL;
      PetscInt        cellClosureSize, cl;
      PetscInt       *facetClosure = NULL;
      PetscInt        facetClosureSize, cl2;   
      const PetscInt *facetList;
      PetscInt        facetListSize, f;
      
      cell = &ccells[(c-cStart)*(dim+1)]; // pointing to the current cell in the ccell table
      ierr = DMPlexGetTransitiveClosure(dm->coarseMesh, c, PETSC_TRUE, &cellClosureSize, &cellClosure);CHKERRQ(ierr);
      // first, encode the permutation of the vertices indices betwenn the cell closure and pragmatic ordering
      for (cl = 0; cl < cellClosureSize*2; cl+=2 ) {
        if ((cellClosure[cl] < vStart) || (cellClosure[cl] >= vEnd)) continue;
        iVer = cellClosure[cl] - vStart;
        for (i=0; i<dim+1; ++i) {
          if ( iVer == cell[i] ) {
            perm[cl] = i;    // the cl-th element of the closure is the i-th vertex of the cell
            break;
          }
        }
      }
      ierr = DMPlexGetCone(dm->coarseMesh, c, &facetList);CHKERRQ(ierr);     // list of edges/facets of the cell
      ierr = DMPlexGetConeSize(dm->coarseMesh, c, &facetListSize);CHKERRQ(ierr);
      // then, for each edge/facet of the cell, find the opposite vertex (ie the one not in the closure of the facet/edge)
      for (f=0; f<facetListSize; ++f){
        facet = facetList[f];
        ierr = DMPlexGetTransitiveClosure(dm->coarseMesh, facet, PETSC_TRUE, &facetClosureSize, &facetClosure);CHKERRQ(ierr);
        // loop over the vertices of the cell closure, and check if each vertex belongs to the edge/facet closure
        PetscMemzero(isInFacetClosure, sizeof(isInFacetClosure));
        for (cl = 0; cl < cellClosureSize*2; cl+=2) {
          if ((cellClosure[cl] < vStart) || (cellClosure[cl] >= vEnd)) isInFacetClosure[cl] = 1;
          for (cl2 =0; cl2<facetClosureSize*2; cl2+=2){
            if ((facetClosure[cl2] < vStart) || (facetClosure[cl2] >= vEnd)) continue;
            if ( cellClosure[cl] == facetClosure[cl2] ) {
              isInFacetClosure[cl] = 1;
            }
          }
        }
        // the vertex that was not marked is the vertex opposite to the edge/facet, ie the one giving the edge/facet boundary tag in pragmatic
        for (cl = 0; cl < cellClosureSize*2; cl+=2 ) {
          if ( !isInFacetClosure[cl] ) {
            idx = (c-cStart)*(dim+1) + perm[cl];
            if ( boundaryTags[idx] ) 
//              ierr = DMSetLabelValue(dm->coarseMesh, bdyLabelName, facet, boundaryTags[idx]);CHKERRQ(ierr);
              ierr = DMLabelSetValue(bd, facet, boundaryTags[idx]);CHKERRQ(ierr);
            break;
          }
        }
        ierr = DMPlexRestoreTransitiveClosure(dm->coarseMesh, facet, PETSC_TRUE, &facetClosureSize, &facetClosure);CHKERRQ(ierr);
      }
      ierr = DMPlexRestoreTransitiveClosure(dm->coarseMesh, c, PETSC_TRUE, &cellClosureSize, &cellClosure);CHKERRQ(ierr);
    }
    pragmatic_finalize();
    ierr = PetscFree5(x, y, z, cells, met);CHKERRQ(ierr);
    ierr = PetscFree2(bdFaces, bdFaceIds);CHKERRQ(ierr);
    ierr = PetscFree(coarseCoords);CHKERRQ(ierr);
    ierr = PetscFree(ccells);CHKERRQ(ierr);
  }
#endif
  ierr = PetscObjectReference((PetscObject) dm->coarseMesh);CHKERRQ(ierr);
  *dmCoarsened = dm->coarseMesh;
  PetscFunctionReturn(0);
}
