#include <petsc-private/pleximpl.h>   /*I      "petscdmplex.h"   I*/

extern PetscErrorCode DMPlexGetNumFaceVertices_Internal(DM, PetscInt, PetscInt *);

#undef __FUNCT__
#define __FUNCT__ "DMPlexMarkSubmesh_Uninterpolated"
/* Here we need the explicit assumption that:

     For any marked cell, the marked vertices constitute a single face
*/
static PetscErrorCode DMPlexMarkSubmesh_Uninterpolated(DM dm, DMLabel vertexLabel, DMLabel subpointMap, PetscInt *numFaces, PetscInt *nFV, DM subdm)
{
  IS               subvertexIS;
  const PetscInt  *subvertices;
  PetscInt        *pStart, *pEnd, *pMax;
  PetscInt         depth, dim, d, numSubVerticesInitial = 0, v;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  *numFaces = 0;
  *nFV      = 0;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim+1,PetscInt,&pStart,dim+1,PetscInt,&pEnd,dim+1,PetscInt,&pMax);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &pMax[depth], depth>1 ? &pMax[depth-1] : PETSC_NULL, depth > 2 ? &pMax[1] : PETSC_NULL, &pMax[0]);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart[d], &pEnd[d]);CHKERRQ(ierr);
    if (pMax[d] >= 0) pEnd[d] = PetscMin(pEnd[d], pMax[d]);
  }
  /* Loop over initial vertices and mark all faces in the collective star() */
  ierr = DMLabelGetStratumIS(vertexLabel, 1, &subvertexIS);CHKERRQ(ierr);
  if (subvertexIS) {
    ierr = ISGetSize(subvertexIS, &numSubVerticesInitial);CHKERRQ(ierr);
    ierr = ISGetIndices(subvertexIS, &subvertices);CHKERRQ(ierr);
  }
  for (v = 0; v < numSubVerticesInitial; ++v) {
    const PetscInt vertex = subvertices[v];
    PetscInt      *star   = PETSC_NULL;
    PetscInt       starSize, s, numCells = 0, c;

    ierr = DMPlexGetTransitiveClosure(dm, vertex, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
    for (s = 0; s < starSize*2; s += 2) {
      const PetscInt point = star[s];
      if ((point >= pStart[depth]) && (point < pEnd[depth])) star[numCells++] = point;
    }
    for (c = 0; c < numCells; ++c) {
      const PetscInt cell    = star[c];
      PetscInt      *closure = PETSC_NULL;
      PetscInt       closureSize, cl;
      PetscInt       cellLoc, numCorners = 0, faceSize = 0;

      ierr = DMLabelGetValue(subpointMap, cell, &cellLoc);CHKERRQ(ierr);
      if (cellLoc == dim) continue;
      if (cellLoc >= 0) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Cell %d has dimension %d in the surface label", cell, cellLoc);
      ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (cl = 0; cl < closureSize*2; cl += 2) {
        const PetscInt point = closure[cl];
        PetscInt       vertexLoc;

        if ((point >= pStart[0]) && (point < pEnd[0])) {
          ++numCorners;
          ierr = DMLabelGetValue(vertexLabel, point, &vertexLoc);CHKERRQ(ierr);
          if (vertexLoc >= 0) closure[faceSize++] = point;
        }
      }
      if (!(*nFV)) {ierr = DMPlexGetNumFaceVertices_Internal(dm, numCorners, nFV);CHKERRQ(ierr);}
      if (faceSize > *nFV) SETERRQ1(((PetscObject) dm)->comm, PETSC_ERR_ARG_WRONG, "Invalid submesh: Too many vertices %d of an element on the surface", faceSize);
      if (faceSize == *nFV) {
        ++(*numFaces);
        for (cl = 0; cl < faceSize; ++cl) {
          ierr = DMLabelSetValue(subpointMap, closure[cl], 0);CHKERRQ(ierr);
        }
        ierr = DMLabelSetValue(subpointMap, cell, 2);CHKERRQ(ierr);
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, vertex, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
  }
  if (subvertexIS) {
    ierr = ISRestoreIndices(subvertexIS, &subvertices);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&subvertexIS);CHKERRQ(ierr);
  ierr = PetscFree3(pStart,pEnd,pMax);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexMarkSubmesh_Interpolated"
static PetscErrorCode DMPlexMarkSubmesh_Interpolated(DM dm, DMLabel vertexLabel, DMLabel subpointMap, DM subdm)
{
  IS               subvertexIS;
  const PetscInt  *subvertices;
  PetscInt        *pStart, *pEnd, *pMax;
  PetscInt         dim, d, numSubVerticesInitial = 0, v;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim+1,PetscInt,&pStart,dim+1,PetscInt,&pEnd,dim+1,PetscInt,&pMax);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &pMax[dim], dim>1 ? &pMax[dim-1] : PETSC_NULL, dim > 2 ? &pMax[1] : PETSC_NULL, &pMax[0]);CHKERRQ(ierr);
  for (d = 0; d <= dim; ++d) {
    ierr = DMPlexGetDepthStratum(dm, d, &pStart[d], &pEnd[d]);CHKERRQ(ierr);
    if (pMax[d] >= 0) pEnd[d] = PetscMin(pEnd[d], pMax[d]);
  }
  /* Loop over initial vertices and mark all faces in the collective star() */
  ierr = DMLabelGetStratumIS(vertexLabel, 1, &subvertexIS);CHKERRQ(ierr);
  if (subvertexIS) {
    ierr = ISGetSize(subvertexIS, &numSubVerticesInitial);CHKERRQ(ierr);
    ierr = ISGetIndices(subvertexIS, &subvertices);CHKERRQ(ierr);
  }
  for (v = 0; v < numSubVerticesInitial; ++v) {
    const PetscInt vertex = subvertices[v];
    PetscInt      *star   = PETSC_NULL;
    PetscInt       starSize, s, numFaces = 0, f;

    ierr = DMPlexGetTransitiveClosure(dm, vertex, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
    for (s = 0; s < starSize*2; s += 2) {
      const PetscInt point = star[s];
      if ((point >= pStart[dim-1]) && (point < pEnd[dim-1])) star[numFaces++] = point;
    }
    for (f = 0; f < numFaces; ++f) {
      const PetscInt face    = star[f];
      PetscInt      *closure = PETSC_NULL;
      PetscInt       closureSize, c;
      PetscInt       faceLoc;

      ierr = DMLabelGetValue(subpointMap, face, &faceLoc);CHKERRQ(ierr);
      if (faceLoc == dim-1) continue;
      if (faceLoc >= 0) SETERRQ2(((PetscObject) dm)->comm, PETSC_ERR_PLIB, "Face %d has dimension %d in the surface label", face, faceLoc);
      ierr = DMPlexGetTransitiveClosure(dm, face, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      for (c = 0; c < closureSize*2; c += 2) {
        const PetscInt point = closure[c];
        PetscInt       vertexLoc;

        if ((point >= pStart[0]) && (point < pEnd[0])) {
          ierr = DMLabelGetValue(vertexLabel, point, &vertexLoc);CHKERRQ(ierr);
          if (vertexLoc < 0) break;
        }
      }
      if (c == closureSize*2) {
        const PetscInt *support;
        PetscInt        supportSize, s;

        for (c = 0; c < closureSize*2; c += 2) {
          const PetscInt point = closure[c];

          for (d = 0; d < dim; ++d) {
            if ((point >= pStart[d]) && (point < pEnd[d])) {
              ierr = DMLabelSetValue(subpointMap, point, d);CHKERRQ(ierr);
              break;
            }
          }
        }
        ierr = DMPlexGetSupportSize(dm, face, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, face, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMLabelSetValue(subpointMap, support[s], dim);CHKERRQ(ierr);
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, face, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, vertex, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
  }
  if (subvertexIS) {
    ierr = ISRestoreIndices(subvertexIS, &subvertices);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&subvertexIS);CHKERRQ(ierr);
  ierr = PetscFree3(pStart,pEnd,pMax);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetFaceOrientation"
PetscErrorCode DMPlexGetFaceOrientation(DM dm, PetscInt cell, PetscInt numCorners, PetscInt indices[], PetscInt oppositeVertex, PetscInt origVertices[], PetscInt faceVertices[], PetscBool *posOriented)
{
  MPI_Comm       comm      = ((PetscObject) dm)->comm;
  PetscBool      posOrient = PETSC_FALSE;
  const PetscInt debug     = 0;
  PetscInt       cellDim, faceSize, f;
  PetscErrorCode ierr;

  ierr = DMPlexGetDimension(dm, &cellDim);CHKERRQ(ierr);
  if (debug) {PetscPrintf(comm, "cellDim: %d numCorners: %d\n", cellDim, numCorners);CHKERRQ(ierr);}

  if (cellDim == numCorners-1) {
    /* Simplices */
    faceSize  = numCorners-1;
    posOrient = !(oppositeVertex%2) ? PETSC_TRUE : PETSC_FALSE;
  } else if (cellDim == 1 && numCorners == 3) {
    /* Quadratic line */
    faceSize  = 1;
    posOrient = PETSC_TRUE;
  } else if (cellDim == 2 && numCorners == 4) {
    /* Quads */
    faceSize = 2;
    if ((indices[1] > indices[0]) && (indices[1] - indices[0] == 1)) {
      posOrient = PETSC_TRUE;
    } else if ((indices[0] == 3) && (indices[1] == 0)) {
      posOrient = PETSC_TRUE;
    } else {
      if (((indices[0] > indices[1]) && (indices[0] - indices[1] == 1)) || ((indices[0] == 0) && (indices[1] == 3))) {
        posOrient = PETSC_FALSE;
      } else SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid quad crossedge");
    }
  } else if (cellDim == 2 && numCorners == 6) {
    /* Quadratic triangle (I hate this) */
    /* Edges are determined by the first 2 vertices (corners of edges) */
    const PetscInt faceSizeTri = 3;
    PetscInt       sortedIndices[3], i, iFace;
    PetscBool      found                    = PETSC_FALSE;
    PetscInt       faceVerticesTriSorted[9] = {
      0, 3,  4, /* bottom */
      1, 4,  5, /* right */
      2, 3,  5, /* left */
    };
    PetscInt       faceVerticesTri[9] = {
      0, 3,  4, /* bottom */
      1, 4,  5, /* right */
      2, 5,  3, /* left */
    };

    faceSize = faceSizeTri;
    for (i = 0; i < faceSizeTri; ++i) sortedIndices[i] = indices[i];
    ierr = PetscSortInt(faceSizeTri, sortedIndices);CHKERRQ(ierr);
    for (iFace = 0; iFace < 3; ++iFace) {
      const PetscInt ii = iFace*faceSizeTri;
      PetscInt       fVertex, cVertex;

      if ((sortedIndices[0] == faceVerticesTriSorted[ii+0]) &&
          (sortedIndices[1] == faceVerticesTriSorted[ii+1])) {
        for (fVertex = 0; fVertex < faceSizeTri; ++fVertex) {
          for (cVertex = 0; cVertex < faceSizeTri; ++cVertex) {
            if (indices[cVertex] == faceVerticesTri[ii+fVertex]) {
              faceVertices[fVertex] = origVertices[cVertex];
              break;
            }
          }
        }
        found = PETSC_TRUE;
        break;
      }
    }
    if (!found) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid tri crossface");
    if (posOriented) *posOriented = PETSC_TRUE;
    PetscFunctionReturn(0);
  } else if (cellDim == 2 && numCorners == 9) {
    /* Quadratic quad (I hate this) */
    /* Edges are determined by the first 2 vertices (corners of edges) */
    const PetscInt faceSizeQuad = 3;
    PetscInt       sortedIndices[3], i, iFace;
    PetscBool      found                      = PETSC_FALSE;
    PetscInt       faceVerticesQuadSorted[12] = {
      0, 1,  4, /* bottom */
      1, 2,  5, /* right */
      2, 3,  6, /* top */
      0, 3,  7, /* left */
    };
    PetscInt       faceVerticesQuad[12] = {
      0, 1,  4, /* bottom */
      1, 2,  5, /* right */
      2, 3,  6, /* top */
      3, 0,  7, /* left */
    };

    faceSize = faceSizeQuad;
    for (i = 0; i < faceSizeQuad; ++i) sortedIndices[i] = indices[i];
    ierr = PetscSortInt(faceSizeQuad, sortedIndices);CHKERRQ(ierr);
    for (iFace = 0; iFace < 4; ++iFace) {
      const PetscInt ii = iFace*faceSizeQuad;
      PetscInt       fVertex, cVertex;

      if ((sortedIndices[0] == faceVerticesQuadSorted[ii+0]) &&
          (sortedIndices[1] == faceVerticesQuadSorted[ii+1])) {
        for (fVertex = 0; fVertex < faceSizeQuad; ++fVertex) {
          for (cVertex = 0; cVertex < faceSizeQuad; ++cVertex) {
            if (indices[cVertex] == faceVerticesQuad[ii+fVertex]) {
              faceVertices[fVertex] = origVertices[cVertex];
              break;
            }
          }
        }
        found = PETSC_TRUE;
        break;
      }
    }
    if (!found) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid quad crossface");
    if (posOriented) *posOriented = PETSC_TRUE;
    PetscFunctionReturn(0);
  } else if (cellDim == 3 && numCorners == 8) {
    /* Hexes
       A hex is two oriented quads with the normal of the first
       pointing up at the second.

          7---6
         /|  /|
        4---5 |
        | 3-|-2
        |/  |/
        0---1

        Faces are determined by the first 4 vertices (corners of faces) */
    const PetscInt faceSizeHex = 4;
    PetscInt       sortedIndices[4], i, iFace;
    PetscBool      found                     = PETSC_FALSE;
    PetscInt       faceVerticesHexSorted[24] = {
      0, 1, 2, 3,  /* bottom */
      4, 5, 6, 7,  /* top */
      0, 1, 4, 5,  /* front */
      1, 2, 5, 6,  /* right */
      2, 3, 6, 7,  /* back */
      0, 3, 4, 7,  /* left */
    };
    PetscInt       faceVerticesHex[24] = {
      3, 2, 1, 0,  /* bottom */
      4, 5, 6, 7,  /* top */
      0, 1, 5, 4,  /* front */
      1, 2, 6, 5,  /* right */
      2, 3, 7, 6,  /* back */
      3, 0, 4, 7,  /* left */
    };

    faceSize = faceSizeHex;
    for (i = 0; i < faceSizeHex; ++i) sortedIndices[i] = indices[i];
    ierr = PetscSortInt(faceSizeHex, sortedIndices);CHKERRQ(ierr);
    for (iFace = 0; iFace < 6; ++iFace) {
      const PetscInt ii = iFace*faceSizeHex;
      PetscInt       fVertex, cVertex;

      if ((sortedIndices[0] == faceVerticesHexSorted[ii+0]) &&
          (sortedIndices[1] == faceVerticesHexSorted[ii+1]) &&
          (sortedIndices[2] == faceVerticesHexSorted[ii+2]) &&
          (sortedIndices[3] == faceVerticesHexSorted[ii+3])) {
        for (fVertex = 0; fVertex < faceSizeHex; ++fVertex) {
          for (cVertex = 0; cVertex < faceSizeHex; ++cVertex) {
            if (indices[cVertex] == faceVerticesHex[ii+fVertex]) {
              faceVertices[fVertex] = origVertices[cVertex];
              break;
            }
          }
        }
        found = PETSC_TRUE;
        break;
      }
    }
    if (!found) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid hex crossface");
    if (posOriented) *posOriented = PETSC_TRUE;
    PetscFunctionReturn(0);
  } else if (cellDim == 3 && numCorners == 10) {
    /* Quadratic tet */
    /* Faces are determined by the first 3 vertices (corners of faces) */
    const PetscInt faceSizeTet = 6;
    PetscInt       sortedIndices[6], i, iFace;
    PetscBool      found                     = PETSC_FALSE;
    PetscInt       faceVerticesTetSorted[24] = {
      0, 1, 2,  6, 7, 8, /* bottom */
      0, 3, 4,  6, 7, 9,  /* front */
      1, 4, 5,  7, 8, 9,  /* right */
      2, 3, 5,  6, 8, 9,  /* left */
    };
    PetscInt       faceVerticesTet[24] = {
      0, 1, 2,  6, 7, 8, /* bottom */
      0, 4, 3,  6, 7, 9,  /* front */
      1, 5, 4,  7, 8, 9,  /* right */
      2, 3, 5,  8, 6, 9,  /* left */
    };

    faceSize = faceSizeTet;
    for (i = 0; i < faceSizeTet; ++i) sortedIndices[i] = indices[i];
    ierr = PetscSortInt(faceSizeTet, sortedIndices);CHKERRQ(ierr);
    for (iFace=0; iFace < 4; ++iFace) {
      const PetscInt ii = iFace*faceSizeTet;
      PetscInt       fVertex, cVertex;

      if ((sortedIndices[0] == faceVerticesTetSorted[ii+0]) &&
          (sortedIndices[1] == faceVerticesTetSorted[ii+1]) &&
          (sortedIndices[2] == faceVerticesTetSorted[ii+2]) &&
          (sortedIndices[3] == faceVerticesTetSorted[ii+3])) {
        for (fVertex = 0; fVertex < faceSizeTet; ++fVertex) {
          for (cVertex = 0; cVertex < faceSizeTet; ++cVertex) {
            if (indices[cVertex] == faceVerticesTet[ii+fVertex]) {
              faceVertices[fVertex] = origVertices[cVertex];
              break;
            }
          }
        }
        found = PETSC_TRUE;
        break;
      }
    }
    if (!found) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid tet crossface");
    if (posOriented) *posOriented = PETSC_TRUE;
    PetscFunctionReturn(0);
  } else if (cellDim == 3 && numCorners == 27) {
    /* Quadratic hexes (I hate this)
       A hex is two oriented quads with the normal of the first
       pointing up at the second.

         7---6
        /|  /|
       4---5 |
       | 3-|-2
       |/  |/
       0---1

       Faces are determined by the first 4 vertices (corners of faces) */
    const PetscInt faceSizeQuadHex = 9;
    PetscInt       sortedIndices[9], i, iFace;
    PetscBool      found                         = PETSC_FALSE;
    PetscInt       faceVerticesQuadHexSorted[54] = {
      0, 1, 2, 3,  8, 9, 10, 11,  24, /* bottom */
      4, 5, 6, 7,  12, 13, 14, 15,  25, /* top */
      0, 1, 4, 5,  8, 12, 16, 17,  22, /* front */
      1, 2, 5, 6,  9, 13, 17, 18,  21, /* right */
      2, 3, 6, 7,  10, 14, 18, 19,  23, /* back */
      0, 3, 4, 7,  11, 15, 16, 19,  20, /* left */
    };
    PetscInt       faceVerticesQuadHex[54] = {
      3, 2, 1, 0,  10, 9, 8, 11,  24, /* bottom */
      4, 5, 6, 7,  12, 13, 14, 15,  25, /* top */
      0, 1, 5, 4,  8, 17, 12, 16,  22, /* front */
      1, 2, 6, 5,  9, 18, 13, 17,  21, /* right */
      2, 3, 7, 6,  10, 19, 14, 18,  23, /* back */
      3, 0, 4, 7,  11, 16, 15, 19,  20 /* left */
    };

    faceSize = faceSizeQuadHex;
    for (i = 0; i < faceSizeQuadHex; ++i) sortedIndices[i] = indices[i];
    ierr = PetscSortInt(faceSizeQuadHex, sortedIndices);CHKERRQ(ierr);
    for (iFace = 0; iFace < 6; ++iFace) {
      const PetscInt ii = iFace*faceSizeQuadHex;
      PetscInt       fVertex, cVertex;

      if ((sortedIndices[0] == faceVerticesQuadHexSorted[ii+0]) &&
          (sortedIndices[1] == faceVerticesQuadHexSorted[ii+1]) &&
          (sortedIndices[2] == faceVerticesQuadHexSorted[ii+2]) &&
          (sortedIndices[3] == faceVerticesQuadHexSorted[ii+3])) {
        for (fVertex = 0; fVertex < faceSizeQuadHex; ++fVertex) {
          for (cVertex = 0; cVertex < faceSizeQuadHex; ++cVertex) {
            if (indices[cVertex] == faceVerticesQuadHex[ii+fVertex]) {
              faceVertices[fVertex] = origVertices[cVertex];
              break;
            }
          }
        }
        found = PETSC_TRUE;
        break;
      }
    }
    if (!found) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid hex crossface");
    if (posOriented) *posOriented = PETSC_TRUE;
    PetscFunctionReturn(0);
  } else SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Unknown cell type for faceOrientation().");
  if (!posOrient) {
    if (debug) {ierr = PetscPrintf(comm, "  Reversing initial face orientation\n");CHKERRQ(ierr);}
    for (f = 0; f < faceSize; ++f) faceVertices[f] = origVertices[faceSize-1 - f];
  } else {
    if (debug) {ierr = PetscPrintf(comm, "  Keeping initial face orientation\n");CHKERRQ(ierr);}
    for (f = 0; f < faceSize; ++f) faceVertices[f] = origVertices[f];
  }
  if (posOriented) *posOriented = posOrient;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetOrientedFace"
/*
    Given a cell and a face, as a set of vertices,
      return the oriented face, as a set of vertices, in faceVertices
    The orientation is such that the face normal points out of the cell
*/
PetscErrorCode DMPlexGetOrientedFace(DM dm, PetscInt cell, PetscInt faceSize, const PetscInt face[], PetscInt numCorners, PetscInt indices[], PetscInt origVertices[], PetscInt faceVertices[], PetscBool *posOriented)
{
  const PetscInt *cone = PETSC_NULL;
  PetscInt        coneSize, v, f, v2;
  PetscInt        oppositeVertex = -1;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetConeSize(dm, cell, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, cell, &cone);CHKERRQ(ierr);
  for (v = 0, v2 = 0; v < coneSize; ++v) {
    PetscBool found = PETSC_FALSE;

    for (f = 0; f < faceSize; ++f) {
      if (face[f] == cone[v]) {
        found = PETSC_TRUE; break;
      }
    }
    if (found) {
      indices[v2]      = v;
      origVertices[v2] = cone[v];
      ++v2;
    } else {
      oppositeVertex = v;
    }
  }
  ierr = DMPlexGetFaceOrientation(dm, cell, numCorners, indices, oppositeVertex, origVertices, faceVertices, posOriented);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexInsertFace_Private"
/*
  DMPlexInsertFace_Private - Puts a face into the mesh

  Not collective

  Input Parameters:
  + dm              - The DMPlex
  . numFaceVertex   - The number of vertices in the face
  . faceVertices    - The vertices in the face for dm
  . subfaceVertices - The vertices in the face for subdm
  . numCorners      - The number of vertices in the cell
  . cell            - A cell in dm containing the face
  . subcell         - A cell in subdm containing the face
  . firstFace       - First face in the mesh
  - newFacePoint    - Next face in the mesh

  Output Parameters:
  . newFacePoint - Contains next face point number on input, updated on output

  Level: developer
*/
PetscErrorCode DMPlexInsertFace_Private(DM dm, DM subdm, PetscInt numFaceVertices, const PetscInt faceVertices[], const PetscInt subfaceVertices[], PetscInt numCorners, PetscInt cell, PetscInt subcell, PetscInt firstFace, PetscInt *newFacePoint)
{
  MPI_Comm        comm    = ((PetscObject) dm)->comm;
  DM_Plex        *submesh = (DM_Plex*) subdm->data;
  const PetscInt *faces;
  PetscInt        numFaces, coneSize;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetConeSize(subdm, subcell, &coneSize);CHKERRQ(ierr);
  if (coneSize != 1) SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cone size of cell %d is %d != 1", cell, coneSize);
#if 0
  /* Cannot use this because support() has not been constructed yet */
  ierr = DMPlexGetJoin(subdm, numFaceVertices, subfaceVertices, &numFaces, &faces);CHKERRQ(ierr);
#else
  {
    PetscInt f;

    numFaces = 0;
    ierr     = DMGetWorkArray(subdm, 1, PETSC_INT, (void**) &faces);CHKERRQ(ierr);
    for (f = firstFace; f < *newFacePoint; ++f) {
      PetscInt dof, off, d;

      ierr = PetscSectionGetDof(submesh->coneSection, f, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(submesh->coneSection, f, &off);CHKERRQ(ierr);
      /* Yes, I know this is quadratic, but I expect the sizes to be <5 */
      for (d = 0; d < dof; ++d) {
        const PetscInt p = submesh->cones[off+d];
        PetscInt       v;

        for (v = 0; v < numFaceVertices; ++v) {
          if (subfaceVertices[v] == p) break;
        }
        if (v == numFaceVertices) break;
      }
      if (d == dof) {
        numFaces               = 1;
        ((PetscInt*) faces)[0] = f;
      }
    }
  }
#endif
  if (numFaces > 1) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Vertex set had %d faces, not one", numFaces);
  else if (numFaces == 1) {
    /* Add the other cell neighbor for this face */
    ierr = DMPlexSetCone(subdm, cell, faces);CHKERRQ(ierr);
  } else {
    PetscInt *indices, *origVertices, *orientedVertices, *orientedSubVertices, v, ov;
    PetscBool posOriented;

    ierr                = DMGetWorkArray(subdm, 4*numFaceVertices * sizeof(PetscInt), PETSC_INT, &orientedVertices);CHKERRQ(ierr);
    origVertices        = &orientedVertices[numFaceVertices];
    indices             = &orientedVertices[numFaceVertices*2];
    orientedSubVertices = &orientedVertices[numFaceVertices*3];
    ierr                = DMPlexGetOrientedFace(dm, cell, numFaceVertices, faceVertices, numCorners, indices, origVertices, orientedVertices, &posOriented);CHKERRQ(ierr);
    /* TODO: I know that routine should return a permutation, not the indices */
    for (v = 0; v < numFaceVertices; ++v) {
      const PetscInt vertex = faceVertices[v], subvertex = subfaceVertices[v];
      for (ov = 0; ov < numFaceVertices; ++ov) {
        if (orientedVertices[ov] == vertex) {
          orientedSubVertices[ov] = subvertex;
          break;
        }
      }
      if (ov == numFaceVertices) SETERRQ1(comm, PETSC_ERR_PLIB, "Could not find face vertex %d in orientated set", vertex);
    }
    ierr = DMPlexSetCone(subdm, *newFacePoint, orientedSubVertices);CHKERRQ(ierr);
    ierr = DMPlexSetCone(subdm, subcell, newFacePoint);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(subdm, 4*numFaceVertices * sizeof(PetscInt), PETSC_INT, &orientedVertices);CHKERRQ(ierr);
    ++(*newFacePoint);
  }
  ierr = DMPlexRestoreJoin(subdm, numFaceVertices, subfaceVertices, &numFaces, &faces);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateSubmesh_Uninterpolated"
static PetscErrorCode DMPlexCreateSubmesh_Uninterpolated(DM dm, const char vertexLabelName[], DM subdm)
{
  MPI_Comm        comm = ((PetscObject) dm)->comm;
  DMLabel         vertexLabel, subpointMap;
  IS              subvertexIS,  subcellIS;
  const PetscInt *subVertices, *subCells;
  PetscInt        numSubVertices, firstSubVertex, numSubCells;
  PetscInt       *subface, maxConeSize, numSubFaces, firstSubFace, newFacePoint, nFV;
  PetscInt        vStart, vEnd, c, f;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Create subpointMap which marks the submesh */
  ierr = DMLabelCreate("subpoint_map", &subpointMap);CHKERRQ(ierr);
  ierr = DMPlexSetSubpointMap(subdm, subpointMap);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&subpointMap);CHKERRQ(ierr);
  ierr = DMPlexGetLabel(dm, vertexLabelName, &vertexLabel);CHKERRQ(ierr);
  ierr = DMPlexMarkSubmesh_Uninterpolated(dm, vertexLabel, subpointMap, &numSubFaces, &nFV, subdm);CHKERRQ(ierr);
  /* Setup chart */
  ierr = DMLabelGetStratumSize(subpointMap, 0, &numSubVertices);CHKERRQ(ierr);
  ierr = DMLabelGetStratumSize(subpointMap, 2, &numSubCells);CHKERRQ(ierr);
  ierr = DMPlexSetChart(subdm, 0, numSubCells+numSubFaces+numSubVertices);CHKERRQ(ierr);
  ierr = DMPlexSetVTKCellHeight(subdm, 1);CHKERRQ(ierr);
  /* Set cone sizes */
  firstSubVertex = numSubCells;
  firstSubFace   = numSubCells+numSubVertices;
  newFacePoint   = firstSubFace;
  ierr = DMLabelGetStratumIS(subpointMap, 0, &subvertexIS);CHKERRQ(ierr);
  if (subvertexIS) {ierr = ISGetIndices(subvertexIS, &subVertices);CHKERRQ(ierr);}
  ierr = DMLabelGetStratumIS(subpointMap, 2, &subcellIS);CHKERRQ(ierr);
  if (subcellIS) {ierr = ISGetIndices(subcellIS, &subCells);CHKERRQ(ierr);}
  for (c = 0; c < numSubCells; ++c) {
    ierr = DMPlexSetConeSize(subdm, c, 1);CHKERRQ(ierr);
  }
  for (f = firstSubFace; f < firstSubFace+numSubFaces; ++f) {
    ierr = DMPlexSetConeSize(subdm, f, nFV);CHKERRQ(ierr);
  }
  ierr = DMSetUp(subdm);CHKERRQ(ierr);
  /* Create face cones */
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, PETSC_NULL);CHKERRQ(ierr);
  ierr = DMGetWorkArray(subdm, maxConeSize, PETSC_INT, (void**) &subface);CHKERRQ(ierr);
  for (c = 0; c < numSubCells; ++c) {
    const PetscInt cell    = subCells[c];
    const PetscInt subcell = c;
    PetscInt      *closure = PETSC_NULL;
    PetscInt       closureSize, cl, numCorners = 0, faceSize = 0;

    ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    for (cl = 0; cl < closureSize*2; cl += 2) {
      const PetscInt point = closure[cl];
      PetscInt       subVertex;

      if ((point >= vStart) && (point < vEnd)) {
        ++numCorners;
        ierr = PetscFindInt(point, numSubVertices, subVertices, &subVertex);CHKERRQ(ierr);
        if (subVertex >= 0) {
          closure[faceSize] = point;
          subface[faceSize] = subVertex;
          ++faceSize;
        }
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    if (faceSize > nFV) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Invalid submesh: Too many vertices %d of an element on the surface", faceSize);
    if (faceSize == nFV) {
      ierr = DMPlexInsertFace_Private(dm, subdm, faceSize, closure, subface, numCorners, cell, subcell, firstSubFace, &newFacePoint);CHKERRQ(ierr);
    }
  }
  ierr = DMRestoreWorkArray(subdm, maxConeSize, PETSC_INT, (void**) &subface);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(subdm);CHKERRQ(ierr);
  ierr = DMPlexStratify(subdm);CHKERRQ(ierr);
  /* Build coordinates */
  {
    PetscSection coordSection, subCoordSection;
    Vec          coordinates, subCoordinates;
    PetscScalar *coords, *subCoords;
    PetscInt     coordSize, v;

    ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMPlexGetCoordinateSection(subdm, &subCoordSection);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(subCoordSection, firstSubVertex, firstSubVertex+numSubVertices);CHKERRQ(ierr);
    for (v = 0; v < numSubVertices; ++v) {
      const PetscInt vertex    = subVertices[v];
      const PetscInt subvertex = firstSubVertex+v;
      PetscInt       dof;

      ierr = PetscSectionGetDof(coordSection, vertex, &dof);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(subCoordSection, subvertex, dof);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(subCoordSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(subCoordSection, &coordSize);CHKERRQ(ierr);
    ierr = VecCreate(comm, &subCoordinates);CHKERRQ(ierr);
    ierr = VecSetSizes(subCoordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(subCoordinates);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates,    &coords);CHKERRQ(ierr);
    ierr = VecGetArray(subCoordinates, &subCoords);CHKERRQ(ierr);
    for (v = 0; v < numSubVertices; ++v) {
      const PetscInt vertex    = subVertices[v];
      const PetscInt subvertex = firstSubVertex+v;
      PetscInt       dof, off, sdof, soff, d;

      ierr = PetscSectionGetDof(coordSection, vertex, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSection, vertex, &off);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(subCoordSection, subvertex, &sdof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(subCoordSection, subvertex, &soff);CHKERRQ(ierr);
      if (dof != sdof) SETERRQ4(comm, PETSC_ERR_PLIB, "Coordinate dimension %d on subvertex %d, vertex %d should be %d", sdof, subvertex, vertex, dof);
      for (d = 0; d < dof; ++d) subCoords[soff+d] = coords[off+d];
    }
    ierr = VecRestoreArray(coordinates,    &coords);CHKERRQ(ierr);
    ierr = VecRestoreArray(subCoordinates, &subCoords);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(subdm, subCoordinates);CHKERRQ(ierr);
    ierr = VecDestroy(&subCoordinates);CHKERRQ(ierr);
  }
  /* Cleanup */
  if (subvertexIS) {ierr = ISRestoreIndices(subvertexIS, &subVertices);CHKERRQ(ierr);}
  ierr = ISDestroy(&subvertexIS);CHKERRQ(ierr);
  if (subcellIS) {ierr = ISRestoreIndices(subcellIS, &subCells);CHKERRQ(ierr);}
  ierr = ISDestroy(&subcellIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateSubmesh_Interpolated"
static PetscErrorCode DMPlexCreateSubmesh_Interpolated(DM dm, const char vertexLabelName[], DM subdm)
{
  MPI_Comm         comm = ((PetscObject) dm)->comm;
  DMLabel          subpointMap, vertexLabel;
  IS              *subpointIS;
  const PetscInt **subpoints;
  PetscInt        *numSubPoints, *firstSubPoint, *coneNew;
  PetscInt         totSubPoints = 0, maxConeSize, dim, p, d, v;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  /* Create subpointMap which marks the submesh */
  ierr = DMLabelCreate("subpoint_map", &subpointMap);CHKERRQ(ierr);
  ierr = DMPlexSetSubpointMap(subdm, subpointMap);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&subpointMap);CHKERRQ(ierr);
  ierr = DMPlexGetLabel(dm, vertexLabelName, &vertexLabel);CHKERRQ(ierr);
  ierr = DMPlexMarkSubmesh_Interpolated(dm, vertexLabel, subpointMap, subdm);CHKERRQ(ierr);
  /* Setup chart */
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscMalloc4(dim+1,PetscInt,&numSubPoints,dim+1,PetscInt,&firstSubPoint,dim+1,IS,&subpointIS,dim+1,const PetscInt *,&subpoints);CHKERRQ(ierr);
  for (d = 0; d <= dim; ++d) {
    ierr = DMLabelGetStratumSize(subpointMap, d, &numSubPoints[d]);CHKERRQ(ierr);
    totSubPoints += numSubPoints[d];
  }
  ierr = DMPlexSetChart(subdm, 0, totSubPoints);CHKERRQ(ierr);
  ierr = DMPlexSetVTKCellHeight(subdm, 1);CHKERRQ(ierr);
  /* Set cone sizes */
  firstSubPoint[dim] = 0;
  firstSubPoint[0]   = firstSubPoint[dim] + numSubPoints[dim];
  if (dim > 1) {firstSubPoint[dim-1] = firstSubPoint[0]     + numSubPoints[0];}
  if (dim > 2) {firstSubPoint[dim-2] = firstSubPoint[dim-1] + numSubPoints[dim-1];}
  for (d = 0; d <= dim; ++d) {
    ierr = DMLabelGetStratumIS(subpointMap, d, &subpointIS[d]);CHKERRQ(ierr);
    ierr = ISGetIndices(subpointIS[d], &subpoints[d]);CHKERRQ(ierr);
  }
  for (d = 0; d <= dim; ++d) {
    for (p = 0; p < numSubPoints[d]; ++p) {
      const PetscInt  point    = subpoints[d][p];
      const PetscInt  subpoint = firstSubPoint[d] + p;
      const PetscInt *cone;
      PetscInt        coneSize, coneSizeNew, c, val;

      ierr = DMPlexGetConeSize(dm, point, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexSetConeSize(subdm, subpoint, coneSize);CHKERRQ(ierr);
      if (d == dim) {
        ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
        for (c = 0, coneSizeNew = 0; c < coneSize; ++c) {
          ierr = DMLabelGetValue(subpointMap, cone[c], &val);CHKERRQ(ierr);
          if (val >= 0) coneSizeNew++;
        }
        ierr = DMPlexSetConeSize(subdm, subpoint, coneSizeNew);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMSetUp(subdm);CHKERRQ(ierr);
  /* Set cones */
  ierr = DMPlexGetMaxSizes(dm, &maxConeSize, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscMalloc(maxConeSize * sizeof(PetscInt), &coneNew);CHKERRQ(ierr);
  for (d = 0; d <= dim; ++d) {
    for (p = 0; p < numSubPoints[d]; ++p) {
      const PetscInt  point    = subpoints[d][p];
      const PetscInt  subpoint = firstSubPoint[d] + p;
      const PetscInt *cone;
      PetscInt        coneSize, subconeSize, coneSizeNew, c, subc;

      ierr = DMPlexGetConeSize(dm, point, &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetConeSize(subdm, subpoint, &subconeSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, point, &cone);CHKERRQ(ierr);
      for (c = 0, coneSizeNew = 0; c < coneSize; ++c) {
        ierr = PetscFindInt(cone[c], numSubPoints[d-1], subpoints[d-1], &subc);CHKERRQ(ierr);
        if (subc >= 0) coneNew[coneSizeNew++] = firstSubPoint[d-1] + subc;
      }
      if (coneSizeNew != subconeSize) SETERRQ2(comm, PETSC_ERR_PLIB, "Number of cone points located %d does not match subcone size %d", coneSizeNew, subconeSize);
      ierr = DMPlexSetCone(subdm, subpoint, coneNew);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(coneNew);CHKERRQ(ierr);
  ierr = DMPlexSymmetrize(subdm);CHKERRQ(ierr);
  ierr = DMPlexStratify(subdm);CHKERRQ(ierr);
  /* Build coordinates */
  {
    PetscSection coordSection, subCoordSection;
    Vec          coordinates, subCoordinates;
    PetscScalar *coords, *subCoords;
    PetscInt     coordSize;

    ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMPlexGetCoordinateSection(subdm, &subCoordSection);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(subCoordSection, firstSubPoint[0], firstSubPoint[0]+numSubPoints[0]);CHKERRQ(ierr);
    for (v = 0; v < numSubPoints[0]; ++v) {
      const PetscInt vertex    = subpoints[0][v];
      const PetscInt subvertex = firstSubPoint[0]+v;
      PetscInt       dof;

      ierr = PetscSectionGetDof(coordSection, vertex, &dof);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(subCoordSection, subvertex, dof);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(subCoordSection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(subCoordSection, &coordSize);CHKERRQ(ierr);
    ierr = VecCreate(comm, &subCoordinates);CHKERRQ(ierr);
    ierr = VecSetSizes(subCoordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetFromOptions(subCoordinates);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates,    &coords);CHKERRQ(ierr);
    ierr = VecGetArray(subCoordinates, &subCoords);CHKERRQ(ierr);
    for (v = 0; v < numSubPoints[0]; ++v) {
      const PetscInt vertex    = subpoints[0][v];
      const PetscInt subvertex = firstSubPoint[0]+v;
      PetscInt dof, off, sdof, soff, d;

      ierr = PetscSectionGetDof(coordSection, vertex, &dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSection, vertex, &off);CHKERRQ(ierr);
      ierr = PetscSectionGetDof(subCoordSection, subvertex, &sdof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(subCoordSection, subvertex, &soff);CHKERRQ(ierr);
      if (dof != sdof) SETERRQ4(comm, PETSC_ERR_PLIB, "Coordinate dimension %d on subvertex %d, vertex %d should be %d", sdof, subvertex, vertex, dof);
      for (d = 0; d < dof; ++d) subCoords[soff+d] = coords[off+d];
    }
    ierr = VecRestoreArray(coordinates,    &coords);CHKERRQ(ierr);
    ierr = VecRestoreArray(subCoordinates, &subCoords);CHKERRQ(ierr);
    ierr = DMSetCoordinatesLocal(subdm, subCoordinates);CHKERRQ(ierr);
    ierr = VecDestroy(&subCoordinates);CHKERRQ(ierr);
  }
  /* Cleanup */
  for (d = 0; d <= dim; ++d) {
    ierr = ISRestoreIndices(subpointIS[d], &subpoints[d]);CHKERRQ(ierr);
    ierr = ISDestroy(&subpointIS[d]);CHKERRQ(ierr);
  }
  ierr = PetscFree4(numSubPoints,firstSubPoint,subpointIS,subpoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateSubmesh"
/*
  DMPlexCreateSubmesh - Extract a hypersurface from the mesh using vertices defined by a label

  Input Parameters:
+ dm           - The original mesh
- vertexLabel  - The DMLabel marking vertices contained in the surface

  Output Parameter:
. subdm - The surface mesh

  Note: This function produces a DMLabel mapping original points in the submesh to their depth. This can be obtained using DMPlexGetSubpointMap().

  Level: developer

.seealso: DMPlexGetSubpointMap(), DMPlexGetLabel(), DMLabelSetValue()
*/
PetscErrorCode DMPlexCreateSubmesh(DM dm, const char vertexLabel[], DM *subdm)
{
  PetscInt       dim, depth;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidCharPointer(vertexLabel, 2);
  PetscValidPointer(subdm, 4);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMCreate(((PetscObject) dm)->comm, subdm);CHKERRQ(ierr);
  ierr = DMSetType(*subdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(*subdm, dim-1);CHKERRQ(ierr);
  if (depth == dim) {
    ierr = DMPlexCreateSubmesh_Interpolated(dm, vertexLabel, *subdm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateSubmesh_Uninterpolated(dm, vertexLabel, *subdm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetSubpointMap"
PetscErrorCode DMPlexGetSubpointMap(DM dm, DMLabel *subpointMap)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(subpointMap, 2);
  *subpointMap = mesh->subpointMap;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetSubpointMap"
/* Note: Should normally not be called by the user, since it is set in DMPlexCreateSubmesh() */
PetscErrorCode DMPlexSetSubpointMap(DM dm, DMLabel subpointMap)
{
  DM_Plex       *mesh = (DM_Plex *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMLabelDestroy(&mesh->subpointMap);CHKERRQ(ierr);
  mesh->subpointMap = subpointMap;
  ++mesh->subpointMap->refct;CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateSubpointIS"
/*
  DMPlexCreateSubpointIS - Creates an IS covering the entire subdm chart with the original points as data

  Input Parameter:
. dm - The submesh DM

  Output Parameter:
. subpointIS - The IS of all the points from the original mesh in this submesh, or PETSC_NULL if this is not a submesh

  Note: This is IS is guaranteed to be sorted by the construction of the submesh
*/
PetscErrorCode DMPlexCreateSubpointIS(DM dm, IS *subpointIS)
{
  MPI_Comm        comm = ((PetscObject) dm)->comm;
  DMLabel         subpointMap;
  IS              is;
  const PetscInt *opoints;
  PetscInt       *points, *depths;
  PetscInt        depth, depStart, depEnd, d, pStart, pEnd, p, n, off;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(subpointIS, 2);
  *subpointIS = PETSC_NULL;
  ierr = DMPlexGetSubpointMap(dm, &subpointMap);CHKERRQ(ierr);
  if (subpointMap) {
    ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
    ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    if (pStart) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Submeshes must start the point numbering at 0, not %d", pStart);
    ierr = DMGetWorkArray(dm, depth+1, PETSC_INT, &depths);CHKERRQ(ierr);
    depths[0] = depth;
    depths[1] = 0;
    for(d = 2; d <= depth; ++d) {depths[d] = depth+1 - d;}
    ierr = PetscMalloc(pEnd * sizeof(PetscInt), &points);CHKERRQ(ierr);
    for(d = 0, off = 0; d <= depth; ++d) {
      const PetscInt dep = depths[d];

      ierr = DMPlexGetDepthStratum(dm, dep, &depStart, &depEnd);CHKERRQ(ierr);
      ierr = DMLabelGetStratumSize(subpointMap, dep, &n);CHKERRQ(ierr);
      if (d < 2) { /* Only check vertices and cells for now since the map is broken for others */
        if (n != depEnd-depStart) SETERRQ3(comm, PETSC_ERR_ARG_WRONG, "The number of mapped submesh points %d at depth %d should be %d", n, dep, depEnd-depStart);
      } else {
        if (!n) for(p = 0; p < depEnd-depStart; ++p, ++off) points[off] = PETSC_MAX_INT;
      }
      if (n) {
        ierr = DMLabelGetStratumIS(subpointMap, dep, &is);CHKERRQ(ierr);
        ierr = ISGetIndices(is, &opoints);CHKERRQ(ierr);
        for(p = 0; p < n; ++p, ++off) points[off] = opoints[p];
        ierr = ISRestoreIndices(is, &opoints);CHKERRQ(ierr);
        ierr = ISDestroy(&is);CHKERRQ(ierr);
      }
    }
    ierr = DMRestoreWorkArray(dm, depth+1, PETSC_INT, &depths);CHKERRQ(ierr);
    if (off != pEnd) SETERRQ2(comm, PETSC_ERR_ARG_WRONG, "The number of mapped submesh points %d should be %d", off, pEnd);
    ierr = ISCreateGeneral(comm, pEnd, points, PETSC_OWN_POINTER, subpointIS);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
