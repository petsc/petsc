#include <petsc-private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/
#include <petscsf.h>

#undef __FUNCT__
#define __FUNCT__ "GetDepthStart_Private"
PETSC_STATIC_INLINE PetscErrorCode GetDepthStart_Private(PetscInt depth, PetscInt depthSize[], PetscInt *cStart, PetscInt *fStart, PetscInt *eStart, PetscInt *vStart)
{
  PetscFunctionBegin;
  if (cStart) *cStart = 0;
  if (vStart) *vStart = depthSize[depth];
  if (fStart) *fStart = depthSize[depth] + depthSize[0];
  if (eStart) *eStart = depthSize[depth] + depthSize[0] + depthSize[depth-1];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetDepthEnd_Private"
PETSC_STATIC_INLINE PetscErrorCode GetDepthEnd_Private(PetscInt depth, PetscInt depthSize[], PetscInt *cEnd, PetscInt *fEnd, PetscInt *eEnd, PetscInt *vEnd)
{
  PetscFunctionBegin;
  if (cEnd) *cEnd = depthSize[depth];
  if (vEnd) *vEnd = depthSize[depth] + depthSize[0];
  if (fEnd) *fEnd = depthSize[depth] + depthSize[0] + depthSize[depth-1];
  if (eEnd) *eEnd = depthSize[depth] + depthSize[0] + depthSize[depth-1] + depthSize[1];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CellRefinerGetSizes"
PetscErrorCode CellRefinerGetSizes(CellRefiner refiner, DM dm, PetscInt depthSize[])
{
  PetscInt       cStart, cEnd, cMax, vStart, vEnd, vMax, fStart, fEnd, fMax, eStart, eEnd, eMax;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, &fMax, &eMax, &vMax);CHKERRQ(ierr);
  switch (refiner) {
  case 0:
    break;
  case 1:
    /* Simplicial 2D */
    depthSize[0] = vEnd - vStart + fEnd - fStart;         /* Add a vertex on every face */
    depthSize[1] = 2*(fEnd - fStart) + 3*(cEnd - cStart); /* Every face is split into 2 faces and 3 faces are added for each cell */
    depthSize[2] = 4*(cEnd - cStart);                     /* Every cell split into 4 cells */
    break;
  case 3:
    /* Hybrid Simplicial 2D */
    if (cMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No cell maximum specified in hybrid mesh");
    if (fMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No face maximum specified in hybrid mesh");
    depthSize[0] = vEnd - vStart + fMax - fStart;                                         /* Add a vertex on every face, but not hybrid faces */
    depthSize[1] = 2*(fMax - fStart) + 3*(cMax - cStart) + (fEnd - fMax) + (cEnd - cMax); /* Every interior face is split into 2 faces, 3 faces are added for each interior cell, and one in each hybrid cell */
    depthSize[2] = 4*(cMax - cStart) + 2*(cEnd - cMax);                                   /* Interior cells split into 4 cells, Hybrid cells split into 2 cells */
    break;
  case 2:
    /* Hex 2D */
    depthSize[0] = vEnd - vStart + cEnd - cStart + fEnd - fStart; /* Add a vertex on every face and cell */
    depthSize[1] = 2*(fEnd - fStart) + 4*(cEnd - cStart);         /* Every face is split into 2 faces and 4 faces are added for each cell */
    depthSize[2] = 4*(cEnd - cStart);                             /* Every cell split into 4 cells */
    break;
  case 5:
    /* Simplicial 3D */
    depthSize[0] =    vEnd - vStart  +    eEnd - eStart;                    /* Add a vertex on every edge */
    depthSize[1] = 2*(eEnd - eStart) + 3*(fEnd - fStart) + (cEnd - cStart); /* Every edge is split into 2 edges, 3 edges are added for each face, and 1 edge for each cell */
    depthSize[2] = 4*(fEnd - fStart) + 8*(cEnd - cStart);                   /* Every face split into 4 faces and 8 faces are added for each cell */
    depthSize[3] = 8*(cEnd - cStart);                                       /* Every cell split into 8 cells */
    break;
  case 7:
    /* Hybrid Simplicial 3D */
    if (cMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No cell maximum specified in hybrid mesh");
    if (fMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No face maximum specified in hybrid mesh");
    if (eMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No edge maximum specified in hybrid mesh");
    /* Tetrahedra */
    depthSize[0]  =    vEnd - vStart  +    eMax - eStart;                    /* Add a vertex on every interior edge */
    depthSize[1]  = 2*(eMax - eStart) + 3*(fMax - fStart) + (cMax - cStart); /* Every interior edge split into 2 edges, 3 edges added for each interior face, 1 edge for each interior cell */
    depthSize[2]  = 4*(fMax - fStart) + 8*(cMax - cStart);                   /* Every interior face split into 4 faces, 8 faces added for each interior cell */
    depthSize[3]  = 8*(cMax - cStart);                                       /* Every interior cell split into 8 cells */
    /* Triangular Prisms */
    depthSize[0] += 0;                                                       /* No hybrid vertices */
    depthSize[1] +=   (eEnd - eMax)   +   (fEnd - fMax);                     /* Every hybrid edge remains, 1 edge for every hybrid face */
    depthSize[2] += 2*(fEnd - fMax)   + 3*(cEnd - cMax);                     /* Every hybrid face split into 2 faces and 3 faces are added for each hybrid cell */
    depthSize[3] += 4*(cEnd - cMax);                                         /* Every hybrid cell split into 4 cells */
    break;
  case 6:
    /* Hex 3D */
    depthSize[0] = vEnd - vStart + eEnd - eStart + fEnd - fStart + cEnd - cStart; /* Add a vertex on every edge, face and cell */
    depthSize[1] = 2*(eEnd - eStart) +  4*(fEnd - fStart) + 6*(cEnd - cStart);    /* Every edge is split into 2 edge, 4 edges are added for each face, and 6 edges for each cell */
    depthSize[2] = 4*(fEnd - fStart) + 12*(cEnd - cStart);                        /* Every face is split into 4 faces, and 12 faces are added for each cell */
    depthSize[3] = 8*(cEnd - cStart);                                             /* Every cell split into 8 cells */
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell refiner %d", refiner);
  }
  PetscFunctionReturn(0);
}

/* Return triangle edge for orientation o, if it is r for o == 0 */
PETSC_STATIC_INLINE PetscInt GetTriEdge_Static(PetscInt o, PetscInt r) {
  return (o < 0 ? 2-(o+r) : o+r)%3;
}

/* Return triangle subface for orientation o, if it is r for o == 0 */
PETSC_STATIC_INLINE PetscInt GetTriSubface_Static(PetscInt o, PetscInt r) {
  return (o < 0 ? 3-(o+r) : o+r)%3;
}

/* Return quad edge for orientation o, if it is r for o == 0 */
PETSC_STATIC_INLINE PetscInt GetQuadEdge_Static(PetscInt o, PetscInt r) {
  return (o < 0 ? 3-(o+r) : o+r)%4;
}

/* Return quad subface for orientation o, if it is r for o == 0 */
PETSC_STATIC_INLINE PetscInt GetQuadSubface_Static(PetscInt o, PetscInt r) {
  return (o < 0 ? 4-(o+r) : o+r)%4;
}

#undef __FUNCT__
#define __FUNCT__ "CellRefinerSetConeSizes"
PetscErrorCode CellRefinerSetConeSizes(CellRefiner refiner, DM dm, PetscInt depthSize[], DM rdm)
{
  PetscInt       depth, cStart, cStartNew, cEnd, cMax, c, vStart, vStartNew, vEnd, vMax, v, fStart, fStartNew, fEnd, fMax, f, eStart, eStartNew, eEnd, eMax, e, r;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, &fMax, &eMax, &vMax);CHKERRQ(ierr);
  if (refiner) {ierr = GetDepthStart_Private(depth, depthSize, &cStartNew, &fStartNew, &eStartNew, &vStartNew);CHKERRQ(ierr);}
  switch (refiner) {
  case 0: break;
  case 1:
    /* Simplicial 2D */
    /* All cells have 3 faces */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = (c - cStart)*4 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 3);CHKERRQ(ierr);
      }
    }
    /* Split faces have 2 vertices and the same cells as the parent */
    for (f = fStart; f < fEnd; ++f) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = fStartNew + (f - fStart)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior faces have 2 vertices and 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 3; ++r) {
        const PetscInt newp = fStartNew + (fEnd - fStart)*2 + (c - cStart)*3 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newp = vStartNew + (v - vStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Face vertices have 2 + cells*2 supports */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (f - fStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2 + size*2);CHKERRQ(ierr);
    }
    break;
  case 2:
    /* Hex 2D */
    /* All cells have 4 faces */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = (c - cStart)*4 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
      }
    }
    /* Split faces have 2 vertices and the same cells as the parent */
    for (f = fStart; f < fEnd; ++f) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = fStartNew + (f - fStart)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior faces have 2 vertices and 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = fStartNew + (fEnd - fStart)*2 + (c - cStart)*4 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newp = vStartNew + (v - vStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Face vertices have 2 + cells supports */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (f - fStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2 + size);CHKERRQ(ierr);
    }
    /* Cell vertices have 4 supports */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (fEnd - fStart) + (c - cStart);

      ierr = DMPlexSetSupportSize(rdm, newp, 4);CHKERRQ(ierr);
    }
    break;
  case 3:
    /* Hybrid Simplicial 2D */
    if (cMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No cell maximum specified in hybrid mesh");
    cMax = PetscMin(cEnd, cMax);
    if (fMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No face maximum specified in hybrid mesh");
    fMax = PetscMin(fEnd, fMax);
    ierr = DMPlexSetHybridBounds(rdm, cStartNew + (cMax - cStart)*4, fStartNew + (fMax - fStart)*2 + (cMax - cStart)*3, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
    /* Interior cells have 3 faces */
    for (c = cStart; c < cMax; ++c) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = cStartNew + (c - cStart)*4 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 3);CHKERRQ(ierr);
      }
    }
    /* Hybrid cells have 4 faces */
    for (c = cMax; c < cEnd; ++c) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = cStartNew + (cMax - cStart)*4 + (c - cMax)*2 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
      }
    }
    /* Interior split faces have 2 vertices and the same cells as the parent */
    for (f = fStart; f < fMax; ++f) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = fStartNew + (f - fStart)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior cell faces have 2 vertices and 2 cells */
    for (c = cStart; c < cMax; ++c) {
      for (r = 0; r < 3; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*2 + (c - cStart)*3 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Hybrid faces have 2 vertices and the same cells */
    for (f = fMax; f < fEnd; ++f) {
      const PetscInt newp = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*3 + (f - fMax);
      PetscInt       size;

      ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Hybrid cell faces have 2 vertices and 2 cells */
    for (c = cMax; c < cEnd; ++c) {
      const PetscInt newp = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*3 + (fEnd - fMax) + (c - cMax);

      ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newp = vStartNew + (v - vStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Face vertices have 2 + (2 interior, 1 hybrid) supports */
    for (f = fStart; f < fMax; ++f) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (f - fStart);
      const PetscInt *support;
      PetscInt       size, newSize = 2, s;

      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        if (support[s] >= cMax) newSize += 1;
        else newSize += 2;
      }
      ierr = DMPlexSetSupportSize(rdm, newp, newSize);CHKERRQ(ierr);
    }
    break;
  case 5:
    /* Simplicial 3D */
    /* All cells have 4 faces */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 8; ++r) {
        const PetscInt newp = cStartNew + (c - cStart)*8 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
      }
    }
    /* Split faces have 3 edges and the same cells as the parent */
    for (f = fStart; f < fEnd; ++f) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = fStartNew + (f - fStart)*4 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 3);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior faces have 3 edges and 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 8; ++r) {
        const PetscInt newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*8 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 3);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Split edges have 2 vertices and the same faces */
    for (e = eStart; e < eEnd; ++e) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = eStartNew + (e - eStart)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Face edges have 2 vertices and 2+cells*(1/2) faces */
    for (f = fStart; f < fEnd; ++f) {
      for (r = 0; r < 3; ++r) {
        const PetscInt  newp = eStartNew + (eEnd - eStart)*2 + (f - fStart)*3 + r;
        const PetscInt *cone, *ornt, *support, eint[4] = {1, 0, 2, 0};
        PetscInt        coneSize, c, supportSize, s, er, intFaces = 0;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {if (cone[c] == f) break;}
          /* Here we want to determine whether edge newp contains a vertex which is part of the cross-tet edge */
          er = ornt[c] < 0 ? (-(ornt[c]+1) + 2-r)%3 : (ornt[c] + r)%3;
          if (er == eint[c]) {
            intFaces += 1;
          } else {
            intFaces += 2;
          }
        }
        ierr = DMPlexSetSupportSize(rdm, newp, 2+intFaces);CHKERRQ(ierr);
      }
    }
    /* Interior edges have 2 vertices and 4 faces */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt newp = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart);

      ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 4);CHKERRQ(ierr);
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newp = vStartNew + (v - vStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Edge vertices have 2 + faces*2 + cells*0/1 supports */
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (e - eStart);
      PetscInt       size, *star = NULL, starSize, s, cellSize = 0;

      ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr = DMPlexGetTransitiveClosure(dm, e, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
      for (s = 0; s < starSize*2; s += 2) {
        const PetscInt *cone, *ornt;
        PetscInt        e01, e23;

        if ((star[s] >= cStart) && (star[s] < cEnd)) {
          /* Check edge 0-1 */
          ierr = DMPlexGetCone(dm, star[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, star[s], &ornt);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, cone[0], &cone);CHKERRQ(ierr);
          e01  = cone[GetTriEdge_Static(ornt[0], 0)];
          /* Check edge 2-3 */
          ierr = DMPlexGetCone(dm, star[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, star[s], &ornt);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, cone[2], &cone);CHKERRQ(ierr);
          e23  = cone[GetTriEdge_Static(ornt[2], 1)];
          if ((e01 == e) || (e23 == e)) ++cellSize;
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, e, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2 + size*2 + cellSize);CHKERRQ(ierr);
    }
    break;
  case 7:
    /* Hybrid Simplicial 3D */
    ierr = DMPlexSetHybridBounds(rdm, cStartNew + 8*(cMax-cStart), fStartNew + 4*(fMax - fStart) + 8*(cMax - cStart),
                                 eStartNew + 2*(eMax - eStart) + 3*(fMax - fStart) + (cMax - cStart), PETSC_DETERMINE);CHKERRQ(ierr);
    /* Interior cells have 4 faces */
    for (c = cStart; c < cMax; ++c) {
      for (r = 0; r < 8; ++r) {
        const PetscInt newp = cStartNew + (c - cStart)*8 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
      }
    }
    /* Hybrid cells have 5 faces */
    for (c = cMax; c < cEnd; ++c) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = cStartNew + (cMax - cStart)*8 + (c - cMax)*4 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 5);CHKERRQ(ierr);
      }
    }
    /* Interior split faces have 3 edges and the same cells as the parent */
    for (f = fStart; f < fMax; ++f) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = fStartNew + (f - fStart)*4 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 3);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior cell faces have 3 edges and 2 cells */
    for (c = cStart; c < cMax; ++c) {
      for (r = 0; r < 8; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*4 + (c - cStart)*8 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 3);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Hybrid split faces have 4 edges and the same cells as the parent */
    for (f = fMax; f < fEnd; ++f) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (f - fMax)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Hybrid cells faces have 4 edges and 2 cells */
    for (c = cMax; c < cEnd; ++c) {
      for (r = 0; r < 3; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (fEnd - fMax)*2 + (c - cMax)*3 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Interior split edges have 2 vertices and the same faces */
    for (e = eStart; e < eMax; ++e) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = eStartNew + (e - eStart)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior face edges have 2 vertices and 2+cells*(1/2) faces */
    for (f = fStart; f < fMax; ++f) {
      for (r = 0; r < 3; ++r) {
        const PetscInt  newp = eStartNew + (eMax - eStart)*2 + (f - fStart)*3 + r;
        const PetscInt *cone, *ornt, *support, eint[4] = {1, 0, 2, 0};
        PetscInt        coneSize, c, supportSize, s, er, intFaces = 0;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {if (cone[c] == f) break;}
          if (support[s] < cMax) {
            /* Here we want to determine whether edge newp contains a vertex which is part of the cross-tet edge */
            er = ornt[c] < 0 ? (-(ornt[c]+1) + 2-r)%3 : (ornt[c] + r)%3;
            if (er == eint[c]) {
              intFaces += 1;
            } else {
              intFaces += 2;
            }
          } else {
            intFaces += 1;
          }
        }
        ierr = DMPlexSetSupportSize(rdm, newp, 2+intFaces);CHKERRQ(ierr);
      }
    }
    /* Interior cell edges have 2 vertices and 4 faces */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart);

      ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 4);CHKERRQ(ierr);
    }
    /* Hybrid edges have 2 vertices and the same faces */
    for (e = eMax; e < eEnd; ++e) {
      const PetscInt newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (e - eMax);
      PetscInt       size;

      ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Hybrid face edges have 2 vertices and 2+2*cells faces */
    for (f = fMax; f < fEnd; ++f) {
      const PetscInt newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (eEnd - eMax) + (f - fMax);
      PetscInt       size;

      ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2+2*size);CHKERRQ(ierr);
    }
    /* Interior vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newp = vStartNew + (v - vStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Interior edge vertices have 2 + interior face*2 + hybrid face + cells*0/1 supports */
    for (e = eStart; e < eMax; ++e) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (e - eStart);
      const PetscInt *support;
      PetscInt        size, *star = NULL, starSize, s, faceSize = 0, cellSize = 0;

      ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        if (support[s] < fMax) faceSize += 2;
        else                   faceSize += 1;
      }
      ierr = DMPlexGetTransitiveClosure(dm, e, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
      for (s = 0; s < starSize*2; s += 2) {
        const PetscInt *cone, *ornt;
        PetscInt        e01, e23;

        if ((star[s] >= cStart) && (star[s] < cMax)) {
          /* Check edge 0-1 */
          ierr = DMPlexGetCone(dm, star[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, star[s], &ornt);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, cone[0], &cone);CHKERRQ(ierr);
          e01  = cone[GetTriEdge_Static(ornt[0], 0)];
          /* Check edge 2-3 */
          ierr = DMPlexGetCone(dm, star[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, star[s], &ornt);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, cone[2], &cone);CHKERRQ(ierr);
          e23  = cone[GetTriEdge_Static(ornt[2], 1)];
          if ((e01 == e) || (e23 == e)) ++cellSize;
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, e, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2 + faceSize + cellSize);CHKERRQ(ierr);
    }
    break;
  case 6:
    /* Hex 3D */
    /* All cells have 6 faces */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 8; ++r) {
        const PetscInt newp = (c - cStart)*8 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 6);CHKERRQ(ierr);
      }
    }
    /* Split faces have 4 edges and the same cells as the parent */
    for (f = fStart; f < fEnd; ++f) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = fStartNew + (f - fStart)*4 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Interior faces have 4 edges and 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 12; ++r) {
        const PetscInt newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 4);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2);CHKERRQ(ierr);
      }
    }
    /* Split edges have 2 vertices and the same faces as the parent */
    for (e = eStart; e < eEnd; ++e) {
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = eStartNew + (e - eStart)*2 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
      }
    }
    /* Face edges have 2 vertices and 2+cells faces */
    for (f = fStart; f < fEnd; ++f) {
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = eStartNew + (eEnd - eStart)*2 + (f - fStart)*4 + r;
        PetscInt       size;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 2+size);CHKERRQ(ierr);
      }
    }
    /* Cell edges have 2 vertices and 4 faces */
    for (c = cStart; c < cEnd; ++c) {
      for (r = 0; r < 6; ++r) {
        const PetscInt newp = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*4 + (c - cStart)*6 + r;

        ierr = DMPlexSetConeSize(rdm, newp, 2);CHKERRQ(ierr);
        ierr = DMPlexSetSupportSize(rdm, newp, 4);CHKERRQ(ierr);
      }
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newp = vStartNew + (v - vStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, size);CHKERRQ(ierr);
    }
    /* Edge vertices have 2 + faces supports */
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (e - eStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 2 + size);CHKERRQ(ierr);
    }
    /* Face vertices have 4 + cells supports */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (f - fStart);
      PetscInt       size;

      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexSetSupportSize(rdm, newp, 4 + size);CHKERRQ(ierr);
    }
    /* Cell vertices have 6 supports */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (fEnd - fStart) + (c - cStart);

      ierr = DMPlexSetSupportSize(rdm, newp, 6);CHKERRQ(ierr);
    }
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell refiner %d", refiner);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CellRefinerSetCones"
PetscErrorCode CellRefinerSetCones(CellRefiner refiner, DM dm, PetscInt depthSize[], DM rdm)
{
  const PetscInt *faces, cellInd[4] = {0, 1, 2, 3};
  PetscInt        cStart,    cEnd,    cMax,    vStart,    vEnd, vMax, fStart,    fEnd,    fMax,    eStart,    eEnd,    eMax;
  PetscInt        cStartNew, cEndNew, cMaxNew, vStartNew, vEndNew,    fStartNew, fEndNew, fMaxNew, eStartNew, eEndNew, eMaxNew;
  PetscInt        depth, maxSupportSize, *supportRef, c, f, e, v, r, p;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, &fMax, &eMax, &vMax);CHKERRQ(ierr);
  if (refiner) {
    ierr = GetDepthStart_Private(depth, depthSize, &cStartNew, &fStartNew, &eStartNew, &vStartNew);CHKERRQ(ierr);
    ierr = GetDepthEnd_Private(depth, depthSize, &cEndNew, &fEndNew, &eEndNew, &vEndNew);CHKERRQ(ierr);
  }
  switch (refiner) {
  case 0: break;
  case 1:
    /* Simplicial 2D */
    /*
     2
     |\
     | \
     |  \
     |   \
     | C  \
     |     \
     |      \
     2---1---1
     |\  D  / \
     | 2   0   \
     |A \ /  B  \
     0---0-------1
     */
    /* All cells have 3 faces */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt  newp = cStartNew + (c - cStart)*4;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[3], orntNew[3];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A triangle */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 1 : 0);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 2;
      orntNew[1] = -2;
      coneNew[2] = fStartNew + (cone[2] - fStart)*2 + (ornt[2] < 0 ? 0 : 1);
      orntNew[2] = ornt[2];
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+0 < cStartNew) || (newp+0 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+0, cStartNew, cEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* B triangle */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 0 : 1);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 1 : 0);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 0;
      orntNew[2] = -2;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+1 < cStartNew) || (newp+1 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+1, cStartNew, cEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* C triangle */
      coneNew[0] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 1;
      orntNew[0] = -2;
      coneNew[1] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 0 : 1);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*2 + (ornt[2] < 0 ? 1 : 0);
      orntNew[2] = ornt[2];
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+2 < cStartNew) || (newp+2 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+2, cStartNew, cEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* D triangle */
      coneNew[0] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 0;
      orntNew[0] = 0;
      coneNew[1] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 1;
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*3 + 2;
      orntNew[2] = 0;
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+3 < cStartNew) || (newp+3 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+3, cStartNew, cEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Split faces have 2 vertices and the same cells as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscMalloc((2 + maxSupportSize*2) * sizeof(PetscInt), &supportRef);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (f - fStart);

      for (r = 0; r < 2; ++r) {
        const PetscInt  newp = fStartNew + (f - fStart)*2 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[2], coneSize, c, supportSize, s;

        ierr             = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
        coneNew[0]       = vStartNew + (cone[0] - vStart);
        coneNew[1]       = vStartNew + (cone[1] - vStart);
        coneNew[(r+1)%2] = newv;
        ierr             = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if 1
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == f) break;
          }
          supportRef[s] = cStartNew + (support[s] - cStart)*4 + (ornt[c] < 0 ? (c+1-r)%3 : (c+r)%3);
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportRef[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior faces have 2 vertices and 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt *cone;

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      for (r = 0; r < 3; ++r) {
        const PetscInt newp = fStartNew + (fEnd - fStart)*2 + (c - cStart)*3 + r;
        PetscInt       coneNew[2];
        PetscInt       supportNew[2];

        coneNew[0] = vStartNew + (vEnd - vStart) + (cone[r]       - fStart);
        coneNew[1] = vStartNew + (vEnd - vStart) + (cone[(r+1)%3] - fStart);
        ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if 1
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        supportNew[0] = (c - cStart)*4 + (r+1)%3;
        supportNew[1] = (c - cStart)*4 + 3;
        ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt  newp = vStartNew + (v - vStart);
      const PetscInt *support, *cone;
      PetscInt        size, s;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, v, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if (cone[1] == v) r = 1;
        supportRef[s] = fStartNew + (support[s] - fStart)*2 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", newp, vStartNew, vEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", supportRef[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Face vertices have 2 + cells*2 supports */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (f - fStart);
      const PetscInt *cone, *support;
      PetscInt        size, s;

      ierr          = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      supportRef[0] = fStartNew + (f - fStart)*2 + 0;
      supportRef[1] = fStartNew + (f - fStart)*2 + 1;
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if      (cone[1] == f) r = 1;
        else if (cone[2] == f) r = 2;
        supportRef[2+s*2+0] = fStartNew + (fEnd - fStart)*2 + (support[s] - cStart)*3 + (r+2)%3;
        supportRef[2+s*2+1] = fStartNew + (fEnd - fStart)*2 + (support[s] - cStart)*3 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", newp, vStartNew, vEndNew);
      for (p = 0; p < 2+size*2; ++p) {
        if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", supportRef[p], fStartNew, fEndNew);
      }
#endif
    }
    ierr = PetscFree(supportRef);CHKERRQ(ierr);
    break;
  case 2:
    /* Hex 2D */
    /*
     3---------2---------2
     |         |         |
     |    D    2    C    |
     |         |         |
     3----3----0----1----1
     |         |         |
     |    A    0    B    |
     |         |         |
     0---------0---------1
     */
    /* All cells have 4 faces */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt  newp = (c - cStart)*4;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[4], orntNew[4];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A quad */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 1 : 0);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*4 + 0;
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*4 + 3;
      orntNew[2] = -2;
      coneNew[3] = fStartNew + (cone[3] - fStart)*2 + (ornt[3] < 0 ? 0 : 1);
      orntNew[3] = ornt[3];
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+0 < cStartNew) || (newp+0 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+0, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* B quad */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 0 : 1);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 1 : 0);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*4 + 1;
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*4 + 0;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+1 < cStartNew) || (newp+1 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+1, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* C quad */
      coneNew[0] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*4 + 1;
      orntNew[0] = -2;
      coneNew[1] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 0 : 1);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*2 + (ornt[2] < 0 ? 1 : 0);
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*4 + 2;
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+2 < cStartNew) || (newp+2 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+2, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* D quad */
      coneNew[0] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*4 + 3;
      orntNew[0] = 0;
      coneNew[1] = fStartNew + (fEnd    - fStart)*2 + (c - cStart)*4 + 2;
      orntNew[1] = -2;
      coneNew[2] = fStartNew + (cone[2] - fStart)*2 + (ornt[2] < 0 ? 0 : 1);
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (cone[3] - fStart)*2 + (ornt[3] < 0 ? 1 : 0);
      orntNew[3] = ornt[3];
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+3 < cStartNew) || (newp+3 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+3, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Split faces have 2 vertices and the same cells as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscMalloc((2 + maxSupportSize*2) * sizeof(PetscInt), &supportRef);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (f - fStart);

      for (r = 0; r < 2; ++r) {
        const PetscInt  newp = fStartNew + (f - fStart)*2 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[2], coneSize, c, supportSize, s;

        ierr             = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
        coneNew[0]       = vStartNew + (cone[0] - vStart);
        coneNew[1]       = vStartNew + (cone[1] - vStart);
        coneNew[(r+1)%2] = newv;
        ierr             = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if 1
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == f) break;
          }
          supportRef[s] = cStartNew + (support[s] - cStart)*4 + (ornt[c] < 0 ? (c+1-r)%4 : (c+r)%4);
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportRef[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior faces have 2 vertices and 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt *cone;
      PetscInt        coneNew[2], supportNew[2];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = fStartNew + (fEnd - fStart)*2 + (c - cStart)*4 + r;

        coneNew[0] = vStartNew + (vEnd - vStart) + (cone[r] - fStart);
        coneNew[1] = vStartNew + (vEnd - vStart) + (fEnd    - fStart) + (c - cStart);
        ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if 1
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        supportNew[0] = (c - cStart)*4 + r;
        supportNew[1] = (c - cStart)*4 + (r+1)%4;
        ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt  newp = vStartNew + (v - vStart);
      const PetscInt *support, *cone;
      PetscInt        size, s;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, v, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if (cone[1] == v) r = 1;
        supportRef[s] = fStartNew + (support[s] - fStart)*2 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", newp, vStartNew, vEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", supportRef[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Face vertices have 2 + cells supports */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (f - fStart);
      const PetscInt *cone, *support;
      PetscInt        size, s;

      ierr          = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      supportRef[0] = fStartNew + (f - fStart)*2 + 0;
      supportRef[1] = fStartNew + (f - fStart)*2 + 1;
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if      (cone[1] == f) r = 1;
        else if (cone[2] == f) r = 2;
        else if (cone[3] == f) r = 3;
        supportRef[2+s] = fStartNew + (fEnd - fStart)*2 + (support[s] - cStart)*4 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", newp, vStartNew, vEndNew);
      for (p = 0; p < 2+size; ++p) {
        if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", supportRef[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Cell vertices have 4 supports */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (fEnd - fStart) + (c - cStart);
      PetscInt       supportNew[4];

      for (r = 0; r < 4; ++r) {
        supportNew[r] = fStartNew + (fEnd - fStart)*2 + (c - cStart)*4 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
    }
    ierr = PetscFree(supportRef);CHKERRQ(ierr);
    break;
  case 3:
    if (cMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No cell maximum specified in hybrid mesh");
    cMax = PetscMin(cEnd, cMax);
    if (fMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No face maximum specified in hybrid mesh");
    fMax = PetscMin(fEnd, fMax);
    /* Interior cells have 3 faces */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt  newp = cStartNew + (c - cStart)*4;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[3], orntNew[3];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A triangle */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 1 : 0);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fMax    - fStart)*2 + (c - cStart)*3 + 2;
      orntNew[1] = -2;
      coneNew[2] = fStartNew + (cone[2] - fStart)*2 + (ornt[2] < 0 ? 0 : 1);
      orntNew[2] = ornt[2];
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+0 < cStartNew) || (newp+0 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+0, cStartNew, cEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* B triangle */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 0 : 1);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 1 : 0);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fMax    - fStart)*2 + (c - cStart)*3 + 0;
      orntNew[2] = -2;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+1 < cStartNew) || (newp+1 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+1, cStartNew, cEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* C triangle */
      coneNew[0] = fStartNew + (fMax    - fStart)*2 + (c - cStart)*3 + 1;
      orntNew[0] = -2;
      coneNew[1] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 0 : 1);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*2 + (ornt[2] < 0 ? 1 : 0);
      orntNew[2] = ornt[2];
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+2 < cStartNew) || (newp+2 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+2, cStartNew, cEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* D triangle */
      coneNew[0] = fStartNew + (fMax    - fStart)*2 + (c - cStart)*3 + 0;
      orntNew[0] = 0;
      coneNew[1] = fStartNew + (fMax    - fStart)*2 + (c - cStart)*3 + 1;
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (fMax    - fStart)*2 + (c - cStart)*3 + 2;
      orntNew[2] = 0;
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+3 < cStartNew) || (newp+3 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+3, cStartNew, cEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
    }
    /*
     2----3----3
     |         |
     |    B    |
     |         |
     0----4--- 1
     |         |
     |    A    |
     |         |
     0----2----1
     */
    /* Hybrid cells have 4 faces */
    for (c = cMax; c < cEnd; ++c) {
      const PetscInt  newp = cStartNew + (cMax - cStart)*4 + (c - cMax)*2;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[4], orntNew[4];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A quad */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 1 : 0);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 1 : 0);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fMax    - fStart)*2 + (cMax - cStart)*3 + (cone[2] - fMax);
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (fMax    - fStart)*2 + (cMax - cStart)*3 + (fEnd    - fMax) + (c - cMax);
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+0 < cStartNew) || (newp+0 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+0, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* B quad */
      coneNew[0] = fStartNew + (cone[0] - fStart)*2 + (ornt[0] < 0 ? 0 : 1);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*2 + (ornt[1] < 0 ? 0 : 1);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fMax    - fStart)*2 + (cMax - cStart)*3 + (fEnd    - fMax) + (c - cMax);
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (fMax    - fStart)*2 + (cMax - cStart)*3 + (cone[3] - fMax);
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+1 < cStartNew) || (newp+1 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+1, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Interior split faces have 2 vertices and the same cells as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscMalloc((2 + maxSupportSize*2) * sizeof(PetscInt), &supportRef);CHKERRQ(ierr);
    for (f = fStart; f < fMax; ++f) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (f - fStart);

      for (r = 0; r < 2; ++r) {
        const PetscInt  newp = fStartNew + (f - fStart)*2 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[2], coneSize, c, supportSize, s;

        ierr             = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
        coneNew[0]       = vStartNew + (cone[0] - vStart);
        coneNew[1]       = vStartNew + (cone[1] - vStart);
        coneNew[(r+1)%2] = newv;
        ierr             = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if 1
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          if (support[s] >= cMax) {
            supportRef[s] = cStartNew + (cMax - cStart)*4 + (support[s] - cMax)*2 + r;
          } else {
            ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
            ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
            ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
            for (c = 0; c < coneSize; ++c) {
              if (cone[c] == f) break;
            }
            supportRef[s] = cStartNew + (support[s] - cStart)*4 + (ornt[c] < 0 ? (c+1-r)%3 : (c+r)%3);
          }
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportRef[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior cell faces have 2 vertices and 2 cells */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt *cone;

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      for (r = 0; r < 3; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*2 + (c - cStart)*3 + r;
        PetscInt       coneNew[2];
        PetscInt       supportNew[2];

        coneNew[0] = vStartNew + (vEnd - vStart) + (cone[r]       - fStart);
        coneNew[1] = vStartNew + (vEnd - vStart) + (cone[(r+1)%3] - fStart);
        ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if 1
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        supportNew[0] = (c - cStart)*4 + (r+1)%3;
        supportNew[1] = (c - cStart)*4 + 3;
        ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior hybrid faces have 2 vertices and the same cells */
    for (f = fMax; f < fEnd; ++f) {
      const PetscInt  newp = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*3 + (f - fMax);
      const PetscInt *cone;
      const PetscInt *support;
      PetscInt        coneNew[2];
      PetscInt        supportNew[2];
      PetscInt        size, s, r;

      ierr       = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
      coneNew[0] = vStartNew + (cone[0] - vStart);
      coneNew[1] = vStartNew + (cone[1] - vStart);
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", coneNew[p], vStartNew, vEndNew);
      }
#endif
      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        for (r = 0; r < 2; ++r) {
          if (cone[r+2] == f) break;
        }
        supportNew[s] = (cMax - cStart)*4 + (support[s] - cMax)*2 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cEndNew);
      }
#endif
    }
    /* Cell hybrid faces have 2 vertices and 2 cells */
    for (c = cMax; c < cEnd; ++c) {
      const PetscInt  newp = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*3 + (fEnd - fMax) + (c - cMax);
      const PetscInt *cone;
      PetscInt        coneNew[2];
      PetscInt        supportNew[2];

      ierr       = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      coneNew[0] = vStartNew + (vEnd - vStart) + (cone[0] - fStart);
      coneNew[1] = vStartNew + (vEnd - vStart) + (cone[1] - fStart);
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", coneNew[p], vStartNew, vEndNew);
      }
#endif
      supportNew[0] = (cMax - cStart)*4 + (c - cMax)*2 + 0;
      supportNew[1] = (cMax - cStart)*4 + (c - cMax)*2 + 1;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cEndNew);
      }
#endif
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt  newp = vStartNew + (v - vStart);
      const PetscInt *support, *cone;
      PetscInt        size, s;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, v, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        if (support[s] >= fMax) {
          supportRef[s] = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*3 + (support[s] - fMax);
        } else {
          PetscInt r = 0;

          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          if (cone[1] == v) r = 1;
          supportRef[s] = fStartNew + (support[s] - fStart)*2 + r;
        }
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", newp, vStartNew, vEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", supportRef[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Face vertices have 2 + (2 interior, 1 hybrid) supports */
    for (f = fStart; f < fMax; ++f) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (f - fStart);
      const PetscInt *cone, *support;
      PetscInt        size, newSize = 2, s;

      ierr          = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      supportRef[0] = fStartNew + (f - fStart)*2 + 0;
      supportRef[1] = fStartNew + (f - fStart)*2 + 1;
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if (support[s] >= cMax) {
          supportRef[newSize+0] = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*3 + (fEnd - fMax) + (support[s] - cMax);

          newSize += 1;
        } else {
          if      (cone[1] == f) r = 1;
          else if (cone[2] == f) r = 2;
          supportRef[newSize+0] = fStartNew + (fMax - fStart)*2 + (support[s] - cStart)*3 + (r+2)%3;
          supportRef[newSize+1] = fStartNew + (fMax - fStart)*2 + (support[s] - cStart)*3 + r;

          newSize += 2;
        }
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", newp, vStartNew, vEndNew);
      for (p = 0; p < newSize; ++p) {
        if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", supportRef[p], fStartNew, fEndNew);
      }
#endif
    }
    ierr = PetscFree(supportRef);CHKERRQ(ierr);
    break;
  case 5:
    /* Simplicial 3D */
    /* All cells have 4 faces: Tet face order is prescribed in DMPlexGetFaces_Internal() */
    ierr = DMPlexGetRawFaces_Internal(dm, 3, 4, cellInd, NULL, NULL, &faces);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt  newp = cStartNew + (c - cStart)*8;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[4], orntNew[4];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A tetrahedron: {0, a, c, d} */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetTriSubface_Static(ornt[0], 0); /* A */
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetTriSubface_Static(ornt[1], 0); /* A */
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetTriSubface_Static(ornt[2], 0); /* A */
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 0;
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+0 < cStartNew) || (newp+0 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+0, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* B tetrahedron: {a, 1, b, e} */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetTriSubface_Static(ornt[0], 1); /* B */
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetTriSubface_Static(ornt[1], 2); /* C */
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 1;
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetTriSubface_Static(ornt[3], 1); /* B */
      orntNew[3] = ornt[3];
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+1 < cStartNew) || (newp+1 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+1, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* C tetrahedron: {c, b, 2, f} */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetTriSubface_Static(ornt[0], 2); /* C */
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 2;
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetTriSubface_Static(ornt[2], 1); /* B */
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetTriSubface_Static(ornt[3], 0); /* A */
      orntNew[3] = ornt[3];
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+2 < cStartNew) || (newp+2 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+2, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* D tetrahedron: {d, e, f, 3} */
      coneNew[0] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 3;
      orntNew[0] = 0;
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetTriSubface_Static(ornt[1], 1); /* B */
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetTriSubface_Static(ornt[2], 2); /* C */
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetTriSubface_Static(ornt[3], 2); /* C */
      orntNew[3] = ornt[3];
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+3 < cStartNew) || (newp+3 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+3, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* A' tetrahedron: {d, a, c, f} */
      coneNew[0] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 0;
      orntNew[0] = -3;
      coneNew[1] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 4;
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + 3;
      orntNew[2] = ornt[2] < 0 ? -((-(ornt[2]+1)+2)%3+1) : (ornt[2]+2)%3;
      coneNew[3] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 5;
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp+4, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+4, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+4 < cStartNew) || (newp+4 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+4, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* B' tetrahedron: {e, b, a, f} */
      coneNew[0] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 1;
      orntNew[0] = -3;
      coneNew[1] = fStartNew + (cone[3] - fStart)*4 + 3;
      orntNew[1] = ornt[3] < 0 ? -((-(ornt[3]+1)+1)%3+1) : (ornt[3]+1)%3;
      coneNew[2] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 6;
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 7;
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp+5, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+5, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+5 < cStartNew) || (newp+5 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+5, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* C' tetrahedron: {b, f, c, a} */
      coneNew[0] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 2;
      orntNew[0] = -3;
      coneNew[1] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 7;
      orntNew[1] = -2;
      coneNew[2] = fStartNew + (cone[0] - fStart)*4 + 3;
      orntNew[2] = ornt[0] < 0 ? (-(ornt[0]+1)+1)%3 : -((ornt[0]+1)%3+1);
      coneNew[3] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 5;
      orntNew[3] = -1;
      ierr       = DMPlexSetCone(rdm, newp+6, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+6, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+6 < cStartNew) || (newp+6 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+6, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* D' tetrahedron: {f, e, d, a} */
      coneNew[0] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 3;
      orntNew[0] = -3;
      coneNew[1] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 6;
      orntNew[1] = -3;
      coneNew[2] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*8 + 4;
      orntNew[2] = -2;
      coneNew[3] = fStartNew + (cone[1] - fStart)*4 + 3;
      orntNew[3] = ornt[2];
      ierr       = DMPlexSetCone(rdm, newp+7, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+7, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+7 < cStartNew) || (newp+7 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+7, cStartNew, cEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Split faces have 3 edges and the same cells as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscMalloc((2 + maxSupportSize*2) * sizeof(PetscInt), &supportRef);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt  newp = fStartNew + (f - fStart)*4;
      const PetscInt *cone, *ornt, *support;
      PetscInt        coneNew[3], orntNew[3], coneSize, supportSize, s;

      ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, f, &ornt);CHKERRQ(ierr);
      /* A triangle */
      coneNew[0] = eStartNew + (cone[0] - eStart)*2 + (ornt[0] < 0 ? 1 : 0);
      orntNew[0] = ornt[0];
      coneNew[1] = eStartNew + (eEnd    - eStart)*2 + (f - fStart)*3 + 2;
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (cone[2] - eStart)*2 + (ornt[2] < 0 ? 0 : 1);
      orntNew[2] = ornt[2];
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+0 < fStartNew) || (newp+0 >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp+0, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* B triangle */
      coneNew[0] = eStartNew + (cone[0] - eStart)*2 + (ornt[0] < 0 ? 0 : 1);
      orntNew[0] = ornt[0];
      coneNew[1] = eStartNew + (cone[1] - eStart)*2 + (ornt[1] < 0 ? 1 : 0);
      orntNew[1] = ornt[1];
      coneNew[2] = eStartNew + (eEnd    - eStart)*2 + (f - fStart)*3 + 0;
      orntNew[2] = -2;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+1 < fStartNew) || (newp+1 >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp+1, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* C triangle */
      coneNew[0] = eStartNew + (eEnd    - eStart)*2 + (f - fStart)*3 + 1;
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (cone[1] - eStart)*2 + (ornt[1] < 0 ? 0 : 1);
      orntNew[1] = ornt[1];
      coneNew[2] = eStartNew + (cone[2] - eStart)*2 + (ornt[2] < 0 ? 1 : 0);
      orntNew[2] = ornt[2];
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+2 < fStartNew) || (newp+2 >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp+2, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* D triangle */
      coneNew[0] = eStartNew + (eEnd    - eStart)*2 + (f - fStart)*3 + 0;
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eEnd    - eStart)*2 + (f - fStart)*3 + 1;
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd    - eStart)*2 + (f - fStart)*3 + 2;
      orntNew[2] = 0;
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+3 < fStartNew) || (newp+3 >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp+3, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (r = 0; r < 4; ++r) {
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == f) break;
          }
          supportRef[s] = cStartNew + (support[s] - cStart)*8 + (r==3 ? (c+2)%4 + 4 : (ornt[c] < 0 ? faces[c*3+(-(ornt[c]+1)+1+3-r)%3] : faces[c*3+r]));
        }
        ierr = DMPlexSetSupport(rdm, newp+r, supportRef);CHKERRQ(ierr);
#if 1
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportRef[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior faces have 3 edges and 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      PetscInt        newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*8;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[3], orntNew[3];
      PetscInt        supportNew[2];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* Face A: {c, a, d} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*3 + (ornt[0] < 0 ? (-(ornt[0]+1)+0)%3 : (ornt[0]+2)%3);
      orntNew[0] = ornt[0] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*3 + (ornt[1] < 0 ? (-(ornt[1]+1)+0)%3 : (ornt[1]+2)%3);
      orntNew[1] = ornt[1] < 0 ? -2 : 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*3 + (ornt[2] < 0 ? (-(ornt[2]+1)+0)%3 : (ornt[2]+2)%3);
      orntNew[2] = ornt[2] < 0 ? -2 : 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 0;
      supportNew[1] = (c - cStart)*8 + 0+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face B: {a, b, e} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*3 + (ornt[0] < 0 ? (-(ornt[0]+1)+2)%3 : (ornt[0]+0)%3);
      orntNew[0] = ornt[0] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*3 + (ornt[3] < 0 ? (-(ornt[3]+1)+2)%3 : (ornt[3]+0)%3);
      orntNew[1] = ornt[3] < 0 ? -2 : 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*3 + (ornt[1] < 0 ? (-(ornt[1]+1)+1)%3 : (ornt[1]+1)%3);
      orntNew[2] = ornt[1] < 0 ? -2 : 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+1 < fStartNew) || (newp+1 >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp+1, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 1;
      supportNew[1] = (c - cStart)*8 + 1+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face C: {c, f, b} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*3 + (ornt[2] < 0 ? (-(ornt[2]+1)+2)%3 : (ornt[2]+0)%3);
      orntNew[0] = ornt[2] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*3 + (ornt[3] < 0 ? (-(ornt[3]+1)+0)%3 : (ornt[3]+2)%3);
      orntNew[1] = ornt[3] < 0 ? -2 : 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*3 + (ornt[0] < 0 ? (-(ornt[0]+1)+1)%3 : (ornt[0]+1)%3);
      orntNew[2] = ornt[0] < 0 ? -2 : 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 2;
      supportNew[1] = (c - cStart)*8 + 2+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face D: {d, e, f} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*3 + (ornt[1] < 0 ? (-(ornt[1]+1)+2)%3 : (ornt[1]+0)%3);
      orntNew[0] = ornt[1] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*3 + (ornt[3] < 0 ? (-(ornt[3]+1)+1)%3 : (ornt[3]+1)%3);
      orntNew[1] = ornt[3] < 0 ? -2 : 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*3 + (ornt[2] < 0 ? (-(ornt[2]+1)+1)%3 : (ornt[2]+1)%3);
      orntNew[2] = ornt[2] < 0 ? -2 : 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 3;
      supportNew[1] = (c - cStart)*8 + 3+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face E: {d, f, a} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*3 + (ornt[2] < 0 ? (-(ornt[2]+1)+1)%3 : (ornt[2]+1)%3);
      orntNew[0] = ornt[2] < 0 ? 0 : -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*3 + (ornt[1] < 0 ? (-(ornt[1]+1)+0)%3 : (ornt[1]+2)%3);
      orntNew[2] = ornt[1] < 0 ? -2 : 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 0+4;
      supportNew[1] = (c - cStart)*8 + 3+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face F: {c, a, f} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*3 + (ornt[0] < 0 ? (-(ornt[0]+1)+0)%3 : (ornt[0]+2)%3);
      orntNew[0] = ornt[0] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart);
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*3 + (ornt[2] < 0 ? (-(ornt[2]+1)+2)%3 : (ornt[2]+0)%3);
      orntNew[2] = ornt[1] < 0 ? 0 : -2;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 0+4;
      supportNew[1] = (c - cStart)*8 + 2+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face G: {e, a, f} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*3 + (ornt[1] < 0 ? (-(ornt[1]+1)+1)%3 : (ornt[1]+1)%3);
      orntNew[0] = ornt[1] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart);
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*3 + (ornt[3] < 0 ? (-(ornt[3]+1)+1)%3 : (ornt[3]+1)%3);
      orntNew[2] = ornt[3] < 0 ? 0 : -2;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 1+4;
      supportNew[1] = (c - cStart)*8 + 3+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
      /* Face H: {a, b, f} */
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*3 + (ornt[0] < 0 ? (-(ornt[0]+1)+2)%3 : (ornt[0]+0)%3);
      orntNew[0] = ornt[0] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*3 + (ornt[3] < 0 ? (-(ornt[3]+1)+0)%3 : (ornt[3]+2)%3);
      orntNew[1] = ornt[3] < 0 ? 0 : -2;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart);
      orntNew[2] = 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 1+4;
      supportNew[1] = (c - cStart)*8 + 2+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cEndNew);
      }
#endif
      ++newp;
    }
    /* Split Edges have 2 vertices and the same faces as the parent */
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (e - eStart);

      for (r = 0; r < 2; ++r) {
        const PetscInt  newp = eStartNew + (e - eStart)*2 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[2], coneSize, c, supportSize, s;

        ierr             = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
        coneNew[0]       = vStartNew + (cone[0] - vStart);
        coneNew[1]       = vStartNew + (cone[1] - vStart);
        coneNew[(r+1)%2] = newv;
        ierr             = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if 1
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", newp, eStartNew, eEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, e, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == e) break;
          }
          supportRef[s] = fStartNew + (support[s] - fStart)*4 + (c + (ornt[c] < 0 ? 1-r : r))%3;
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", newp, eStartNew, eEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", supportRef[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Face edges have 2 vertices and 2+cells*(1/2) faces */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt *cone, *ornt, *support;
      PetscInt        coneSize, supportSize, s;

      ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (r = 0; r < 3; ++r) {
        const PetscInt  newp = eStartNew + (eEnd - eStart)*2 + (f - fStart)*3 + r;
        PetscInt        coneNew[2], intFaces = 0, er, eint[4] = {1, 0, 2, 0};
        PetscInt        fint[24] = { 1,  7, -1, -1,  0,  5,
                                    -1, -1,  1,  6,  0,  4,
                                     2,  5,  3,  4, -1, -1,
                                    -1, -1,  3,  6,  2,  7};

        ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
        coneNew[0] = vStartNew + (vEnd - vStart) + (cone[(r+0)%3] - eStart);
        coneNew[1] = vStartNew + (vEnd - vStart) + (cone[(r+1)%3] - eStart);
        ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if 1
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", newp, eStartNew, eEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        supportRef[0] = fStartNew + (f - fStart)*4 + (r+1)%3;
        supportRef[1] = fStartNew + (f - fStart)*4 + 3;
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {if (cone[c] == f) break;}
          /* Here we want to determine whether edge newp contains a vertex which is part of the cross-tet edge */
          er   = ornt[c] < 0 ? (-(ornt[c]+1) + 2-r)%3 : (ornt[c] + r)%3;
          if (er == eint[c]) {
            supportRef[2+intFaces++] = fStartNew + (fEnd - fStart)*4 + (support[s] - cStart)*8 + (c + 2)%4;
          } else {
            supportRef[2+intFaces++] = fStartNew + (fEnd - fStart)*4 + (support[s] - cStart)*8 + fint[(c*3 + er)*2 + 0];
            supportRef[2+intFaces++] = fStartNew + (fEnd - fStart)*4 + (support[s] - cStart)*8 + fint[(c*3 + er)*2 + 1];
          }
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", newp, eStartNew, eEndNew);
        for (p = 0; p < intFaces; ++p) {
          if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", supportRef[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Interior edges have 2 vertices and 4 faces */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt  newp = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (c - cStart);
      const PetscInt *cone, *ornt, *fcone;
      PetscInt        coneNew[2], supportNew[4], find;

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, cone[0], &fcone);CHKERRQ(ierr);
      find = GetTriEdge_Static(ornt[0], 0);
      coneNew[0] = vStartNew + (vEnd - vStart) + (fcone[find] - eStart);
      ierr = DMPlexGetCone(dm, cone[2], &fcone);CHKERRQ(ierr);
      find = GetTriEdge_Static(ornt[2], 1);
      coneNew[1] = vStartNew + (vEnd - vStart) + (fcone[find] - eStart);
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if 1
      if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", newp, eStartNew, eEndNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", coneNew[p], vStartNew, vEndNew);
      }
#endif
      supportNew[0] = fStartNew + (fEnd - fStart)*4 + (c - cStart)*8 + 4;
      supportNew[1] = fStartNew + (fEnd - fStart)*4 + (c - cStart)*8 + 5;
      supportNew[2] = fStartNew + (fEnd - fStart)*4 + (c - cStart)*8 + 6;
      supportNew[3] = fStartNew + (fEnd - fStart)*4 + (c - cStart)*8 + 7;
      ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
      if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", newp, eStartNew, eEndNew);
      for (p = 0; p < 4; ++p) {
        if ((supportNew[p] < fStartNew) || (supportNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", supportNew[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt  newp = vStartNew + (v - vStart);
      const PetscInt *support, *cone;
      PetscInt        size, s;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, v, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if (cone[1] == v) r = 1;
        supportRef[s] = eStartNew + (support[s] - eStart)*2 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", newp, vStartNew, vEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    /* Edge vertices have 2 + face*2 + 0/1 supports */
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (e - eStart);
      const PetscInt *cone, *support;
      PetscInt       *star = NULL, starSize, cellSize = 0, coneSize, size, s;

      ierr          = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
      supportRef[0] = eStartNew + (e - eStart)*2 + 0;
      supportRef[1] = eStartNew + (e - eStart)*2 + 1;
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        for (r = 0; r < coneSize; ++r) {if (cone[r] == e) break;}
        supportRef[2+s*2+0] = eStartNew + (eEnd - eStart)*2 + (support[s] - fStart)*3 + (r+0)%3;
        supportRef[2+s*2+1] = eStartNew + (eEnd - eStart)*2 + (support[s] - fStart)*3 + (r+2)%3;
      }
      ierr = DMPlexGetTransitiveClosure(dm, e, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
      for (s = 0; s < starSize*2; s += 2) {
        const PetscInt *cone, *ornt;
        PetscInt        e01, e23;

        if ((star[s] >= cStart) && (star[s] < cEnd)) {
          /* Check edge 0-1 */
          ierr = DMPlexGetCone(dm, star[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, star[s], &ornt);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, cone[0], &cone);CHKERRQ(ierr);
          e01  = cone[GetTriEdge_Static(ornt[0], 0)];
          /* Check edge 2-3 */
          ierr = DMPlexGetCone(dm, star[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, star[s], &ornt);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, cone[2], &cone);CHKERRQ(ierr);
          e23  = cone[GetTriEdge_Static(ornt[2], 1)];
          if ((e01 == e) || (e23 == e)) {supportRef[2+size*2+cellSize++] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (star[s] - cStart);}
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, e, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", newp, vStartNew, vEndNew);
      for (p = 0; p < 2+size*2+cellSize; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    ierr = PetscFree(supportRef);CHKERRQ(ierr);
    ierr = DMPlexRestoreFaces_Internal(dm, 3, cStart, NULL, NULL, &faces);CHKERRQ(ierr);
    break;
  case 7:
    /* Hybrid Simplicial 3D */
    ierr = DMPlexGetHybridBounds(rdm, &cMaxNew, &fMaxNew, &eMaxNew, NULL);CHKERRQ(ierr);
    /* Interior cells have 4 faces: Tet face order is prescribed in DMPlexGetFaces_Internal() */
    ierr = DMPlexGetRawFaces_Internal(dm, 3, 4, cellInd, NULL, NULL, &faces);CHKERRQ(ierr);
    for (c = cStart; c < cMax; ++c) {
      const PetscInt  newp = cStartNew + (c - cStart)*8;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[4], orntNew[4];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A tetrahedron: {0, a, c, d} */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetTriSubface_Static(ornt[0], 0); /* A */
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetTriSubface_Static(ornt[1], 0); /* A */
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetTriSubface_Static(ornt[2], 0); /* A */
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 0;
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+0 < cStartNew) || (newp+0 >= cMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+0, cStartNew, cMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* B tetrahedron: {a, 1, b, e} */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetTriSubface_Static(ornt[0], 1); /* B */
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetTriSubface_Static(ornt[1], 2); /* C */
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 1;
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetTriSubface_Static(ornt[3], 1); /* B */
      orntNew[3] = ornt[3];
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+1 < cStartNew) || (newp+1 >= cMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+1, cStartNew, cMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* C tetrahedron: {c, b, 2, f} */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetTriSubface_Static(ornt[0], 2); /* C */
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 2;
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetTriSubface_Static(ornt[2], 1); /* B */
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetTriSubface_Static(ornt[3], 0); /* A */
      orntNew[3] = ornt[3];
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+2 < cStartNew) || (newp+2 >= cMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+2, cStartNew, cMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* D tetrahedron: {d, e, f, 3} */
      coneNew[0] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 3;
      orntNew[0] = 0;
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetTriSubface_Static(ornt[1], 1); /* B */
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetTriSubface_Static(ornt[2], 2); /* C */
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetTriSubface_Static(ornt[3], 2); /* C */
      orntNew[3] = ornt[3];
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+3 < cStartNew) || (newp+3 >= cMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+3, cStartNew, cMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* A' tetrahedron: {d, a, c, f} */
      coneNew[0] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 0;
      orntNew[0] = -3;
      coneNew[1] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 4;
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + 3;
      orntNew[2] = ornt[2] < 0 ? -((-(ornt[2]+1)+2)%3+1) : (ornt[2]+2)%3;
      coneNew[3] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 5;
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp+4, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+4, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+4 < cStartNew) || (newp+4 >= cMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+4, cStartNew, cMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* B' tetrahedron: {e, b, a, f} */
      coneNew[0] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 1;
      orntNew[0] = -3;
      coneNew[1] = fStartNew + (cone[3] - fStart)*4 + 3;
      orntNew[1] = ornt[3] < 0 ? -((-(ornt[3]+1)+1)%3+1) : (ornt[3]+1)%3;
      coneNew[2] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 6;
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 7;
      orntNew[3] = 0;
      ierr       = DMPlexSetCone(rdm, newp+5, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+5, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+5 < cStartNew) || (newp+5 >= cMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+5, cStartNew, cMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* C' tetrahedron: {b, f, c, a} */
      coneNew[0] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 2;
      orntNew[0] = -3;
      coneNew[1] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 7;
      orntNew[1] = -2;
      coneNew[2] = fStartNew + (cone[0] - fStart)*4 + 3;
      orntNew[2] = ornt[0] < 0 ? (-(ornt[0]+1)+1)%3 : -((ornt[0]+1)%3+1);
      coneNew[3] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 5;
      orntNew[3] = -1;
      ierr       = DMPlexSetCone(rdm, newp+6, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+6, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+6 < cStartNew) || (newp+6 >= cMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+6, cStartNew, cMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
      /* D' tetrahedron: {f, e, d, a} */
      coneNew[0] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 3;
      orntNew[0] = -3;
      coneNew[1] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 6;
      orntNew[1] = -3;
      coneNew[2] = fStartNew + (fMax    - fStart)*4 + (c - cStart)*8 + 4;
      orntNew[2] = -2;
      coneNew[3] = fStartNew + (cone[1] - fStart)*4 + 3;
      orntNew[3] = ornt[2];
      ierr       = DMPlexSetCone(rdm, newp+7, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+7, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+7 < cStartNew) || (newp+7 >= cMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+7, cStartNew, cMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fMaxNew);
      }
#endif
    }
    /* Hybrid cells have 5 faces */
    for (c = cMax; c < cEnd; ++c) {
      const PetscInt  newp = cStartNew + (cMax - cStart)*8 + (c - cMax)*4;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[5], orntNew[5];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      for (r = 0; r < 3; ++r) {
        coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetTriSubface_Static(ornt[0], r);
        orntNew[0] = ornt[0];
        coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetTriSubface_Static(ornt[1], r);
        orntNew[1] = ornt[1];
        coneNew[2] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (cone[2+GetTriSubface_Static(ornt[0], (r+2)%3)] - fMax)*2 + (ornt[2+GetTriSubface_Static(ornt[0], (r+2)%3)] < 0 ? 0 : 1);
        orntNew[2] = 0;
        coneNew[3] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (cone[2+GetTriSubface_Static(ornt[0], r)]       - fMax)*2 + (ornt[2+GetTriSubface_Static(ornt[0], r)]       < 0 ? 1 : 0);
        orntNew[3] = 0;
        coneNew[4] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (fEnd - fMax)*2 + (c - cMax)*3 + GetTriSubface_Static(ornt[0], r);
        orntNew[4] = 0;
        ierr = DMPlexSetCone(rdm, newp+r, coneNew);CHKERRQ(ierr);
        ierr = DMPlexSetConeOrientation(rdm, newp+r, orntNew);CHKERRQ(ierr);
#if 1
        if ((newp+r < cMaxNew) || (newp+r >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a hybrid cell [%d, %d)", newp+r, cMaxNew, cEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fMaxNew);
        }
        for (p = 2; p < 5; ++p) {
          if ((coneNew[p] < fMaxNew)   || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a hybrid face [%d, %d)", coneNew[p], fMaxNew, fEndNew);
        }
#endif
      }
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + 3;
      orntNew[0] = 0;
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + 3;
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (fEnd - fMax)*2 + (c - cMax)*3 + 0;
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (fEnd - fMax)*2 + (c - cMax)*3 + 1;
      orntNew[3] = 0;
      coneNew[4] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (fEnd - fMax)*2 + (c - cMax)*3 + 2;
      orntNew[4] = 0;
      ierr = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+3 < cMaxNew) || (newp+3 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a hybrid cell [%d, %d)", newp+3, cMaxNew, cEndNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fMaxNew);
      }
      for (p = 2; p < 5; ++p) {
        if ((coneNew[p] < fMaxNew)   || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a hybrid face [%d, %d)", coneNew[p], fMaxNew, fEndNew);
      }
#endif
    }
    /* Split faces have 3 edges and the same cells as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscMalloc((2 + maxSupportSize*2) * sizeof(PetscInt), &supportRef);CHKERRQ(ierr);
    for (f = fStart; f < fMax; ++f) {
      const PetscInt  newp = fStartNew + (f - fStart)*4;
      const PetscInt *cone, *ornt, *support;
      PetscInt        coneNew[3], orntNew[3], coneSize, supportSize, s;

      ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, f, &ornt);CHKERRQ(ierr);
      /* A triangle */
      coneNew[0] = eStartNew + (cone[0] - eStart)*2 + (ornt[0] < 0 ? 1 : 0);
      orntNew[0] = ornt[0];
      coneNew[1] = eStartNew + (eMax    - eStart)*2 + (f - fStart)*3 + 2;
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (cone[2] - eStart)*2 + (ornt[2] < 0 ? 0 : 1);
      orntNew[2] = ornt[2];
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+0 < fStartNew) || (newp+0 >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp+0, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      /* B triangle */
      coneNew[0] = eStartNew + (cone[0] - eStart)*2 + (ornt[0] < 0 ? 0 : 1);
      orntNew[0] = ornt[0];
      coneNew[1] = eStartNew + (cone[1] - eStart)*2 + (ornt[1] < 0 ? 1 : 0);
      orntNew[1] = ornt[1];
      coneNew[2] = eStartNew + (eMax    - eStart)*2 + (f - fStart)*3 + 0;
      orntNew[2] = -2;
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+1 < fStartNew) || (newp+1 >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp+1, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      /* C triangle */
      coneNew[0] = eStartNew + (eMax    - eStart)*2 + (f - fStart)*3 + 1;
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (cone[1] - eStart)*2 + (ornt[1] < 0 ? 0 : 1);
      orntNew[1] = ornt[1];
      coneNew[2] = eStartNew + (cone[2] - eStart)*2 + (ornt[2] < 0 ? 1 : 0);
      orntNew[2] = ornt[2];
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+2 < fStartNew) || (newp+2 >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp+2, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      /* D triangle */
      coneNew[0] = eStartNew + (eMax    - eStart)*2 + (f - fStart)*3 + 0;
      orntNew[0] = 0;
      coneNew[1] = eStartNew + (eMax    - eStart)*2 + (f - fStart)*3 + 1;
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eMax    - eStart)*2 + (f - fStart)*3 + 2;
      orntNew[2] = 0;
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+3 < fStartNew) || (newp+3 >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp+3, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (r = 0; r < 4; ++r) {
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == f) break;
          }
          if (support[s] < cMax) {
            supportRef[s] = cStartNew + (support[s] - cStart)*8 + (r==3 ? (c+2)%4 + 4 : (ornt[c] < 0 ? faces[c*3+(-(ornt[c]+1)+1+3-r)%3] : faces[c*3+r]));
          } else {
            supportRef[s] = cStartNew + (cMax - cStart)*8 + (support[s] - cMax)*4 + r;
          }
        }
        ierr = DMPlexSetSupport(rdm, newp+r, supportRef);CHKERRQ(ierr);
#if 1
        if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fMaxNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an interior or hybrid cell [%d, %d)", supportRef[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior cell faces have 3 edges and 2 cells */
    for (c = cStart; c < cMax; ++c) {
      PetscInt        newp = fStartNew + (fMax - fStart)*4 + (c - cStart)*8;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[3], orntNew[3];
      PetscInt        supportNew[2];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* Face A: {c, a, d} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*3 + (ornt[0] < 0 ? (-(ornt[0]+1)+0)%3 : (ornt[0]+2)%3);
      orntNew[0] = ornt[0] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*3 + (ornt[1] < 0 ? (-(ornt[1]+1)+0)%3 : (ornt[1]+2)%3);
      orntNew[1] = ornt[1] < 0 ? -2 : 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[2] - fStart)*3 + (ornt[2] < 0 ? (-(ornt[2]+1)+0)%3 : (ornt[2]+2)%3);
      orntNew[2] = ornt[2] < 0 ? -2 : 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 0;
      supportNew[1] = (c - cStart)*8 + 0+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cMaxNew);
      }
#endif
      ++newp;
      /* Face B: {a, b, e} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*3 + (ornt[0] < 0 ? (-(ornt[0]+1)+2)%3 : (ornt[0]+0)%3);
      orntNew[0] = ornt[0] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[3] - fStart)*3 + (ornt[3] < 0 ? (-(ornt[3]+1)+2)%3 : (ornt[3]+0)%3);
      orntNew[1] = ornt[3] < 0 ? -2 : 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*3 + (ornt[1] < 0 ? (-(ornt[1]+1)+1)%3 : (ornt[1]+1)%3);
      orntNew[2] = ornt[1] < 0 ? -2 : 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+1 < fStartNew) || (newp+1 >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp+1, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 1;
      supportNew[1] = (c - cStart)*8 + 1+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cMaxNew);
      }
#endif
      ++newp;
      /* Face C: {c, f, b} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[2] - fStart)*3 + (ornt[2] < 0 ? (-(ornt[2]+1)+2)%3 : (ornt[2]+0)%3);
      orntNew[0] = ornt[2] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[3] - fStart)*3 + (ornt[3] < 0 ? (-(ornt[3]+1)+0)%3 : (ornt[3]+2)%3);
      orntNew[1] = ornt[3] < 0 ? -2 : 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*3 + (ornt[0] < 0 ? (-(ornt[0]+1)+1)%3 : (ornt[0]+1)%3);
      orntNew[2] = ornt[0] < 0 ? -2 : 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 2;
      supportNew[1] = (c - cStart)*8 + 2+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cMaxNew);
      }
#endif
      ++newp;
      /* Face D: {d, e, f} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*3 + (ornt[1] < 0 ? (-(ornt[1]+1)+2)%3 : (ornt[1]+0)%3);
      orntNew[0] = ornt[1] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[3] - fStart)*3 + (ornt[3] < 0 ? (-(ornt[3]+1)+1)%3 : (ornt[3]+1)%3);
      orntNew[1] = ornt[3] < 0 ? -2 : 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[2] - fStart)*3 + (ornt[2] < 0 ? (-(ornt[2]+1)+1)%3 : (ornt[2]+1)%3);
      orntNew[2] = ornt[2] < 0 ? -2 : 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 3;
      supportNew[1] = (c - cStart)*8 + 3+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cMaxNew);
      }
#endif
      ++newp;
      /* Face E: {d, f, a} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[2] - fStart)*3 + (ornt[2] < 0 ? (-(ornt[2]+1)+1)%3 : (ornt[2]+1)%3);
      orntNew[0] = ornt[2] < 0 ? 0 : -2;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*3 + (ornt[1] < 0 ? (-(ornt[1]+1)+0)%3 : (ornt[1]+2)%3);
      orntNew[2] = ornt[1] < 0 ? -2 : 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 0+4;
      supportNew[1] = (c - cStart)*8 + 3+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cMaxNew);
      }
#endif
      ++newp;
      /* Face F: {c, a, f} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*3 + (ornt[0] < 0 ? (-(ornt[0]+1)+0)%3 : (ornt[0]+2)%3);
      orntNew[0] = ornt[0] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart);
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[2] - fStart)*3 + (ornt[2] < 0 ? (-(ornt[2]+1)+2)%3 : (ornt[2]+0)%3);
      orntNew[2] = ornt[1] < 0 ? 0 : -2;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 0+4;
      supportNew[1] = (c - cStart)*8 + 2+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cMaxNew);
      }
#endif
      ++newp;
      /* Face G: {e, a, f} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*3 + (ornt[1] < 0 ? (-(ornt[1]+1)+1)%3 : (ornt[1]+1)%3);
      orntNew[0] = ornt[1] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart);
      orntNew[1] = -2;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (cone[3] - fStart)*3 + (ornt[3] < 0 ? (-(ornt[3]+1)+1)%3 : (ornt[3]+1)%3);
      orntNew[2] = ornt[3] < 0 ? 0 : -2;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 1+4;
      supportNew[1] = (c - cStart)*8 + 3+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cMaxNew);
      }
#endif
      ++newp;
      /* Face H: {a, b, f} */
      coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*3 + (ornt[0] < 0 ? (-(ornt[0]+1)+2)%3 : (ornt[0]+0)%3);
      orntNew[0] = ornt[0] < 0 ? -2 : 0;
      coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[3] - fStart)*3 + (ornt[3] < 0 ? (-(ornt[3]+1)+0)%3 : (ornt[3]+2)%3);
      orntNew[1] = ornt[3] < 0 ? 0 : -2;
      coneNew[2] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart);
      orntNew[2] = 0;
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 3; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eMaxNew);
      }
#endif
      supportNew[0] = (c - cStart)*8 + 1+4;
      supportNew[1] = (c - cStart)*8 + 2+4;
      ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fMaxNew);
      for (p = 0; p < 2; ++p) {
        if ((supportNew[p] < cStartNew) || (supportNew[p] >= cMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cMaxNew);
      }
#endif
      ++newp;
    }
    /* Hybrid split faces have 4 edges and same cells */
    for (f = fMax; f < fEnd; ++f) {
      const PetscInt *cone, *ornt, *support;
      PetscInt        coneNew[4], orntNew[4];
      PetscInt        supportNew[2], size, s, c;

      ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, f, &ornt);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (r = 0; r < 2; ++r) {
        const PetscInt newp = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (f - fMax)*2 + r;

        coneNew[0]   = eStartNew + (cone[0] - eStart)*2 + (ornt[0] < 0 ? 1-r : r);
        orntNew[0]   = ornt[0];
        coneNew[1]   = eStartNew + (cone[1] - eStart)*2 + (ornt[1] < 0 ? 1-r : r);
        orntNew[1]   = ornt[1];
        coneNew[2+r] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (cone[2+r] - eMax);
        orntNew[2+r] = 0;
        coneNew[3-r] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (eEnd      - eMax) + (f - fMax);
        orntNew[3-r] = 0;
        ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
        ierr = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
        if ((newp < fMaxNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a hybrid face [%d, %d)", newp, fMaxNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eMaxNew);
        }
        for (p = 2; p < 4; ++p) {
          if ((coneNew[p] < eMaxNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a hybrid edge [%d, %d)", coneNew[p], eMaxNew, eEndNew);
        }
#endif
        for (s = 0; s < size; ++s) {
          const PetscInt *coneCell, *orntCell;

          ierr = DMPlexGetCone(dm, support[s], &coneCell);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &orntCell);CHKERRQ(ierr);
          for (c = 2; c < 5; ++c) if (coneCell[c] == f) break;
          if (c >= 5) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not find face %d in cone of cell %d", f, support[s]);
          supportNew[s] = cStartNew + (cMax - cStart)*8 + (support[s] - cMax)*4 + GetTriSubface_Static(orntCell[c], (c-2+r)%3);
        }
        ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
        if ((newp < fMaxNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a hybrid face [%d, %d)", newp, fMaxNew, fEndNew);
        for (p = 0; p < size; ++p) {
          if ((supportNew[p] < cMaxNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a hybrid cell [%d, %d)", supportNew[p], cMaxNew, cEndNew);
        }
#endif
      }
    }
    /* Hybrid cell faces have 4 edges and 2 cells */
    for (c = cMax; c < cEnd; ++c) {
      PetscInt        newp = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (fEnd - fMax)*2 + (c - cMax)*3;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[4], orntNew[4];
      PetscInt        supportNew[2];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      for (r = 0; r < 3; ++r) {
        coneNew[0] = eStartNew + (eMax - eStart)*2 + (cone[0] - fStart)*3 + GetTriSubface_Static(ornt[0], (r+2)%3);
        orntNew[0] = 0;
        coneNew[1] = eStartNew + (eMax - eStart)*2 + (cone[1] - fStart)*3 + GetTriSubface_Static(ornt[1], (r+2)%3);
        orntNew[1] = 0;
        coneNew[2] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (eEnd - eMax) + (cone[2+GetTriSubface_Static(ornt[0], (r+2)%3)] - fMax);
        orntNew[2] = 0;
        coneNew[3] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (eEnd - eMax) + (cone[2+GetTriSubface_Static(ornt[0], r)]       - fMax);
        orntNew[3] = 0;
        ierr = DMPlexSetCone(rdm, newp+r, coneNew);CHKERRQ(ierr);
        ierr = DMPlexSetConeOrientation(rdm, newp+r, orntNew);CHKERRQ(ierr);
#if 1
        if ((newp+r < fMaxNew) || (newp+r >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a hybrid face [%d, %d)", newp+r, fMaxNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < eStartNew) || (coneNew[p] >= eMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eMaxNew);
        }
        for (p = 2; p < 4; ++p) {
          if ((coneNew[p] < eMaxNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a hybrid edge [%d, %d)", coneNew[p], eMaxNew, eEndNew);
        }
#endif
        supportNew[0] = cStartNew + (cMax - cStart)*8 + (c - cMax)*4 + GetTriSubface_Static(ornt[0], r);
        supportNew[1] = cStartNew + (cMax - cStart)*8 + (c - cMax)*4 + 3;
        ierr          = DMPlexSetSupport(rdm, newp+r, supportNew);CHKERRQ(ierr);
#if 1
        if ((newp+r < fMaxNew) || (newp+r >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a hybrid face [%d, %d)", newp+r, fMaxNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((supportNew[p] < cMaxNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a hybrid cell [%d, %d)", supportNew[p], cMaxNew, cEndNew);
        }
#endif
      }
    }
    /* Interior split edges have 2 vertices and the same faces as the parent */
    for (e = eStart; e < eMax; ++e) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (e - eStart);

      for (r = 0; r < 2; ++r) {
        const PetscInt  newp = eStartNew + (e - eStart)*2 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[2], coneSize, c, supportSize, s;

        ierr             = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
        coneNew[0]       = vStartNew + (cone[0] - vStart);
        coneNew[1]       = vStartNew + (cone[1] - vStart);
        coneNew[(r+1)%2] = newv;
        ierr             = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if 1
        if ((newp < eStartNew) || (newp >= eMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", newp, eStartNew, eMaxNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, e, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) if (cone[c] == e) break;
          if (support[s] < fMax) {
            supportRef[s] = fStartNew + (support[s] - fStart)*4 + (c + (ornt[c] < 0 ? 1-r : r))%3;
          } else {
            supportRef[s] = fStartNew + (fMax       - fStart)*4 + (cMax - cStart)*8 + (support[s] - fMax)*2 + (ornt[c] < 0 ? 1-r : r);
          }
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
        if ((newp < eStartNew) || (newp >= eMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", newp, eStartNew, eMaxNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an interior or hybrid face [%d, %d)", supportRef[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Interior face edges have 2 vertices and 2+cells*(1/2) faces */
    for (f = fStart; f < fMax; ++f) {
      const PetscInt *cone, *ornt, *support;
      PetscInt        coneSize, supportSize, s;

      ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (r = 0; r < 3; ++r) {
        const PetscInt  newp = eStartNew + (eMax - eStart)*2 + (f - fStart)*3 + r;
        PetscInt        coneNew[2], intFaces = 0, er, eint[4] = {1, 0, 2, 0};
        PetscInt        fint[24] = { 1,  7, -1, -1,  0,  5,
                                    -1, -1,  1,  6,  0,  4,
                                     2,  5,  3,  4, -1, -1,
                                    -1, -1,  3,  6,  2,  7};

        ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
        coneNew[0] = vStartNew + (vEnd - vStart) + (cone[(r+0)%3] - eStart);
        coneNew[1] = vStartNew + (vEnd - vStart) + (cone[(r+1)%3] - eStart);
        ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if 1
        if ((newp < eStartNew) || (newp >= eMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", newp, eStartNew, eMaxNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        supportRef[0] = fStartNew + (f - fStart)*4 + (r+1)%3;
        supportRef[1] = fStartNew + (f - fStart)*4 + 3;
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {if (cone[c] == f) break;}
          if (support[s] < cMax) {
            /* Here we want to determine whether edge newp contains a vertex which is part of the cross-tet edge */
            er   = ornt[c] < 0 ? (-(ornt[c]+1) + 2-r)%3 : (ornt[c] + r)%3;
            if (er == eint[c]) {
              supportRef[2+intFaces++] = fStartNew + (fMax - fStart)*4 + (support[s] - cStart)*8 + (c + 2)%4;
            } else {
              supportRef[2+intFaces++] = fStartNew + (fMax - fStart)*4 + (support[s] - cStart)*8 + fint[(c*3 + er)*2 + 0];
              supportRef[2+intFaces++] = fStartNew + (fMax - fStart)*4 + (support[s] - cStart)*8 + fint[(c*3 + er)*2 + 1];
            }
          } else {
            supportRef[2+intFaces++] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (fEnd - fMax)*2 + (support[s] - cMax)*3 + (r+1)%3;
          }
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
        if ((newp < eStartNew) || (newp >= eMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", newp, eStartNew, eMaxNew);
        for (p = 0; p < intFaces; ++p) {
          if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an interior or hybrid face [%d, %d)", supportRef[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Interior cell edges have 2 vertices and 4 faces */
    for (c = cStart; c < cMax; ++c) {
      const PetscInt  newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (c - cStart);
      const PetscInt *cone, *ornt, *fcone;
      PetscInt        coneNew[2], supportNew[4], find;

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, cone[0], &fcone);CHKERRQ(ierr);
      find = GetTriEdge_Static(ornt[0], 0);
      coneNew[0] = vStartNew + (vEnd - vStart) + (fcone[find] - eStart);
      ierr = DMPlexGetCone(dm, cone[2], &fcone);CHKERRQ(ierr);
      find = GetTriEdge_Static(ornt[2], 1);
      coneNew[1] = vStartNew + (vEnd - vStart) + (fcone[find] - eStart);
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if 1
      if ((newp < eStartNew) || (newp >= eMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", newp, eStartNew, eMaxNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", coneNew[p], vStartNew, vEndNew);
      }
#endif
      supportNew[0] = fStartNew + (fMax - fStart)*4 + (c - cStart)*8 + 4;
      supportNew[1] = fStartNew + (fMax - fStart)*4 + (c - cStart)*8 + 5;
      supportNew[2] = fStartNew + (fMax - fStart)*4 + (c - cStart)*8 + 6;
      supportNew[3] = fStartNew + (fMax - fStart)*4 + (c - cStart)*8 + 7;
      ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
      if ((newp < eStartNew) || (newp >= eMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", newp, eStartNew, eMaxNew);
      for (p = 0; p < 4; ++p) {
        if ((supportNew[p] < fStartNew) || (supportNew[p] >= fMaxNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", supportNew[p], fStartNew, fMaxNew);
      }
#endif
    }
    /* Hybrid edges have two vertices and the same faces */
    for (e = eMax; e < eEnd; ++e) {
      const PetscInt  newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (e - eMax);
      const PetscInt *cone, *support, *fcone;
      PetscInt        coneNew[2], size, fsize, s;

      ierr = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
      coneNew[0] = vStartNew + (cone[0] - vStart);
      coneNew[1] = vStartNew + (cone[1] - vStart);
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if 1
      if ((newp < eMaxNew) || (newp >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a hybrid edge [%d, %d)", newp, eMaxNew, eEndNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", coneNew[p], vStartNew, vEndNew);
      }
#endif
      for (s = 0; s < size; ++s) {
        ierr = DMPlexGetConeSize(dm, support[s], &fsize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, support[s], &fcone);CHKERRQ(ierr);
        for (c = 0; c < fsize; ++c) if (fcone[c] == e) break;
        if ((c < 2) || (c > 3)) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Edge %d not found in cone of face %d", e, support[s]);
        supportRef[s] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (support[s] - fMax)*2 + c-2;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
      if ((newp < eMaxNew) || (newp >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a hybrid edge [%d, %d)", newp, eMaxNew, eEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < fMaxNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a hybrid face [%d, %d)", supportRef[p], fMaxNew, fEndNew);
      }
#endif
    }
    /* Hybrid face edges have 2 vertices and 2+2*cells faces */
    for (f = fMax; f < fEnd; ++f) {
      const PetscInt  newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (eEnd - eMax) + (f - fMax);
      const PetscInt *cone, *support, *ccone;
      PetscInt        coneNew[2], size, csize, s;

      ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      coneNew[0] = vStartNew + (vEnd - vStart) + (cone[0] - eStart);
      coneNew[1] = vStartNew + (vEnd - vStart) + (cone[1] - eStart);
      ierr = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if 1
      if ((newp < eMaxNew) || (newp >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a hybrid edge [%d, %d)", newp, eMaxNew, eEndNew);
      for (p = 0; p < 2; ++p) {
        if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", coneNew[p], vStartNew, vEndNew);
      }
#endif
      supportRef[0] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (f - fMax)*2 + 0;
      supportRef[1] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (f - fMax)*2 + 1;
      for (s = 0; s < size; ++s) {
        ierr = DMPlexGetConeSize(dm, support[s], &csize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, support[s], &ccone);CHKERRQ(ierr);
        for (c = 0; c < csize; ++c) if (ccone[c] == f) break;
        if ((c < 2) || (c >= csize)) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Hybrid face %d is not in cone of hybrid cell %d", f, support[s]);
        supportRef[2+s*2+0] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (fEnd - fMax)*2 + (support[s] - cMax)*3 + (c-2)%3;
        supportRef[2+s*2+1] = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (fEnd - fMax)*2 + (support[s] - cMax)*3 + (c-1)%3;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
      if ((newp < eMaxNew) || (newp >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a hybrid edge [%d, %d)", newp, eMaxNew, eEndNew);
      for (p = 0; p < 2+size*2; ++p) {
        if ((supportRef[p] < fMaxNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a hybrid face [%d, %d)", supportRef[p], fMaxNew, fEndNew);
      }
#endif
    }
    /* Interior vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt  newp = vStartNew + (v - vStart);
      const PetscInt *support, *cone;
      PetscInt        size, s;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, v, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if (cone[1] == v) r = 1;
        if (support[s] < eMax) supportRef[s] = eStartNew + (support[s] - eStart)*2 + r;
        else                   supportRef[s] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (support[s] - eMax);
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", newp, vStartNew, vEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an interior or hybrid edge [%d, %d)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    /* Interior edge vertices have 2 + interior face*2 + hybrid face + cells*0/1 supports */
    for (e = eStart; e < eMax; ++e) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (e - eStart);
      const PetscInt *cone, *support;
      PetscInt       *star = NULL, starSize, faceSize = 0, cellSize = 0, coneSize, size, s;

      ierr          = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
      supportRef[0] = eStartNew + (e - eStart)*2 + 0;
      supportRef[1] = eStartNew + (e - eStart)*2 + 1;
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        if (support[s] < fMax) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          for (r = 0; r < coneSize; ++r) {if (cone[r] == e) break;}
          supportRef[2+faceSize+0] = eStartNew + (eMax - eStart)*2 + (support[s] - fStart)*3 + (r+0)%3;
          supportRef[2+faceSize+1] = eStartNew + (eMax - eStart)*2 + (support[s] - fStart)*3 + (r+2)%3;
          faceSize += 2;
        } else {
          supportRef[2+faceSize+0] = eStartNew + (eMax - eStart)*2 + (fMax       - fStart)*3 + (cMax - cStart) + (eEnd - eMax) + (support[s] - fMax);
          ++faceSize;
        }
      }
      ierr = DMPlexGetTransitiveClosure(dm, e, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
      for (s = 0; s < starSize*2; s += 2) {
        const PetscInt *cone, *ornt;
        PetscInt        e01, e23;

        if ((star[s] >= cStart) && (star[s] < cMax)) {
          /* Check edge 0-1 */
          ierr = DMPlexGetCone(dm, star[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, star[s], &ornt);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, cone[0], &cone);CHKERRQ(ierr);
          e01  = cone[GetTriEdge_Static(ornt[0], 0)];
          /* Check edge 2-3 */
          ierr = DMPlexGetCone(dm, star[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, star[s], &ornt);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, cone[2], &cone);CHKERRQ(ierr);
          e23  = cone[GetTriEdge_Static(ornt[2], 1)];
          if ((e01 == e) || (e23 == e)) {supportRef[2+faceSize+cellSize++] = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (star[s] - cStart);}
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, e, PETSC_FALSE, &starSize, &star);CHKERRQ(ierr);
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", newp, vStartNew, vEndNew);
      for (p = 0; p < 2+faceSize+cellSize; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an interior or hybrid edge [%d, %d)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    ierr = PetscFree(supportRef);CHKERRQ(ierr);
    ierr = DMPlexRestoreFaces_Internal(dm, 3, cStart, NULL, NULL, &faces);CHKERRQ(ierr);
    break;
  case 6:
    /* Hex 3D */
    /*
     Bottom (viewed from top)    Top
     1---------2---------2       7---------2---------6
     |         |         |       |         |         |
     |    B    2    C    |       |    H    2    G    |
     |         |         |       |         |         |
     3----3----0----1----1       3----3----0----1----1
     |         |         |       |         |         |
     |    A    0    D    |       |    E    0    F    |
     |         |         |       |         |         |
     0---------0---------3       4---------0---------5
     */
    /* All cells have 6 faces: Bottom, Top, Front, Back, Right, Left */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt  newp = (c - cStart)*8;
      const PetscInt *cone, *ornt;
      PetscInt        coneNew[6], orntNew[6];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetQuadSubface_Static(ornt[0], 0);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  8; /* AE */
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetQuadSubface_Static(ornt[2], 0);
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  3; /* AB */
      orntNew[3] = 0;
      coneNew[4] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  0; /* AD */
      orntNew[4] = 0;
      coneNew[5] = fStartNew + (cone[5] - fStart)*4 + GetQuadSubface_Static(ornt[5], 0);
      orntNew[5] = ornt[5];
      ierr       = DMPlexSetCone(rdm, newp+0, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+0, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+0 < cStartNew) || (newp+0 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+0, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* B hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetQuadSubface_Static(ornt[0], 1);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 + 11; /* BH */
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  3; /* AB */
      orntNew[2] = -3;
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetQuadSubface_Static(ornt[3], 1);
      orntNew[3] = ornt[3];
      coneNew[4] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  2; /* BC */
      orntNew[4] = 0;
      coneNew[5] = fStartNew + (cone[5] - fStart)*4 + GetQuadSubface_Static(ornt[5], 3);
      orntNew[5] = ornt[5];
      ierr       = DMPlexSetCone(rdm, newp+1, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+1, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+1 < cStartNew) || (newp+1 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+1, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* C hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetQuadSubface_Static(ornt[0], 2);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 + 10; /* CG */
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  1; /* CD */
      orntNew[2] = 0;
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetQuadSubface_Static(ornt[3], 0);
      orntNew[3] = ornt[3];
      coneNew[4] = fStartNew + (cone[4] - fStart)*4 + GetQuadSubface_Static(ornt[4], 1);
      orntNew[4] = ornt[4];
      coneNew[5] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  2; /* BC */
      orntNew[5] = -3;
      ierr       = DMPlexSetCone(rdm, newp+2, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+2, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+2 < cStartNew) || (newp+2 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+2, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* D hex */
      coneNew[0] = fStartNew + (cone[0] - fStart)*4 + GetQuadSubface_Static(ornt[0], 3);
      orntNew[0] = ornt[0];
      coneNew[1] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  9; /* DF */
      orntNew[1] = 0;
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetQuadSubface_Static(ornt[2], 1);
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  1; /* CD */
      orntNew[3] = -3;
      coneNew[4] = fStartNew + (cone[4] - fStart)*4 + GetQuadSubface_Static(ornt[4], 0);
      orntNew[4] = ornt[4];
      coneNew[5] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  0; /* AD */
      orntNew[5] = -3;
      ierr       = DMPlexSetCone(rdm, newp+3, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+3, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+3 < cStartNew) || (newp+3 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+3, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* E hex */
      coneNew[0] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  8; /* AE */
      orntNew[0] = -3;
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetQuadSubface_Static(ornt[1], 0);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetQuadSubface_Static(ornt[2], 3);
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  7; /* EH */
      orntNew[3] = 0;
      coneNew[4] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  4; /* EF */
      orntNew[4] = 0;
      coneNew[5] = fStartNew + (cone[5] - fStart)*4 + GetQuadSubface_Static(ornt[5], 1);
      orntNew[5] = ornt[5];
      ierr       = DMPlexSetCone(rdm, newp+4, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+4, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+4 < cStartNew) || (newp+4 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+4, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* F hex */
      coneNew[0] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  9; /* DF */
      orntNew[0] = -3;
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetQuadSubface_Static(ornt[1], 1);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (cone[2] - fStart)*4 + GetQuadSubface_Static(ornt[2], 2);
      orntNew[2] = ornt[2];
      coneNew[3] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  5; /* FG */
      orntNew[3] = 0;
      coneNew[4] = fStartNew + (cone[4] - fStart)*4 + GetQuadSubface_Static(ornt[4], 3);
      orntNew[4] = ornt[4];
      coneNew[5] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  4; /* EF */
      orntNew[5] = -3;
      ierr       = DMPlexSetCone(rdm, newp+5, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+5, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+5 < cStartNew) || (newp+5 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+5, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* G hex */
      coneNew[0] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 + 10; /* CG */
      orntNew[0] = -3;
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetQuadSubface_Static(ornt[1], 2);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  5; /* FG */
      orntNew[2] = -3;
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetQuadSubface_Static(ornt[3], 3);
      orntNew[3] = ornt[3];
      coneNew[4] = fStartNew + (cone[4] - fStart)*4 + GetQuadSubface_Static(ornt[4], 2);
      orntNew[4] = ornt[4];
      coneNew[5] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  6; /* GH */
      orntNew[5] = 0;
      ierr       = DMPlexSetCone(rdm, newp+6, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+6, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+6 < cStartNew) || (newp+6 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+6, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
      /* H hex */
      coneNew[0] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 + 11; /* BH */
      orntNew[0] = -3;
      coneNew[1] = fStartNew + (cone[1] - fStart)*4 + GetQuadSubface_Static(ornt[1], 3);
      orntNew[1] = ornt[1];
      coneNew[2] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  7; /* EH */
      orntNew[2] = -3;
      coneNew[3] = fStartNew + (cone[3] - fStart)*4 + GetQuadSubface_Static(ornt[3], 2);
      orntNew[3] = ornt[3];
      coneNew[4] = fStartNew + (fEnd    - fStart)*4 + (c - cStart)*12 +  6; /* GH */
      orntNew[4] = -3;
      coneNew[5] = fStartNew + (cone[5] - fStart)*4 + GetQuadSubface_Static(ornt[5], 2);
      orntNew[5] = ornt[5];
      ierr       = DMPlexSetCone(rdm, newp+7, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp+7, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp+7 < cStartNew) || (newp+7 >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", newp+7, cStartNew, cEndNew);
      for (p = 0; p < 6; ++p) {
        if ((coneNew[p] < fStartNew) || (coneNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", coneNew[p], fStartNew, fEndNew);
      }
#endif
    }
    /* Split faces have 4 edges and the same cells as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    ierr = PetscMalloc((4 + maxSupportSize*2) * sizeof(PetscInt), &supportRef);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      for (r = 0; r < 4; ++r) {
        /* TODO: This can come from GetFaces_Internal() */
        const PetscInt  newCells[24] = {0, 1, 2, 3,  4, 5, 6, 7,  0, 3, 5, 4,  2, 1, 7, 6,  3, 2, 6, 5,  0, 4, 7, 1};
        const PetscInt  newp = fStartNew + (f - fStart)*4 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[4], orntNew[4], coneSize, c, supportSize, s;

        ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
        ierr = DMPlexGetConeOrientation(dm, f, &ornt);CHKERRQ(ierr);
        coneNew[0] = eStartNew + (cone[(r+3)%4] - eStart)*2 + (ornt[(r+3)%4] < 0 ? 0 : 1);
        orntNew[0] = ornt[(r+3)%4];
        coneNew[1] = eStartNew + (cone[r]       - eStart)*2 + (ornt[r] < 0 ? 1 : 0);
        orntNew[1] = ornt[r];
        coneNew[2] = eStartNew + (eEnd - eStart)*2 + (f - fStart)*4 + r;
        orntNew[2] = 0;
        coneNew[3] = eStartNew + (eEnd - eStart)*2 + (f - fStart)*4 + (r+3)%4;
        orntNew[3] = -2;
        ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
        ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
        for (p = 0; p < 4; ++p) {
          if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == f) break;
          }
          supportRef[s] = cStartNew + (support[s] - cStart)*8 + newCells[c*4+GetQuadSubface_Static(ornt[c], r)];
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < cStartNew) || (supportRef[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportRef[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Interior faces have 4 edges and 2 cells */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt  newCells[24] = {0, 3,  2, 3,  1, 2,  0, 1,  4, 5,  5, 6,  6, 7,  4, 7,  0, 4,  3, 5,  2, 6,  1, 7};
      const PetscInt *cone, *ornt;
      PetscInt        newp, coneNew[4], orntNew[4], supportNew[2];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, c, &ornt);CHKERRQ(ierr);
      /* A-D face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 0;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*4 + GetQuadEdge_Static(ornt[2], 0);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*4 + GetQuadEdge_Static(ornt[0], 3);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 0;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 2;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* C-D face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 1;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[4] - fStart)*4 + GetQuadEdge_Static(ornt[4], 0);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*4 + GetQuadEdge_Static(ornt[0], 2);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 0;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 4;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* B-C face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 2;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*4 + GetQuadEdge_Static(ornt[0], 1);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*4 + GetQuadEdge_Static(ornt[3], 0);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 3;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 0;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* A-B face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 3;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[0] - fStart)*4 + GetQuadEdge_Static(ornt[0], 0);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[5] - fStart)*4 + GetQuadEdge_Static(ornt[5], 3);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 5;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 0;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* E-F face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 4;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*4 + GetQuadEdge_Static(ornt[2], 2);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*4 + GetQuadEdge_Static(ornt[1], 0);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 1;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 2;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* F-G face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 5;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[4] - fStart)*4 + GetQuadEdge_Static(ornt[4], 2);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*4 + GetQuadEdge_Static(ornt[1], 1);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 1;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 4;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* G-H face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 6;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*4 + GetQuadEdge_Static(ornt[3], 2);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*4 + GetQuadEdge_Static(ornt[1], 2);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 1;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 3;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* E-H face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 7;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[5] - fStart)*4 + GetQuadEdge_Static(ornt[5], 1);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[1] - fStart)*4 + GetQuadEdge_Static(ornt[1], 3);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 1;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 5;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* A-E face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 8;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[5] - fStart)*4 + GetQuadEdge_Static(ornt[5], 0);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*4 + GetQuadEdge_Static(ornt[2], 3);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 2;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 5;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* D-F face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 9;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[2] - fStart)*4 + GetQuadEdge_Static(ornt[2], 1);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[4] - fStart)*4 + GetQuadEdge_Static(ornt[4], 3);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 4;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 2;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* C-G face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 10;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[4] - fStart)*4 + GetQuadEdge_Static(ornt[4], 1);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*4 + GetQuadEdge_Static(ornt[3], 3);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 3;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 4;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      /* B-H face */
      newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + 11;
      coneNew[0] = eStartNew + (eEnd - eStart)*2 + (cone[3] - fStart)*4 + GetQuadEdge_Static(ornt[3], 1);
      orntNew[0] = -2;
      coneNew[1] = eStartNew + (eEnd - eStart)*2 + (cone[5] - fStart)*4 + GetQuadEdge_Static(ornt[5], 2);
      orntNew[1] = 0;
      coneNew[2] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 5;
      orntNew[2] = 0;
      coneNew[3] = eStartNew + (eEnd - eStart)*2 + (fEnd    - fStart)*4 + (c - cStart)*6 + 3;
      orntNew[3] = -2;
      ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
      ierr       = DMPlexSetConeOrientation(rdm, newp, orntNew);CHKERRQ(ierr);
#if 1
      if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
      for (p = 0; p < 4; ++p) {
        if ((coneNew[p] < eStartNew) || (coneNew[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", coneNew[p], eStartNew, eEndNew);
      }
#endif
      for (r = 0; r < 12; ++r) {
        newp = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + r;
        supportNew[0] = cStartNew + (c - cStart)*8 + newCells[r*2+0];
        supportNew[1] = cStartNew + (c - cStart)*8 + newCells[r*2+1];
        ierr          = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
        if ((newp < fStartNew) || (newp >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", newp, fStartNew, fEndNew);
        for (p = 0; p < 2; ++p) {
          if ((supportNew[p] < cStartNew) || (supportNew[p] >= cEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a cell [%d, %d)", supportNew[p], cStartNew, cEndNew);
        }
#endif
      }
    }
    /* Split edges have 2 vertices and the same faces as the parent */
    ierr = DMPlexGetMaxSizes(dm, NULL, &maxSupportSize);CHKERRQ(ierr);
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (e - eStart);

      for (r = 0; r < 2; ++r) {
        const PetscInt  newp = eStartNew + (e - eStart)*2 + r;
        const PetscInt *cone, *ornt, *support;
        PetscInt        coneNew[2], coneSize, c, supportSize, s;

        ierr             = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
        coneNew[0]       = vStartNew + (cone[0] - vStart);
        coneNew[1]       = vStartNew + (cone[1] - vStart);
        coneNew[(r+1)%2] = newv;
        ierr             = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if 1
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", newp, eStartNew, eEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, e, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &ornt);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) {
            if (cone[c] == e) break;
          }
          supportRef[s] = fStartNew + (support[s] - fStart)*4 + (ornt[c] < 0 ? (c+1-r)%4 : (c+r)%4);
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", newp, eStartNew, eEndNew);
        for (p = 0; p < supportSize; ++p) {
          if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", supportRef[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Face edges have 2 vertices and 2+cells faces */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt  newFaces[24] = {3, 2, 1, 0,  4, 5, 6, 7,  0, 9, 4, 8,  2, 11, 6, 10,  1, 10, 5, 9,  8, 7, 11, 3};
      const PetscInt  newv = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (f - fStart);
      const PetscInt *cone, *coneCell, *orntCell, *support;
      PetscInt        coneNew[2], coneSize, c, supportSize, s;

      ierr = DMPlexGetCone(dm, f, &cone);CHKERRQ(ierr);
      for (r = 0; r < 4; ++r) {
        const PetscInt newp = eStartNew + (eEnd - eStart)*2 + (f - fStart)*4 + r;

        coneNew[0] = vStartNew + (vEnd - vStart) + (cone[r] - eStart);
        coneNew[1] = newv;
        ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if 1
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", newp, eStartNew, eEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        ierr = DMPlexGetSupportSize(dm, f, &supportSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
        supportRef[0] = fStartNew + (f - fStart)*4 + r;
        supportRef[1] = fStartNew + (f - fStart)*4 + (r+1)%4;
        for (s = 0; s < supportSize; ++s) {
          ierr = DMPlexGetConeSize(dm, support[s], &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, support[s], &coneCell);CHKERRQ(ierr);
          ierr = DMPlexGetConeOrientation(dm, support[s], &orntCell);CHKERRQ(ierr);
          for (c = 0; c < coneSize; ++c) if (coneCell[c] == f) break;
          supportRef[2+s] = fStartNew + (fEnd - fStart)*4 + (support[s] - cStart)*12 + newFaces[c*4 + GetQuadEdge_Static(orntCell[c], r)];
        }
        ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", newp, eStartNew, eEndNew);
        for (p = 0; p < 2+supportSize; ++p) {
          if ((supportRef[p] < fStartNew) || (supportRef[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", supportRef[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Cell edges have 2 vertices and 4 faces */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt  newFaces[24] = {0, 1, 2, 3,  4, 5, 6, 7,  0, 9, 4, 8,  2, 11, 6, 10,  1, 10, 5, 9,  3, 8, 7, 11};
      const PetscInt  newv = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (fEnd - fStart) + (c - cStart);
      const PetscInt *cone;
      PetscInt        coneNew[2], supportNew[4];

      ierr = DMPlexGetCone(dm, c, &cone);CHKERRQ(ierr);
      for (r = 0; r < 6; ++r) {
        const PetscInt newp = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*4 + (c - cStart)*6 + r;

        coneNew[0] = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (cone[r] - fStart);
        coneNew[1] = newv;
        ierr       = DMPlexSetCone(rdm, newp, coneNew);CHKERRQ(ierr);
#if 1
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", newp, eStartNew, eEndNew);
        for (p = 0; p < 2; ++p) {
          if ((coneNew[p] < vStartNew) || (coneNew[p] >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", coneNew[p], vStartNew, vEndNew);
        }
#endif
        for (f = 0; f < 4; ++f) supportNew[f] = fStartNew + (fEnd - fStart)*4 + (c - cStart)*12 + newFaces[r*4+f];
        ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
#if 1
        if ((newp < eStartNew) || (newp >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", newp, eStartNew, eEndNew);
        for (p = 0; p < 4; ++p) {
          if ((supportNew[p] < fStartNew) || (supportNew[p] >= fEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a face [%d, %d)", supportNew[p], fStartNew, fEndNew);
        }
#endif
      }
    }
    /* Old vertices have identical supports */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt  newp = vStartNew + (v - vStart);
      const PetscInt *support, *cone;
      PetscInt        size, s;

      ierr = DMPlexGetSupportSize(dm, v, &size);CHKERRQ(ierr);
      ierr = DMPlexGetSupport(dm, v, &support);CHKERRQ(ierr);
      for (s = 0; s < size; ++s) {
        PetscInt r = 0;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        if (cone[1] == v) r = 1;
        supportRef[s] = eStartNew + (support[s] - eStart)*2 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", newp, vStartNew, vEndNew);
      for (p = 0; p < size; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    /* Edge vertices have 2 + faces supports */
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (e - eStart);
      const PetscInt *cone, *support;
      PetscInt        size, s;

      ierr          = DMPlexGetSupportSize(dm, e, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, e, &support);CHKERRQ(ierr);
      supportRef[0] = eStartNew + (e - eStart)*2 + 0;
      supportRef[1] = eStartNew + (e - eStart)*2 + 1;
      for (s = 0; s < size; ++s) {
        PetscInt r;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        for (r = 0; r < 4; ++r) if (cone[r] == e) break;
        supportRef[2+s] = eStartNew + (eEnd - eStart)*2 + (support[s] - fStart)*4 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", newp, vStartNew, vEndNew);
      for (p = 0; p < 2+size; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    /* Face vertices have 4 + cells supports */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt  newp = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (f - fStart);
      const PetscInt *cone, *support;
      PetscInt        size, s;

      ierr          = DMPlexGetSupportSize(dm, f, &size);CHKERRQ(ierr);
      ierr          = DMPlexGetSupport(dm, f, &support);CHKERRQ(ierr);
      for (r = 0; r < 4; ++r) supportRef[r] = eStartNew + (e - eStart)*2 +  (f - fStart)*4 + r;
      for (s = 0; s < size; ++s) {
        PetscInt r;

        ierr = DMPlexGetCone(dm, support[s], &cone);CHKERRQ(ierr);
        for (r = 0; r < 6; ++r) if (cone[r] == f) break;
        supportRef[4+s] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*4 + (support[s] - cStart)*6 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportRef);CHKERRQ(ierr);
#if 1
      if ((newp < vStartNew) || (newp >= vEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not a vertex [%d, %d)", newp, vStartNew, vEndNew);
      for (p = 0; p < 4+size; ++p) {
        if ((supportRef[p] < eStartNew) || (supportRef[p] >= eEndNew)) SETERRQ3(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Point %d is not an edge [%d, %d)", supportRef[p], eStartNew, eEndNew);
      }
#endif
    }
    /* Cell vertices have 6 supports */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt newp = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (fEnd - fStart) + (c - cStart);
      PetscInt       supportNew[6];

      for (r = 0; r < 6; ++r) {
        supportNew[r] = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*4 + (c - cStart)*6 + r;
      }
      ierr = DMPlexSetSupport(rdm, newp, supportNew);CHKERRQ(ierr);
    }
    ierr = PetscFree(supportRef);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell refiner %d", refiner);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CellRefinerSetCoordinates"
PetscErrorCode CellRefinerSetCoordinates(CellRefiner refiner, DM dm, PetscInt depthSize[], DM rdm)
{
  PetscSection   coordSection, coordSectionNew;
  Vec            coordinates, coordinatesNew;
  PetscScalar   *coords, *coordsNew;
  const PetscInt numVertices = depthSize ? depthSize[0] : 0;
  PetscInt       dim, depth, coordSizeNew, cStart, cEnd, c, vStart, vStartNew, vEnd, v, eStart, eEnd, eMax, e, fStart, fEnd, fMax, f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, NULL, &fMax, &eMax, NULL);CHKERRQ(ierr);
  if (refiner) {ierr = GetDepthStart_Private(depth, depthSize, NULL, NULL, NULL, &vStartNew);CHKERRQ(ierr);}
  ierr = GetDepthStart_Private(depth, depthSize, NULL, NULL, NULL, &vStartNew);CHKERRQ(ierr);
  ierr = DMPlexGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dm), &coordSectionNew);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSectionNew, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSectionNew, 0, dim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSectionNew, vStartNew, vStartNew+numVertices);CHKERRQ(ierr);
  if (fMax < 0) fMax = fEnd;
  if (eMax < 0) eMax = eEnd;
  /* All vertices have the dim coordinates */
  for (v = vStartNew; v < vStartNew+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSectionNew, v, dim);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSectionNew, v, 0, dim);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSectionNew);CHKERRQ(ierr);
  ierr = DMPlexSetCoordinateSection(rdm, coordSectionNew);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSectionNew, &coordSizeNew);CHKERRQ(ierr);
  ierr = VecCreate(PetscObjectComm((PetscObject)dm), &coordinatesNew);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinatesNew, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinatesNew, coordSizeNew, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(coordinatesNew);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(coordinatesNew, &coordsNew);CHKERRQ(ierr);
  switch (refiner) {
  case 0: break;
  case 6: /* Hex 3D */
    /* Face vertices have the average of corner coordinates */
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (f - fStart);
      PetscInt      *cone = NULL;
      PetscInt       closureSize, coneSize = 0, off[8], offnew, p, d;

      ierr = DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &cone);CHKERRQ(ierr);
      for (p = 0; p < closureSize*2; p += 2) {
        const PetscInt point = cone[p];
        if ((point >= vStart) && (point < vEnd)) cone[coneSize++] = point;
      }
      for (v = 0; v < coneSize; ++v) {
        ierr = PetscSectionGetOffset(coordSection, cone[v], &off[v]);CHKERRQ(ierr);
      }
      ierr = PetscSectionGetOffset(coordSectionNew, newv, &offnew);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        coordsNew[offnew+d] = 0.0;
        for (v = 0; v < coneSize; ++v) coordsNew[offnew+d] += coords[off[v]+d];
        coordsNew[offnew+d] /= coneSize;
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &closureSize, &cone);CHKERRQ(ierr);
    }
  case 2: /* Hex 2D */
    /* Cell vertices have the average of corner coordinates */
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt newv = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (c - cStart) + (dim > 2 ? (fEnd - fStart) : 0);
      PetscInt      *cone = NULL;
      PetscInt       closureSize, coneSize = 0, off[8], offnew, p, d;

      ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &cone);CHKERRQ(ierr);
      for (p = 0; p < closureSize*2; p += 2) {
        const PetscInt point = cone[p];
        if ((point >= vStart) && (point < vEnd)) cone[coneSize++] = point;
      }
      for (v = 0; v < coneSize; ++v) {
        ierr = PetscSectionGetOffset(coordSection, cone[v], &off[v]);CHKERRQ(ierr);
      }
      ierr = PetscSectionGetOffset(coordSectionNew, newv, &offnew);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        coordsNew[offnew+d] = 0.0;
        for (v = 0; v < coneSize; ++v) coordsNew[offnew+d] += coords[off[v]+d];
        coordsNew[offnew+d] /= coneSize;
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &cone);CHKERRQ(ierr);
    }
  case 1: /* Simplicial 2D */
  case 3: /* Hybrid Simplicial 2D */
  case 5: /* Simplicial 3D */
  case 7: /* Hybrid Simplicial 3D */
    /* Edge vertices have the average of endpoint coordinates */
    for (e = eStart; e < eMax; ++e) {
      const PetscInt  newv = vStartNew + (vEnd - vStart) + (e - eStart);
      const PetscInt *cone;
      PetscInt        coneSize, offA, offB, offnew, d;

      ierr = DMPlexGetConeSize(dm, e, &coneSize);CHKERRQ(ierr);
      if (coneSize != 2) SETERRQ2(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Edge %d cone should have two vertices, not %d", e, coneSize);
      ierr = DMPlexGetCone(dm, e, &cone);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSection, cone[0], &offA);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSection, cone[1], &offB);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSectionNew, newv, &offnew);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        coordsNew[offnew+d] = 0.5*(coords[offA+d] + coords[offB+d]);
      }
    }
    /* Old vertices have the same coordinates */
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt newv = vStartNew + (v - vStart);
      PetscInt       off, offnew, d;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordSectionNew, newv, &offnew);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        coordsNew[offnew+d] = coords[off+d];
      }
    }
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell refiner %d", refiner);
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = VecRestoreArray(coordinatesNew, &coordsNew);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(rdm, coordinatesNew);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinatesNew);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&coordSectionNew);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexCreateProcessSF"
PetscErrorCode DMPlexCreateProcessSF(DM dm, PetscSF sfPoint, IS *processRanks, PetscSF *sfProcess)
{
  PetscInt           numRoots, numLeaves, l;
  const PetscInt    *localPoints;
  const PetscSFNode *remotePoints;
  PetscInt          *localPointsNew;
  PetscSFNode       *remotePointsNew;
  PetscInt          *ranks, *ranksNew;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscSFGetGraph(sfPoint, &numRoots, &numLeaves, &localPoints, &remotePoints);CHKERRQ(ierr);
  ierr = PetscMalloc(numLeaves * sizeof(PetscInt), &ranks);CHKERRQ(ierr);
  for (l = 0; l < numLeaves; ++l) {
    ranks[l] = remotePoints[l].rank;
  }
  ierr = PetscSortRemoveDupsInt(&numLeaves, ranks);CHKERRQ(ierr);
  ierr = PetscMalloc(numLeaves * sizeof(PetscInt),    &ranksNew);CHKERRQ(ierr);
  ierr = PetscMalloc(numLeaves * sizeof(PetscInt),    &localPointsNew);CHKERRQ(ierr);
  ierr = PetscMalloc(numLeaves * sizeof(PetscSFNode), &remotePointsNew);CHKERRQ(ierr);
  for (l = 0; l < numLeaves; ++l) {
    ranksNew[l]              = ranks[l];
    localPointsNew[l]        = l;
    remotePointsNew[l].index = 0;
    remotePointsNew[l].rank  = ranksNew[l];
  }
  ierr = PetscFree(ranks);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dm), numLeaves, ranksNew, PETSC_OWN_POINTER, processRanks);CHKERRQ(ierr);
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)dm), sfProcess);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(*sfProcess);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*sfProcess, 1, numLeaves, localPointsNew, PETSC_OWN_POINTER, remotePointsNew, PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CellRefinerCreateSF"
PetscErrorCode CellRefinerCreateSF(CellRefiner refiner, DM dm, PetscInt depthSize[], DM rdm)
{
  PetscSF            sf, sfNew, sfProcess;
  IS                 processRanks;
  MPI_Datatype       depthType;
  PetscInt           numRoots, numLeaves, numLeavesNew = 0, l, m;
  const PetscInt    *localPoints, *neighbors;
  const PetscSFNode *remotePoints;
  PetscInt          *localPointsNew;
  PetscSFNode       *remotePointsNew;
  PetscInt          *depthSizeOld, *rdepthSize, *rdepthSizeOld, *rdepthMaxOld, *rvStart, *rvStartNew, *reStart, *reStartNew, *rfStart, *rfStartNew, *rcStart, *rcStartNew;
  PetscInt           depth, numNeighbors, pStartNew, pEndNew, cStart, cStartNew, cEnd, cMax, vStart, vStartNew, vEnd, vMax, fStart, fStartNew, fEnd, fMax, eStart, eStartNew, eEnd, eMax, r, n;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetChart(rdm, &pStartNew, &pEndNew);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, &fMax, &eMax, &vMax);CHKERRQ(ierr);
  if (refiner) {ierr = GetDepthStart_Private(depth, depthSize, &cStartNew, &fStartNew, &eStartNew, &vStartNew);CHKERRQ(ierr);}
  ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
  ierr = DMGetPointSF(rdm, &sfNew);CHKERRQ(ierr);
  /* Caculate size of new SF */
  ierr = PetscSFGetGraph(sf, &numRoots, &numLeaves, &localPoints, &remotePoints);CHKERRQ(ierr);
  if (numRoots < 0) PetscFunctionReturn(0);
  for (l = 0; l < numLeaves; ++l) {
    const PetscInt p = localPoints[l];

    switch (refiner) {
    case 1:
      /* Simplicial 2D */
      if ((p >= vStart) && (p < vEnd)) {
        /* Old vertices stay the same */
        ++numLeavesNew;
      } else if ((p >= fStart) && (p < fEnd)) {
        /* Old faces add new faces and vertex */
        numLeavesNew += 2 + 1;
      } else if ((p >= cStart) && (p < cEnd)) {
        /* Old cells add new cells and interior faces */
        numLeavesNew += 4 + 3;
      }
      break;
    case 2:
      /* Hex 2D */
      if ((p >= vStart) && (p < vEnd)) {
        /* Old vertices stay the same */
        ++numLeavesNew;
      } else if ((p >= fStart) && (p < fEnd)) {
        /* Old faces add new faces and vertex */
        numLeavesNew += 2 + 1;
      } else if ((p >= cStart) && (p < cEnd)) {
        /* Old cells add new cells, interior faces, and vertex */
        numLeavesNew += 4 + 4 + 1;
      }
      break;
    case 5:
      /* Simplicial 3D */
      if ((p >= vStart) && (p < vEnd)) {
        /* Old vertices stay the same */
        ++numLeavesNew;
      } else if ((p >= eStart) && (p < eEnd)) {
        /* Old edges add new edges and vertex */
        numLeavesNew += 2 + 1;
      } else if ((p >= fStart) && (p < fEnd)) {
        /* Old faces add new faces and face edges */
        numLeavesNew += 4 + 3;
      } else if ((p >= cStart) && (p < cEnd)) {
        /* Old cells add new cells and interior faces and edges */
        numLeavesNew += 8 + 8 + 1;
      }
      break;
    case 7:
      /* Hybrid Simplicial 3D */
      if ((p >= vStart) && (p < vEnd)) {
        /* Interior vertices stay the same */
        ++numLeavesNew;
      } else if ((p >= eStart) && (p < eMax)) {
        /* Interior edges add new edges and vertex */
        numLeavesNew += 2 + 1;
      } else if ((p >= eMax) && (p < eEnd)) {
        /* Hybrid edges stay the same */
        ++numLeavesNew;
      } else if ((p >= fStart) && (p < fMax)) {
        /* Interior faces add new faces and edges */
        numLeavesNew += 4 + 3;
      } else if ((p >= fMax) && (p < fEnd)) {
        /* Hybrid faces add new faces and edges */
        numLeavesNew += 2 + 1;
      } else if ((p >= cStart) && (p < cMax)) {
        /* Interior cells add new cells, faces, and edges */
        numLeavesNew += 8 + 8 + 1;
      } else if ((p >= cMax) && (p < cEnd)) {
        /* Hybrid cells add new cells and faces */
        numLeavesNew += 4 + 3;
      }
      break;
    case 6:
      /* Hex 3D */
      if ((p >= vStart) && (p < vEnd)) {
        /* Old vertices stay the same */
        ++numLeavesNew;
      } else if ((p >= eStart) && (p < eEnd)) {
        /* Old edges add new edges, and vertex */
        numLeavesNew += 2 + 1;
      } else if ((p >= fStart) && (p < fEnd)) {
        /* Old faces add new faces, edges, and vertex */
        numLeavesNew += 4 + 4 + 1;
      } else if ((p >= cStart) && (p < cEnd)) {
        /* Old cells add new cells, faces, edges, and vertex */
        numLeavesNew += 8 + 12 + 6 + 1;
      }
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell refiner %d", refiner);
    }
  }
  /* Communicate depthSizes for each remote rank */
  ierr = DMPlexCreateProcessSF(dm, sf, &processRanks, &sfProcess);CHKERRQ(ierr);
  ierr = ISGetLocalSize(processRanks, &numNeighbors);CHKERRQ(ierr);
  ierr = PetscMalloc5((depth+1)*numNeighbors,PetscInt,&rdepthSize,numNeighbors,PetscInt,&rvStartNew,numNeighbors,PetscInt,&reStartNew,numNeighbors,PetscInt,&rfStartNew,numNeighbors,PetscInt,&rcStartNew);CHKERRQ(ierr);
  ierr = PetscMalloc7(depth+1,PetscInt,&depthSizeOld,(depth+1)*numNeighbors,PetscInt,&rdepthSizeOld,(depth+1)*numNeighbors,PetscInt,&rdepthMaxOld,numNeighbors,PetscInt,&rvStart,numNeighbors,PetscInt,&reStart,numNeighbors,PetscInt,&rfStart,numNeighbors,PetscInt,&rcStart);CHKERRQ(ierr);
  ierr = MPI_Type_contiguous(depth+1, MPIU_INT, &depthType);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&depthType);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(sfProcess, depthType, depthSize, rdepthSize);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sfProcess, depthType, depthSize, rdepthSize);CHKERRQ(ierr);
  for (n = 0; n < numNeighbors; ++n) {
    ierr = GetDepthStart_Private(depth, &rdepthSize[n*(depth+1)], &rcStartNew[n], &rfStartNew[n], &reStartNew[n], &rvStartNew[n]);CHKERRQ(ierr);
  }
  depthSizeOld[depth]   = cMax;
  depthSizeOld[0]       = vMax;
  depthSizeOld[depth-1] = fMax;
  depthSizeOld[1]       = eMax;

  ierr = PetscSFBcastBegin(sfProcess, depthType, depthSizeOld, rdepthMaxOld);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sfProcess, depthType, depthSizeOld, rdepthMaxOld);CHKERRQ(ierr);

  depthSizeOld[depth]   = cEnd - cStart;
  depthSizeOld[0]       = vEnd - vStart;
  depthSizeOld[depth-1] = fEnd - fStart;
  depthSizeOld[1]       = eEnd - eStart;

  ierr = PetscSFBcastBegin(sfProcess, depthType, depthSizeOld, rdepthSizeOld);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(sfProcess, depthType, depthSizeOld, rdepthSizeOld);CHKERRQ(ierr);
  for (n = 0; n < numNeighbors; ++n) {
    ierr = GetDepthStart_Private(depth, &rdepthSizeOld[n*(depth+1)], &rcStart[n], &rfStart[n], &reStart[n], &rvStart[n]);CHKERRQ(ierr);
  }
  ierr = MPI_Type_free(&depthType);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sfProcess);CHKERRQ(ierr);
  /* Calculate new point SF */
  ierr = PetscMalloc(numLeavesNew * sizeof(PetscInt),    &localPointsNew);CHKERRQ(ierr);
  ierr = PetscMalloc(numLeavesNew * sizeof(PetscSFNode), &remotePointsNew);CHKERRQ(ierr);
  ierr = ISGetIndices(processRanks, &neighbors);CHKERRQ(ierr);
  for (l = 0, m = 0; l < numLeaves; ++l) {
    PetscInt    p     = localPoints[l];
    PetscInt    rp    = remotePoints[l].index, n;
    PetscMPIInt rrank = remotePoints[l].rank;

    ierr = PetscFindInt(rrank, numNeighbors, neighbors, &n);CHKERRQ(ierr);
    if (n < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Could not locate remote rank %d", rrank);
    switch (refiner) {
    case 1:
      /* Simplicial 2D */
      if ((p >= vStart) && (p < vEnd)) {
        /* Old vertices stay the same */
        localPointsNew[m]        = vStartNew     + (p  - vStart);
        remotePointsNew[m].index = rvStartNew[n] + (rp - rvStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= fStart) && (p < fEnd)) {
        /* Old faces add new faces and vertex */
        localPointsNew[m]        = vStartNew     + (vEnd - vStart)              + (p  - fStart);
        remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+0] + (rp - rfStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
        for (r = 0; r < 2; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (p  - fStart)*2     + r;
          remotePointsNew[m].index = rfStartNew[n] + (rp - rfStart[n])*2 + r;
          remotePointsNew[m].rank  = rrank;
        }
      } else if ((p >= cStart) && (p < cEnd)) {
        /* Old cells add new cells and interior faces */
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (p  - cStart)*4     + r;
          remotePointsNew[m].index = rcStartNew[n] + (rp - rcStart[n])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 3; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fEnd - fStart)*2                    + (p  - cStart)*3     + r;
          remotePointsNew[m].index = rfStartNew[n] + rdepthSizeOld[n*(depth+1)+depth-1]*2 + (rp - rcStart[n])*3 + r;
          remotePointsNew[m].rank  = rrank;
        }
      }
      break;
    case 2:
      /* Hex 2D */
      if ((p >= vStart) && (p < vEnd)) {
        /* Old vertices stay the same */
        localPointsNew[m]        = vStartNew     + (p  - vStart);
        remotePointsNew[m].index = rvStartNew[n] + (rp - rvStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= fStart) && (p < fEnd)) {
        /* Old faces add new faces and vertex */
        localPointsNew[m]        = vStartNew     + (vEnd - vStart)              + (p  - fStart);
        remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+0] + (rp - rfStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
        for (r = 0; r < 2; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (p  - fStart)*2     + r;
          remotePointsNew[m].index = rfStartNew[n] + (rp - rfStart[n])*2 + r;
          remotePointsNew[m].rank  = rrank;
        }
      } else if ((p >= cStart) && (p < cEnd)) {
        /* Old cells add new cells, interior faces, and vertex */
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (p  - cStart)*4     + r;
          remotePointsNew[m].index = rcStartNew[n] + (rp - rcStart[n])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fEnd - fStart)*2                    + (p  - cStart)*4     + r;
          remotePointsNew[m].index = rfStartNew[n] + rdepthSizeOld[n*(depth+1)+depth-1]*2 + (rp - rcStart[n])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 1; ++r, ++m) {
          localPointsNew[m]        = vStartNew     + (fEnd - fStart)                    + (p  - cStart)     + r;
          remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+depth-1] + (rp - rcStart[n]) + r;
          remotePointsNew[m].rank  = rrank;
        }
      }
      break;
    case 3:
      /* Hybrid simplicial 2D */
      if ((p >= vStart) && (p < vEnd)) {
        /* Old vertices stay the same */
        localPointsNew[m]        = vStartNew     + (p  - vStart);
        remotePointsNew[m].index = rvStartNew[n] + (rp - rvStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= fStart) && (p < fMax)) {
        /* Old interior faces add new faces and vertex */
        localPointsNew[m]        = vStartNew     + (vEnd - vStart)              + (p  - fStart);
        remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+0] + (rp - rfStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
        for (r = 0; r < 2; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (p  - fStart)*2     + r;
          remotePointsNew[m].index = rfStartNew[n] + (rp - rfStart[n])*2 + r;
          remotePointsNew[m].rank  = rrank;
        }
      } else if ((p >= fMax) && (p < fEnd)) {
        /* Old hybrid faces stay the same */
        localPointsNew[m]        = fStartNew     + (fMax                              - fStart)*2     + (p  - fMax);
        remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*2 + (rp - rdepthMaxOld[n*(depth+1)+depth-1]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= cStart) && (p < cMax)) {
        /* Old interior cells add new cells and interior faces */
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (p  - cStart)*4     + r;
          remotePointsNew[m].index = rcStartNew[n] + (rp - rcStart[n])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 3; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fMax                              - fStart)*2     + (p  - cStart)*3     + r;
          remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*2 + (rp - rcStart[n])*3 + r;
          remotePointsNew[m].rank  = rrank;
        }
      } else if ((p >= cStart) && (p < cMax)) {
        /* Old hybrid cells add new cells and hybrid face */
        for (r = 0; r < 2; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (p  - cStart)*4     + r;
          remotePointsNew[m].index = rcStartNew[n] + (rp - rcStart[n])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = fStartNew     + (fMax                              - fStart)*2     + (cMax                            - cStart)*3     + (p  - cMax);
        remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*2 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*3 + (rp - rdepthMaxOld[n*(depth+1)+depth]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      }
      break;
    case 5:
      /* Simplicial 3D */
      if ((p >= vStart) && (p < vEnd)) {
        /* Old vertices stay the same */
        localPointsNew[m]        = vStartNew     + (p  - vStart);
        remotePointsNew[m].index = rvStartNew[n] + (rp - rvStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= eStart) && (p < eEnd)) {
        /* Old edges add new edges and vertex */
        for (r = 0; r < 2; ++r, ++m) {
          localPointsNew[m]        = eStartNew     + (p  - eStart)*2     + r;
          remotePointsNew[m].index = reStartNew[n] + (rp - reStart[n])*2 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = vStartNew     + (vEnd - vStart)              + (p  - eStart);
        remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+0] + (rp - reStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= fStart) && (p < fEnd)) {
        /* Old faces add new faces and face edges */
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (p  - fStart)*4     + r;
          remotePointsNew[m].index = rfStartNew[n] + (rp - rfStart[n])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 3; ++r, ++m) {
          localPointsNew[m]        = eStartNew     + (eEnd - eStart)*2              + (p  - fStart)*3     + r;
          remotePointsNew[m].index = reStartNew[n] + rdepthSizeOld[n*(depth+1)+1]*2 + (rp - rfStart[n])*3 + r;
          remotePointsNew[m].rank  = rrank;
        }
      } else if ((p >= cStart) && (p < cEnd)) {
        /* Old cells add new cells and interior faces and edges */
        for (r = 0; r < 8; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (p  - cStart)*8     + r;
          remotePointsNew[m].index = rcStartNew[n] + (rp - rcStart[n])*8 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 8; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fEnd - fStart)*4                    + (p  - cStart)*8     + r;
          remotePointsNew[m].index = rfStartNew[n] + rdepthSizeOld[n*(depth+1)+depth-1]*4 + (rp - rcStart[n])*8 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 1; ++r, ++m) {
          localPointsNew[m]        = eStartNew     + (eEnd - eStart)*2              + (fEnd - fStart)*3                    + (p  - cStart)*1     + r;
          remotePointsNew[m].index = reStartNew[n] + rdepthSizeOld[n*(depth+1)+1]*2 + rdepthSizeOld[n*(depth+1)+depth-1]*3 + (rp - rcStart[n])*1 + r;
          remotePointsNew[m].rank  = rrank;
        }
      }
      break;
    case 7:
      /* Hybrid Simplicial 3D */
      if ((p >= vStart) && (p < vEnd)) {
        /* Interior vertices stay the same */
        localPointsNew[m]        = vStartNew     + (p  - vStart);
        remotePointsNew[m].index = rvStartNew[n] + (rp - rvStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= eStart) && (p < eMax)) {
        /* Interior edges add new edges and vertex */
        for (r = 0; r < 2; ++r, ++m) {
          localPointsNew[m]        = eStartNew     + (p  - eStart)*2     + r;
          remotePointsNew[m].index = reStartNew[n] + (rp - reStart[n])*2 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = vStartNew     + (vEnd - vStart)              + (p  - eStart);
        remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+0] + (rp - reStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= eMax) && (p < eEnd)) {
        /* Hybrid edges stay the same */
        localPointsNew[m]        = eStartNew     + (eMax                        - eStart)*2     + (fMax                              - fStart)*3     + (cMax                            - cStart)     + (p  - eMax);
        remotePointsNew[m].index = rvStartNew[n] + (rdepthMaxOld[n*(depth+1)+1] - reStart[n])*2 + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*3 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n]) + (rp - rdepthMaxOld[n*(depth+1)+1]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= fStart) && (p < fMax)) {
        /* Interior faces add new faces and edges */
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (p  - fStart)*4     + r;
          remotePointsNew[m].index = rfStartNew[n] + (rp - rfStart[n])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 3; ++r, ++m) {
          localPointsNew[m]        = eStartNew     + (eMax                        - eStart)*2     + (p  - fStart)*3     + r;
          remotePointsNew[m].index = reStartNew[n] + (rdepthMaxOld[n*(depth+1)+1] - reStart[n])*2 + (rp - rfStart[n])*3 + r;
          remotePointsNew[m].rank  = rrank;
        }
      } else if ((p >= fMax) && (p < fEnd)) {
        /* Hybrid faces add new faces and edges */
        for (r = 0; r < 2; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fMax                              - fStart)*4     + (cMax                            - cStart)*8     + (p  - fStart)*2     + r;
          remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*4 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*8 + (rp - rfStart[n])*2 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = eStartNew     + (eMax                        - eStart)*2     + (fMax                              - fStart)     + (cMax                            - cStart)     + (fEnd                                          - fMax);
        remotePointsNew[m].index = reStartNew[n] + (rdepthMaxOld[n*(depth+1)+1] - reStart[n])*2 + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n]) + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n]) + (rdepthSizeOld[n*(depth+1)+depth-1]+rfStart[n] - rdepthMaxOld[n*(depth+1)+depth-1]);
        remotePointsNew[m].rank  = rrank;
      } else if ((p >= cStart) && (p < cMax)) {
        /* Interior cells add new cells, faces, and edges */
        for (r = 0; r < 8; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (p  - cStart)*8     + r;
          remotePointsNew[m].index = rcStartNew[n] + (rp - rcStart[n])*8 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 8; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fMax                              - fStart)*4     + (p  - cStart)*8     + r;
          remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*4 + (rp - rcStart[n])*8 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = eStartNew     + (eMax                        - eStart)*2     + (fMax                              - fStart)*3     + (p  - cStart)*1     + r;
        remotePointsNew[m].index = reStartNew[n] + (rdepthMaxOld[n*(depth+1)+1] - reStart[n])*2 + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*3 + (rp - rcStart[n])*1 + r;
        remotePointsNew[m].rank  = rrank;
      } else if ((p >= cMax) && (p < cEnd)) {
        /* Hybrid cells add new cells and faces */
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (cMax                            - cStart)*8     + (p  - cMax)*4                            + r;
          remotePointsNew[m].index = rcStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*8 + (rp - rdepthMaxOld[n*(depth+1)+depth])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 3; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fMax                              - fStart)*4     + (cMax                            - cStart)*8     + (fEnd                                          - fMax)*4                              + (p  - cMax)*3                            + r;
          remotePointsNew[m].index = rfStartNew[n] + (rdepthMaxOld[n*(depth+1)+depth-1] - rfStart[n])*4 + (rdepthMaxOld[n*(depth+1)+depth] - rcStart[n])*8 + (rdepthSizeOld[n*(depth+1)+depth-1]+rfStart[n] - rdepthMaxOld[n*(depth+1)+depth-1])*4 + (rp - rdepthMaxOld[n*(depth+1)+depth])*3 + r;
          remotePointsNew[m].rank  = rrank;
        }
      }
      break;
    case 6:
      /* Hex 3D */
      if ((p >= vStart) && (p < vEnd)) {
        /* Old vertices stay the same */
        localPointsNew[m]        = vStartNew     + (p  - vStart);
        remotePointsNew[m].index = rvStartNew[n] + (rp - rvStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= eStart) && (p < eEnd)) {
        /* Old edges add new edges and vertex */
        for (r = 0; r < 2; ++r, ++m) {
          localPointsNew[m]        = eStartNew     + (p  - eStart)*2     + r;
          remotePointsNew[m].index = reStartNew[n] + (rp - reStart[n])*2 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = vStartNew     + (vEnd - vStart)              + (p  - eStart);
        remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+0] + (rp - reStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= fStart) && (p < fEnd)) {
        /* Old faces add new faces, edges, and vertex */
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (p  - fStart)*4     + r;
          remotePointsNew[m].index = rfStartNew[n] + (rp - rfStart[n])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 4; ++r, ++m) {
          localPointsNew[m]        = eStartNew     + (eEnd - eStart)*2              + (p  - fStart)*4     + r;
          remotePointsNew[m].index = reStartNew[n] + rdepthSizeOld[n*(depth+1)+1]*2 + (rp - rfStart[n])*4 + r;
          remotePointsNew[m].rank  = rrank;
        }
        localPointsNew[m]        = vStartNew     + (vEnd - vStart)              + (eEnd - eStart)              + (p  - fStart);
        remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+0] + rdepthSizeOld[n*(depth+1)+1] + (rp - rfStart[n]);
        remotePointsNew[m].rank  = rrank;
        ++m;
      } else if ((p >= cStart) && (p < cEnd)) {
        /* Old cells add new cells, faces, edges, and vertex */
        for (r = 0; r < 8; ++r, ++m) {
          localPointsNew[m]        = cStartNew     + (p  - cStart)*8     + r;
          remotePointsNew[m].index = rcStartNew[n] + (rp - rcStart[n])*8 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 12; ++r, ++m) {
          localPointsNew[m]        = fStartNew     + (fEnd - fStart)*4                    + (p  - cStart)*12     + r;
          remotePointsNew[m].index = rfStartNew[n] + rdepthSizeOld[n*(depth+1)+depth-1]*4 + (rp - rcStart[n])*12 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 6; ++r, ++m) {
          localPointsNew[m]        = eStartNew     + (eEnd - eStart)*2              + (fEnd - fStart)*4                    + (p  - cStart)*6     + r;
          remotePointsNew[m].index = reStartNew[n] + rdepthSizeOld[n*(depth+1)+1]*2 + rdepthSizeOld[n*(depth+1)+depth-1]*4 + (rp - rcStart[n])*6 + r;
          remotePointsNew[m].rank  = rrank;
        }
        for (r = 0; r < 1; ++r, ++m) {
          localPointsNew[m]        = vStartNew     + (eEnd - eStart)              + (fEnd - fStart)                    + (p  - cStart)     + r;
          remotePointsNew[m].index = rvStartNew[n] + rdepthSizeOld[n*(depth+1)+1] + rdepthSizeOld[n*(depth+1)+depth-1] + (rp - rcStart[n]) + r;
          remotePointsNew[m].rank  = rrank;
        }
      }
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell refiner %d", refiner);
    }
  }
  ierr = ISRestoreIndices(processRanks, &neighbors);CHKERRQ(ierr);
  ierr = ISDestroy(&processRanks);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sfNew, pEndNew-pStartNew, numLeavesNew, localPointsNew, PETSC_OWN_POINTER, remotePointsNew, PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscFree5(rdepthSize,rvStartNew,reStartNew,rfStartNew,rcStartNew);CHKERRQ(ierr);
  ierr = PetscFree7(depthSizeOld,rdepthSizeOld,rdepthMaxOld,rvStart,reStart,rfStart,rcStart);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CellRefinerCreateLabels"
PetscErrorCode CellRefinerCreateLabels(CellRefiner refiner, DM dm, PetscInt depthSize[], DM rdm)
{
  PetscInt       numLabels, l;
  PetscInt       depth, newp, cStart, cStartNew, cEnd, cMax, vStart, vStartNew, vEnd, vMax, fStart, fStartNew, fEnd, fMax, eStart, eStartNew, eEnd, eMax, r;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  if (refiner) {ierr = GetDepthStart_Private(depth, depthSize, &cStartNew, &fStartNew, &eStartNew, &vStartNew);CHKERRQ(ierr);}
  ierr = DMPlexGetNumLabels(dm, &numLabels);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, &fMax, &eMax, &vMax);CHKERRQ(ierr);
  switch (refiner) {
  case 0: break;
  case 7:
  case 8:
    if (eMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No edge maximum specified in hybrid mesh");
  case 3:
  case 4:
    if (cMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No cell maximum specified in hybrid mesh");
    if (fMax < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No face maximum specified in hybrid mesh");
  }
  for (l = 0; l < numLabels; ++l) {
    DMLabel         label, labelNew;
    const char     *lname;
    PetscBool       isDepth;
    IS              valueIS;
    const PetscInt *values;
    PetscInt        numValues, val;

    ierr = DMPlexGetLabelName(dm, l, &lname);CHKERRQ(ierr);
    ierr = PetscStrcmp(lname, "depth", &isDepth);CHKERRQ(ierr);
    if (isDepth) continue;
    ierr = DMPlexCreateLabel(rdm, lname);CHKERRQ(ierr);
    ierr = DMPlexGetLabel(dm, lname, &label);CHKERRQ(ierr);
    ierr = DMPlexGetLabel(rdm, lname, &labelNew);CHKERRQ(ierr);
    ierr = DMLabelGetValueIS(label, &valueIS);CHKERRQ(ierr);
    ierr = ISGetLocalSize(valueIS, &numValues);CHKERRQ(ierr);
    ierr = ISGetIndices(valueIS, &values);CHKERRQ(ierr);
    for (val = 0; val < numValues; ++val) {
      IS              pointIS;
      const PetscInt *points;
      PetscInt        numPoints, n;

      ierr = DMLabelGetStratumIS(label, values[val], &pointIS);CHKERRQ(ierr);
      ierr = ISGetLocalSize(pointIS, &numPoints);CHKERRQ(ierr);
      ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
      for (n = 0; n < numPoints; ++n) {
        const PetscInt p = points[n];
        switch (refiner) {
        case 1:
          /* Simplicial 2D */
          if ((p >= vStart) && (p < vEnd)) {
            /* Old vertices stay the same */
            newp = vStartNew + (p - vStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= fStart) && (p < fEnd)) {
            /* Old faces add new faces and vertex */
            newp = vStartNew + (vEnd - vStart) + (p - fStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            for (r = 0; r < 2; ++r) {
              newp = fStartNew + (p - fStart)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          } else if ((p >= cStart) && (p < cEnd)) {
            /* Old cells add new cells and interior faces */
            for (r = 0; r < 4; ++r) {
              newp = cStartNew + (p - cStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 3; ++r) {
              newp = fStartNew + (fEnd - fStart)*2 + (p - cStart)*3 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          }
          break;
        case 2:
          /* Hex 2D */
          if ((p >= vStart) && (p < vEnd)) {
            /* Old vertices stay the same */
            newp = vStartNew + (p - vStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= fStart) && (p < fEnd)) {
            /* Old faces add new faces and vertex */
            newp = vStartNew + (vEnd - vStart) + (p - fStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            for (r = 0; r < 2; ++r) {
              newp = fStartNew + (p - fStart)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          } else if ((p >= cStart) && (p < cEnd)) {
            /* Old cells add new cells and interior faces and vertex */
            for (r = 0; r < 4; ++r) {
              newp = cStartNew + (p - cStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 4; ++r) {
              newp = fStartNew + (fEnd - fStart)*2 + (p - cStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = vStartNew + (vEnd - vStart) + (fEnd - fStart) + (p - cStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          }
          break;
        case 3:
          /* Hybrid simplicial 2D */
          if ((p >= vStart) && (p < vEnd)) {
            /* Old vertices stay the same */
            newp = vStartNew + (p - vStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= fStart) && (p < fMax)) {
            /* Old interior faces add new faces and vertex */
            newp = vStartNew + (vEnd - vStart) + (p - fStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            for (r = 0; r < 2; ++r) {
              newp = fStartNew + (p - fStart)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          } else if ((p >= fMax) && (p < fEnd)) {
            /* Old hybrid faces stay the same */
            newp = fStartNew + (fMax - fStart)*2 + (p - fMax);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= cStart) && (p < cMax)) {
            /* Old interior cells add new cells and interior faces */
            for (r = 0; r < 4; ++r) {
              newp = cStartNew + (p - cStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 3; ++r) {
              newp = fStartNew + (fEnd - fStart)*2 + (p - cStart)*3 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          } else if ((p >= cMax) && (p < cEnd)) {
            /* Old hybrid cells add new cells and hybrid face */
            for (r = 0; r < 2; ++r) {
              newp = cStartNew + (cMax - cStart)*4 + (p - cMax)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = fStartNew + (fMax - fStart)*2 + (cMax - cStart)*3 + (p - cMax);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          }
          break;
        case 5:
          /* Simplicial 3D */
          if ((p >= vStart) && (p < vEnd)) {
            /* Old vertices stay the same */
            newp = vStartNew + (p - vStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= eStart) && (p < eEnd)) {
            /* Old edges add new edges and vertex */
            for (r = 0; r < 2; ++r) {
              newp = eStartNew + (p - eStart)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = vStartNew + (vEnd - vStart) + (p - eStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= fStart) && (p < fEnd)) {
            /* Old faces add new faces and edges */
            for (r = 0; r < 4; ++r) {
              newp = fStartNew + (p - fStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 3; ++r) {
              newp = eStartNew + (eEnd - eStart)*2 + (p - fStart)*3 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          } else if ((p >= cStart) && (p < cEnd)) {
            /* Old cells add new cells and interior faces and edges */
            for (r = 0; r < 8; ++r) {
              newp = cStartNew + (p - cStart)*8 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 8; ++r) {
              newp = fStartNew + (fEnd - fStart)*4 + (p - cStart)*8 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 1; ++r) {
              newp = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*3 + (p - cStart)*1 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          }
          break;
        case 7:
          /* Hybrid Simplicial 3D */
          if ((p >= vStart) && (p < vEnd)) {
            /* Interior vertices stay the same */
            newp = vStartNew + (p - vStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= eStart) && (p < eMax)) {
            /* Interior edges add new edges and vertex */
            for (r = 0; r < 2; ++r) {
              newp = eStartNew + (p - eStart)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = vStartNew + (vEnd - vStart) + (p - eStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= eMax) && (p < eEnd)) {
            /* Hybrid edges stay the same */
            newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (p - eMax);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= fStart) && (p < fMax)) {
            /* Interior faces add new faces and edges */
            for (r = 0; r < 4; ++r) {
              newp = fStartNew + (p - fStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 3; ++r) {
              newp = eStartNew + (eMax - eStart)*2 + (p - fStart)*3 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          } else if ((p >= fMax) && (p < fEnd)) {
            /* Hybrid faces add new faces and edges */
            for (r = 0; r < 2; ++r) {
              newp = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (p - fMax)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (cMax - cStart) + (p - fMax);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= cStart) && (p < cMax)) {
            /* Interior cells add new cells, faces, and edges */
            for (r = 0; r < 8; ++r) {
              newp = cStartNew + (p - cStart)*8 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 8; ++r) {
              newp = fStartNew + (fMax - fStart)*4 + (p - cStart)*8 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = eStartNew + (eMax - eStart)*2 + (fMax - fStart)*3 + (p - cStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= cMax) && (p < cEnd)) {
            /* Hybrid cells add new cells and faces */
            for (r = 0; r < 4; ++r) {
              newp = cStartNew + (cMax - cStart)*8 + (p - cMax)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 3; ++r) {
              newp = fStartNew + (fMax - fStart)*4 + (cMax - cStart)*8 + (fEnd - fMax)*2 + (p - cMax)*3 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
          }
          break;
        case 6:
          /* Hex 3D */
          if ((p >= vStart) && (p < vEnd)) {
            /* Old vertices stay the same */
            newp = vStartNew + (p - vStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= eStart) && (p < eEnd)) {
            /* Old edges add new edges and vertex */
            for (r = 0; r < 2; ++r) {
              newp = eStartNew + (p - eStart)*2 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = vStartNew + (vEnd - vStart) + (p - eStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= fStart) && (p < fEnd)) {
            /* Old faces add new faces, edges, and vertex */
            for (r = 0; r < 4; ++r) {
              newp = fStartNew + (p - fStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 4; ++r) {
              newp = eStartNew + (eEnd - eStart)*2 + (p - fStart)*4 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (p - fStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          } else if ((p >= cStart) && (p < cEnd)) {
            /* Old cells add new cells, faces, edges, and vertex */
            for (r = 0; r < 8; ++r) {
              newp = cStartNew + (p - cStart)*8 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 12; ++r) {
              newp = fStartNew + (fEnd - fStart)*4 + (p - cStart)*12 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            for (r = 0; r < 6; ++r) {
              newp = eStartNew + (eEnd - eStart)*2 + (fEnd - fStart)*4 + (p - cStart)*6 + r;
              ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
            }
            newp = vStartNew + (vEnd - vStart) + (eEnd - eStart) + (fEnd - fStart) + (p - cStart);
            ierr = DMLabelSetValue(labelNew, newp, values[val]);CHKERRQ(ierr);
          }
          break;
        default:
          SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown cell refiner %d", refiner);
        }
      }
      ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(valueIS, &values);CHKERRQ(ierr);
    ierr = ISDestroy(&valueIS);CHKERRQ(ierr);
    if (0) {
      ierr = PetscViewerASCIISynchronizedAllow(PETSC_VIEWER_STDOUT_WORLD, PETSC_TRUE);CHKERRQ(ierr);
      ierr = DMLabelView(labelNew, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexRefineUniform_Internal"
/* This will only work for interpolated meshes */
PetscErrorCode DMPlexRefineUniform_Internal(DM dm, CellRefiner cellRefiner, DM *dmRefined)
{
  DM             rdm;
  PetscInt      *depthSize;
  PetscInt       dim, depth = 0, d, pStart = 0, pEnd = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(PetscObjectComm((PetscObject)dm), &rdm);CHKERRQ(ierr);
  ierr = DMSetType(rdm, DMPLEX);CHKERRQ(ierr);
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexSetDimension(rdm, dim);CHKERRQ(ierr);
  /* Calculate number of new points of each depth */
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = PetscMalloc((depth+1) * sizeof(PetscInt), &depthSize);CHKERRQ(ierr);
  ierr = PetscMemzero(depthSize, (depth+1) * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = CellRefinerGetSizes(cellRefiner, dm, depthSize);CHKERRQ(ierr);
  /* Step 1: Set chart */
  for (d = 0; d <= depth; ++d) pEnd += depthSize[d];
  ierr = DMPlexSetChart(rdm, pStart, pEnd);CHKERRQ(ierr);
  /* Step 2: Set cone/support sizes */
  ierr = CellRefinerSetConeSizes(cellRefiner, dm, depthSize, rdm);CHKERRQ(ierr);
  /* Step 3: Setup refined DM */
  ierr = DMSetUp(rdm);CHKERRQ(ierr);
  /* Step 4: Set cones and supports */
  ierr = CellRefinerSetCones(cellRefiner, dm, depthSize, rdm);CHKERRQ(ierr);
  /* Step 5: Stratify */
  ierr = DMPlexStratify(rdm);CHKERRQ(ierr);
  /* Step 6: Set coordinates for vertices */
  ierr = CellRefinerSetCoordinates(cellRefiner, dm, depthSize, rdm);CHKERRQ(ierr);
  /* Step 7: Create pointSF */
  ierr = CellRefinerCreateSF(cellRefiner, dm, depthSize, rdm);CHKERRQ(ierr);
  /* Step 8: Create labels */
  ierr = CellRefinerCreateLabels(cellRefiner, dm, depthSize, rdm);CHKERRQ(ierr);
  ierr = PetscFree(depthSize);CHKERRQ(ierr);

  *dmRefined = rdm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetRefinementUniform"
PetscErrorCode DMPlexSetRefinementUniform(DM dm, PetscBool refinementUniform)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->refinementUniform = refinementUniform;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetRefinementUniform"
PetscErrorCode DMPlexGetRefinementUniform(DM dm, PetscBool *refinementUniform)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(refinementUniform,  2);
  *refinementUniform = mesh->refinementUniform;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexSetRefinementLimit"
PetscErrorCode DMPlexSetRefinementLimit(DM dm, PetscReal refinementLimit)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->refinementLimit = refinementLimit;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetRefinementLimit"
PetscErrorCode DMPlexGetRefinementLimit(DM dm, PetscReal *refinementLimit)
{
  DM_Plex *mesh = (DM_Plex*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(refinementLimit,  2);
  /* if (mesh->refinementLimit < 0) = getMaxVolume()/2.0; */
  *refinementLimit = mesh->refinementLimit;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPlexGetCellRefiner_Internal"
PetscErrorCode DMPlexGetCellRefiner_Internal(DM dm, CellRefiner *cellRefiner)
{
  PetscInt       dim, cStart, cEnd, coneSize, cMax;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  if (cEnd <= cStart) {*cellRefiner = 0; PetscFunctionReturn(0);}
  ierr = DMPlexGetConeSize(dm, cStart, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cMax, NULL, NULL, NULL);CHKERRQ(ierr);
  switch (dim) {
  case 2:
    switch (coneSize) {
    case 3:
      if (cMax >= 0) *cellRefiner = 3; /* Hybrid */
      else *cellRefiner = 1; /* Triangular */
      break;
    case 4:
      if (cMax >= 0) *cellRefiner = 4; /* Hybrid */
      else *cellRefiner = 2; /* Quadrilateral */
      break;
    default:
      SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown coneSize %d in dimension %d for cell refiner", coneSize, dim);
    }
    break;
  case 3:
    switch (coneSize) {
    case 4:
      if (cMax >= 0) *cellRefiner = 7; /* Hybrid */
      else *cellRefiner = 5; /* Tetrahedral */
      break;
    case 6:
      if (cMax >= 0) *cellRefiner = 8; /* Hybrid */
      else *cellRefiner = 6; /* hexahedral */
      break;
    default:
      SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown coneSize %d in dimension %d for cell refiner", coneSize, dim);
    }
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown dimension %d for cell refiner", dim);
  }
  PetscFunctionReturn(0);
}
