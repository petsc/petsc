#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/
#include <petscsf.h>

PetscBool  SBRcite       = PETSC_FALSE;
const char SBRCitation[] = "@article{PlazaCarey2000,\n"
                           "  title   = {Local refinement of simplicial grids based on the skeleton},\n"
                           "  journal = {Applied Numerical Mathematics},\n"
                           "  author  = {A. Plaza and Graham F. Carey},\n"
                           "  volume  = {32},\n"
                           "  number  = {3},\n"
                           "  pages   = {195--218},\n"
                           "  doi     = {10.1016/S0168-9274(99)00022-7},\n"
                           "  year    = {2000}\n}\n";

/*
  Tetrahedron combinatorics, in the canonical local numbering used throughout this file (matching
  the convention documented in DMPlexTransformCellRefine_Regular() for DM_POLYTOPE_TETRAHEDRON):

    e0 = (v0,v1), e1 = (v1,v2), e2 = (v2,v0), e3 = (v0,v3), e4 = (v1,v3), e5 = (v2,v3)
    f0 = (v0,v1,v2), f1 = (v0,v3,v1), f2 = (v0,v2,v3), f3 = (v2,v1,v3)

  tetEdgeVert[e]    - the 2 vertices of edge e, in its canonical direction
  tetFaceVert[f]    - the 3 vertices of face f, in its canonical order
  tetFaceEdge[f]    - the edges of face f, at local indices 0,1,2 (i.e. (v_0,v_1),(v_1,v_2),(v_2,v_0) of that face)
  tetEdgeFaceLoc[e] - a (face, local edge index) path that reaches edge e in its canonical direction
  tetVertPath[v]    - a (face, local edge index, local vertex index) path that reaches vertex v
  tetOppFace[x]     - the 3 local vertex positions (not values) of the face opposite position x, i.e. f[3-x]
*/
static const PetscInt tetEdgeVert[6][2] = {
  {0, 1},
  {1, 2},
  {2, 0},
  {0, 3},
  {1, 3},
  {2, 3}
};
static const PetscInt tetFaceVert[4][3] = {
  {0, 1, 2},
  {0, 3, 1},
  {0, 2, 3},
  {2, 1, 3}
};
static const PetscInt tetFaceEdge[4][3] = {
  {0, 1, 2},
  {3, 4, 0},
  {2, 5, 3},
  {1, 4, 5}
};
static const PetscInt tetEdgeFaceLoc[6][2] = {
  {0, 0},
  {0, 1},
  {0, 2},
  {1, 0},
  {3, 1},
  {2, 1}
};
static const PetscInt tetVertPath[4][3] = {
  {0, 0, 0},
  {0, 0, 1},
  {0, 2, 0},
  {1, 0, 1}
};
static const PetscInt tetOppFace[4][3] = {
  {2, 1, 3},
  {0, 2, 3},
  {0, 3, 1},
  {0, 1, 2}
};

/* Find the tetrahedron edge index (0-5) connecting the two given (unordered) vertices */
static PetscInt SBREdgeIndexFromVerts_Private(PetscInt v0, PetscInt v1)
{
  for (PetscInt k = 0; k < 6; ++k) {
    if ((tetEdgeVert[k][0] == v0 && tetEdgeVert[k][1] == v1) || (tetEdgeVert[k][0] == v1 && tetEdgeVert[k][1] == v0)) return k;
  }
  return -1;
}

/* Read off the 3 vertices of the tetrahedron face opposite local position missingPos (0-3), from a
   general 4-tuple T (which may hold original vertices 0-3 and/or edge-midpoint markers >= 4) */
static void SBRTetFaceOpposite_Private(const PetscInt T[4], PetscInt missingPos, PetscInt tri[3])
{
  for (PetscInt i = 0; i < 3; ++i) tri[i] = T[tetOppFace[missingPos][i]];
}

static PetscErrorCode SBRGetEdgeLen_Private(DMPlexTransform tr, PetscInt edge, PetscReal *len)
{
  DMPlexRefine_SBR *sbr = (DMPlexRefine_SBR *)tr->data;
  DM                dm;
  PetscInt          off;

  PetscFunctionBeginHot;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(PetscSectionGetOffset(sbr->secEdgeLen, edge, &off));
  if (sbr->edgeLen[off] <= 0.0) {
    DM                 cdm;
    Vec                coordsLocal;
    const PetscScalar *coords;
    const PetscInt    *cone;
    PetscScalar       *cA, *cB;
    PetscInt           coneSize, cdim;

    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMPlexGetCone(dm, edge, &cone));
    PetscCall(DMPlexGetConeSize(dm, edge, &coneSize));
    PetscCheck(coneSize == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Edge %" PetscInt_FMT " cone size must be 2, not %" PetscInt_FMT, edge, coneSize);
    PetscCall(DMGetCoordinateDim(dm, &cdim));
    PetscCall(DMGetCoordinatesLocalNoncollective(dm, &coordsLocal));
    PetscCall(VecGetArrayRead(coordsLocal, &coords));
    PetscCall(DMPlexPointLocalRead(cdm, cone[0], coords, &cA));
    PetscCall(DMPlexPointLocalRead(cdm, cone[1], coords, &cB));
    sbr->edgeLen[off] = DMPlex_DistD_Internal(cdim, cA, cB);
    PetscCall(VecRestoreArrayRead(coordsLocal, &coords));
  }
  *len = sbr->edgeLen[off];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Get the 6 edges of a tetrahedron, in the canonical local order used throughout this file:
  e0 = (v0,v1), e1 = (v1,v2), e2 = (v2,v0), e3 = (v0,v3), e4 = (v1,v3), e5 = (v2,v3), where
  v0..v3 are the tetrahedron's vertices in the order given by its own transitive closure. This
  matches the convention documented in DMPlexTransformCellRefine_Regular() for DM_POLYTOPE_TETRAHEDRON.
  The faces of the tetrahedron, in DMPlexGetCone() order, are f0 = (v0,v1,v2), f1 = (v0,v3,v1),
  f2 = (v0,v2,v3), f3 = (v2,v1,v3), so each e_k below is read off face fIdx[k] at local edge index
  lIdx[k], reoriented for that face's actual orientation as seen from the tetrahedron.
*/
static PetscErrorCode SBRGetTetEdges_Private(DM dm, PetscInt tet, PetscInt edges[6])
{
  const PetscInt *fcone, *forient;

  PetscFunctionBeginHot;
  PetscCall(DMPlexGetCone(dm, tet, &fcone));
  PetscCall(DMPlexGetConeOrientation(dm, tet, &forient));
  for (PetscInt k = 0; k < 6; ++k) {
    const PetscInt  floc = tetEdgeFaceLoc[k][0];
    const PetscInt  face = fcone[floc];
    const PetscInt *arr  = DMPolytopeTypeGetArrangement(DM_POLYTOPE_TRIANGLE, forient[floc]);
    const PetscInt  li   = arr[tetEdgeFaceLoc[k][1] * 2];
    const PetscInt *econe;

    PetscCall(DMPlexGetCone(dm, face, &econe));
    edges[k] = econe[li];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Find the longest of the 6 edges of a tetrahedron */
static PetscErrorCode SBRGetTetMaxEdge_Private(DMPlexTransform tr, PetscInt tet, PetscInt *maxedge)
{
  DM        dm;
  PetscInt  edges[6];
  PetscReal maxlen, len;

  PetscFunctionBeginHot;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(SBRGetTetEdges_Private(dm, tet, edges));
  PetscCall(SBRGetEdgeLen_Private(tr, edges[0], &maxlen));
  *maxedge = edges[0];
  for (PetscInt k = 1; k < 6; ++k) {
    PetscCall(SBRGetEdgeLen_Private(tr, edges[k], &len));
    if (len > maxlen) {
      maxlen   = len;
      *maxedge = edges[k];
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Mark local edges that should be split, ensuring conformity of the mesh skeleton.

  This implements the closure step of Plaza & Carey, Section 3.1, generalized from 2D to 3D:
  whenever an edge is marked, every non-conforming face containing it has its own longest edge
  marked (as in the original 2D algorithm), and every tetrahedron touching such a face has its
  own longest edge marked in turn (the "2.1/2.2" steps of the paper's 3D outline). In a 2D mesh,
  the tetrahedron support of a face is empty and this reduces exactly to the original algorithm.
*/
static PetscErrorCode SBRSplitLocalEdges_Private(DMPlexTransform tr, DMPlexPointQueue queue)
{
  DMPlexRefine_SBR *sbr = (DMPlexRefine_SBR *)tr->data;
  DM                dm;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  while (!DMPlexPointQueueEmpty(queue)) {
    PetscInt        p = -1;
    const PetscInt *support;
    PetscInt        supportSize;

    PetscCall(DMPlexPointQueueDequeue(queue, &p));
    PetscCall(DMPlexGetSupport(dm, p, &support));
    PetscCall(DMPlexGetSupportSize(dm, p, &supportSize));
    for (PetscInt s = 0; s < supportSize; ++s) {
      const PetscInt  cell = support[s];
      const PetscInt *cone;
      const PetscInt *tsupport;
      PetscInt        coneSize, tsupportSize, c;
      PetscInt        cval, eval, maxedge;
      PetscReal       len, maxlen;

      PetscCall(DMLabelGetValue(sbr->splitPoints, cell, &cval));
      if (cval == 2) continue;
      PetscCall(DMPlexGetCone(dm, cell, &cone));
      PetscCall(DMPlexGetConeSize(dm, cell, &coneSize));
      PetscCall(SBRGetEdgeLen_Private(tr, cone[0], &maxlen));
      maxedge = cone[0];
      for (c = 1; c < coneSize; ++c) {
        PetscCall(SBRGetEdgeLen_Private(tr, cone[c], &len));
        if (len > maxlen) {
          maxlen  = len;
          maxedge = cone[c];
        }
      }
      PetscCall(DMLabelGetValue(sbr->splitPoints, maxedge, &eval));
      if (eval != 1) {
        PetscCall(DMLabelSetValue(sbr->splitPoints, maxedge, 1));
        PetscCall(DMPlexPointQueueEnqueue(queue, maxedge));
      }
      PetscCall(DMLabelSetValue(sbr->splitPoints, cell, 2));
      /* Propagate to the tetrahedra above this face, if any (empty in a 2D mesh) */
      PetscCall(DMPlexGetSupport(dm, cell, &tsupport));
      PetscCall(DMPlexGetSupportSize(dm, cell, &tsupportSize));
      for (PetscInt ts = 0; ts < tsupportSize; ++ts) {
        const PetscInt tet = tsupport[ts];
        PetscInt       tval, tmaxedge = -1, teval;

        PetscCall(DMLabelGetValue(sbr->splitPoints, tet, &tval));
        if (tval == 3) continue;
        PetscCall(SBRGetTetMaxEdge_Private(tr, tet, &tmaxedge));
        PetscCall(DMLabelGetValue(sbr->splitPoints, tmaxedge, &teval));
        if (teval != 1) {
          PetscCall(DMLabelSetValue(sbr->splitPoints, tmaxedge, 1));
          PetscCall(DMPlexPointQueueEnqueue(queue, tmaxedge));
        }
        PetscCall(DMLabelSetValue(sbr->splitPoints, tet, 3));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode splitPoint(PETSC_UNUSED DMLabel label, PetscInt p, PETSC_UNUSED PetscInt val, PetscCtx ctx)
{
  DMPlexPointQueue queue = (DMPlexPointQueue)ctx;

  PetscFunctionBegin;
  PetscCall(DMPlexPointQueueEnqueue(queue, p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  The 'splitPoints' label marks mesh points to be divided. It marks edges with 1, triangles with 2, and tetrahedra with 3.
  Then the refinement type is calculated as follows:

    vertex:                   0
    edge unsplit:             1
    edge split:               2
    triangle unsplit:         3
    triangle split all edges: 4
    triangle split edges 0 1: 5
    triangle split edges 1 0: 6
    triangle split edges 1 2: 7
    triangle split edges 2 1: 8
    triangle split edges 2 0: 9
    triangle split edges 0 2: 10
    triangle split edge 0:    11
    triangle split edge 1:    12
    triangle split edge 2:    13
    tetrahedron unsplit:      14
    tetrahedron split all edges: 15
    tetrahedron split, edges (e0,...,e5) marked by bit mask: 16 + mask, mask in [1, 62]

  For a tetrahedron, edges e0..e5 are numbered as documented at SBRGetTetEdges_Private(). The mask
  values 0 and 63 are handled by RT_TET and RT_TET_SPLIT respectively, rather than as RT_TET_SPLIT_BASE
  offsets, since they need no reference to which particular edges are marked.
*/
typedef enum {
  RT_VERTEX,
  RT_EDGE,
  RT_EDGE_SPLIT,
  RT_TRIANGLE,
  RT_TRIANGLE_SPLIT,
  RT_TRIANGLE_SPLIT_01,
  RT_TRIANGLE_SPLIT_10,
  RT_TRIANGLE_SPLIT_12,
  RT_TRIANGLE_SPLIT_21,
  RT_TRIANGLE_SPLIT_20,
  RT_TRIANGLE_SPLIT_02,
  RT_TRIANGLE_SPLIT_0,
  RT_TRIANGLE_SPLIT_1,
  RT_TRIANGLE_SPLIT_2,
  RT_TET,
  RT_TET_SPLIT,
  RT_TET_SPLIT_BASE
} RefinementType;

static PetscErrorCode DMPlexTransformSetUp_SBR(DMPlexTransform tr)
{
  DMPlexRefine_SBR *sbr = (DMPlexRefine_SBR *)tr->data;
  DM                dm;
  DMLabel           active;
  PetscSF           pointSF;
  DMPlexPointQueue  queue = NULL;
  IS                refineIS;
  const PetscInt   *refineCells;
  PetscInt          pStart, pEnd, p, eStart, eEnd, e, edgeLenSize, Nc, c;
  PetscBool         empty;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Split Points", &sbr->splitPoints));
  /* Create edge lengths */
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF, &sbr->secEdgeLen));
  PetscCall(PetscSectionSetChart(sbr->secEdgeLen, eStart, eEnd));
  for (e = eStart; e < eEnd; ++e) PetscCall(PetscSectionSetDof(sbr->secEdgeLen, e, 1));
  PetscCall(PetscSectionSetUp(sbr->secEdgeLen));
  PetscCall(PetscSectionGetStorageSize(sbr->secEdgeLen, &edgeLenSize));
  PetscCall(PetscCalloc1(edgeLenSize, &sbr->edgeLen));
  /* Add edges of cells that are marked for refinement to edge queue */
  PetscCall(DMPlexTransformGetActive(tr, &active));
  PetscCheck(active, PetscObjectComm((PetscObject)tr), PETSC_ERR_ARG_WRONGSTATE, "DMPlexTransform must have an adaptation label in order to use SBR algorithm");
  PetscCall(DMPlexPointQueueCreate(1024, &queue));
  PetscCall(DMLabelGetStratumIS(active, DM_ADAPT_REFINE, &refineIS));
  PetscCall(DMLabelGetStratumSize(active, DM_ADAPT_REFINE, &Nc));
  if (refineIS) PetscCall(ISGetIndices(refineIS, &refineCells));
  for (c = 0; c < Nc; ++c) {
    const PetscInt cell = refineCells[c];
    PetscInt       depth;

    PetscCall(DMPlexGetPointDepth(dm, cell, &depth));
    if (depth == 1) {
      PetscCall(DMLabelSetValue(sbr->splitPoints, cell, 1));
      PetscCall(DMPlexPointQueueEnqueue(queue, cell));
    } else {
      PetscInt *closure = NULL;
      PetscInt  Ncl;

      PetscCall(DMLabelSetValue(sbr->splitPoints, cell, depth));
      PetscCall(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure));
      for (PetscInt cl = 0; cl < 2 * Ncl; cl += 2) {
        const PetscInt edge = closure[cl];

        if (edge >= eStart && edge < eEnd) {
          PetscCall(DMLabelSetValue(sbr->splitPoints, edge, 1));
          PetscCall(DMPlexPointQueueEnqueue(queue, edge));
        }
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure));
    }
  }
  if (refineIS) PetscCall(ISRestoreIndices(refineIS, &refineCells));
  PetscCall(ISDestroy(&refineIS));
  /* Setup communication */
  PetscCall(DMGetPointSF(dm, &pointSF));
  PetscCall(DMLabelPropagateBegin(sbr->splitPoints, pointSF));
  /* While edge queue is not empty: */
  PetscCall(DMPlexPointQueueEmptyCollective((PetscObject)dm, queue, &empty));
  while (!empty) {
    PetscCall(SBRSplitLocalEdges_Private(tr, queue));
    /* Communicate marked edges
         An easy implementation is to allocate an array the size of the number of points. We put the splitPoints marks into the
         array, and then call PetscSFReduce()+PetscSFBcast() to make the marks consistent.

         TODO: We could use in-place communication with a different SF
           We use MPI_SUM for the Reduce, and check the result against the rootdegree. If sum >= rootdegree+1, then the edge has
           already been marked. If not, it might have been handled on the process in this round, but we add it anyway.

           In order to update the queue with the new edges from the label communication, we use BcastAnOp(MPI_SUM), so that new
           values will have 1+0=1 and old values will have 1+1=2. Loop over these, resetting the values to 1, and adding any new
           edge to the queue.
    */
    PetscCall(DMLabelPropagatePush(sbr->splitPoints, pointSF, MPI_MAX, splitPoint, queue));
    PetscCall(DMPlexPointQueueEmptyCollective((PetscObject)dm, queue, &empty));
  }
  PetscCall(DMLabelPropagateEnd(sbr->splitPoints, pointSF));
  /* Calculate refineType for each cell */
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Refine Type", &tr->trType));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    DMLabel        trType = tr->trType;
    DMPolytopeType ct;
    PetscInt       val;

    PetscCall(DMPlexGetCellType(dm, p, &ct));
    switch (ct) {
    case DM_POLYTOPE_POINT:
      PetscCall(DMLabelSetValue(trType, p, RT_VERTEX));
      break;
    case DM_POLYTOPE_SEGMENT:
      PetscCall(DMLabelGetValue(sbr->splitPoints, p, &val));
      if (val == 1) PetscCall(DMLabelSetValue(trType, p, RT_EDGE_SPLIT));
      else PetscCall(DMLabelSetValue(trType, p, RT_EDGE));
      break;
    case DM_POLYTOPE_TRIANGLE:
      PetscCall(DMLabelGetValue(sbr->splitPoints, p, &val));
      if (val == 2) {
        const PetscInt *cone;
        PetscReal       lens[3];
        PetscInt        vals[3], i;

        PetscCall(DMPlexGetCone(dm, p, &cone));
        for (i = 0; i < 3; ++i) {
          PetscCall(DMLabelGetValue(sbr->splitPoints, cone[i], &vals[i]));
          vals[i] = vals[i] < 0 ? 0 : vals[i];
          PetscCall(SBRGetEdgeLen_Private(tr, cone[i], &lens[i]));
        }
        if (vals[0] && vals[1] && vals[2]) PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT));
        else if (vals[0] && vals[1]) PetscCall(DMLabelSetValue(trType, p, lens[0] > lens[1] ? RT_TRIANGLE_SPLIT_01 : RT_TRIANGLE_SPLIT_10));
        else if (vals[1] && vals[2]) PetscCall(DMLabelSetValue(trType, p, lens[1] > lens[2] ? RT_TRIANGLE_SPLIT_12 : RT_TRIANGLE_SPLIT_21));
        else if (vals[2] && vals[0]) PetscCall(DMLabelSetValue(trType, p, lens[2] > lens[0] ? RT_TRIANGLE_SPLIT_20 : RT_TRIANGLE_SPLIT_02));
        else if (vals[0]) PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT_0));
        else if (vals[1]) PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT_1));
        else if (vals[2]) PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT_2));
        else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cell %" PetscInt_FMT " does not fit any refinement type (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ")", p, vals[0], vals[1], vals[2]);
      } else PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE));
      break;
    case DM_POLYTOPE_TETRAHEDRON:
      PetscCall(DMLabelGetValue(sbr->splitPoints, p, &val));
      if (val == 3) {
        PetscInt edges[6], mask = 0;

        PetscCall(SBRGetTetEdges_Private(dm, p, edges));
        for (PetscInt k = 0; k < 6; ++k) {
          PetscInt eval;

          PetscCall(DMLabelGetValue(sbr->splitPoints, edges[k], &eval));
          if (eval == 1) mask |= 1 << k;
        }
        if (mask == 63) PetscCall(DMLabelSetValue(trType, p, RT_TET_SPLIT));
        else if (mask == 0) PetscCall(DMLabelSetValue(trType, p, RT_TET));
        else PetscCall(DMLabelSetValue(trType, p, RT_TET_SPLIT_BASE + mask));
      } else PetscCall(DMLabelSetValue(trType, p, RT_TET));
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle points of type %s", DMPolytopeTypes[ct]);
    }
    PetscCall(DMLabelGetValue(sbr->splitPoints, p, &val));
  }
  /* Cleanup */
  PetscCall(DMPlexPointQueueDestroy(&queue));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformGetSubcellOrientation_SBR(DMPlexTransform tr, DMPolytopeType sct, PetscInt sp, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  PetscInt rt;

  PetscFunctionBeginHot;
  PetscCall(DMLabelGetValue(tr->trType, sp, &rt));
  *rnew = r;
  *onew = o;
  switch (rt) {
  case RT_TRIANGLE_SPLIT_01:
  case RT_TRIANGLE_SPLIT_10:
  case RT_TRIANGLE_SPLIT_12:
  case RT_TRIANGLE_SPLIT_21:
  case RT_TRIANGLE_SPLIT_20:
  case RT_TRIANGLE_SPLIT_02:
    switch (tct) {
    case DM_POLYTOPE_SEGMENT:
      break;
    case DM_POLYTOPE_TRIANGLE:
      break;
    default:
      break;
    }
    break;
  case RT_TRIANGLE_SPLIT_0:
  case RT_TRIANGLE_SPLIT_1:
  case RT_TRIANGLE_SPLIT_2:
    switch (tct) {
    case DM_POLYTOPE_SEGMENT:
      break;
    case DM_POLYTOPE_TRIANGLE:
      *onew = so < 0 ? -(o + 1) : o;
      *rnew = so < 0 ? (r + 1) % 2 : r;
      break;
    default:
      break;
    }
    break;
  case RT_EDGE_SPLIT:
  case RT_TRIANGLE_SPLIT:
  case RT_TET_SPLIT:
    PetscCall(DMPlexTransformGetSubcellOrientation_Regular(tr, sct, sp, so, tct, r, o, rnew, onew));
    break;
  default:
    PetscCall(DMPlexTransformGetSubcellOrientationIdentity(tr, sct, sp, so, tct, r, o, rnew, onew));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Add 1 edge inside this triangle, making 2 new triangles.
 2
 |\
 | \
 |  \
 |   \
 |    1
 |     \
 |  B   \
 2       1
 |      / \
 | ____/   0
 |/    A    \
 0-----0-----1
*/
static PetscErrorCode SBRGetTriangleSplitSingle(PetscInt o, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  const PetscInt       *arr     = DMPolytopeTypeGetArrangement(DM_POLYTOPE_TRIANGLE, o);
  static DMPolytopeType triT1[] = {DM_POLYTOPE_SEGMENT, DM_POLYTOPE_TRIANGLE};
  static PetscInt       triS1[] = {1, 2};
  static PetscInt       triC1[] = {DM_POLYTOPE_POINT,   2, 0, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_SEGMENT, 1, 2, 0,
                                   DM_POLYTOPE_SEGMENT, 0, 0};
  static PetscInt       triO1[] = {0, 0, 0, 0, -1, 0, 0, 0};

  PetscFunctionBeginHot;
  /* To get the other divisions, we reorient the triangle */
  triC1[2]  = arr[0 * 2];
  triC1[7]  = arr[1 * 2];
  triC1[11] = arr[0 * 2];
  triC1[15] = arr[1 * 2];
  triC1[22] = arr[1 * 2];
  triC1[26] = arr[2 * 2];
  *Nt       = 2;
  *target   = triT1;
  *size     = triS1;
  *cone     = triC1;
  *ornt     = triO1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Add 2 edges inside this triangle, making 3 new triangles.
 RT_TRIANGLE_SPLIT_12
 2
 |\
 | \
 |  \
 0   \
 |    1
 |     \
 |  B   \
 2-------1
 |   C  / \
 1 ____/   0
 |/    A    \
 0-----0-----1
 RT_TRIANGLE_SPLIT_10
 2
 |\
 | \
 |  \
 0   \
 |    1
 |     \
 |  A   \
 2       1
 |      /|\
 1 ____/ / 0
 |/ C   / B \
 0-----0-----1
 RT_TRIANGLE_SPLIT_20
 2
 |\
 | \
 |  \
 0   \
 |    \
 |     \
 |      \
 2   A   1
 |\       \
 1 ---\    \
 |B \_C----\\
 0-----0-----1
 RT_TRIANGLE_SPLIT_21
 2
 |\
 | \
 |  \
 0   \
 |    \
 |  B  \
 |      \
 2-------1
 |\     C \
 1 ---\    \
 |  A  ----\\
 0-----0-----1
 RT_TRIANGLE_SPLIT_01
 2
 |\
 |\\
 || \
 | \ \
 |  | \
 |  |  \
 |  |   \
 2   \ C 1
 |  A | / \
 |    | |B \
 |     \/   \
 0-----0-----1
 RT_TRIANGLE_SPLIT_02
 2
 |\
 |\\
 || \
 | \ \
 |  | \
 |  |  \
 |  |   \
 2 C \   1
 |\   |   \
 | \__|  A \
 | B  \\    \
 0-----0-----1
*/
static PetscErrorCode SBRGetTriangleSplitDouble(PetscInt o, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  PetscInt              e0, e1;
  const PetscInt       *arr     = DMPolytopeTypeGetArrangement(DM_POLYTOPE_TRIANGLE, o);
  static DMPolytopeType triT2[] = {DM_POLYTOPE_SEGMENT, DM_POLYTOPE_TRIANGLE};
  static PetscInt       triS2[] = {2, 3};
  static PetscInt triC2[] = {DM_POLYTOPE_POINT, 2, 0, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0, DM_POLYTOPE_POINT, 1, 1, 0, DM_POLYTOPE_POINT, 1, 2, 0, DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_SEGMENT, 1, 2, 0, DM_POLYTOPE_SEGMENT, 0, 1, DM_POLYTOPE_SEGMENT, 1, 2, 1, DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 0, 1};
  static PetscInt triO2[] = {0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0};

  PetscFunctionBeginHot;
  /* To get the other divisions, we reorient the triangle */
  triC2[2]  = arr[0 * 2];
  triC2[3]  = arr[0 * 2 + 1] ? 1 : 0;
  triC2[7]  = arr[1 * 2];
  triC2[11] = arr[1 * 2];
  triC2[15] = arr[2 * 2];
  /* Swap the first two edges if the triangle is reversed */
  e0            = o < 0 ? 23 : 19;
  e1            = o < 0 ? 19 : 23;
  triC2[e0]     = arr[0 * 2];
  triC2[e0 + 1] = 0;
  triC2[e1]     = arr[1 * 2];
  triC2[e1 + 1] = o < 0 ? 1 : 0;
  triO2[6]      = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, -1, arr[2 * 2 + 1]);
  /* Swap the first two edges if the triangle is reversed */
  e0            = o < 0 ? 34 : 30;
  e1            = o < 0 ? 30 : 34;
  triC2[e0]     = arr[1 * 2];
  triC2[e0 + 1] = o < 0 ? 0 : 1;
  triC2[e1]     = arr[2 * 2];
  triC2[e1 + 1] = o < 0 ? 1 : 0;
  triO2[9]      = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, -1, arr[2 * 2 + 1]);
  /* Swap the last two edges if the triangle is reversed */
  triC2[41] = arr[2 * 2];
  triC2[42] = o < 0 ? 0 : 1;
  triC2[45] = o < 0 ? 1 : 0;
  triC2[48] = o < 0 ? 0 : 1;
  triO2[11] = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, 0, arr[1 * 2 + 1]);
  triO2[12] = DMPolytopeTypeComposeOrientation(DM_POLYTOPE_SEGMENT, 0, arr[2 * 2 + 1]);
  *Nt       = 2;
  *target   = triT2;
  *size     = triS2;
  *cone     = triC2;
  *ornt     = triO2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Append one cone entry {ft, fn, acp[0..fn-1], r} plus its matching ornt entry o */
static void SBRAppendCone_Private(PetscInt cone[], PetscInt *coff, PetscInt ornt[], PetscInt *ooff, DMPolytopeType ft, PetscInt fn, const PetscInt acp[], PetscInt r, PetscInt o)
{
  cone[(*coff)++] = (PetscInt)ft;
  cone[(*coff)++] = fn;
  for (PetscInt i = 0; i < fn; ++i) cone[(*coff)++] = acp[i];
  cone[(*coff)++] = r;
  ornt[(*ooff)++] = o;
}

/*
  Bisect a tetrahedron by a single marked edge e (the only marked edge, which by the closure
  invariant is necessarily the tetrahedron's own longest edge). This produces 2 sub-tetrahedra
  separated by 1 new interior triangle, itself bounded by 2 new diagonal segments running from the
  new edge midpoint to the tetrahedron's other 2 vertices, e.g. for e = e0 = (v0,v1), with m the
  midpoint and c,d the other two vertices:

    child1 = (v0, m, c, d)     child2 = (m, v1, c, d)     interior face = (m, c, d)

  The 2 faces of the tetrahedron containing e are each classified RT_TRIANGLE_SPLIT_k on their own
  (e is their own longest edge too, since it is the longest of the whole tetrahedron), and split
  into 2 replicas by the existing 2D logic; the 2 faces not containing e pass through unsplit. This
  routine references those replicas using the same formula empirically verified against
  SBRGetTriangleSplitSingle(): for a face with marked local edge (tail,head) and opposite vertex
  opp, replica 0 = (opp,tail,mid), replica 1 = (mid,head,opp).

  Vertex-slot values 0-3 denote the tetrahedron's own original vertices; the value 4+e denotes the
  midpoint of edge e (only e's own midpoint, called m below, ever appears here since only one edge
  is marked). Orientations are computed with DMPolytopeGetVertexOrientation() rather than derived
  by hand, so this routine does not depend on guessing PETSc's arrangement sign conventions.

  Known limitation: DMPlexTransformGetSubcellOrientation_SBR()'s RT_TRIANGLE_SPLIT_k case only
  adjusts the referenced replica for a reflected relative orientation (so < 0), not the full
  arrangement group, so referencing one of the tetrahedron's 2 split side-faces here is only
  correct when that face's orientation as seen by the tetrahedron is 0 -- true for every face of a
  freshly-created reference cell, but not guaranteed in a general mesh. This is a pre-existing gap
  in the 2D code (the same reason the sbr_triangle_* tests in ex11.c restrict -ornts), not something
  introduced here; fixing it is future work, tracked alongside the recursive generator needed for
  na = 2..5 (multiple marked edges) in plan-sbr-refine.md.
*/
static PetscErrorCode SBRGetTetSplitSingleEdge_Private(PetscInt e, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  static DMPolytopeType tetT[] = {DM_POLYTOPE_SEGMENT, DM_POLYTOPE_TRIANGLE, DM_POLYTOPE_TETRAHEDRON};
  static PetscInt       tetS[] = {2, 1, 2};
  static PetscInt       tetC[96];
  static PetscInt       tetO[16];
  const PetscInt        a = tetEdgeVert[e][0], b = tetEdgeVert[e][1];
  const PetscInt        m = 4 + e;
  PetscInt              c = -1, d = -1, n = 0;
  PetscInt              child1[4], child2[4], triCanon[3];
  PetscInt              coff = 0, ooff = 0;

  PetscFunctionBeginHot;
  for (PetscInt x = 0; x < 4; ++x)
    if (x != a && x != b) {
      if (n == 0) c = x;
      else d = x;
      n++;
    }
  for (PetscInt x = 0; x < 4; ++x) {
    child1[x] = (x == b) ? m : x;
    child2[x] = (x == a) ? m : x;
  }

  /* new SEGMENT replica 0: (m, c) and replica 1: (m, d) */
  for (PetscInt r = 0; r < 2; ++r) {
    const PetscInt other = r == 0 ? c : d;

    SBRAppendCone_Private(tetC, &coff, tetO, &ooff, DM_POLYTOPE_POINT, 2, tetEdgeFaceLoc[e], 0, 0);
    SBRAppendCone_Private(tetC, &coff, tetO, &ooff, DM_POLYTOPE_POINT, 3, tetVertPath[other], 0, 0);
  }

  /* new TRIANGLE replica 0: the interior face, canonical order = child2's own face opposite b */
  SBRTetFaceOpposite_Private(child2, b, triCanon);
  for (PetscInt k = 0; k < 3; ++k) {
    const PetscInt v0 = triCanon[k], v1 = triCanon[(k + 1) % 3];
    PetscInt       o;

    if (v0 == m || v1 == m) {
      const PetscInt other     = v0 == m ? v1 : v0;
      const PetscInt segIdx    = other == c ? 0 : 1;
      const PetscInt canon2[2] = {m, other}, desired2[2] = {v0, v1};

      PetscCall(DMPolytopeGetVertexOrientation(DM_POLYTOPE_SEGMENT, canon2, desired2, &o));
      SBRAppendCone_Private(tetC, &coff, tetO, &ooff, DM_POLYTOPE_SEGMENT, 0, NULL, segIdx, o);
    } else {
      const PetscInt edgeIdx   = SBREdgeIndexFromVerts_Private(v0, v1);
      const PetscInt canon2[2] = {tetEdgeVert[edgeIdx][0], tetEdgeVert[edgeIdx][1]}, desired2[2] = {v0, v1};

      PetscCall(DMPolytopeGetVertexOrientation(DM_POLYTOPE_SEGMENT, canon2, desired2, &o));
      SBRAppendCone_Private(tetC, &coff, tetO, &ooff, DM_POLYTOPE_SEGMENT, 2, tetEdgeFaceLoc[edgeIdx], 0, o);
    }
  }

  /* the 2 sub-tetrahedra, near a (child1) and near b (child2) */
  for (PetscInt r = 0; r < 2; ++r) {
    const PetscInt *child = r == 0 ? child1 : child2;
    const PetscInt  near = r == 0 ? a : b, far = r == 0 ? b : a;

    for (PetscInt f = 0; f < 4; ++f) {
      /* DMPlexGetRawFaces_Internal()'s face f is missing closure position 3-f, not f */
      const PetscInt mp = 3 - f;
      PetscInt       face3[3], o;

      SBRTetFaceOpposite_Private(child, mp, face3);
      if (mp == near) {
        /* the interior face shared with the other child */
        PetscCall(DMPolytopeGetVertexOrientation(DM_POLYTOPE_TRIANGLE, triCanon, face3, &o));
        SBRAppendCone_Private(tetC, &coff, tetO, &ooff, DM_POLYTOPE_TRIANGLE, 0, NULL, 0, o);
      } else if (mp == far) {
        /* the whole original face opposite 'far', unmarked since it does not contain e */
        const PetscInt faceIdx = 3 - far, acp[1] = {faceIdx};

        PetscCall(DMPolytopeGetVertexOrientation(DM_POLYTOPE_TRIANGLE, tetFaceVert[faceIdx], face3, &o));
        SBRAppendCone_Private(tetC, &coff, tetO, &ooff, DM_POLYTOPE_TRIANGLE, 1, acp, 0, o);
      } else {
        /* a single-split replica of the original face opposite 'mp', which does contain e */
        const PetscInt faceIdx = 3 - mp, acp[1] = {faceIdx};
        PetscInt       localE = -1;

        for (PetscInt i = 0; i < 3; ++i)
          if (tetFaceEdge[faceIdx][i] == e) localE = i;
        {
          const PetscInt tail = tetFaceVert[faceIdx][localE], head = tetFaceVert[faceIdx][(localE + 1) % 3], opp = tetFaceVert[faceIdx][(localE + 2) % 3];
          const PetscInt replicaIdx = near == tail ? 0 : 1;
          PetscInt       canon3[3];

          if (replicaIdx == 0) {
            canon3[0] = opp;
            canon3[1] = tail;
            canon3[2] = m;
          } else {
            canon3[0] = m;
            canon3[1] = head;
            canon3[2] = opp;
          }
          PetscCall(DMPolytopeGetVertexOrientation(DM_POLYTOPE_TRIANGLE, canon3, face3, &o));
          SBRAppendCone_Private(tetC, &coff, tetO, &ooff, DM_POLYTOPE_TRIANGLE, 1, acp, replicaIdx, o);
        }
      }
    }
  }

  *Nt     = 3;
  *target = tetT;
  *size   = tetS;
  *cone   = tetC;
  *ornt   = tetO;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformCellTransform_SBR(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  DMLabel  trType = tr->trType;
  PetscInt val;

  PetscFunctionBeginHot;
  PetscCheck(p >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point argument is invalid");
  PetscCall(DMLabelGetValue(trType, p, &val));
  if (rt) *rt = val;
  switch (source) {
  case DM_POLYTOPE_POINT:
  case DM_POLYTOPE_POINT_PRISM_TENSOR:
  case DM_POLYTOPE_QUADRILATERAL:
  case DM_POLYTOPE_SEG_PRISM_TENSOR:
  case DM_POLYTOPE_HEXAHEDRON:
  case DM_POLYTOPE_TRI_PRISM:
  case DM_POLYTOPE_TRI_PRISM_TENSOR:
  case DM_POLYTOPE_QUAD_PRISM_TENSOR:
  case DM_POLYTOPE_PYRAMID:
    PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt));
    break;
  case DM_POLYTOPE_SEGMENT:
    if (val == RT_EDGE) PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt));
    else PetscCall(DMPlexTransformCellRefine_Regular(tr, source, p, NULL, Nt, target, size, cone, ornt));
    break;
  case DM_POLYTOPE_TRIANGLE:
    switch (val) {
    case RT_TRIANGLE_SPLIT_0:
      PetscCall(SBRGetTriangleSplitSingle(2, Nt, target, size, cone, ornt));
      break;
    case RT_TRIANGLE_SPLIT_1:
      PetscCall(SBRGetTriangleSplitSingle(0, Nt, target, size, cone, ornt));
      break;
    case RT_TRIANGLE_SPLIT_2:
      PetscCall(SBRGetTriangleSplitSingle(1, Nt, target, size, cone, ornt));
      break;
    case RT_TRIANGLE_SPLIT_21:
      PetscCall(SBRGetTriangleSplitDouble(-3, Nt, target, size, cone, ornt));
      break;
    case RT_TRIANGLE_SPLIT_10:
      PetscCall(SBRGetTriangleSplitDouble(-2, Nt, target, size, cone, ornt));
      break;
    case RT_TRIANGLE_SPLIT_02:
      PetscCall(SBRGetTriangleSplitDouble(-1, Nt, target, size, cone, ornt));
      break;
    case RT_TRIANGLE_SPLIT_12:
      PetscCall(SBRGetTriangleSplitDouble(0, Nt, target, size, cone, ornt));
      break;
    case RT_TRIANGLE_SPLIT_20:
      PetscCall(SBRGetTriangleSplitDouble(1, Nt, target, size, cone, ornt));
      break;
    case RT_TRIANGLE_SPLIT_01:
      PetscCall(SBRGetTriangleSplitDouble(2, Nt, target, size, cone, ornt));
      break;
    case RT_TRIANGLE_SPLIT:
      PetscCall(DMPlexTransformCellRefine_Regular(tr, source, p, NULL, Nt, target, size, cone, ornt));
      break;
    default:
      PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt));
    }
    break;
  case DM_POLYTOPE_TETRAHEDRON:
    if (val == RT_TET_SPLIT) {
      /* All 6 edges are marked: this is exactly the regular 1-to-8 refinement */
      PetscCall(DMPlexTransformCellRefine_Regular(tr, source, p, NULL, Nt, target, size, cone, ornt));
    } else if (val == RT_TET) {
      PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt));
    } else {
      const PetscInt mask  = val - RT_TET_SPLIT_BASE;
      PetscInt       nbits = 0, e = -1;

      for (PetscInt k = 0; k < 6; ++k)
        if (mask & (1 << k)) {
          nbits++;
          e = k;
        }
      if (nbits == 1) {
        /* Exactly one marked edge, necessarily the tetrahedron's own longest */
        PetscCall(SBRGetTetSplitSingleEdge_Private(e, Nt, target, size, cone, ornt));
      } else {
        /* Splits with 2-5 marked edges are not yet implemented; see plan-sbr-refine.md for the
           recursive bisection generator that will produce the cone/orientation data for these
           cases, extending SBRGetTetSplitSingleEdge_Private() to recurse when a leaf still has
           marked edges left after one bisection. */
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Partial 3D SBR refinement of a tetrahedron with %" PetscInt_FMT " marked edges (mask %" PetscInt_FMT ") is not yet implemented", nbits, mask);
      }
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No refinement strategy for %s", DMPolytopeTypes[source]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformSetFromOptions_SBR(DMPlexTransform tr, PetscOptionItems PetscOptionsObject)
{
  PetscInt  cells[256], n = 256, i;
  PetscBool flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "DMPlex Options");
  PetscCall(PetscOptionsIntArray("-dm_plex_transform_sbr_ref_cell", "Mark cells for refinement", "", cells, &n, &flg));
  if (flg) {
    DMLabel active;

    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", &active));
    for (i = 0; i < n; ++i) PetscCall(DMLabelSetValue(active, cells[i], DM_ADAPT_REFINE));
    PetscCall(DMPlexTransformSetActive(tr, active));
    PetscCall(DMLabelDestroy(&active));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformView_SBR(DMPlexTransform tr, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscViewerFormat format;
    const char       *name;

    PetscCall(PetscObjectGetName((PetscObject)tr, &name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "SBR refinement %s\n", name ? name : ""));
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscCall(DMLabelView(tr->trType, viewer));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)tr), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlexTransform writing", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformDestroy_SBR(DMPlexTransform tr)
{
  DMPlexRefine_SBR *sbr = (DMPlexRefine_SBR *)tr->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(sbr->edgeLen));
  PetscCall(PetscSectionDestroy(&sbr->secEdgeLen));
  PetscCall(DMLabelDestroy(&sbr->splitPoints));
  PetscCall(PetscFree(tr->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformInitialize_SBR(DMPlexTransform tr)
{
  PetscFunctionBegin;
  tr->ops->view                  = DMPlexTransformView_SBR;
  tr->ops->setfromoptions        = DMPlexTransformSetFromOptions_SBR;
  tr->ops->setup                 = DMPlexTransformSetUp_SBR;
  tr->ops->destroy               = DMPlexTransformDestroy_SBR;
  tr->ops->setdimensions         = DMPlexTransformSetDimensions_Internal;
  tr->ops->celltransform         = DMPlexTransformCellTransform_SBR;
  tr->ops->getsubcellorientation = DMPlexTransformGetSubcellOrientation_SBR;
  tr->ops->mapcoordinates        = DMPlexTransformMapCoordinatesBarycenter_Internal;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_SBR(DMPlexTransform tr)
{
  DMPlexRefine_SBR *f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscCall(PetscNew(&f));
  tr->data = f;

  PetscCall(DMPlexTransformInitialize_SBR(tr));
  PetscCall(PetscCitationsRegister(SBRCitation, &SBRcite));
  PetscFunctionReturn(PETSC_SUCCESS);
}
