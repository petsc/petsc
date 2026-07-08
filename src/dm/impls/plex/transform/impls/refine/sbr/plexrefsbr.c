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
/* tetVertFaces[v], tetEdgeFaces[e] - bitmasks over f0..f3 of the faces containing vertex v or edge e */
static const PetscInt tetVertFaces[4] = {7, 11, 13, 14};
static const PetscInt tetEdgeFaces[6] = {3, 9, 5, 6, 10, 12};
/* The edges of the triangle, triEdgeVert[e] = (v_e, v_{e+1}) */
static const PetscInt triEdgeVert[3][2] = {
  {0, 1},
  {1, 2},
  {2, 0}
};

/* Find the tetrahedron edge index (0-5) connecting the two given (unordered) vertices */
static PetscInt SBREdgeIndexFromVerts_Private(PetscInt v0, PetscInt v1)
{
  for (PetscInt k = 0; k < 6; ++k) {
    if ((tetEdgeVert[k][0] == v0 && tetEdgeVert[k][1] == v1) || (tetEdgeVert[k][0] == v1 && tetEdgeVert[k][1] == v0)) return k;
  }
  return -1;
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

static PetscErrorCode SBRGetEdgeMid_Private(DMPlexTransform tr, PetscInt edge, PetscReal mid[3])
{
  DM                 dm, cdm;
  Vec                coordsLocal;
  const PetscScalar *coords;
  const PetscInt    *cone;
  PetscScalar       *cA, *cB;
  PetscInt           cdim;

  PetscFunctionBeginHot;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMPlexGetCone(dm, edge, &cone));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMGetCoordinatesLocalNoncollective(dm, &coordsLocal));
  PetscCall(VecGetArrayRead(coordsLocal, &coords));
  PetscCall(DMPlexPointLocalRead(cdm, cone[0], coords, &cA));
  PetscCall(DMPlexPointLocalRead(cdm, cone[1], coords, &cB));
  for (PetscInt d = 0; d < 3; ++d) mid[d] = d < cdim ? 0.5 * PetscRealPart(cA[d] + cB[d]) : 0.0;
  PetscCall(VecRestoreArrayRead(coordsLocal, &coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Determine whether edge eA is bisected before edge eB: longer edges come first, and exact length
  ties are broken in favor of the lexicographically greater edge midpoint. The tie-breaker depends
  only on the geometry, not on mesh point numbers or cone traversal order, so every face and every
  cell containing the two edges resolves the tie the same way, on every process. This keeps the
  subdivision each face chooses for itself consistent with the subdivision induced on that face by
  the bisection of the cells above it (see the discussion at RefinementType).
*/
static PetscErrorCode SBREdgePrecedes_Private(DMPlexTransform tr, PetscInt eA, PetscInt eB, PetscBool *precedes)
{
  PetscReal lenA, lenB, midA[3], midB[3];

  PetscFunctionBeginHot;
  PetscCall(SBRGetEdgeLen_Private(tr, eA, &lenA));
  PetscCall(SBRGetEdgeLen_Private(tr, eB, &lenB));
  if (lenA != lenB) {
    *precedes = lenA > lenB ? PETSC_TRUE : PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(SBRGetEdgeMid_Private(tr, eA, midA));
  PetscCall(SBRGetEdgeMid_Private(tr, eB, midB));
  *precedes = PETSC_FALSE;
  for (PetscInt d = 0; d < 3; ++d) {
    if (midA[d] != midB[d]) {
      *precedes = midA[d] > midB[d] ? PETSC_TRUE : PETSC_FALSE;
      break;
    }
  }
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

/* Find the first edge of a tetrahedron in the bisection order, that is its longest edge */
static PetscErrorCode SBRGetTetMaxEdge_Private(DMPlexTransform tr, PetscInt tet, PetscInt *maxedge)
{
  DM        dm;
  PetscInt  edges[6];
  PetscBool prec;

  PetscFunctionBeginHot;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(SBRGetTetEdges_Private(dm, tet, edges));
  *maxedge = edges[0];
  for (PetscInt k = 1; k < 6; ++k) {
    PetscCall(SBREdgePrecedes_Private(tr, edges[k], *maxedge, &prec));
    if (prec) *maxedge = edges[k];
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
      PetscBool       prec;

      PetscCall(DMLabelGetValue(sbr->splitPoints, cell, &cval));
      if (cval == 2) continue;
      PetscCall(DMPlexGetCone(dm, cell, &cone));
      PetscCall(DMPlexGetConeSize(dm, cell, &coneSize));
      maxedge = cone[0];
      for (c = 1; c < coneSize; ++c) {
        PetscCall(SBREdgePrecedes_Private(tr, cone[c], maxedge, &prec));
        if (prec) maxedge = cone[c];
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

    RT_VERTEX:                  vertex
    RT_EDGE:                    edge unsplit
    RT_EDGE_SPLIT:              edge split
    RT_TRIANGLE:                triangle unsplit
    RT_TRIANGLE_SPLIT:          maximal (2D) triangle with all edges split, subdivided regularly
    RT_TRIANGLE_SPLIT_ij:       triangle with edges i and j split, i bisected first
    RT_TRIANGLE_SPLIT_k:        triangle with only edge k split
    RT_TRIANGLE_SPLIT_FAN_k:    triangle with all edges split under a 3D cell, fanned around the midpoint of edge k
    RT_TET:                     tetrahedron unsplit
    RT_TET_SPLIT_BASE + code:   tetrahedron with marked edges bisected in a given order

  Following Plaza & Carey (2000), a marked cell is subdivided by bisecting its marked edges one at a
  time, longest first: each bisection splits every subsimplex containing that edge, so the ordered
  list of marked edges determines the whole subdivision, and the subdivision it induces on a face is
  the bisection of the face's own marked edges in the same relative order. The classification below
  therefore records the bisection order: for a pair of split triangle edges as the type
  RT_TRIANGLE_SPLIT_ij, and for a tetrahedron as code = sum_i (e_i + 1) 7^i, the little-endian
  base-7 encoding of its marked edges e_0 > e_1 > ... in bisection order, with edges e0..e5 numbered
  as documented at SBRGetTetEdges_Private(). All length comparisons use SBREdgePrecedes_Private(),
  whose geometric tie-breaking makes the order consistent between each face and the cells above it.

  A fully marked triangle under a tetrahedron is fanned around the midpoint of its first-bisected
  edge, matching the bisection cascade (the 4T partition of Rivara, Fig. 3 of the paper), and a
  fully marked tetrahedron is bisected 6 times like any other marked tetrahedron. The regular
  subdivisions are kept only for maximal (2D) triangles, where no compatibility with higher cells
  constrains the interior of the subdivision, preserving the original 2D behavior of this transform.
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
  RT_TRIANGLE_SPLIT_FAN_0,
  RT_TRIANGLE_SPLIT_FAN_1,
  RT_TRIANGLE_SPLIT_FAN_2,
  RT_TET,
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
        PetscInt        vals[3], i;
        PetscBool       prec;

        PetscCall(DMPlexGetCone(dm, p, &cone));
        for (i = 0; i < 3; ++i) {
          PetscCall(DMLabelGetValue(sbr->splitPoints, cone[i], &vals[i]));
          vals[i] = vals[i] < 0 ? 0 : vals[i];
        }
        if (vals[0] && vals[1] && vals[2]) {
          PetscInt suppSize, k = 0;

          /* A triangle below a 3D cell must be subdivided compatibly with the bisection cascade of
             the cells above it, fanning around the midpoint of its first-bisected edge; a maximal
             (2D) triangle keeps the regular subdivision used since the original 2D implementation */
          PetscCall(DMPlexGetSupportSize(dm, p, &suppSize));
          if (suppSize) {
            for (i = 1; i < 3; ++i) {
              PetscCall(SBREdgePrecedes_Private(tr, cone[i], cone[k], &prec));
              if (prec) k = i;
            }
            PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT_FAN_0 + k));
          } else PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT));
        } else if (vals[0] && vals[1]) {
          PetscCall(SBREdgePrecedes_Private(tr, cone[0], cone[1], &prec));
          PetscCall(DMLabelSetValue(trType, p, prec ? RT_TRIANGLE_SPLIT_01 : RT_TRIANGLE_SPLIT_10));
        } else if (vals[1] && vals[2]) {
          PetscCall(SBREdgePrecedes_Private(tr, cone[1], cone[2], &prec));
          PetscCall(DMLabelSetValue(trType, p, prec ? RT_TRIANGLE_SPLIT_12 : RT_TRIANGLE_SPLIT_21));
        } else if (vals[2] && vals[0]) {
          PetscCall(SBREdgePrecedes_Private(tr, cone[2], cone[0], &prec));
          PetscCall(DMLabelSetValue(trType, p, prec ? RT_TRIANGLE_SPLIT_20 : RT_TRIANGLE_SPLIT_02));
        } else if (vals[0]) PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT_0));
        else if (vals[1]) PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT_1));
        else if (vals[2]) PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE_SPLIT_2));
        else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cell %" PetscInt_FMT " does not fit any refinement type (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ")", p, vals[0], vals[1], vals[2]);
      } else PetscCall(DMLabelSetValue(trType, p, RT_TRIANGLE));
      break;
    case DM_POLYTOPE_TETRAHEDRON:
      PetscCall(DMLabelGetValue(sbr->splitPoints, p, &val));
      if (val == 3) {
        PetscInt edges[6], ord[6], na = 0, code = 0;

        PetscCall(SBRGetTetEdges_Private(dm, p, edges));
        for (PetscInt k = 0; k < 6; ++k) {
          PetscInt eval;

          PetscCall(DMLabelGetValue(sbr->splitPoints, edges[k], &eval));
          if (eval == 1) ord[na++] = k;
        }
        PetscCheck(na, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cell %" PetscInt_FMT " is marked for subdivision but has no marked edges", p);
        /* Insertion sort of the marked edges into bisection order */
        for (PetscInt i = 1; i < na; ++i) {
          const PetscInt ei = ord[i];
          PetscInt       j;

          for (j = i - 1; j >= 0; --j) {
            PetscBool prec;

            PetscCall(SBREdgePrecedes_Private(tr, edges[ei], edges[ord[j]], &prec));
            if (!prec) break;
            ord[j + 1] = ord[j];
          }
          ord[j + 1] = ei;
        }
        for (PetscInt i = na - 1; i >= 0; --i) code = code * 7 + (ord[i] + 1);
        PetscCall(DMLabelSetValue(trType, p, RT_TET_SPLIT_BASE + code));
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

/* Append one cone entry {ft, fn, acp[0..fn-1], r} plus its matching ornt entry o */
static PetscErrorCode SBRAppendCone_Private(PetscInt cone[], PetscInt *coff, PetscInt maxCone, PetscInt ornt[], PetscInt *ooff, PetscInt maxOrnt, DMPolytopeType ft, PetscInt fn, const PetscInt acp[], PetscInt r, PetscInt o)
{
  PetscFunctionBeginHot;
  PetscCheck(*coff + 3 + fn <= maxCone && *ooff + 1 <= maxOrnt, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Cone buffer of size (%" PetscInt_FMT ", %" PetscInt_FMT ") is too small", maxCone, maxOrnt);
  cone[(*coff)++] = (PetscInt)ft;
  cone[(*coff)++] = fn;
  for (PetscInt i = 0; i < fn; ++i) cone[(*coff)++] = acp[i];
  cone[(*coff)++] = r;
  ornt[(*ooff)++] = o;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Symbolic subdivision machinery

  Subdivisions are worked out on a canonical reference cell, in terms of symbols: for a triangle,
  0-2 denote its vertices and 3+e the midpoint of its edge e; for a tetrahedron, 0-3 denote its
  vertices and 4+e the midpoint of its edge e. Everything is purely combinatorial, so the results
  depend only on the refinement type, never on mesh coordinates.
*/

typedef struct {
  PetscInt nseg;      /* Number of segments added inside the triangle */
  PetscInt ntri;      /* Number of subtriangles */
  PetscInt seg[3][2]; /* The endpoints of each added segment, in its canonical direction */
  PetscInt tri[4][3]; /* The vertices of each subtriangle, in canonical cone order */
} SBRTriSubdiv;

/* Map a triangle symbol authored against arrangement o of the triangle to the actual cone numbering */
static PetscInt SBRTriRelabel_Private(PetscInt o, PetscInt s)
{
  const PetscInt *arr = DMPolytopeTypeGetArrangement(DM_POLYTOPE_TRIANGLE, o);

  if (s < 3) return triEdgeVert[arr[s * 2]][arr[s * 2 + 1] ? 1 : 0];
  return 3 + arr[(s - 3) * 2];
}

/*
  Compute the symbolic subdivision of a triangle: nsplit = 0 leaves it whole, nsplit = 1 bisects
  edge 'first', nsplit = 2 bisects edge 'first' and then edge 'second', and nsplit = 3 bisects all
  three edges starting with edge 'first' (the fan, where the order of the remaining two does not
  affect the result). The subcells are listed exactly as produced, in replica order, by the identity
  transform, SBRGetTriangleSplitSingle(), SBRGetTriangleSplitDouble(), and
  SBRGetTriangleSplitFan_Private(): each case is authored here for the canonical arrangement and
  relabeled through the same arrangement its template is dispatched with.
*/
static PetscErrorCode SBRTriangleSubdiv_Private(PetscInt nsplit, PetscInt first, PetscInt second, SBRTriSubdiv *sub)
{
  static const SBRTriSubdiv unsplitS = {0, 1, {{0, 0}}, {{0, 1, 2}}};
  static const SBRTriSubdiv singleS  = {
    1, 2, {{0, 4}},
      {{0, 1, 4}, {4, 2, 0}}
  };
  static const SBRTriSubdiv doubleS = {
    2, 3, {{0, 4}, {4, 5}},
      {{0, 1, 4}, {4, 2, 5}, {5, 0, 4}}
  };
  /* The reflected double split (second = first - 1), authored for first = 2, second = 1: the
     template emits its subtriangle cones starting from different corners than a plain reflection
     of the rotated instances, so it gets its own canonical tuples */
  static const SBRTriSubdiv doubleRS = {
    2, 3, {{1, 5}, {5, 4}},
      {{5, 0, 1}, {4, 2, 5}, {1, 4, 5}}
  };
  static const SBRTriSubdiv fanS = {
    3, 4, {{0, 4}, {4, 3}, {4, 5}},
      {{0, 3, 4}, {3, 1, 4}, {5, 4, 2}, {0, 4, 5}}
  };
  const SBRTriSubdiv *canon;
  PetscInt            o;

  PetscFunctionBeginHot;
  switch (nsplit) {
  case 0:
    canon = &unsplitS;
    o     = 0;
    break;
  case 1:
  case 3:
    canon = nsplit == 1 ? &singleS : &fanS;
    o     = (first + 2) % 3;
    break;
  case 2:
    if (second == (first + 1) % 3) {
      canon = &doubleS;
      o     = (first + 2) % 3;
    } else {
      canon = &doubleRS;
      o     = (first + 1) % 3;
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid number of split edges %" PetscInt_FMT, nsplit);
  }
  sub->nseg = canon->nseg;
  sub->ntri = canon->ntri;
  for (PetscInt s = 0; s < canon->nseg; ++s)
    for (PetscInt i = 0; i < 2; ++i) sub->seg[s][i] = SBRTriRelabel_Private(o, canon->seg[s][i]);
  for (PetscInt t = 0; t < canon->ntri; ++t)
    for (PetscInt i = 0; i < 3; ++i) sub->tri[t][i] = SBRTriRelabel_Private(o, canon->tri[t][i]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define SBR_TET_MAX_SEGS   16
#define SBR_TET_MAX_TRIS   32
#define SBR_TET_MAX_LEAVES 16
#define SBR_TET_MAX_CONE   2048
#define SBR_TET_MAX_ORNT   512

typedef struct {
  PetscInt nseg, ntri, nleaf;
  PetscInt seg[SBR_TET_MAX_SEGS][2];    /* Interior segments, in their canonical direction */
  PetscInt tri[SBR_TET_MAX_TRIS][3];    /* Interior triangles, in canonical cone order */
  PetscInt leaf[SBR_TET_MAX_LEAVES][4]; /* Subtetrahedra, in canonical vertex order */
} SBRTetSubdiv;

/* The faces of the tetrahedron containing a given symbol, as a bitmask over f0..f3 */
static PetscInt SBRTetSymbolFaces_Private(PetscInt s)
{
  return s < 4 ? tetVertFaces[s] : tetEdgeFaces[s - 4];
}

static PetscInt SBRTupleCompare_Private(PetscInt n, const PetscInt a[], const PetscInt b[])
{
  for (PetscInt i = 0; i < n; ++i)
    if (a[i] != b[i]) return a[i] < b[i] ? -1 : 1;
  return 0;
}

static PetscBool SBRTupleMatch_Private(PetscInt n, const PetscInt a[], const PetscInt b[])
{
  for (PetscInt i = 0; i < n; ++i) {
    PetscInt j;

    for (j = 0; j < n; ++j)
      if (a[i] == b[j]) break;
    if (j == n) return PETSC_FALSE;
  }
  return PETSC_TRUE;
}

static void SBRTetDecodeOrder_Private(PetscInt code, PetscInt ord[6], PetscInt *na)
{
  *na = 0;
  while (code) {
    ord[(*na)++] = code % 7 - 1;
    code /= 7;
  }
}

/* Add the interior segment (P, Q) if it is not already present */
static PetscErrorCode SBRTetAddSeg_Private(SBRTetSubdiv *sub, PetscInt P, PetscInt Q)
{
  PetscFunctionBeginHot;
  for (PetscInt s = 0; s < sub->nseg; ++s)
    if ((sub->seg[s][0] == P && sub->seg[s][1] == Q) || (sub->seg[s][0] == Q && sub->seg[s][1] == P)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(sub->nseg < SBR_TET_MAX_SEGS, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Too many interior segments");
  sub->seg[sub->nseg][0] = P;
  sub->seg[sub->nseg][1] = Q;
  ++sub->nseg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Subdivide the reference tetrahedron by bisecting the given edges in the given order. Each
  bisection splits every current subsimplex containing the edge (Lemma 1.1 of Plaza & Carey (2000)):
  the leaf subtetrahedra, the interior triangles created by earlier bisections, and implicitly the
  faces of the tetrahedron, whose subdivision is not tracked here because each face is subdivided by
  its own transform, consistently, since its refinement type orders its marked edges with the same
  comparison (see RefinementType). Only the segments and triangles strictly interior to the
  tetrahedron are recorded as its own subcells; everything on the boundary belongs to a face.
*/
static PetscErrorCode SBRTetSubdivide_Private(const PetscInt ord[], PetscInt na, SBRTetSubdiv *sub)
{
  PetscFunctionBegin;
  sub->nseg  = 0;
  sub->ntri  = 0;
  sub->nleaf = 1;
  for (PetscInt i = 0; i < 4; ++i) sub->leaf[0][i] = i;
  for (PetscInt s = 0; s < na; ++s) {
    const PetscInt e = ord[s], a = tetEdgeVert[e][0], b = tetEdgeVert[e][1], m = 4 + e;

    for (PetscInt t = 0; t < sub->ntri; ++t) {
      PetscInt pa = -1, pb = -1;

      for (PetscInt i = 0; i < 3; ++i) {
        if (sub->tri[t][i] == a) pa = i;
        else if (sub->tri[t][i] == b) pb = i;
      }
      if (pa < 0 || pb < 0) continue;
      PetscCheck(sub->ntri < SBR_TET_MAX_TRIS, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Too many interior triangles");
      PetscCall(PetscArraymove(&sub->tri[t + 2][0], &sub->tri[t + 1][0], 3 * (sub->ntri - t - 1)));
      PetscCall(PetscArraycpy(sub->tri[t + 1], sub->tri[t], 3));
      PetscCall(SBRTetAddSeg_Private(sub, sub->tri[t][3 - pa - pb], m));
      sub->tri[t][pb]     = m;
      sub->tri[t + 1][pa] = m;
      ++sub->ntri;
      ++t;
    }
    for (PetscInt l = 0; l < sub->nleaf; ++l) {
      PetscInt pa = -1, pb = -1, *newtri;

      for (PetscInt i = 0; i < 4; ++i) {
        if (sub->leaf[l][i] == a) pa = i;
        else if (sub->leaf[l][i] == b) pb = i;
      }
      if (pa < 0 || pb < 0) continue;
      PetscCheck(sub->nleaf < SBR_TET_MAX_LEAVES && sub->ntri < SBR_TET_MAX_TRIS, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Too many subcells");
      PetscCall(PetscArraymove(&sub->leaf[l + 2][0], &sub->leaf[l + 1][0], 4 * (sub->nleaf - l - 1)));
      PetscCall(PetscArraycpy(sub->leaf[l + 1], sub->leaf[l], 4));
      sub->leaf[l][pb]     = m;
      sub->leaf[l + 1][pa] = m;
      ++sub->nleaf;
      /* Record the interior triangle separating the two children, and its interior median edges */
      newtri = sub->tri[sub->ntri++];
      for (PetscInt i = 0; i < 3; ++i) newtri[i] = sub->leaf[l + 1][tetOppFace[pb][i]];
      for (PetscInt i = 0; i < 3; ++i) {
        const PetscInt P = newtri[i], Q = newtri[(i + 1) % 3], other = P == m ? Q : P;

        if (P != m && Q != m) continue;
        if (SBRTetSymbolFaces_Private(m) & SBRTetSymbolFaces_Private(other)) continue;
        PetscCall(SBRTetAddSeg_Private(sub, other, m));
      }
      ++l;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The subdivision of face f of the tetrahedron, relabeled from face symbols into tetrahedron symbols */
static PetscErrorCode SBRTetFaceSubdiv_Private(const PetscInt ord[], PetscInt na, PetscInt f, SBRTriSubdiv *sub)
{
  PetscInt fl[3], nfm = 0;

  PetscFunctionBeginHot;
  for (PetscInt s = 0; s < na; ++s)
    for (PetscInt l = 0; l < 3; ++l)
      if (tetFaceEdge[f][l] == ord[s]) fl[nfm++] = l;
  PetscCall(SBRTriangleSubdiv_Private(nfm, nfm > 0 ? fl[0] : -1, nfm > 1 ? fl[1] : -1, sub));
  for (PetscInt s = 0; s < sub->nseg; ++s)
    for (PetscInt i = 0; i < 2; ++i) sub->seg[s][i] = sub->seg[s][i] < 3 ? tetFaceVert[f][sub->seg[s][i]] : 4 + tetFaceEdge[f][sub->seg[s][i] - 3];
  for (PetscInt t = 0; t < sub->ntri; ++t)
    for (PetscInt i = 0; i < 3; ++i) sub->tri[t][i] = sub->tri[t][i] < 3 ? tetFaceVert[f][sub->tri[t][i]] : 4 + tetFaceEdge[f][sub->tri[t][i] - 3];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Emit the cone entry for the segment (P, Q): a whole original edge, half of a split edge, a
   segment added inside a face, or a segment interior to the tetrahedron */
static PetscErrorCode SBRTetEmitSegRef_Private(const SBRTetSubdiv *sub, const SBRTriSubdiv fsub[], PetscInt P, PetscInt Q, PetscInt cone[], PetscInt *coff, PetscInt ornt[], PetscInt *ooff)
{
  PetscFunctionBeginHot;
  if (P < 4 && Q < 4) {
    const PetscInt e = SBREdgeIndexFromVerts_Private(P, Q);

    PetscCall(SBRAppendCone_Private(cone, coff, SBR_TET_MAX_CONE, ornt, ooff, SBR_TET_MAX_ORNT, DM_POLYTOPE_SEGMENT, 2, tetEdgeFaceLoc[e], 0, tetEdgeVert[e][0] == P ? 0 : -1));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (P < 4 || Q < 4) {
    const PetscInt v = P < 4 ? P : Q, e = (P < 4 ? Q : P) - 4;

    if (tetEdgeVert[e][0] == v || tetEdgeVert[e][1] == v) {
      const PetscInt r = tetEdgeVert[e][0] == v ? 0 : 1;

      /* Half of split edge e: replica 0 runs from the edge tail to the midpoint, replica 1 from the midpoint to the head */
      PetscCall(SBRAppendCone_Private(cone, coff, SBR_TET_MAX_CONE, ornt, ooff, SBR_TET_MAX_ORNT, DM_POLYTOPE_SEGMENT, 2, tetEdgeFaceLoc[e], r, (r == 0) == (P == v) ? 0 : -1));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  {
    const PetscInt mask = SBRTetSymbolFaces_Private(P) & SBRTetSymbolFaces_Private(Q);

    if (mask) {
      PetscInt f = 0;

      while (!(mask & (1 << f))) ++f;
      for (PetscInt r = 0; r < fsub[f].nseg; ++r) {
        if ((fsub[f].seg[r][0] == P && fsub[f].seg[r][1] == Q) || (fsub[f].seg[r][0] == Q && fsub[f].seg[r][1] == P)) {
          const PetscInt acp[1] = {f};

          PetscCall(SBRAppendCone_Private(cone, coff, SBR_TET_MAX_CONE, ornt, ooff, SBR_TET_MAX_ORNT, DM_POLYTOPE_SEGMENT, 1, acp, r, fsub[f].seg[r][0] == P ? 0 : -1));
          PetscFunctionReturn(PETSC_SUCCESS);
        }
      }
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Segment (%" PetscInt_FMT ", %" PetscInt_FMT ") not found in the subdivision of face %" PetscInt_FMT, P, Q, f);
    }
    for (PetscInt r = 0; r < sub->nseg; ++r) {
      if ((sub->seg[r][0] == P && sub->seg[r][1] == Q) || (sub->seg[r][0] == Q && sub->seg[r][1] == P)) {
        PetscCall(SBRAppendCone_Private(cone, coff, SBR_TET_MAX_CONE, ornt, ooff, SBR_TET_MAX_ORNT, DM_POLYTOPE_SEGMENT, 0, NULL, r, sub->seg[r][0] == P ? 0 : -1));
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    }
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Interior segment (%" PetscInt_FMT ", %" PetscInt_FMT ") not found", P, Q);
  }
}

/* Emit the cone entry for the triangle with vertex cycle C: a subtriangle of a face, or a triangle
   interior to the tetrahedron */
static PetscErrorCode SBRTetEmitTriRef_Private(const SBRTetSubdiv *sub, const SBRTriSubdiv fsub[], const PetscInt C[3], PetscInt cone[], PetscInt *coff, PetscInt ornt[], PetscInt *ooff)
{
  const PetscInt mask = SBRTetSymbolFaces_Private(C[0]) & SBRTetSymbolFaces_Private(C[1]) & SBRTetSymbolFaces_Private(C[2]);
  PetscInt       o;

  PetscFunctionBeginHot;
  if (mask) {
    PetscInt f = 0;

    while (!(mask & (1 << f))) ++f;
    for (PetscInt r = 0; r < fsub[f].ntri; ++r) {
      if (SBRTupleMatch_Private(3, fsub[f].tri[r], C)) {
        const PetscInt acp[1] = {f};

        PetscCall(DMPolytopeGetVertexOrientation(DM_POLYTOPE_TRIANGLE, fsub[f].tri[r], C, &o));
        PetscCall(SBRAppendCone_Private(cone, coff, SBR_TET_MAX_CONE, ornt, ooff, SBR_TET_MAX_ORNT, DM_POLYTOPE_TRIANGLE, 1, acp, r, o));
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    }
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Triangle (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ") not found in the subdivision of face %" PetscInt_FMT, C[0], C[1], C[2], f);
  }
  for (PetscInt r = 0; r < sub->ntri; ++r) {
    if (SBRTupleMatch_Private(3, sub->tri[r], C)) {
      PetscCall(DMPolytopeGetVertexOrientation(DM_POLYTOPE_TRIANGLE, sub->tri[r], C, &o));
      PetscCall(SBRAppendCone_Private(cone, coff, SBR_TET_MAX_CONE, ornt, ooff, SBR_TET_MAX_ORNT, DM_POLYTOPE_TRIANGLE, 0, NULL, r, o));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Interior triangle (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ") not found", C[0], C[1], C[2]);
}

/* Emit the cone and orientation data for the subdivision of a tetrahedron */
static PetscErrorCode SBRTetEmit_Private(const PetscInt ord[], PetscInt na, const SBRTetSubdiv *sub, PetscInt *Nt, DMPolytopeType target[], PetscInt size[], PetscInt cone[], PetscInt *Ncone, PetscInt ornt[], PetscInt *Nornt)
{
  SBRTriSubdiv fsub[4];
  PetscInt     coff = 0, ooff = 0, n = 0;

  PetscFunctionBegin;
  for (PetscInt f = 0; f < 4; ++f) PetscCall(SBRTetFaceSubdiv_Private(ord, na, f, &fsub[f]));
  for (PetscInt s = 0; s < sub->nseg; ++s) {
    for (PetscInt i = 0; i < 2; ++i) {
      const PetscInt sym = sub->seg[s][i];

      if (sym < 4) PetscCall(SBRAppendCone_Private(cone, &coff, SBR_TET_MAX_CONE, ornt, &ooff, SBR_TET_MAX_ORNT, DM_POLYTOPE_POINT, 3, tetVertPath[sym], 0, 0));
      else PetscCall(SBRAppendCone_Private(cone, &coff, SBR_TET_MAX_CONE, ornt, &ooff, SBR_TET_MAX_ORNT, DM_POLYTOPE_POINT, 2, tetEdgeFaceLoc[sym - 4], 0, 0));
    }
  }
  for (PetscInt t = 0; t < sub->ntri; ++t)
    for (PetscInt i = 0; i < 3; ++i) PetscCall(SBRTetEmitSegRef_Private(sub, fsub, sub->tri[t][i], sub->tri[t][(i + 1) % 3], cone, &coff, ornt, &ooff));
  for (PetscInt l = 0; l < sub->nleaf; ++l) {
    for (PetscInt f = 0; f < 4; ++f) {
      /* The face f of the child tetrahedron is the face opposite its vertex 3 - f */
      const PetscInt mp = 3 - f;
      PetscInt       C[3];

      for (PetscInt i = 0; i < 3; ++i) C[i] = sub->leaf[l][tetOppFace[mp][i]];
      PetscCall(SBRTetEmitTriRef_Private(sub, fsub, C, cone, &coff, ornt, &ooff));
    }
  }
  if (sub->nseg) {
    target[n] = DM_POLYTOPE_SEGMENT;
    size[n]   = sub->nseg;
    ++n;
  }
  target[n] = DM_POLYTOPE_TRIANGLE;
  size[n]   = sub->ntri;
  ++n;
  target[n] = DM_POLYTOPE_TETRAHEDRON;
  size[n]   = sub->nleaf;
  ++n;
  *Nt    = n;
  *Ncone = coff;
  *Nornt = ooff;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Get the subdivision data for the encoded bisection order, generating and caching it on first use */
static PetscErrorCode SBRGetTetSplit_Private(DMPlexTransform tr, PetscInt code, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  DMPlexRefine_SBR *sbr = (DMPlexRefine_SBR *)tr->data;
  PetscInt          i;

  PetscFunctionBeginHot;
  for (i = 0; i < sbr->Ncache; ++i)
    if (sbr->cacheCode[i] == code) break;
  if (i == sbr->Ncache) {
    SBRTetSubdiv   sub;
    DMPolytopeType targetTmp[3];
    PetscInt       sizeTmp[3], coneTmp[SBR_TET_MAX_CONE], orntTmp[SBR_TET_MAX_ORNT];
    PetscInt       ord[6], na, NtTmp, Ncone, Nornt;

    SBRTetDecodeOrder_Private(code, ord, &na);
    PetscCall(SBRTetSubdivide_Private(ord, na, &sub));
    PetscCall(SBRTetEmit_Private(ord, na, &sub, &NtTmp, targetTmp, sizeTmp, coneTmp, &Ncone, orntTmp, &Nornt));
    if (sbr->Ncache == sbr->maxCache) {
      const PetscInt   newmax = PetscMax(2 * sbr->maxCache, 16);
      PetscInt        *newCode, *newNt, **newSize, **newCone, **newOrnt;
      DMPolytopeType **newTarget;

      PetscCall(PetscMalloc6(newmax, &newCode, newmax, &newNt, newmax, &newTarget, newmax, &newSize, newmax, &newCone, newmax, &newOrnt));
      PetscCall(PetscArraycpy(newCode, sbr->cacheCode, sbr->Ncache));
      PetscCall(PetscArraycpy(newNt, sbr->cacheNt, sbr->Ncache));
      PetscCall(PetscArraycpy(newTarget, sbr->cacheTarget, sbr->Ncache));
      PetscCall(PetscArraycpy(newSize, sbr->cacheSize, sbr->Ncache));
      PetscCall(PetscArraycpy(newCone, sbr->cacheCone, sbr->Ncache));
      PetscCall(PetscArraycpy(newOrnt, sbr->cacheOrnt, sbr->Ncache));
      PetscCall(PetscFree6(sbr->cacheCode, sbr->cacheNt, sbr->cacheTarget, sbr->cacheSize, sbr->cacheCone, sbr->cacheOrnt));
      sbr->cacheCode   = newCode;
      sbr->cacheNt     = newNt;
      sbr->cacheTarget = newTarget;
      sbr->cacheSize   = newSize;
      sbr->cacheCone   = newCone;
      sbr->cacheOrnt   = newOrnt;
      sbr->maxCache    = newmax;
    }
    PetscCall(PetscMalloc1(NtTmp, &sbr->cacheTarget[i]));
    PetscCall(PetscMalloc1(NtTmp, &sbr->cacheSize[i]));
    PetscCall(PetscMalloc1(Ncone, &sbr->cacheCone[i]));
    PetscCall(PetscMalloc1(Nornt, &sbr->cacheOrnt[i]));
    PetscCall(PetscArraycpy(sbr->cacheTarget[i], targetTmp, NtTmp));
    PetscCall(PetscArraycpy(sbr->cacheSize[i], sizeTmp, NtTmp));
    PetscCall(PetscArraycpy(sbr->cacheCone[i], coneTmp, Ncone));
    PetscCall(PetscArraycpy(sbr->cacheOrnt[i], orntTmp, Nornt));
    sbr->cacheCode[i] = code;
    sbr->cacheNt[i]   = NtTmp;
    ++sbr->Ncache;
  }
  *Nt     = sbr->cacheNt[i];
  *target = sbr->cacheTarget[i];
  *size   = sbr->cacheSize[i];
  *cone   = sbr->cacheCone[i];
  *ornt   = sbr->cacheOrnt[i];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Map the subcell (r, o) of a split triangle, authored against arrangement so of the triangle, to
  the matching subcell of the triangle's own production: both subdivisions are reconstructed
  symbolically, the authored subcell is relabeled through the arrangement, and matched against the
  actual subcells
*/
static PetscErrorCode SBRTriangleOrient_Private(PetscInt rt, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  const PetscInt *arr = DMPolytopeTypeGetArrangement(DM_POLYTOPE_TRIANGLE, so);
  SBRTriSubdiv    subT, subV;
  const PetscInt *tupleV, *canonT;
  PetscInt        inv[3], tuple[3], n, nlist, nsplit, first, second = -1, r2, dO;

  PetscFunctionBeginHot;
  switch (rt) {
  case RT_TRIANGLE_SPLIT_0:
  case RT_TRIANGLE_SPLIT_1:
  case RT_TRIANGLE_SPLIT_2:
    nsplit = 1;
    first  = rt - RT_TRIANGLE_SPLIT_0;
    break;
  case RT_TRIANGLE_SPLIT_FAN_0:
  case RT_TRIANGLE_SPLIT_FAN_1:
  case RT_TRIANGLE_SPLIT_FAN_2:
    nsplit = 3;
    first  = rt - RT_TRIANGLE_SPLIT_FAN_0;
    break;
  case RT_TRIANGLE_SPLIT_01:
  case RT_TRIANGLE_SPLIT_10:
  case RT_TRIANGLE_SPLIT_12:
  case RT_TRIANGLE_SPLIT_21:
  case RT_TRIANGLE_SPLIT_20:
  case RT_TRIANGLE_SPLIT_02: {
    static const PetscInt pairs[6][2] = {
      {0, 1},
      {1, 0},
      {1, 2},
      {2, 1},
      {2, 0},
      {0, 2}
    };

    nsplit = 2;
    first  = pairs[rt - RT_TRIANGLE_SPLIT_01][0];
    second = pairs[rt - RT_TRIANGLE_SPLIT_01][1];
  } break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid refinement type %" PetscInt_FMT, rt);
  }
  for (PetscInt l = 0; l < 3; ++l) inv[arr[l * 2]] = l;
  PetscCall(SBRTriangleSubdiv_Private(nsplit, first, second, &subT));
  PetscCall(SBRTriangleSubdiv_Private(nsplit, inv[first], second < 0 ? -1 : inv[second], &subV));
  if (tct == DM_POLYTOPE_SEGMENT) {
    n      = 2;
    nlist  = subT.nseg;
    tupleV = subV.seg[r];
  } else {
    n      = 3;
    nlist  = subT.ntri;
    tupleV = subV.tri[r];
  }
  for (PetscInt i = 0; i < n; ++i) tuple[i] = SBRTriRelabel_Private(so, tupleV[i]);
  for (r2 = 0; r2 < nlist; ++r2) {
    canonT = tct == DM_POLYTOPE_SEGMENT ? subT.seg[r2] : subT.tri[r2];
    if (SBRTupleMatch_Private(n, canonT, tuple)) break;
  }
  PetscCheck(r2 < nlist, PETSC_COMM_SELF, PETSC_ERR_PLIB, "No matching %s subcell for replica %" PetscInt_FMT " at orientation %" PetscInt_FMT, DMPolytopeTypes[tct], r, so);
  PetscCall(DMPolytopeGetVertexOrientation(tct, canonT, tuple, &dO));
  *rnew = r2;
  *onew = DMPolytopeTypeComposeOrientation(tct, o, dO);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The images of the tetrahedron edges under the vertex permutation varr */
static void SBRTetEdgeImage_Private(const PetscInt varr[], PetscInt medge[6])
{
  for (PetscInt e = 0; e < 6; ++e) medge[e] = SBREdgeIndexFromVerts_Private(varr[tetEdgeVert[e][0]], varr[tetEdgeVert[e][1]]);
}

/* The tetrahedron analogue of SBRTriangleOrient_Private() */
static PetscErrorCode SBRTetOrient_Private(PetscInt code, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  const PetscInt *varr = DMPolytopeTypeGetVertexArrangement(DM_POLYTOPE_TETRAHEDRON, so);
  SBRTetSubdiv    subT, subV;
  const PetscInt *tupleV, *canonT = NULL;
  PetscInt        medge[6], minv[6], ordT[6], ordV[6], na, tuple[4], n, nlist, r2, dO;

  PetscFunctionBeginHot;
  SBRTetDecodeOrder_Private(code, ordT, &na);
  SBRTetEdgeImage_Private(varr, medge);
  for (PetscInt e = 0; e < 6; ++e) minv[medge[e]] = e;
  for (PetscInt i = 0; i < na; ++i) ordV[i] = minv[ordT[i]];
  PetscCall(SBRTetSubdivide_Private(ordT, na, &subT));
  PetscCall(SBRTetSubdivide_Private(ordV, na, &subV));
  switch (tct) {
  case DM_POLYTOPE_SEGMENT:
    n      = 2;
    nlist  = subT.nseg;
    tupleV = subV.seg[r];
    break;
  case DM_POLYTOPE_TRIANGLE:
    n      = 3;
    nlist  = subT.ntri;
    tupleV = subV.tri[r];
    break;
  case DM_POLYTOPE_TETRAHEDRON:
    n      = 4;
    nlist  = subT.nleaf;
    tupleV = subV.leaf[r];
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid target type %s", DMPolytopeTypes[tct]);
  }
  for (PetscInt i = 0; i < n; ++i) tuple[i] = tupleV[i] < 4 ? varr[tupleV[i]] : 4 + medge[tupleV[i] - 4];
  for (r2 = 0; r2 < nlist; ++r2) {
    canonT = tct == DM_POLYTOPE_SEGMENT ? subT.seg[r2] : (tct == DM_POLYTOPE_TRIANGLE ? subT.tri[r2] : subT.leaf[r2]);
    if (SBRTupleMatch_Private(n, canonT, tuple)) break;
  }
  PetscCheck(r2 < nlist, PETSC_COMM_SELF, PETSC_ERR_PLIB, "No matching %s subcell for replica %" PetscInt_FMT " at orientation %" PetscInt_FMT, DMPolytopeTypes[tct], r, so);
  PetscCall(DMPolytopeGetVertexOrientation(tct, canonT, tuple, &dO));
  *rnew = r2;
  *onew = DMPolytopeTypeComposeOrientation(tct, o, dO);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Validate the subdivision generator against the enumeration of Plaza & Carey (2000): over all
  possible edge length orders and all conforming sets of marked edges, there are exactly 51
  subdivision classes up to rotation, distributed over the number of marked edges na = 1..6 as
  1, 3, 9, 17, 15, 6 (Table 4 of the paper). This runs the generator on every configuration, which
  also exercises all internal consistency checks in the emission of the subdivision data.
*/
static PetscErrorCode DMPlexTransformSBRValidate_Private(DMPlexTransform tr)
{
  const PetscInt expected[6] = {1, 3, 9, 17, 15, 6};
  const PetscInt fact[6]     = {120, 24, 6, 2, 1, 1};
  const PetscInt keyLen      = 2 + 4 * SBR_TET_MAX_LEAVES;
  PetscBT        btCode;
  PetscInt      *codes, *keys, *nas, counts[6] = {0, 0, 0, 0, 0, 0};
  PetscInt       Ncodes = 0, maxCodes = 4096;

  PetscFunctionBegin;
  PetscCall(PetscBTCreate(117649, &btCode)); /* 7^6 possible codes */
  PetscCall(PetscMalloc3(maxCodes, &codes, maxCodes * keyLen, &keys, maxCodes, &nas));
  for (PetscInt idx = 0; idx < 720; ++idx) {
    PetscInt avail[6] = {0, 1, 2, 3, 4, 5}, perm[6], prio[6], k = idx;

    for (PetscInt i = 0; i < 6; ++i) {
      const PetscInt pos = k / fact[i];

      k %= fact[i];
      perm[i] = avail[pos];
      for (PetscInt j = pos; j < 5 - i; ++j) avail[j] = avail[j + 1];
    }
    for (PetscInt i = 0; i < 6; ++i) prio[perm[i]] = i;
    for (PetscInt mask = 1; mask < 64; ++mask) {
      PetscInt  ord[6], na = 0, code = 0;
      PetscBool stable = (mask & (1 << perm[0])) ? PETSC_TRUE : PETSC_FALSE;

      /* The set is conforming when the longest edge overall and of each affected face is marked */
      for (PetscInt f = 0; f < 4 && stable; ++f) {
        PetscInt fmax = tetFaceEdge[f][0], nm = 0;

        for (PetscInt l = 0; l < 3; ++l) {
          if (mask & (1 << tetFaceEdge[f][l])) ++nm;
          if (prio[tetFaceEdge[f][l]] < prio[fmax]) fmax = tetFaceEdge[f][l];
        }
        if (nm && !(mask & (1 << fmax))) stable = PETSC_FALSE;
      }
      if (!stable) continue;
      for (PetscInt i = 0; i < 6; ++i)
        if (mask & (1 << perm[i])) ord[na++] = perm[i];
      for (PetscInt i = na - 1; i >= 0; --i) code = code * 7 + (ord[i] + 1);
      if (!PetscBTLookupSet(btCode, code)) {
        PetscCheck(Ncodes < maxCodes, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Too many configurations");
        codes[Ncodes] = code;
        nas[Ncodes]   = na;
        ++Ncodes;
      }
    }
  }
  for (PetscInt c = 0; c < Ncodes; ++c) {
    SBRTetSubdiv   sub;
    DMPolytopeType targetTmp[3];
    PetscInt       sizeTmp[3], coneTmp[SBR_TET_MAX_CONE], orntTmp[SBR_TET_MAX_ORNT];
    PetscInt       ord[6], na, NtTmp, Ncone, Nornt;
    PetscInt      *key = &keys[c * keyLen];

    SBRTetDecodeOrder_Private(codes[c], ord, &na);
    PetscCall(SBRTetSubdivide_Private(ord, na, &sub));
    PetscCall(SBRTetEmit_Private(ord, na, &sub, &NtTmp, targetTmp, sizeTmp, coneTmp, &Ncone, orntTmp, &Nornt));
    /* The class key is the smallest relabeling, over all 12 rotations, of the leaf set together
       with the first bisected edge: the paper counts length configurations, which distinguish two
       marked sets producing the same subdivision when their longest edges do not correspond */
    for (PetscInt so = 0; so < 12; ++so) {
      const PetscInt *varr = DMPolytopeTypeGetVertexArrangement(DM_POLYTOPE_TETRAHEDRON, so);
      PetscInt        medge[6], cand[2 + 4 * SBR_TET_MAX_LEAVES];

      SBRTetEdgeImage_Private(varr, medge);
      for (PetscInt i = 0; i < keyLen; ++i) cand[i] = -1;
      cand[0] = sub.nleaf;
      cand[1] = medge[ord[0]];
      for (PetscInt l = 0; l < sub.nleaf; ++l) {
        PetscInt *t = &cand[2 + 4 * l];

        for (PetscInt i = 0; i < 4; ++i) t[i] = sub.leaf[l][i] < 4 ? varr[sub.leaf[l][i]] : 4 + medge[sub.leaf[l][i] - 4];
        PetscCall(PetscSortInt(4, t));
      }
      /* Insertion sort of the leaf records, then keep the smallest candidate */
      for (PetscInt l = 1; l < sub.nleaf; ++l) {
        PetscInt t[4], j;

        PetscCall(PetscArraycpy(t, &cand[2 + 4 * l], 4));
        for (j = l - 1; j >= 0 && SBRTupleCompare_Private(4, &cand[2 + 4 * j], t) > 0; --j) PetscCall(PetscArraycpy(&cand[2 + 4 * (j + 1)], &cand[2 + 4 * j], 4));
        PetscCall(PetscArraycpy(&cand[2 + 4 * (j + 1)], t, 4));
      }
      if (so == 0 || SBRTupleCompare_Private(keyLen, cand, key) < 0) PetscCall(PetscArraycpy(key, cand, keyLen));
    }
  }
  for (PetscInt c = 0; c < Ncodes; ++c) {
    PetscInt d;

    for (d = 0; d < c; ++d) {
      PetscBool same;

      if (nas[d] != nas[c]) continue;
      PetscCall(PetscMemcmp(&keys[d * keyLen], &keys[c * keyLen], keyLen * sizeof(PetscInt), &same));
      if (same) break;
    }
    if (d == c) ++counts[nas[c] - 1];
  }
  for (PetscInt na = 1; na <= 6; ++na)
    PetscCheck(counts[na - 1] == expected[na - 1], PetscObjectComm((PetscObject)tr), PETSC_ERR_PLIB, "Found %" PetscInt_FMT " subdivision classes with %" PetscInt_FMT " marked edges, expected %" PetscInt_FMT, counts[na - 1], na, expected[na - 1]);
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tr), "SBR: validated %" PetscInt_FMT " conforming configurations covering 51 subdivision classes (1 3 9 17 15 6 for 1-6 bisected edges)\n", Ncodes));
  PetscCall(PetscFree3(codes, keys, nas));
  PetscCall(PetscBTDestroy(&btCode));
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
  case RT_TRIANGLE_SPLIT_0:
  case RT_TRIANGLE_SPLIT_1:
  case RT_TRIANGLE_SPLIT_2:
  case RT_TRIANGLE_SPLIT_FAN_0:
  case RT_TRIANGLE_SPLIT_FAN_1:
  case RT_TRIANGLE_SPLIT_FAN_2:
    if (so) PetscCall(SBRTriangleOrient_Private(rt, so, tct, r, o, rnew, onew));
    break;
  case RT_EDGE_SPLIT:
  case RT_TRIANGLE_SPLIT:
    PetscCall(DMPlexTransformGetSubcellOrientation_Regular(tr, sct, sp, so, tct, r, o, rnew, onew));
    break;
  default:
    if (rt >= RT_TET_SPLIT_BASE) {
      if (so) PetscCall(SBRTetOrient_Private(rt - RT_TET_SPLIT_BASE, so, tct, r, o, rnew, onew));
    } else PetscCall(DMPlexTransformGetSubcellOrientationIdentity(tr, sct, sp, so, tct, r, o, rnew, onew));
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

/* Add 3 edges inside this triangle, making 4 new triangles fanned around the midpoint of the
   first-bisected edge (the 4T partition of Rivara), e.g. for first = 1:
 2
 |\
 | \
 2   1
 | \  \
 |  \  \
 |   \  \
 2 D  \ B 1
 |\    \  |\
 | \  C \ | 0
 |  2--__\|A \
 0-----0-----1
   This subdivision is used for a fully marked triangle below a 3D cell, where it matches the
   subdivision induced on the triangle by bisecting the cells above it (Lemma 1.2 of Plaza & Carey
   (2000)); a fully marked maximal (2D) triangle is subdivided regularly instead. */
static PetscErrorCode SBRGetTriangleSplitFan_Private(PetscInt first, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  static DMPolytopeType fanT[] = {DM_POLYTOPE_SEGMENT, DM_POLYTOPE_TRIANGLE};
  static PetscInt       fanS[] = {3, 4};
  static PetscInt       fanC[96];
  static PetscInt       fanO[24];
  SBRTriSubdiv          sub;
  PetscInt              coff = 0, ooff = 0;

  PetscFunctionBeginHot;
  PetscCall(SBRTriangleSubdiv_Private(3, first, -1, &sub));
  for (PetscInt s = 0; s < sub.nseg; ++s) {
    for (PetscInt i = 0; i < 2; ++i) {
      const PetscInt sym = sub.seg[s][i];

      if (sym < 3) {
        const PetscInt acp[2] = {sym, 0}; /* vertex v is the first vertex of edge v */

        PetscCall(SBRAppendCone_Private(fanC, &coff, 96, fanO, &ooff, 24, DM_POLYTOPE_POINT, 2, acp, 0, 0));
      } else {
        const PetscInt acp[1] = {sym - 3};

        PetscCall(SBRAppendCone_Private(fanC, &coff, 96, fanO, &ooff, 24, DM_POLYTOPE_POINT, 1, acp, 0, 0));
      }
    }
  }
  for (PetscInt t = 0; t < sub.ntri; ++t) {
    for (PetscInt i = 0; i < 3; ++i) {
      const PetscInt P = sub.tri[t][i], Q = sub.tri[t][(i + 1) % 3];
      const PetscInt v = P < 3 ? P : Q, e = (P < 3 ? Q : P) - 3;
      PetscInt       r;

      if ((P < 3 || Q < 3) && (triEdgeVert[e][0] == v || triEdgeVert[e][1] == v)) {
        /* Half of a split edge: replica 0 runs from the edge tail to the midpoint, replica 1 from the midpoint to the head */
        const PetscInt acp[1] = {e};

        r = triEdgeVert[e][0] == v ? 0 : 1;
        PetscCall(SBRAppendCone_Private(fanC, &coff, 96, fanO, &ooff, 24, DM_POLYTOPE_SEGMENT, 1, acp, r, (r == 0) == (P == v) ? 0 : -1));
      } else {
        for (r = 0; r < sub.nseg; ++r)
          if ((sub.seg[r][0] == P && sub.seg[r][1] == Q) || (sub.seg[r][0] == Q && sub.seg[r][1] == P)) break;
        PetscCheck(r < sub.nseg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Interior segment (%" PetscInt_FMT ", %" PetscInt_FMT ") not found", P, Q);
        PetscCall(SBRAppendCone_Private(fanC, &coff, 96, fanO, &ooff, 24, DM_POLYTOPE_SEGMENT, 0, NULL, r, sub.seg[r][0] == P ? 0 : -1));
      }
    }
  }
  *Nt     = 2;
  *target = fanT;
  *size   = fanS;
  *cone   = fanC;
  *ornt   = fanO;
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
    case RT_TRIANGLE_SPLIT_FAN_0:
    case RT_TRIANGLE_SPLIT_FAN_1:
    case RT_TRIANGLE_SPLIT_FAN_2:
      PetscCall(SBRGetTriangleSplitFan_Private(val - RT_TRIANGLE_SPLIT_FAN_0, Nt, target, size, cone, ornt));
      break;
    case RT_TRIANGLE_SPLIT:
      PetscCall(DMPlexTransformCellRefine_Regular(tr, source, p, NULL, Nt, target, size, cone, ornt));
      break;
    default:
      PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt));
    }
    break;
  case DM_POLYTOPE_TETRAHEDRON:
    if (val == RT_TET) PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt));
    else {
      PetscCheck(val >= RT_TET_SPLIT_BASE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid refinement type %" PetscInt_FMT " for tetrahedron %" PetscInt_FMT, val, p);
      PetscCall(SBRGetTetSplit_Private(tr, val - RT_TET_SPLIT_BASE, Nt, target, size, cone, ornt));
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
  PetscBool flg, validate = PETSC_FALSE;

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
  PetscCall(PetscOptionsBool("-dm_plex_transform_sbr_validate", "Validate the tetrahedron subdivision generator against Plaza & Carey (2000), Table 4", "", validate, &validate, NULL));
  PetscOptionsHeadEnd();
  if (validate) PetscCall(DMPlexTransformSBRValidate_Private(tr));
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
  for (PetscInt i = 0; i < sbr->Ncache; ++i) {
    PetscCall(PetscFree(sbr->cacheTarget[i]));
    PetscCall(PetscFree(sbr->cacheSize[i]));
    PetscCall(PetscFree(sbr->cacheCone[i]));
    PetscCall(PetscFree(sbr->cacheOrnt[i]));
  }
  PetscCall(PetscFree6(sbr->cacheCode, sbr->cacheNt, sbr->cacheTarget, sbr->cacheSize, sbr->cacheCone, sbr->cacheOrnt));
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
