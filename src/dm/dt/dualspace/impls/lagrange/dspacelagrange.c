#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/
#include <petscdmplex.h>
#include <petscblaslapack.h>

PetscErrorCode DMPlexGetTransitiveClosure_Internal(DM, PetscInt, PetscInt, PetscBool, PetscInt *, PetscInt *[]);

struct _n_Petsc1DNodeFamily
{
  PetscInt         refct;
  PetscDTNodeType  nodeFamily;
  PetscReal        gaussJacobiExp;
  PetscInt         nComputed;
  PetscReal      **nodesets;
  PetscBool        endpoints;
};

/* users set node families for PETSCDUALSPACELAGRANGE with just the inputs to this function, but internally we create
 * an object that can cache the computations across multiple dual spaces */
static PetscErrorCode Petsc1DNodeFamilyCreate(PetscDTNodeType family, PetscReal gaussJacobiExp, PetscBool endpoints, Petsc1DNodeFamily *nf)
{
  Petsc1DNodeFamily f;

  PetscFunctionBegin;
  PetscCall(PetscNew(&f));
  switch (family) {
  case PETSCDTNODES_GAUSSJACOBI:
  case PETSCDTNODES_EQUISPACED:
    f->nodeFamily = family;
    break;
  default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unknown 1D node family");
  }
  f->endpoints = endpoints;
  f->gaussJacobiExp = 0.;
  if (family == PETSCDTNODES_GAUSSJACOBI) {
    PetscCheck(gaussJacobiExp > -1.,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Gauss-Jacobi exponent must be > -1.");
    f->gaussJacobiExp = gaussJacobiExp;
  }
  f->refct = 1;
  *nf = f;
  PetscFunctionReturn(0);
}

static PetscErrorCode Petsc1DNodeFamilyReference(Petsc1DNodeFamily nf)
{
  PetscFunctionBegin;
  if (nf) nf->refct++;
  PetscFunctionReturn(0);
}

static PetscErrorCode Petsc1DNodeFamilyDestroy(Petsc1DNodeFamily *nf)
{
  PetscInt       i, nc;

  PetscFunctionBegin;
  if (!(*nf)) PetscFunctionReturn(0);
  if (--(*nf)->refct > 0) {
    *nf = NULL;
    PetscFunctionReturn(0);
  }
  nc = (*nf)->nComputed;
  for (i = 0; i < nc; i++) {
    PetscCall(PetscFree((*nf)->nodesets[i]));
  }
  PetscCall(PetscFree((*nf)->nodesets));
  PetscCall(PetscFree(*nf));
  *nf = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode Petsc1DNodeFamilyGetNodeSets(Petsc1DNodeFamily f, PetscInt degree, PetscReal ***nodesets)
{
  PetscInt       nc;

  PetscFunctionBegin;
  nc = f->nComputed;
  if (degree >= nc) {
    PetscInt    i, j;
    PetscReal **new_nodesets;
    PetscReal  *w;

    PetscCall(PetscMalloc1(degree + 1, &new_nodesets));
    PetscCall(PetscArraycpy(new_nodesets, f->nodesets, nc));
    PetscCall(PetscFree(f->nodesets));
    f->nodesets = new_nodesets;
    PetscCall(PetscMalloc1(degree + 1, &w));
    for (i = nc; i < degree + 1; i++) {
      PetscCall(PetscMalloc1(i + 1, &(f->nodesets[i])));
      if (!i) {
        f->nodesets[i][0] = 0.5;
      } else {
        switch (f->nodeFamily) {
        case PETSCDTNODES_EQUISPACED:
          if (f->endpoints) {
            for (j = 0; j <= i; j++) f->nodesets[i][j] = (PetscReal) j / (PetscReal) i;
          } else {
            /* these nodes are at the centroids of the small simplices created by the equispaced nodes that include
             * the endpoints */
            for (j = 0; j <= i; j++) f->nodesets[i][j] = ((PetscReal) j + 0.5) / ((PetscReal) i + 1.);
          }
          break;
        case PETSCDTNODES_GAUSSJACOBI:
          if (f->endpoints) {
            PetscCall(PetscDTGaussLobattoJacobiQuadrature(i + 1, 0., 1., f->gaussJacobiExp, f->gaussJacobiExp, f->nodesets[i], w));
          } else {
            PetscCall(PetscDTGaussJacobiQuadrature(i + 1, 0., 1., f->gaussJacobiExp, f->gaussJacobiExp, f->nodesets[i], w));
          }
          break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unknown 1D node family");
        }
      }
    }
    PetscCall(PetscFree(w));
    f->nComputed = degree + 1;
  }
  *nodesets = f->nodesets;
  PetscFunctionReturn(0);
}

/* http://arxiv.org/abs/2002.09421 for details */
static PetscErrorCode PetscNodeRecursive_Internal(PetscInt dim, PetscInt degree, PetscReal **nodesets, PetscInt tup[], PetscReal node[])
{
  PetscReal w;
  PetscInt i, j;

  PetscFunctionBeginHot;
  w = 0.;
  if (dim == 1) {
    node[0] = nodesets[degree][tup[0]];
    node[1] = nodesets[degree][tup[1]];
  } else {
    for (i = 0; i < dim + 1; i++) node[i] = 0.;
    for (i = 0; i < dim + 1; i++) {
      PetscReal wi = nodesets[degree][degree-tup[i]];

      for (j = 0; j < dim+1; j++) tup[dim+1+j] = tup[j+(j>=i)];
      PetscCall(PetscNodeRecursive_Internal(dim-1,degree-tup[i],nodesets,&tup[dim+1],&node[dim+1]));
      for (j = 0; j < dim+1; j++) node[j+(j>=i)] += wi * node[dim+1+j];
      w += wi;
    }
    for (i = 0; i < dim+1; i++) node[i] /= w;
  }
  PetscFunctionReturn(0);
}

/* compute simplex nodes for the biunit simplex from the 1D node family */
static PetscErrorCode Petsc1DNodeFamilyComputeSimplexNodes(Petsc1DNodeFamily f, PetscInt dim, PetscInt degree, PetscReal points[])
{
  PetscInt      *tup;
  PetscInt       k;
  PetscInt       npoints;
  PetscReal    **nodesets = NULL;
  PetscInt       worksize;
  PetscReal     *nodework;
  PetscInt      *tupwork;

  PetscFunctionBegin;
  PetscCheck(dim >= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Must have non-negative dimension");
  PetscCheck(degree >= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Must have non-negative degree");
  if (!dim) PetscFunctionReturn(0);
  PetscCall(PetscCalloc1(dim+2, &tup));
  k = 0;
  PetscCall(PetscDTBinomialInt(degree + dim, dim, &npoints));
  PetscCall(Petsc1DNodeFamilyGetNodeSets(f, degree, &nodesets));
  worksize = ((dim + 2) * (dim + 3)) / 2;
  PetscCall(PetscMalloc2(worksize, &nodework, worksize, &tupwork));
  /* loop over the tuples of length dim with sum at most degree */
  for (k = 0; k < npoints; k++) {
    PetscInt i;

    /* turn thm into tuples of length dim + 1 with sum equal to degree (barycentric indice) */
    tup[0] = degree;
    for (i = 0; i < dim; i++) {
      tup[0] -= tup[i+1];
    }
    switch(f->nodeFamily) {
    case PETSCDTNODES_EQUISPACED:
      /* compute equispaces nodes on the unit reference triangle */
      if (f->endpoints) {
        for (i = 0; i < dim; i++) {
          points[dim*k + i] = (PetscReal) tup[i+1] / (PetscReal) degree;
        }
      } else {
        for (i = 0; i < dim; i++) {
          /* these nodes are at the centroids of the small simplices created by the equispaced nodes that include
           * the endpoints */
          points[dim*k + i] = ((PetscReal) tup[i+1] + 1./(dim+1.)) / (PetscReal) (degree + 1.);
        }
      }
      break;
    default:
      /* compute equispaces nodes on the barycentric reference triangle (the trace on the first dim dimensions are the
       * unit reference triangle nodes */
      for (i = 0; i < dim + 1; i++) tupwork[i] = tup[i];
      PetscCall(PetscNodeRecursive_Internal(dim, degree, nodesets, tupwork, nodework));
      for (i = 0; i < dim; i++) points[dim*k + i] = nodework[i + 1];
      break;
    }
    PetscCall(PetscDualSpaceLatticePointLexicographic_Internal(dim, degree, &tup[1]));
  }
  /* map from unit simplex to biunit simplex */
  for (k = 0; k < npoints * dim; k++) points[k] = points[k] * 2. - 1.;
  PetscCall(PetscFree2(nodework, tupwork));
  PetscCall(PetscFree(tup));
  PetscFunctionReturn(0);
}

/* If we need to get the dofs from a mesh point, or add values into dofs at a mesh point, and there is more than one dof
 * on that mesh point, we have to be careful about getting/adding everything in the right place.
 *
 * With nodal dofs like PETSCDUALSPACELAGRANGE makes, the general approach to calculate the value of dofs associate
 * with a node A is
 * - transform the node locations x(A) by the map that takes the mesh point to its reorientation, x' = phi(x(A))
 * - figure out which node was originally at the location of the transformed point, A' = idx(x')
 * - if the dofs are not scalars, figure out how to represent the transformed dofs in terms of the basis
 *   of dofs at A' (using pushforward/pullback rules)
 *
 * The one sticky point with this approach is the "A' = idx(x')" step: trying to go from real valued coordinates
 * back to indices.  I don't want to rely on floating point tolerances.  Additionally, PETSCDUALSPACELAGRANGE may
 * eventually support quasi-Lagrangian dofs, which could involve quadrature at multiple points, so the location "x(A)"
 * would be ambiguous.
 *
 * So each dof gets an integer value coordinate (nodeIdx in the structure below).  The choice of integer coordinates
 * is somewhat arbitrary, as long as all of the relevant symmetries of the mesh point correspond to *permutations* of
 * the integer coordinates, which do not depend on numerical precision.
 *
 * So
 *
 * - DMPlexGetTransitiveClosure_Internal() tells me how an orientation turns into a permutation of the vertices of a
 *   mesh point
 * - The permutation of the vertices, and the nodeIdx values assigned to them, tells what permutation in index space
 *   is associated with the orientation
 * - I uses that permutation to get xi' = phi(xi(A)), the integer coordinate of the transformed dof
 * - I can without numerical issues compute A' = idx(xi')
 *
 * Here are some examples of how the process works
 *
 * - With a triangle:
 *
 *   The triangle has the following integer coordinates for vertices, taken from the barycentric triangle
 *
 *     closure order 2
 *     nodeIdx (0,0,1)
 *      \
 *       +
 *       |\
 *       | \
 *       |  \
 *       |   \    closure order 1
 *       |    \ / nodeIdx (0,1,0)
 *       +-----+
 *        \
 *      closure order 0
 *      nodeIdx (1,0,0)
 *
 *   If I do DMPlexGetTransitiveClosure_Internal() with orientation 1, the vertices would appear
 *   in the order (1, 2, 0)
 *
 *   If I list the nodeIdx of each vertex in closure order for orientation 0 (0, 1, 2) and orientation 1 (1, 2, 0), I
 *   see
 *
 *   orientation 0  | orientation 1
 *
 *   [0] (1,0,0)      [1] (0,1,0)
 *   [1] (0,1,0)      [2] (0,0,1)
 *   [2] (0,0,1)      [0] (1,0,0)
 *          A                B
 *
 *   In other words, B is the result of a row permutation of A.  But, there is also
 *   a column permutation that accomplishes the same result, (2,0,1).
 *
 *   So if a dof has nodeIdx coordinate (a,b,c), after the transformation its nodeIdx coordinate
 *   is (c,a,b), and the transformed degree of freedom will be a linear combination of dofs
 *   that originally had coordinate (c,a,b).
 *
 * - With a quadrilateral:
 *
 *   The quadrilateral has the following integer coordinates for vertices, taken from concatenating barycentric
 *   coordinates for two segments:
 *
 *     closure order 3      closure order 2
 *     nodeIdx (1,0,0,1)    nodeIdx (0,1,0,1)
 *                   \      /
 *                    +----+
 *                    |    |
 *                    |    |
 *                    +----+
 *                   /      \
 *     closure order 0      closure order 1
 *     nodeIdx (1,0,1,0)    nodeIdx (0,1,1,0)
 *
 *   If I do DMPlexGetTransitiveClosure_Internal() with orientation 1, the vertices would appear
 *   in the order (1, 2, 3, 0)
 *
 *   If I list the nodeIdx of each vertex in closure order for orientation 0 (0, 1, 2, 3) and
 *   orientation 1 (1, 2, 3, 0), I see
 *
 *   orientation 0  | orientation 1
 *
 *   [0] (1,0,1,0)    [1] (0,1,1,0)
 *   [1] (0,1,1,0)    [2] (0,1,0,1)
 *   [2] (0,1,0,1)    [3] (1,0,0,1)
 *   [3] (1,0,0,1)    [0] (1,0,1,0)
 *          A                B
 *
 *   The column permutation that accomplishes the same result is (3,2,0,1).
 *
 *   So if a dof has nodeIdx coordinate (a,b,c,d), after the transformation its nodeIdx coordinate
 *   is (d,c,a,b), and the transformed degree of freedom will be a linear combination of dofs
 *   that originally had coordinate (d,c,a,b).
 *
 * Previously PETSCDUALSPACELAGRANGE had hardcoded symmetries for the triangle and quadrilateral,
 * but this approach will work for any polytope, such as the wedge (triangular prism).
 */
struct _n_PetscLagNodeIndices
{
  PetscInt   refct;
  PetscInt   nodeIdxDim;
  PetscInt   nodeVecDim;
  PetscInt   nNodes;
  PetscInt  *nodeIdx;      /* for each node an index of size nodeIdxDim */
  PetscReal *nodeVec;      /* for each node a vector of size nodeVecDim */
  PetscInt  *perm;         /* if these are vertices, perm takes DMPlex point index to closure order;
                              if these are nodes, perm lists nodes in index revlex order */
};

/* this is just here so I can access the values in tests/ex1.c outside the library */
PetscErrorCode PetscLagNodeIndicesGetData_Internal(PetscLagNodeIndices ni, PetscInt *nodeIdxDim, PetscInt *nodeVecDim, PetscInt *nNodes, const PetscInt *nodeIdx[], const PetscReal *nodeVec[])
{
  PetscFunctionBegin;
  *nodeIdxDim = ni->nodeIdxDim;
  *nodeVecDim = ni->nodeVecDim;
  *nNodes = ni->nNodes;
  *nodeIdx = ni->nodeIdx;
  *nodeVec = ni->nodeVec;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLagNodeIndicesReference(PetscLagNodeIndices ni)
{
  PetscFunctionBegin;
  if (ni) ni->refct++;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLagNodeIndicesDuplicate(PetscLagNodeIndices ni, PetscLagNodeIndices *niNew)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(niNew));
  (*niNew)->refct = 1;
  (*niNew)->nodeIdxDim = ni->nodeIdxDim;
  (*niNew)->nodeVecDim = ni->nodeVecDim;
  (*niNew)->nNodes = ni->nNodes;
  PetscCall(PetscMalloc1(ni->nNodes * ni->nodeIdxDim, &((*niNew)->nodeIdx)));
  PetscCall(PetscArraycpy((*niNew)->nodeIdx, ni->nodeIdx, ni->nNodes * ni->nodeIdxDim));
  PetscCall(PetscMalloc1(ni->nNodes * ni->nodeVecDim, &((*niNew)->nodeVec)));
  PetscCall(PetscArraycpy((*niNew)->nodeVec, ni->nodeVec, ni->nNodes * ni->nodeVecDim));
  (*niNew)->perm = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLagNodeIndicesDestroy(PetscLagNodeIndices *ni)
{
  PetscFunctionBegin;
  if (!(*ni)) PetscFunctionReturn(0);
  if (--(*ni)->refct > 0) {
    *ni = NULL;
    PetscFunctionReturn(0);
  }
  PetscCall(PetscFree((*ni)->nodeIdx));
  PetscCall(PetscFree((*ni)->nodeVec));
  PetscCall(PetscFree((*ni)->perm));
  PetscCall(PetscFree(*ni));
  *ni = NULL;
  PetscFunctionReturn(0);
}

/* The vertices are given nodeIdx coordinates (e.g. the corners of the barycentric triangle).  Those coordinates are
 * in some other order, and to understand the effect of different symmetries, we need them to be in closure order.
 *
 * If sortIdx is PETSC_FALSE, the coordinates are already in revlex order, otherwise we must sort them
 * to that order before we do the real work of this function, which is
 *
 * - mark the vertices in closure order
 * - sort them in revlex order
 * - use the resulting permutation to list the vertex coordinates in closure order
 */
static PetscErrorCode PetscLagNodeIndicesComputeVertexOrder(DM dm, PetscLagNodeIndices ni, PetscBool sortIdx)
{
  PetscInt        v, w, vStart, vEnd, c, d;
  PetscInt        nVerts;
  PetscInt        closureSize = 0;
  PetscInt       *closure = NULL;
  PetscInt       *closureOrder;
  PetscInt       *invClosureOrder;
  PetscInt       *revlexOrder;
  PetscInt       *newNodeIdx;
  PetscInt        dim;
  Vec             coordVec;
  const PetscScalar *coords;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  nVerts = vEnd - vStart;
  PetscCall(PetscMalloc1(nVerts, &closureOrder));
  PetscCall(PetscMalloc1(nVerts, &invClosureOrder));
  PetscCall(PetscMalloc1(nVerts, &revlexOrder));
  if (sortIdx) { /* bubble sort nodeIdx into revlex order */
    PetscInt nodeIdxDim = ni->nodeIdxDim;
    PetscInt *idxOrder;

    PetscCall(PetscMalloc1(nVerts * nodeIdxDim, &newNodeIdx));
    PetscCall(PetscMalloc1(nVerts, &idxOrder));
    for (v = 0; v < nVerts; v++) idxOrder[v] = v;
    for (v = 0; v < nVerts; v++) {
      for (w = v + 1; w < nVerts; w++) {
        const PetscInt *iv = &(ni->nodeIdx[idxOrder[v] * nodeIdxDim]);
        const PetscInt *iw = &(ni->nodeIdx[idxOrder[w] * nodeIdxDim]);
        PetscInt diff = 0;

        for (d = nodeIdxDim - 1; d >= 0; d--) if ((diff = (iv[d] - iw[d]))) break;
        if (diff > 0) {
          PetscInt swap = idxOrder[v];

          idxOrder[v] = idxOrder[w];
          idxOrder[w] = swap;
        }
      }
    }
    for (v = 0; v < nVerts; v++) {
      for (d = 0; d < nodeIdxDim; d++) {
        newNodeIdx[v * ni->nodeIdxDim + d] = ni->nodeIdx[idxOrder[v] * nodeIdxDim + d];
      }
    }
    PetscCall(PetscFree(ni->nodeIdx));
    ni->nodeIdx = newNodeIdx;
    newNodeIdx = NULL;
    PetscCall(PetscFree(idxOrder));
  }
  PetscCall(DMPlexGetTransitiveClosure(dm, 0, PETSC_TRUE, &closureSize, &closure));
  c = closureSize - nVerts;
  for (v = 0; v < nVerts; v++) closureOrder[v] = closure[2 * (c + v)] - vStart;
  for (v = 0; v < nVerts; v++) invClosureOrder[closureOrder[v]] = v;
  PetscCall(DMPlexRestoreTransitiveClosure(dm, 0, PETSC_TRUE, &closureSize, &closure));
  PetscCall(DMGetCoordinatesLocal(dm, &coordVec));
  PetscCall(VecGetArrayRead(coordVec, &coords));
  /* bubble sort closure vertices by coordinates in revlex order */
  for (v = 0; v < nVerts; v++) revlexOrder[v] = v;
  for (v = 0; v < nVerts; v++) {
    for (w = v + 1; w < nVerts; w++) {
      const PetscScalar *cv = &coords[closureOrder[revlexOrder[v]] * dim];
      const PetscScalar *cw = &coords[closureOrder[revlexOrder[w]] * dim];
      PetscReal diff = 0;

      for (d = dim - 1; d >= 0; d--) if ((diff = PetscRealPart(cv[d] - cw[d])) != 0.) break;
      if (diff > 0.) {
        PetscInt swap = revlexOrder[v];

        revlexOrder[v] = revlexOrder[w];
        revlexOrder[w] = swap;
      }
    }
  }
  PetscCall(VecRestoreArrayRead(coordVec, &coords));
  PetscCall(PetscMalloc1(ni->nodeIdxDim * nVerts, &newNodeIdx));
  /* reorder nodeIdx to be in closure order */
  for (v = 0; v < nVerts; v++) {
    for (d = 0; d < ni->nodeIdxDim; d++) {
      newNodeIdx[revlexOrder[v] * ni->nodeIdxDim + d] = ni->nodeIdx[v * ni->nodeIdxDim + d];
    }
  }
  PetscCall(PetscFree(ni->nodeIdx));
  ni->nodeIdx = newNodeIdx;
  ni->perm = invClosureOrder;
  PetscCall(PetscFree(revlexOrder));
  PetscCall(PetscFree(closureOrder));
  PetscFunctionReturn(0);
}

/* the coordinates of the simplex vertices are the corners of the barycentric simplex.
 * When we stack them on top of each other in revlex order, they look like the identity matrix */
static PetscErrorCode PetscLagNodeIndicesCreateSimplexVertices(DM dm, PetscLagNodeIndices *nodeIndices)
{
  PetscLagNodeIndices ni;
  PetscInt       dim, d;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ni));
  PetscCall(DMGetDimension(dm, &dim));
  ni->nodeIdxDim = dim + 1;
  ni->nodeVecDim = 0;
  ni->nNodes = dim + 1;
  ni->refct = 1;
  PetscCall(PetscCalloc1((dim + 1)*(dim + 1), &(ni->nodeIdx)));
  for (d = 0; d < dim + 1; d++) ni->nodeIdx[d*(dim + 2)] = 1;
  PetscCall(PetscLagNodeIndicesComputeVertexOrder(dm, ni, PETSC_FALSE));
  *nodeIndices = ni;
  PetscFunctionReturn(0);
}

/* A polytope that is a tensor product of a facet and a segment.
 * We take whatever coordinate system was being used for the facet
 * and we concatenate the barycentric coordinates for the vertices
 * at the end of the segment, (1,0) and (0,1), to get a coordinate
 * system for the tensor product element */
static PetscErrorCode PetscLagNodeIndicesCreateTensorVertices(DM dm, PetscLagNodeIndices facetni, PetscLagNodeIndices *nodeIndices)
{
  PetscLagNodeIndices ni;
  PetscInt       nodeIdxDim, subNodeIdxDim = facetni->nodeIdxDim;
  PetscInt       nVerts, nSubVerts = facetni->nNodes;
  PetscInt       dim, d, e, f, g;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ni));
  PetscCall(DMGetDimension(dm, &dim));
  ni->nodeIdxDim = nodeIdxDim = subNodeIdxDim + 2;
  ni->nodeVecDim = 0;
  ni->nNodes = nVerts = 2 * nSubVerts;
  ni->refct = 1;
  PetscCall(PetscCalloc1(nodeIdxDim * nVerts, &(ni->nodeIdx)));
  for (f = 0, d = 0; d < 2; d++) {
    for (e = 0; e < nSubVerts; e++, f++) {
      for (g = 0; g < subNodeIdxDim; g++) {
        ni->nodeIdx[f * nodeIdxDim + g] = facetni->nodeIdx[e * subNodeIdxDim + g];
      }
      ni->nodeIdx[f * nodeIdxDim + subNodeIdxDim] = (1 - d);
      ni->nodeIdx[f * nodeIdxDim + subNodeIdxDim + 1] = d;
    }
  }
  PetscCall(PetscLagNodeIndicesComputeVertexOrder(dm, ni, PETSC_TRUE));
  *nodeIndices = ni;
  PetscFunctionReturn(0);
}

/* This helps us compute symmetries, and it also helps us compute coordinates for dofs that are being pushed
 * forward from a boundary mesh point.
 *
 * Input:
 *
 * dm - the target reference cell where we want new coordinates and dof directions to be valid
 * vert - the vertex coordinate system for the target reference cell
 * p - the point in the target reference cell that the dofs are coming from
 * vertp - the vertex coordinate system for p's reference cell
 * ornt - the resulting coordinates and dof vectors will be for p under this orientation
 * nodep - the node coordinates and dof vectors in p's reference cell
 * formDegree - the form degree that the dofs transform as
 *
 * Output:
 *
 * pfNodeIdx - the node coordinates for p's dofs, in the dm reference cell, from the ornt perspective
 * pfNodeVec - the node dof vectors for p's dofs, in the dm reference cell, from the ornt perspective
 */
static PetscErrorCode PetscLagNodeIndicesPushForward(DM dm, PetscLagNodeIndices vert, PetscInt p, PetscLagNodeIndices vertp, PetscLagNodeIndices nodep, PetscInt ornt, PetscInt formDegree, PetscInt pfNodeIdx[], PetscReal pfNodeVec[])
{
  PetscInt       *closureVerts;
  PetscInt        closureSize = 0;
  PetscInt       *closure = NULL;
  PetscInt        dim, pdim, c, i, j, k, n, v, vStart, vEnd;
  PetscInt        nSubVert = vertp->nNodes;
  PetscInt        nodeIdxDim = vert->nodeIdxDim;
  PetscInt        subNodeIdxDim = vertp->nodeIdxDim;
  PetscInt        nNodes = nodep->nNodes;
  const PetscInt  *vertIdx = vert->nodeIdx;
  const PetscInt  *subVertIdx = vertp->nodeIdx;
  const PetscInt  *nodeIdx = nodep->nodeIdx;
  const PetscReal *nodeVec = nodep->nodeVec;
  PetscReal       *J, *Jstar;
  PetscReal       detJ;
  PetscInt        depth, pdepth, Nk, pNk;
  Vec             coordVec;
  PetscScalar      *newCoords = NULL;
  const PetscScalar *oldCoords = NULL;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMGetCoordinatesLocal(dm, &coordVec));
  PetscCall(DMPlexGetPointDepth(dm, p, &pdepth));
  pdim = pdepth != depth ? pdepth != 0 ? pdepth : 0 : dim;
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMGetWorkArray(dm, nSubVert, MPIU_INT, &closureVerts));
  PetscCall(DMPlexGetTransitiveClosure_Internal(dm, p, ornt, PETSC_TRUE, &closureSize, &closure));
  c = closureSize - nSubVert;
  /* we want which cell closure indices the closure of this point corresponds to */
  for (v = 0; v < nSubVert; v++) closureVerts[v] = vert->perm[closure[2 * (c + v)] - vStart];
  PetscCall(DMPlexRestoreTransitiveClosure(dm, p, PETSC_TRUE, &closureSize, &closure));
  /* push forward indices */
  for (i = 0; i < nodeIdxDim; i++) { /* for every component of the target index space */
    /* check if this is a component that all vertices around this point have in common */
    for (j = 1; j < nSubVert; j++) {
      if (vertIdx[closureVerts[j] * nodeIdxDim + i] != vertIdx[closureVerts[0] * nodeIdxDim + i]) break;
    }
    if (j == nSubVert) { /* all vertices have this component in common, directly copy to output */
      PetscInt val = vertIdx[closureVerts[0] * nodeIdxDim + i];
      for (n = 0; n < nNodes; n++) pfNodeIdx[n * nodeIdxDim + i] = val;
    } else {
      PetscInt subi = -1;
      /* there must be a component in vertp that looks the same */
      for (k = 0; k < subNodeIdxDim; k++) {
        for (j = 0; j < nSubVert; j++) {
          if (vertIdx[closureVerts[j] * nodeIdxDim + i] != subVertIdx[j * subNodeIdxDim + k]) break;
        }
        if (j == nSubVert) {
          subi = k;
          break;
        }
      }
      PetscCheck(subi >= 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Did not find matching coordinate");
      /* that component in the vertp system becomes component i in the vert system for each dof */
      for (n = 0; n < nNodes; n++) pfNodeIdx[n * nodeIdxDim + i] = nodeIdx[n * subNodeIdxDim + subi];
    }
  }
  /* push forward vectors */
  PetscCall(DMGetWorkArray(dm, dim * dim, MPIU_REAL, &J));
  if (ornt != 0) { /* temporarily change the coordinate vector so
                      DMPlexComputeCellGeometryAffineFEM gives us the Jacobian we want */
    PetscInt        closureSize2 = 0;
    PetscInt       *closure2 = NULL;

    PetscCall(DMPlexGetTransitiveClosure_Internal(dm, p, 0, PETSC_TRUE, &closureSize2, &closure2));
    PetscCall(PetscMalloc1(dim * nSubVert, &newCoords));
    PetscCall(VecGetArrayRead(coordVec, &oldCoords));
    for (v = 0; v < nSubVert; v++) {
      PetscInt d;
      for (d = 0; d < dim; d++) {
        newCoords[(closure2[2 * (c + v)] - vStart) * dim + d] = oldCoords[closureVerts[v] * dim + d];
      }
    }
    PetscCall(VecRestoreArrayRead(coordVec, &oldCoords));
    PetscCall(DMPlexRestoreTransitiveClosure(dm, p, PETSC_TRUE, &closureSize2, &closure2));
    PetscCall(VecPlaceArray(coordVec, newCoords));
  }
  PetscCall(DMPlexComputeCellGeometryAffineFEM(dm, p, NULL, J, NULL, &detJ));
  if (ornt != 0) {
    PetscCall(VecResetArray(coordVec));
    PetscCall(PetscFree(newCoords));
  }
  PetscCall(DMRestoreWorkArray(dm, nSubVert, MPIU_INT, &closureVerts));
  /* compactify */
  for (i = 0; i < dim; i++) for (j = 0; j < pdim; j++) J[i * pdim + j] = J[i * dim + j];
  /* We have the Jacobian mapping the point's reference cell to this reference cell:
   * pulling back a function to the point and applying the dof is what we want,
   * so we get the pullback matrix and multiply the dof by that matrix on the right */
  PetscCall(PetscDTBinomialInt(dim, PetscAbsInt(formDegree), &Nk));
  PetscCall(PetscDTBinomialInt(pdim, PetscAbsInt(formDegree), &pNk));
  PetscCall(DMGetWorkArray(dm, pNk * Nk, MPIU_REAL, &Jstar));
  PetscCall(PetscDTAltVPullbackMatrix(pdim, dim, J, formDegree, Jstar));
  for (n = 0; n < nNodes; n++) {
    for (i = 0; i < Nk; i++) {
      PetscReal val = 0.;
      for (j = 0; j < pNk; j++) val += nodeVec[n * pNk + j] * Jstar[j * Nk + i];
      pfNodeVec[n * Nk + i] = val;
    }
  }
  PetscCall(DMRestoreWorkArray(dm, pNk * Nk, MPIU_REAL, &Jstar));
  PetscCall(DMRestoreWorkArray(dm, dim * dim, MPIU_REAL, &J));
  PetscFunctionReturn(0);
}

/* given to sets of nodes, take the tensor product, where the product of the dof indices is concatenation and the
 * product of the dof vectors is the wedge product */
static PetscErrorCode PetscLagNodeIndicesTensor(PetscLagNodeIndices tracei, PetscInt dimT, PetscInt kT, PetscLagNodeIndices fiberi, PetscInt dimF, PetscInt kF, PetscLagNodeIndices *nodeIndices)
{
  PetscInt       dim = dimT + dimF;
  PetscInt       nodeIdxDim, nNodes;
  PetscInt       formDegree = kT + kF;
  PetscInt       Nk, NkT, NkF;
  PetscInt       MkT, MkF;
  PetscLagNodeIndices ni;
  PetscInt       i, j, l;
  PetscReal      *projF, *projT;
  PetscReal      *projFstar, *projTstar;
  PetscReal      *workF, *workF2, *workT, *workT2, *work, *work2;
  PetscReal      *wedgeMat;
  PetscReal      sign;

  PetscFunctionBegin;
  PetscCall(PetscDTBinomialInt(dim, PetscAbsInt(formDegree), &Nk));
  PetscCall(PetscDTBinomialInt(dimT, PetscAbsInt(kT), &NkT));
  PetscCall(PetscDTBinomialInt(dimF, PetscAbsInt(kF), &NkF));
  PetscCall(PetscDTBinomialInt(dim, PetscAbsInt(kT), &MkT));
  PetscCall(PetscDTBinomialInt(dim, PetscAbsInt(kF), &MkF));
  PetscCall(PetscNew(&ni));
  ni->nodeIdxDim = nodeIdxDim = tracei->nodeIdxDim + fiberi->nodeIdxDim;
  ni->nodeVecDim = Nk;
  ni->nNodes = nNodes = tracei->nNodes * fiberi->nNodes;
  ni->refct = 1;
  PetscCall(PetscMalloc1(nNodes * nodeIdxDim, &(ni->nodeIdx)));
  /* first concatenate the indices */
  for (l = 0, j = 0; j < fiberi->nNodes; j++) {
    for (i = 0; i < tracei->nNodes; i++, l++) {
      PetscInt m, n = 0;

      for (m = 0; m < tracei->nodeIdxDim; m++) ni->nodeIdx[l * nodeIdxDim + n++] = tracei->nodeIdx[i * tracei->nodeIdxDim + m];
      for (m = 0; m < fiberi->nodeIdxDim; m++) ni->nodeIdx[l * nodeIdxDim + n++] = fiberi->nodeIdx[j * fiberi->nodeIdxDim + m];
    }
  }

  /* now wedge together the push-forward vectors */
  PetscCall(PetscMalloc1(nNodes * Nk, &(ni->nodeVec)));
  PetscCall(PetscCalloc2(dimT*dim, &projT, dimF*dim, &projF));
  for (i = 0; i < dimT; i++) projT[i * (dim + 1)] = 1.;
  for (i = 0; i < dimF; i++) projF[i * (dim + dimT + 1) + dimT] = 1.;
  PetscCall(PetscMalloc2(MkT*NkT, &projTstar, MkF*NkF, &projFstar));
  PetscCall(PetscDTAltVPullbackMatrix(dim, dimT, projT, kT, projTstar));
  PetscCall(PetscDTAltVPullbackMatrix(dim, dimF, projF, kF, projFstar));
  PetscCall(PetscMalloc6(MkT, &workT, MkT, &workT2, MkF, &workF, MkF, &workF2, Nk, &work, Nk, &work2));
  PetscCall(PetscMalloc1(Nk * MkT, &wedgeMat));
  sign = (PetscAbsInt(kT * kF) & 1) ? -1. : 1.;
  for (l = 0, j = 0; j < fiberi->nNodes; j++) {
    PetscInt d, e;

    /* push forward fiber k-form */
    for (d = 0; d < MkF; d++) {
      PetscReal val = 0.;
      for (e = 0; e < NkF; e++) val += projFstar[d * NkF + e] * fiberi->nodeVec[j * NkF + e];
      workF[d] = val;
    }
    /* Hodge star to proper form if necessary */
    if (kF < 0) {
      for (d = 0; d < MkF; d++) workF2[d] = workF[d];
      PetscCall(PetscDTAltVStar(dim, PetscAbsInt(kF), 1, workF2, workF));
    }
    /* Compute the matrix that wedges this form with one of the trace k-form */
    PetscCall(PetscDTAltVWedgeMatrix(dim, PetscAbsInt(kF), PetscAbsInt(kT), workF, wedgeMat));
    for (i = 0; i < tracei->nNodes; i++, l++) {
      /* push forward trace k-form */
      for (d = 0; d < MkT; d++) {
        PetscReal val = 0.;
        for (e = 0; e < NkT; e++) val += projTstar[d * NkT + e] * tracei->nodeVec[i * NkT + e];
        workT[d] = val;
      }
      /* Hodge star to proper form if necessary */
      if (kT < 0) {
        for (d = 0; d < MkT; d++) workT2[d] = workT[d];
        PetscCall(PetscDTAltVStar(dim, PetscAbsInt(kT), 1, workT2, workT));
      }
      /* compute the wedge product of the push-forward trace form and firer forms */
      for (d = 0; d < Nk; d++) {
        PetscReal val = 0.;
        for (e = 0; e < MkT; e++) val += wedgeMat[d * MkT + e] * workT[e];
        work[d] = val;
      }
      /* inverse Hodge star from proper form if necessary */
      if (formDegree < 0) {
        for (d = 0; d < Nk; d++) work2[d] = work[d];
        PetscCall(PetscDTAltVStar(dim, PetscAbsInt(formDegree), -1, work2, work));
      }
      /* insert into the array (adjusting for sign) */
      for (d = 0; d < Nk; d++) ni->nodeVec[l * Nk + d] = sign * work[d];
    }
  }
  PetscCall(PetscFree(wedgeMat));
  PetscCall(PetscFree6(workT, workT2, workF, workF2, work, work2));
  PetscCall(PetscFree2(projTstar, projFstar));
  PetscCall(PetscFree2(projT, projF));
  *nodeIndices = ni;
  PetscFunctionReturn(0);
}

/* simple union of two sets of nodes */
static PetscErrorCode PetscLagNodeIndicesMerge(PetscLagNodeIndices niA, PetscLagNodeIndices niB, PetscLagNodeIndices *nodeIndices)
{
  PetscLagNodeIndices ni;
  PetscInt            nodeIdxDim, nodeVecDim, nNodes;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ni));
  ni->nodeIdxDim = nodeIdxDim = niA->nodeIdxDim;
  PetscCheck(niB->nodeIdxDim == nodeIdxDim,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Cannot merge PetscLagNodeIndices with different nodeIdxDim");
  ni->nodeVecDim = nodeVecDim = niA->nodeVecDim;
  PetscCheck(niB->nodeVecDim == nodeVecDim,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Cannot merge PetscLagNodeIndices with different nodeVecDim");
  ni->nNodes = nNodes = niA->nNodes + niB->nNodes;
  ni->refct = 1;
  PetscCall(PetscMalloc1(nNodes * nodeIdxDim, &(ni->nodeIdx)));
  PetscCall(PetscMalloc1(nNodes * nodeVecDim, &(ni->nodeVec)));
  PetscCall(PetscArraycpy(ni->nodeIdx, niA->nodeIdx, niA->nNodes * nodeIdxDim));
  PetscCall(PetscArraycpy(ni->nodeVec, niA->nodeVec, niA->nNodes * nodeVecDim));
  PetscCall(PetscArraycpy(&(ni->nodeIdx[niA->nNodes * nodeIdxDim]), niB->nodeIdx, niB->nNodes * nodeIdxDim));
  PetscCall(PetscArraycpy(&(ni->nodeVec[niA->nNodes * nodeVecDim]), niB->nodeVec, niB->nNodes * nodeVecDim));
  *nodeIndices = ni;
  PetscFunctionReturn(0);
}

#define PETSCTUPINTCOMPREVLEX(N)                                   \
static int PetscConcat_(PetscTupIntCompRevlex_,N)(const void *a, const void *b) \
{                                                                  \
  const PetscInt *A = (const PetscInt *) a;                        \
  const PetscInt *B = (const PetscInt *) b;                        \
  int i;                                                           \
  PetscInt diff = 0;                                               \
  for (i = 0; i < N; i++) {                                        \
    diff = A[N - i] - B[N - i];                                    \
    if (diff) break;                                               \
  }                                                                \
  return (diff <= 0) ? (diff < 0) ? -1 : 0 : 1;                    \
}

PETSCTUPINTCOMPREVLEX(3)
PETSCTUPINTCOMPREVLEX(4)
PETSCTUPINTCOMPREVLEX(5)
PETSCTUPINTCOMPREVLEX(6)
PETSCTUPINTCOMPREVLEX(7)

static int PetscTupIntCompRevlex_N(const void *a, const void *b)
{
  const PetscInt *A = (const PetscInt *) a;
  const PetscInt *B = (const PetscInt *) b;
  int i;
  int N = A[0];
  PetscInt diff = 0;
  for (i = 0; i < N; i++) {
    diff = A[N - i] - B[N - i];
    if (diff) break;
  }
  return (diff <= 0) ? (diff < 0) ? -1 : 0 : 1;
}

/* The nodes are not necessarily in revlex order wrt nodeIdx: get the permutation
 * that puts them in that order */
static PetscErrorCode PetscLagNodeIndicesGetPermutation(PetscLagNodeIndices ni, PetscInt *perm[])
{
  PetscFunctionBegin;
  if (!(ni->perm)) {
    PetscInt *sorter;
    PetscInt m = ni->nNodes;
    PetscInt nodeIdxDim = ni->nodeIdxDim;
    PetscInt i, j, k, l;
    PetscInt *prm;
    int (*comp) (const void *, const void *);

    PetscCall(PetscMalloc1((nodeIdxDim + 2) * m, &sorter));
    for (k = 0, l = 0, i = 0; i < m; i++) {
      sorter[k++] = nodeIdxDim + 1;
      sorter[k++] = i;
      for (j = 0; j < nodeIdxDim; j++) sorter[k++] = ni->nodeIdx[l++];
    }
    switch (nodeIdxDim) {
    case 2:
      comp = PetscTupIntCompRevlex_3;
      break;
    case 3:
      comp = PetscTupIntCompRevlex_4;
      break;
    case 4:
      comp = PetscTupIntCompRevlex_5;
      break;
    case 5:
      comp = PetscTupIntCompRevlex_6;
      break;
    case 6:
      comp = PetscTupIntCompRevlex_7;
      break;
    default:
      comp = PetscTupIntCompRevlex_N;
      break;
    }
    qsort(sorter, m, (nodeIdxDim + 2) * sizeof(PetscInt), comp);
    PetscCall(PetscMalloc1(m, &prm));
    for (i = 0; i < m; i++) prm[i] = sorter[(nodeIdxDim + 2) * i + 1];
    ni->perm = prm;
    PetscCall(PetscFree(sorter));
  }
  *perm = ni->perm;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceDestroy_Lagrange(PetscDualSpace sp)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;

  PetscFunctionBegin;
  if (lag->symperms) {
    PetscInt **selfSyms = lag->symperms[0];

    if (selfSyms) {
      PetscInt i, **allocated = &selfSyms[-lag->selfSymOff];

      for (i = 0; i < lag->numSelfSym; i++) {
        PetscCall(PetscFree(allocated[i]));
      }
      PetscCall(PetscFree(allocated));
    }
    PetscCall(PetscFree(lag->symperms));
  }
  if (lag->symflips) {
    PetscScalar **selfSyms = lag->symflips[0];

    if (selfSyms) {
      PetscInt i;
      PetscScalar **allocated = &selfSyms[-lag->selfSymOff];

      for (i = 0; i < lag->numSelfSym; i++) {
        PetscCall(PetscFree(allocated[i]));
      }
      PetscCall(PetscFree(allocated));
    }
    PetscCall(PetscFree(lag->symflips));
  }
  PetscCall(Petsc1DNodeFamilyDestroy(&(lag->nodeFamily)));
  PetscCall(PetscLagNodeIndicesDestroy(&(lag->vertIndices)));
  PetscCall(PetscLagNodeIndicesDestroy(&(lag->intNodeIndices)));
  PetscCall(PetscLagNodeIndicesDestroy(&(lag->allNodeIndices)));
  PetscCall(PetscFree(lag));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetContinuity_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetContinuity_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetTensor_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetTensor_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetTrimmed_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetTrimmed_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetNodeType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetNodeType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetUseMoments_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetUseMoments_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetMomentOrder_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetMomentOrder_C", NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceLagrangeView_Ascii(PetscDualSpace sp, PetscViewer viewer)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(viewer, "%s %s%sLagrange dual space\n", lag->continuous ? "Continuous" : "Discontinuous", lag->tensorSpace ? "tensor " : "", lag->trimmed ? "trimmed " : ""));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceView_Lagrange(PetscDualSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscDualSpaceLagrangeView_Ascii(sp, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceSetFromOptions_Lagrange(PetscOptionItems *PetscOptionsObject,PetscDualSpace sp)
{
  PetscBool      continuous, tensor, trimmed, flg, flg2, flg3;
  PetscDTNodeType nodeType;
  PetscReal      nodeExponent;
  PetscInt       momentOrder;
  PetscBool      nodeEndpoints, useMoments;

  PetscFunctionBegin;
  PetscCall(PetscDualSpaceLagrangeGetContinuity(sp, &continuous));
  PetscCall(PetscDualSpaceLagrangeGetTensor(sp, &tensor));
  PetscCall(PetscDualSpaceLagrangeGetTrimmed(sp, &trimmed));
  PetscCall(PetscDualSpaceLagrangeGetNodeType(sp, &nodeType, &nodeEndpoints, &nodeExponent));
  if (nodeType == PETSCDTNODES_DEFAULT) nodeType = PETSCDTNODES_GAUSSJACOBI;
  PetscCall(PetscDualSpaceLagrangeGetUseMoments(sp, &useMoments));
  PetscCall(PetscDualSpaceLagrangeGetMomentOrder(sp, &momentOrder));
  PetscCall(PetscOptionsHead(PetscOptionsObject,"PetscDualSpace Lagrange Options"));
  PetscCall(PetscOptionsBool("-petscdualspace_lagrange_continuity", "Flag for continuous element", "PetscDualSpaceLagrangeSetContinuity", continuous, &continuous, &flg));
  if (flg) PetscCall(PetscDualSpaceLagrangeSetContinuity(sp, continuous));
  PetscCall(PetscOptionsBool("-petscdualspace_lagrange_tensor", "Flag for tensor dual space", "PetscDualSpaceLagrangeSetTensor", tensor, &tensor, &flg));
  if (flg) PetscCall(PetscDualSpaceLagrangeSetTensor(sp, tensor));
  PetscCall(PetscOptionsBool("-petscdualspace_lagrange_trimmed", "Flag for trimmed dual space", "PetscDualSpaceLagrangeSetTrimmed", trimmed, &trimmed, &flg));
  if (flg) PetscCall(PetscDualSpaceLagrangeSetTrimmed(sp, trimmed));
  PetscCall(PetscOptionsEnum("-petscdualspace_lagrange_node_type", "Lagrange node location type", "PetscDualSpaceLagrangeSetNodeType", PetscDTNodeTypes, (PetscEnum)nodeType, (PetscEnum *)&nodeType, &flg));
  PetscCall(PetscOptionsBool("-petscdualspace_lagrange_node_endpoints", "Flag for nodes that include endpoints", "PetscDualSpaceLagrangeSetNodeType", nodeEndpoints, &nodeEndpoints, &flg2));
  flg3 = PETSC_FALSE;
  if (nodeType == PETSCDTNODES_GAUSSJACOBI) {
    PetscCall(PetscOptionsReal("-petscdualspace_lagrange_node_exponent", "Gauss-Jacobi weight function exponent", "PetscDualSpaceLagrangeSetNodeType", nodeExponent, &nodeExponent, &flg3));
  }
  if (flg || flg2 || flg3) PetscCall(PetscDualSpaceLagrangeSetNodeType(sp, nodeType, nodeEndpoints, nodeExponent));
  PetscCall(PetscOptionsBool("-petscdualspace_lagrange_use_moments", "Use moments (where appropriate) for functionals", "PetscDualSpaceLagrangeSetUseMoments", useMoments, &useMoments, &flg));
  if (flg) PetscCall(PetscDualSpaceLagrangeSetUseMoments(sp, useMoments));
  PetscCall(PetscOptionsInt("-petscdualspace_lagrange_moment_order", "Quadrature order for moment functionals", "PetscDualSpaceLagrangeSetMomentOrder", momentOrder, &momentOrder, &flg));
  if (flg) PetscCall(PetscDualSpaceLagrangeSetMomentOrder(sp, momentOrder));
  PetscCall(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceDuplicate_Lagrange(PetscDualSpace sp, PetscDualSpace spNew)
{
  PetscBool           cont, tensor, trimmed, boundary;
  PetscDTNodeType     nodeType;
  PetscReal           exponent;
  PetscDualSpace_Lag *lag    = (PetscDualSpace_Lag *) sp->data;

  PetscFunctionBegin;
  PetscCall(PetscDualSpaceLagrangeGetContinuity(sp, &cont));
  PetscCall(PetscDualSpaceLagrangeSetContinuity(spNew, cont));
  PetscCall(PetscDualSpaceLagrangeGetTensor(sp, &tensor));
  PetscCall(PetscDualSpaceLagrangeSetTensor(spNew, tensor));
  PetscCall(PetscDualSpaceLagrangeGetTrimmed(sp, &trimmed));
  PetscCall(PetscDualSpaceLagrangeSetTrimmed(spNew, trimmed));
  PetscCall(PetscDualSpaceLagrangeGetNodeType(sp, &nodeType, &boundary, &exponent));
  PetscCall(PetscDualSpaceLagrangeSetNodeType(spNew, nodeType, boundary, exponent));
  if (lag->nodeFamily) {
    PetscDualSpace_Lag *lagnew = (PetscDualSpace_Lag *) spNew->data;

    PetscCall(Petsc1DNodeFamilyReference(lag->nodeFamily));
    lagnew->nodeFamily = lag->nodeFamily;
  }
  PetscFunctionReturn(0);
}

/* for making tensor product spaces: take a dual space and product a segment space that has all the same
 * specifications (trimmed, continuous, order, node set), except for the form degree */
static PetscErrorCode PetscDualSpaceCreateEdgeSubspace_Lagrange(PetscDualSpace sp, PetscInt order, PetscInt k, PetscInt Nc, PetscBool interiorOnly, PetscDualSpace *bdsp)
{
  DM                 K;
  PetscDualSpace_Lag *newlag;

  PetscFunctionBegin;
  PetscCall(PetscDualSpaceDuplicate(sp,bdsp));
  PetscCall(PetscDualSpaceSetFormDegree(*bdsp, k));
  PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_SELF, DMPolytopeTypeSimpleShape(1, PETSC_TRUE), &K));
  PetscCall(PetscDualSpaceSetDM(*bdsp, K));
  PetscCall(DMDestroy(&K));
  PetscCall(PetscDualSpaceSetOrder(*bdsp, order));
  PetscCall(PetscDualSpaceSetNumComponents(*bdsp, Nc));
  newlag = (PetscDualSpace_Lag *) (*bdsp)->data;
  newlag->interiorOnly = interiorOnly;
  PetscCall(PetscDualSpaceSetUp(*bdsp));
  PetscFunctionReturn(0);
}

/* just the points, weights aren't handled */
static PetscErrorCode PetscQuadratureCreateTensor(PetscQuadrature trace, PetscQuadrature fiber, PetscQuadrature *product)
{
  PetscInt         dimTrace, dimFiber;
  PetscInt         numPointsTrace, numPointsFiber;
  PetscInt         dim, numPoints;
  const PetscReal *pointsTrace;
  const PetscReal *pointsFiber;
  PetscReal       *points;
  PetscInt         i, j, k, p;

  PetscFunctionBegin;
  PetscCall(PetscQuadratureGetData(trace, &dimTrace, NULL, &numPointsTrace, &pointsTrace, NULL));
  PetscCall(PetscQuadratureGetData(fiber, &dimFiber, NULL, &numPointsFiber, &pointsFiber, NULL));
  dim = dimTrace + dimFiber;
  numPoints = numPointsFiber * numPointsTrace;
  PetscCall(PetscMalloc1(numPoints * dim, &points));
  for (p = 0, j = 0; j < numPointsFiber; j++) {
    for (i = 0; i < numPointsTrace; i++, p++) {
      for (k = 0; k < dimTrace; k++) points[p * dim +            k] = pointsTrace[i * dimTrace + k];
      for (k = 0; k < dimFiber; k++) points[p * dim + dimTrace + k] = pointsFiber[j * dimFiber + k];
    }
  }
  PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, product));
  PetscCall(PetscQuadratureSetData(*product, dim, 0, numPoints, points, NULL));
  PetscFunctionReturn(0);
}

/* Kronecker tensor product where matrix is considered a matrix of k-forms, so that
 * the entries in the product matrix are wedge products of the entries in the original matrices */
static PetscErrorCode MatTensorAltV(Mat trace, Mat fiber, PetscInt dimTrace, PetscInt kTrace, PetscInt dimFiber, PetscInt kFiber, Mat *product)
{
  PetscInt mTrace, nTrace, mFiber, nFiber, m, n, k, i, j, l;
  PetscInt dim, NkTrace, NkFiber, Nk;
  PetscInt dT, dF;
  PetscInt *nnzTrace, *nnzFiber, *nnz;
  PetscInt iT, iF, jT, jF, il, jl;
  PetscReal *workT, *workT2, *workF, *workF2, *work, *workstar;
  PetscReal *projT, *projF;
  PetscReal *projTstar, *projFstar;
  PetscReal *wedgeMat;
  PetscReal sign;
  PetscScalar *workS;
  Mat prod;
  /* this produces dof groups that look like the identity */

  PetscFunctionBegin;
  PetscCall(MatGetSize(trace, &mTrace, &nTrace));
  PetscCall(PetscDTBinomialInt(dimTrace, PetscAbsInt(kTrace), &NkTrace));
  PetscCheck(nTrace % NkTrace == 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "point value space of trace matrix is not a multiple of k-form size");
  PetscCall(MatGetSize(fiber, &mFiber, &nFiber));
  PetscCall(PetscDTBinomialInt(dimFiber, PetscAbsInt(kFiber), &NkFiber));
  PetscCheck(nFiber % NkFiber == 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "point value space of fiber matrix is not a multiple of k-form size");
  PetscCall(PetscMalloc2(mTrace, &nnzTrace, mFiber, &nnzFiber));
  for (i = 0; i < mTrace; i++) {
    PetscCall(MatGetRow(trace, i, &(nnzTrace[i]), NULL, NULL));
    PetscCheck(nnzTrace[i] % NkTrace == 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "nonzeros in trace matrix are not in k-form size blocks");
  }
  for (i = 0; i < mFiber; i++) {
    PetscCall(MatGetRow(fiber, i, &(nnzFiber[i]), NULL, NULL));
    PetscCheck(nnzFiber[i] % NkFiber == 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "nonzeros in fiber matrix are not in k-form size blocks");
  }
  dim = dimTrace + dimFiber;
  k = kFiber + kTrace;
  PetscCall(PetscDTBinomialInt(dim, PetscAbsInt(k), &Nk));
  m = mTrace * mFiber;
  PetscCall(PetscMalloc1(m, &nnz));
  for (l = 0, j = 0; j < mFiber; j++) for (i = 0; i < mTrace; i++, l++) nnz[l] = (nnzTrace[i] / NkTrace) * (nnzFiber[j] / NkFiber) * Nk;
  n = (nTrace / NkTrace) * (nFiber / NkFiber) * Nk;
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, m, n, 0, nnz, &prod));
  PetscCall(PetscFree(nnz));
  PetscCall(PetscFree2(nnzTrace,nnzFiber));
  /* reasoning about which points each dof needs depends on having zeros computed at points preserved */
  PetscCall(MatSetOption(prod, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE));
  /* compute pullbacks */
  PetscCall(PetscDTBinomialInt(dim, PetscAbsInt(kTrace), &dT));
  PetscCall(PetscDTBinomialInt(dim, PetscAbsInt(kFiber), &dF));
  PetscCall(PetscMalloc4(dimTrace * dim, &projT, dimFiber * dim, &projF, dT * NkTrace, &projTstar, dF * NkFiber, &projFstar));
  PetscCall(PetscArrayzero(projT, dimTrace * dim));
  for (i = 0; i < dimTrace; i++) projT[i * (dim + 1)] = 1.;
  PetscCall(PetscArrayzero(projF, dimFiber * dim));
  for (i = 0; i < dimFiber; i++) projF[i * (dim + 1) + dimTrace] = 1.;
  PetscCall(PetscDTAltVPullbackMatrix(dim, dimTrace, projT, kTrace, projTstar));
  PetscCall(PetscDTAltVPullbackMatrix(dim, dimFiber, projF, kFiber, projFstar));
  PetscCall(PetscMalloc5(dT, &workT, dF, &workF, Nk, &work, Nk, &workstar, Nk, &workS));
  PetscCall(PetscMalloc2(dT, &workT2, dF, &workF2));
  PetscCall(PetscMalloc1(Nk * dT, &wedgeMat));
  sign = (PetscAbsInt(kTrace * kFiber) & 1) ? -1. : 1.;
  for (i = 0, iF = 0; iF < mFiber; iF++) {
    PetscInt           ncolsF, nformsF;
    const PetscInt    *colsF;
    const PetscScalar *valsF;

    PetscCall(MatGetRow(fiber, iF, &ncolsF, &colsF, &valsF));
    nformsF = ncolsF / NkFiber;
    for (iT = 0; iT < mTrace; iT++, i++) {
      PetscInt           ncolsT, nformsT;
      const PetscInt    *colsT;
      const PetscScalar *valsT;

      PetscCall(MatGetRow(trace, iT, &ncolsT, &colsT, &valsT));
      nformsT = ncolsT / NkTrace;
      for (j = 0, jF = 0; jF < nformsF; jF++) {
        PetscInt colF = colsF[jF * NkFiber] / NkFiber;

        for (il = 0; il < dF; il++) {
          PetscReal val = 0.;
          for (jl = 0; jl < NkFiber; jl++) val += projFstar[il * NkFiber + jl] * PetscRealPart(valsF[jF * NkFiber + jl]);
          workF[il] = val;
        }
        if (kFiber < 0) {
          for (il = 0; il < dF; il++) workF2[il] = workF[il];
          PetscCall(PetscDTAltVStar(dim, PetscAbsInt(kFiber), 1, workF2, workF));
        }
        PetscCall(PetscDTAltVWedgeMatrix(dim, PetscAbsInt(kFiber), PetscAbsInt(kTrace), workF, wedgeMat));
        for (jT = 0; jT < nformsT; jT++, j++) {
          PetscInt colT = colsT[jT * NkTrace] / NkTrace;
          PetscInt col = colF * (nTrace / NkTrace) + colT;
          const PetscScalar *vals;

          for (il = 0; il < dT; il++) {
            PetscReal val = 0.;
            for (jl = 0; jl < NkTrace; jl++) val += projTstar[il * NkTrace + jl] * PetscRealPart(valsT[jT * NkTrace + jl]);
            workT[il] = val;
          }
          if (kTrace < 0) {
            for (il = 0; il < dT; il++) workT2[il] = workT[il];
            PetscCall(PetscDTAltVStar(dim, PetscAbsInt(kTrace), 1, workT2, workT));
          }

          for (il = 0; il < Nk; il++) {
            PetscReal val = 0.;
            for (jl = 0; jl < dT; jl++) val += sign * wedgeMat[il * dT + jl] * workT[jl];
            work[il] = val;
          }
          if (k < 0) {
            PetscCall(PetscDTAltVStar(dim, PetscAbsInt(k), -1, work, workstar));
#if defined(PETSC_USE_COMPLEX)
            for (l = 0; l < Nk; l++) workS[l] = workstar[l];
            vals = &workS[0];
#else
            vals = &workstar[0];
#endif
          } else {
#if defined(PETSC_USE_COMPLEX)
            for (l = 0; l < Nk; l++) workS[l] = work[l];
            vals = &workS[0];
#else
            vals = &work[0];
#endif
          }
          for (l = 0; l < Nk; l++) {
            PetscCall(MatSetValue(prod, i, col * Nk + l, vals[l], INSERT_VALUES));
          } /* Nk */
        } /* jT */
      } /* jF */
      PetscCall(MatRestoreRow(trace, iT, &ncolsT, &colsT, &valsT));
    } /* iT */
    PetscCall(MatRestoreRow(fiber, iF, &ncolsF, &colsF, &valsF));
  } /* iF */
  PetscCall(PetscFree(wedgeMat));
  PetscCall(PetscFree4(projT, projF, projTstar, projFstar));
  PetscCall(PetscFree2(workT2, workF2));
  PetscCall(PetscFree5(workT, workF, work, workstar, workS));
  PetscCall(MatAssemblyBegin(prod, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(prod, MAT_FINAL_ASSEMBLY));
  *product = prod;
  PetscFunctionReturn(0);
}

/* Union of quadrature points, with an attempt to identify commont points in the two sets */
static PetscErrorCode PetscQuadraturePointsMerge(PetscQuadrature quadA, PetscQuadrature quadB, PetscQuadrature *quadJoint, PetscInt *aToJoint[], PetscInt *bToJoint[])
{
  PetscInt         dimA, dimB;
  PetscInt         nA, nB, nJoint, i, j, d;
  const PetscReal *pointsA;
  const PetscReal *pointsB;
  PetscReal       *pointsJoint;
  PetscInt        *aToJ, *bToJ;
  PetscQuadrature  qJ;

  PetscFunctionBegin;
  PetscCall(PetscQuadratureGetData(quadA, &dimA, NULL, &nA, &pointsA, NULL));
  PetscCall(PetscQuadratureGetData(quadB, &dimB, NULL, &nB, &pointsB, NULL));
  PetscCheck(dimA == dimB,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Quadrature points must be in the same dimension");
  nJoint = nA;
  PetscCall(PetscMalloc1(nA, &aToJ));
  for (i = 0; i < nA; i++) aToJ[i] = i;
  PetscCall(PetscMalloc1(nB, &bToJ));
  for (i = 0; i < nB; i++) {
    for (j = 0; j < nA; j++) {
      bToJ[i] = -1;
      for (d = 0; d < dimA; d++) if (PetscAbsReal(pointsB[i * dimA + d] - pointsA[j * dimA + d]) > PETSC_SMALL) break;
      if (d == dimA) {
        bToJ[i] = j;
        break;
      }
    }
    if (bToJ[i] == -1) {
      bToJ[i] = nJoint++;
    }
  }
  *aToJoint = aToJ;
  *bToJoint = bToJ;
  PetscCall(PetscMalloc1(nJoint * dimA, &pointsJoint));
  PetscCall(PetscArraycpy(pointsJoint, pointsA, nA * dimA));
  for (i = 0; i < nB; i++) {
    if (bToJ[i] >= nA) {
      for (d = 0; d < dimA; d++) pointsJoint[bToJ[i] * dimA + d] = pointsB[i * dimA + d];
    }
  }
  PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, &qJ));
  PetscCall(PetscQuadratureSetData(qJ, dimA, 0, nJoint, pointsJoint, NULL));
  *quadJoint = qJ;
  PetscFunctionReturn(0);
}

/* Matrices matA and matB are both quadrature -> dof matrices: produce a matrix that is joint quadrature -> union of
 * dofs, where the joint quadrature was produced by PetscQuadraturePointsMerge */
static PetscErrorCode MatricesMerge(Mat matA, Mat matB, PetscInt dim, PetscInt k, PetscInt numMerged, const PetscInt aToMerged[], const PetscInt bToMerged[], Mat *matMerged)
{
  PetscInt m, n, mA, nA, mB, nB, Nk, i, j, l;
  Mat      M;
  PetscInt *nnz;
  PetscInt maxnnz;
  PetscInt *work;

  PetscFunctionBegin;
  PetscCall(PetscDTBinomialInt(dim, PetscAbsInt(k), &Nk));
  PetscCall(MatGetSize(matA, &mA, &nA));
  PetscCheck(nA % Nk == 0,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "matA column space not a multiple of k-form size");
  PetscCall(MatGetSize(matB, &mB, &nB));
  PetscCheck(nB % Nk == 0,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "matB column space not a multiple of k-form size");
  m = mA + mB;
  n = numMerged * Nk;
  PetscCall(PetscMalloc1(m, &nnz));
  maxnnz = 0;
  for (i = 0; i < mA; i++) {
    PetscCall(MatGetRow(matA, i, &(nnz[i]), NULL, NULL));
    PetscCheck(nnz[i] % Nk == 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "nonzeros in matA are not in k-form size blocks");
    maxnnz = PetscMax(maxnnz, nnz[i]);
  }
  for (i = 0; i < mB; i++) {
    PetscCall(MatGetRow(matB, i, &(nnz[i+mA]), NULL, NULL));
    PetscCheck(nnz[i+mA] % Nk == 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "nonzeros in matB are not in k-form size blocks");
    maxnnz = PetscMax(maxnnz, nnz[i+mA]);
  }
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, m, n, 0, nnz, &M));
  PetscCall(PetscFree(nnz));
  /* reasoning about which points each dof needs depends on having zeros computed at points preserved */
  PetscCall(MatSetOption(M, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE));
  PetscCall(PetscMalloc1(maxnnz, &work));
  for (i = 0; i < mA; i++) {
    const PetscInt *cols;
    const PetscScalar *vals;
    PetscInt nCols;
    PetscCall(MatGetRow(matA, i, &nCols, &cols, &vals));
    for (j = 0; j < nCols / Nk; j++) {
      PetscInt newCol = aToMerged[cols[j * Nk] / Nk];
      for (l = 0; l < Nk; l++) work[j * Nk + l] = newCol * Nk + l;
    }
    PetscCall(MatSetValuesBlocked(M, 1, &i, nCols, work, vals, INSERT_VALUES));
    PetscCall(MatRestoreRow(matA, i, &nCols, &cols, &vals));
  }
  for (i = 0; i < mB; i++) {
    const PetscInt *cols;
    const PetscScalar *vals;

    PetscInt row = i + mA;
    PetscInt nCols;
    PetscCall(MatGetRow(matB, i, &nCols, &cols, &vals));
    for (j = 0; j < nCols / Nk; j++) {
      PetscInt newCol = bToMerged[cols[j * Nk] / Nk];
      for (l = 0; l < Nk; l++) work[j * Nk + l] = newCol * Nk + l;
    }
    PetscCall(MatSetValuesBlocked(M, 1, &row, nCols, work, vals, INSERT_VALUES));
    PetscCall(MatRestoreRow(matB, i, &nCols, &cols, &vals));
  }
  PetscCall(PetscFree(work));
  PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));
  *matMerged = M;
  PetscFunctionReturn(0);
}

/* Take a dual space and product a segment space that has all the same specifications (trimmed, continuous, order,
 * node set), except for the form degree.  For computing boundary dofs and for making tensor product spaces */
static PetscErrorCode PetscDualSpaceCreateFacetSubspace_Lagrange(PetscDualSpace sp, DM K, PetscInt f, PetscInt k, PetscInt Ncopies, PetscBool interiorOnly, PetscDualSpace *bdsp)
{
  PetscInt           Nknew, Ncnew;
  PetscInt           dim, pointDim = -1;
  PetscInt           depth;
  DM                 dm;
  PetscDualSpace_Lag *newlag;

  PetscFunctionBegin;
  PetscCall(PetscDualSpaceGetDM(sp,&dm));
  PetscCall(DMGetDimension(dm,&dim));
  PetscCall(DMPlexGetDepth(dm,&depth));
  PetscCall(PetscDualSpaceDuplicate(sp,bdsp));
  PetscCall(PetscDualSpaceSetFormDegree(*bdsp,k));
  if (!K) {
    if (depth == dim) {
      DMPolytopeType ct;

      pointDim = dim - 1;
      PetscCall(DMPlexGetCellType(dm, f, &ct));
      PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_SELF, ct, &K));
    } else if (depth == 1) {
      pointDim = 0;
      PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_SELF, DM_POLYTOPE_POINT, &K));
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unsupported interpolation state of reference element");
  } else {
    PetscCall(PetscObjectReference((PetscObject)K));
    PetscCall(DMGetDimension(K, &pointDim));
  }
  PetscCall(PetscDualSpaceSetDM(*bdsp, K));
  PetscCall(DMDestroy(&K));
  PetscCall(PetscDTBinomialInt(pointDim, PetscAbsInt(k), &Nknew));
  Ncnew = Nknew * Ncopies;
  PetscCall(PetscDualSpaceSetNumComponents(*bdsp, Ncnew));
  newlag = (PetscDualSpace_Lag *) (*bdsp)->data;
  newlag->interiorOnly = interiorOnly;
  PetscCall(PetscDualSpaceSetUp(*bdsp));
  PetscFunctionReturn(0);
}

/* Construct simplex nodes from a nodefamily, add Nk dof vectors of length Nk at each node.
 * Return the (quadrature, matrix) form of the dofs and the nodeIndices form as well.
 *
 * Sometimes we want a set of nodes to be contained in the interior of the element,
 * even when the node scheme puts nodes on the boundaries.  numNodeSkip tells
 * the routine how many "layers" of nodes need to be skipped.
 * */
static PetscErrorCode PetscDualSpaceLagrangeCreateSimplexNodeMat(Petsc1DNodeFamily nodeFamily, PetscInt dim, PetscInt sum, PetscInt Nk, PetscInt numNodeSkip, PetscQuadrature *iNodes, Mat *iMat, PetscLagNodeIndices *nodeIndices)
{
  PetscReal *extraNodeCoords, *nodeCoords;
  PetscInt nNodes, nExtraNodes;
  PetscInt i, j, k, extraSum = sum + numNodeSkip * (1 + dim);
  PetscQuadrature intNodes;
  Mat intMat;
  PetscLagNodeIndices ni;

  PetscFunctionBegin;
  PetscCall(PetscDTBinomialInt(dim + sum, dim, &nNodes));
  PetscCall(PetscDTBinomialInt(dim + extraSum, dim, &nExtraNodes));

  PetscCall(PetscMalloc1(dim * nExtraNodes, &extraNodeCoords));
  PetscCall(PetscNew(&ni));
  ni->nodeIdxDim = dim + 1;
  ni->nodeVecDim = Nk;
  ni->nNodes = nNodes * Nk;
  ni->refct = 1;
  PetscCall(PetscMalloc1(nNodes * Nk * (dim + 1), &(ni->nodeIdx)));
  PetscCall(PetscMalloc1(nNodes * Nk * Nk, &(ni->nodeVec)));
  for (i = 0; i < nNodes; i++) for (j = 0; j < Nk; j++) for (k = 0; k < Nk; k++) ni->nodeVec[(i * Nk + j) * Nk + k] = (j == k) ? 1. : 0.;
  PetscCall(Petsc1DNodeFamilyComputeSimplexNodes(nodeFamily, dim, extraSum, extraNodeCoords));
  if (numNodeSkip) {
    PetscInt k;
    PetscInt *tup;

    PetscCall(PetscMalloc1(dim * nNodes, &nodeCoords));
    PetscCall(PetscMalloc1(dim + 1, &tup));
    for (k = 0; k < nNodes; k++) {
      PetscInt j, c;
      PetscInt index;

      PetscCall(PetscDTIndexToBary(dim + 1, sum, k, tup));
      for (j = 0; j < dim + 1; j++) tup[j] += numNodeSkip;
      for (c = 0; c < Nk; c++) {
        for (j = 0; j < dim + 1; j++) {
          ni->nodeIdx[(k * Nk + c) * (dim + 1) + j] = tup[j] + 1;
        }
      }
      PetscCall(PetscDTBaryToIndex(dim + 1, extraSum, tup, &index));
      for (j = 0; j < dim; j++) nodeCoords[k * dim + j] = extraNodeCoords[index * dim + j];
    }
    PetscCall(PetscFree(tup));
    PetscCall(PetscFree(extraNodeCoords));
  } else {
    PetscInt k;
    PetscInt *tup;

    nodeCoords = extraNodeCoords;
    PetscCall(PetscMalloc1(dim + 1, &tup));
    for (k = 0; k < nNodes; k++) {
      PetscInt j, c;

      PetscCall(PetscDTIndexToBary(dim + 1, sum, k, tup));
      for (c = 0; c < Nk; c++) {
        for (j = 0; j < dim + 1; j++) {
          /* barycentric indices can have zeros, but we don't want to push forward zeros because it makes it harder to
           * determine which nodes correspond to which under symmetries, so we increase by 1.  This is fine
           * because the nodeIdx coordinates don't have any meaning other than helping to identify symmetries */
          ni->nodeIdx[(k * Nk + c) * (dim + 1) + j] = tup[j] + 1;
        }
      }
    }
    PetscCall(PetscFree(tup));
  }
  PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, &intNodes));
  PetscCall(PetscQuadratureSetData(intNodes, dim, 0, nNodes, nodeCoords, NULL));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, nNodes * Nk, nNodes * Nk, Nk, NULL, &intMat));
  PetscCall(MatSetOption(intMat,MAT_IGNORE_ZERO_ENTRIES,PETSC_FALSE));
  for (j = 0; j < nNodes * Nk; j++) {
    PetscInt rem = j % Nk;
    PetscInt a, aprev = j - rem;
    PetscInt anext = aprev + Nk;

    for (a = aprev; a < anext; a++) {
      PetscCall(MatSetValue(intMat, j, a, (a == j) ? 1. : 0., INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(intMat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(intMat, MAT_FINAL_ASSEMBLY));
  *iNodes = intNodes;
  *iMat = intMat;
  *nodeIndices = ni;
  PetscFunctionReturn(0);
}

/* once the nodeIndices have been created for the interior of the reference cell, and for all of the boundary cells,
 * push forward the boundary dofs and concatenate them into the full node indices for the dual space */
static PetscErrorCode PetscDualSpaceLagrangeCreateAllNodeIdx(PetscDualSpace sp)
{
  DM             dm;
  PetscInt       dim, nDofs;
  PetscSection   section;
  PetscInt       pStart, pEnd, p;
  PetscInt       formDegree, Nk;
  PetscInt       nodeIdxDim, spintdim;
  PetscDualSpace_Lag *lag;
  PetscLagNodeIndices ni, verti;

  PetscFunctionBegin;
  lag = (PetscDualSpace_Lag *) sp->data;
  verti = lag->vertIndices;
  PetscCall(PetscDualSpaceGetDM(sp, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscDualSpaceGetFormDegree(sp, &formDegree));
  PetscCall(PetscDTBinomialInt(dim, PetscAbsInt(formDegree), &Nk));
  PetscCall(PetscDualSpaceGetSection(sp, &section));
  PetscCall(PetscSectionGetStorageSize(section, &nDofs));
  PetscCall(PetscNew(&ni));
  ni->nodeIdxDim = nodeIdxDim = verti->nodeIdxDim;
  ni->nodeVecDim = Nk;
  ni->nNodes = nDofs;
  ni->refct = 1;
  PetscCall(PetscMalloc1(nodeIdxDim * nDofs, &(ni->nodeIdx)));
  PetscCall(PetscMalloc1(Nk * nDofs, &(ni->nodeVec)));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscSectionGetDof(section, 0, &spintdim));
  if (spintdim) {
    PetscCall(PetscArraycpy(ni->nodeIdx, lag->intNodeIndices->nodeIdx, spintdim * nodeIdxDim));
    PetscCall(PetscArraycpy(ni->nodeVec, lag->intNodeIndices->nodeVec, spintdim * Nk));
  }
  for (p = pStart + 1; p < pEnd; p++) {
    PetscDualSpace psp = sp->pointSpaces[p];
    PetscDualSpace_Lag *plag;
    PetscInt dof, off;

    PetscCall(PetscSectionGetDof(section, p, &dof));
    if (!dof) continue;
    plag = (PetscDualSpace_Lag *) psp->data;
    PetscCall(PetscSectionGetOffset(section, p, &off));
    PetscCall(PetscLagNodeIndicesPushForward(dm, verti, p, plag->vertIndices, plag->intNodeIndices, 0, formDegree, &(ni->nodeIdx[off * nodeIdxDim]), &(ni->nodeVec[off * Nk])));
  }
  lag->allNodeIndices = ni;
  PetscFunctionReturn(0);
}

/* once the (quadrature, Matrix) forms of the dofs have been created for the interior of the
 * reference cell and for the boundary cells, jk
 * push forward the boundary data and concatenate them into the full (quadrature, matrix) data
 * for the dual space */
static PetscErrorCode PetscDualSpaceCreateAllDataFromInteriorData(PetscDualSpace sp)
{
  DM               dm;
  PetscSection     section;
  PetscInt         pStart, pEnd, p, k, Nk, dim, Nc;
  PetscInt         nNodes;
  PetscInt         countNodes;
  Mat              allMat;
  PetscQuadrature  allNodes;
  PetscInt         nDofs;
  PetscInt         maxNzforms, j;
  PetscScalar      *work;
  PetscReal        *L, *J, *Jinv, *v0, *pv0;
  PetscInt         *iwork;
  PetscReal        *nodes;

  PetscFunctionBegin;
  PetscCall(PetscDualSpaceGetDM(sp, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscDualSpaceGetSection(sp, &section));
  PetscCall(PetscSectionGetStorageSize(section, &nDofs));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscDualSpaceGetFormDegree(sp, &k));
  PetscCall(PetscDualSpaceGetNumComponents(sp, &Nc));
  PetscCall(PetscDTBinomialInt(dim, PetscAbsInt(k), &Nk));
  for (p = pStart, nNodes = 0, maxNzforms = 0; p < pEnd; p++) {
    PetscDualSpace  psp;
    DM              pdm;
    PetscInt        pdim, pNk;
    PetscQuadrature intNodes;
    Mat intMat;

    PetscCall(PetscDualSpaceGetPointSubspace(sp, p, &psp));
    if (!psp) continue;
    PetscCall(PetscDualSpaceGetDM(psp, &pdm));
    PetscCall(DMGetDimension(pdm, &pdim));
    if (pdim < PetscAbsInt(k)) continue;
    PetscCall(PetscDTBinomialInt(pdim, PetscAbsInt(k), &pNk));
    PetscCall(PetscDualSpaceGetInteriorData(psp, &intNodes, &intMat));
    if (intNodes) {
      PetscInt nNodesp;

      PetscCall(PetscQuadratureGetData(intNodes, NULL, NULL, &nNodesp, NULL, NULL));
      nNodes += nNodesp;
    }
    if (intMat) {
      PetscInt maxNzsp;
      PetscInt maxNzformsp;

      PetscCall(MatSeqAIJGetMaxRowNonzeros(intMat, &maxNzsp));
      PetscCheck(maxNzsp % pNk == 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "interior matrix is not laid out as blocks of k-forms");
      maxNzformsp = maxNzsp / pNk;
      maxNzforms = PetscMax(maxNzforms, maxNzformsp);
    }
  }
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, nDofs, nNodes * Nc, maxNzforms * Nk, NULL, &allMat));
  PetscCall(MatSetOption(allMat,MAT_IGNORE_ZERO_ENTRIES,PETSC_FALSE));
  PetscCall(PetscMalloc7(dim, &v0, dim, &pv0, dim * dim, &J, dim * dim, &Jinv, Nk * Nk, &L, maxNzforms * Nk, &work, maxNzforms * Nk, &iwork));
  for (j = 0; j < dim; j++) pv0[j] = -1.;
  PetscCall(PetscMalloc1(dim * nNodes, &nodes));
  for (p = pStart, countNodes = 0; p < pEnd; p++) {
    PetscDualSpace  psp;
    PetscQuadrature intNodes;
    DM pdm;
    PetscInt pdim, pNk;
    PetscInt countNodesIn = countNodes;
    PetscReal detJ;
    Mat intMat;

    PetscCall(PetscDualSpaceGetPointSubspace(sp, p, &psp));
    if (!psp) continue;
    PetscCall(PetscDualSpaceGetDM(psp, &pdm));
    PetscCall(DMGetDimension(pdm, &pdim));
    if (pdim < PetscAbsInt(k)) continue;
    PetscCall(PetscDualSpaceGetInteriorData(psp, &intNodes, &intMat));
    if (intNodes == NULL && intMat == NULL) continue;
    PetscCall(PetscDTBinomialInt(pdim, PetscAbsInt(k), &pNk));
    if (p) {
      PetscCall(DMPlexComputeCellGeometryAffineFEM(dm, p, v0, J, Jinv, &detJ));
    } else { /* identity */
      PetscInt i,j;

      for (i = 0; i < dim; i++) for (j = 0; j < dim; j++) J[i * dim + j] = Jinv[i * dim + j] = 0.;
      for (i = 0; i < dim; i++) J[i * dim + i] = Jinv[i * dim + i] = 1.;
      for (i = 0; i < dim; i++) v0[i] = -1.;
    }
    if (pdim != dim) { /* compactify Jacobian */
      PetscInt i, j;

      for (i = 0; i < dim; i++) for (j = 0; j < pdim; j++) J[i * pdim + j] = J[i * dim + j];
    }
    PetscCall(PetscDTAltVPullbackMatrix(pdim, dim, J, k, L));
    if (intNodes) { /* push forward quadrature locations by the affine transformation */
      PetscInt nNodesp;
      const PetscReal *nodesp;
      PetscInt j;

      PetscCall(PetscQuadratureGetData(intNodes, NULL, NULL, &nNodesp, &nodesp, NULL));
      for (j = 0; j < nNodesp; j++, countNodes++) {
        PetscInt d, e;

        for (d = 0; d < dim; d++) {
          nodes[countNodes * dim + d] = v0[d];
          for (e = 0; e < pdim; e++) {
            nodes[countNodes * dim + d] += J[d * pdim + e] * (nodesp[j * pdim + e] - pv0[e]);
          }
        }
      }
    }
    if (intMat) {
      PetscInt nrows;
      PetscInt off;

      PetscCall(PetscSectionGetDof(section, p, &nrows));
      PetscCall(PetscSectionGetOffset(section, p, &off));
      for (j = 0; j < nrows; j++) {
        PetscInt ncols;
        const PetscInt *cols;
        const PetscScalar *vals;
        PetscInt l, d, e;
        PetscInt row = j + off;

        PetscCall(MatGetRow(intMat, j, &ncols, &cols, &vals));
        PetscCheck(ncols % pNk == 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "interior matrix is not laid out as blocks of k-forms");
        for (l = 0; l < ncols / pNk; l++) {
          PetscInt blockcol;

          for (d = 0; d < pNk; d++) {
            PetscCheck((cols[l * pNk + d] % pNk) == d,PETSC_COMM_SELF, PETSC_ERR_PLIB, "interior matrix is not laid out as blocks of k-forms");
          }
          blockcol = cols[l * pNk] / pNk;
          for (d = 0; d < Nk; d++) {
            iwork[l * Nk + d] = (blockcol + countNodesIn) * Nk + d;
          }
          for (d = 0; d < Nk; d++) work[l * Nk + d] = 0.;
          for (d = 0; d < Nk; d++) {
            for (e = 0; e < pNk; e++) {
              /* "push forward" dof by pulling back a k-form to be evaluated on the point: multiply on the right by L */
              work[l * Nk + d] += vals[l * pNk + e] * L[e * Nk + d];
            }
          }
        }
        PetscCall(MatSetValues(allMat, 1, &row, (ncols / pNk) * Nk, iwork, work, INSERT_VALUES));
        PetscCall(MatRestoreRow(intMat, j, &ncols, &cols, &vals));
      }
    }
  }
  PetscCall(MatAssemblyBegin(allMat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(allMat, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, &allNodes));
  PetscCall(PetscQuadratureSetData(allNodes, dim, 0, nNodes, nodes, NULL));
  PetscCall(PetscFree7(v0, pv0, J, Jinv, L, work, iwork));
  PetscCall(MatDestroy(&(sp->allMat)));
  sp->allMat = allMat;
  PetscCall(PetscQuadratureDestroy(&(sp->allNodes)));
  sp->allNodes = allNodes;
  PetscFunctionReturn(0);
}

/* rather than trying to get all data from the functionals, we create
 * the functionals from rows of the quadrature -> dof matrix.
 *
 * Ideally most of the uses of PetscDualSpace in PetscFE will switch
 * to using intMat and allMat, so that the individual functionals
 * don't need to be constructed at all */
static PetscErrorCode PetscDualSpaceComputeFunctionalsFromAllData(PetscDualSpace sp)
{
  PetscQuadrature allNodes;
  Mat             allMat;
  PetscInt        nDofs;
  PetscInt        dim, k, Nk, Nc, f;
  DM              dm;
  PetscInt        nNodes, spdim;
  const PetscReal *nodes = NULL;
  PetscSection    section;
  PetscBool       useMoments;

  PetscFunctionBegin;
  PetscCall(PetscDualSpaceGetDM(sp, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscDualSpaceGetNumComponents(sp, &Nc));
  PetscCall(PetscDualSpaceGetFormDegree(sp, &k));
  PetscCall(PetscDTBinomialInt(dim, PetscAbsInt(k), &Nk));
  PetscCall(PetscDualSpaceGetAllData(sp, &allNodes, &allMat));
  nNodes = 0;
  if (allNodes) {
    PetscCall(PetscQuadratureGetData(allNodes, NULL, NULL, &nNodes, &nodes, NULL));
  }
  PetscCall(MatGetSize(allMat, &nDofs, NULL));
  PetscCall(PetscDualSpaceGetSection(sp, &section));
  PetscCall(PetscSectionGetStorageSize(section, &spdim));
  PetscCheck(spdim == nDofs,PETSC_COMM_SELF, PETSC_ERR_PLIB, "incompatible all matrix size");
  PetscCall(PetscMalloc1(nDofs, &(sp->functional)));
  PetscCall(PetscDualSpaceLagrangeGetUseMoments(sp, &useMoments));
  if (useMoments) {
    Mat              allMat;
    PetscInt         momentOrder, i;
    PetscBool        tensor;
    const PetscReal *weights;
    PetscScalar     *array;

    PetscCheck(nDofs == 1,PETSC_COMM_SELF, PETSC_ERR_SUP, "We do not yet support moments beyond P0, nDofs == %D", nDofs);
    PetscCall(PetscDualSpaceLagrangeGetMomentOrder(sp, &momentOrder));
    PetscCall(PetscDualSpaceLagrangeGetTensor(sp, &tensor));
    if (!tensor) PetscCall(PetscDTStroudConicalQuadrature(dim, Nc, PetscMax(momentOrder + 1,1), -1.0, 1.0, &(sp->functional[0])));
    else         PetscCall(PetscDTGaussTensorQuadrature(dim, Nc, PetscMax(momentOrder + 1,1), -1.0, 1.0, &(sp->functional[0])));
    /* Need to replace allNodes and allMat */
    PetscCall(PetscObjectReference((PetscObject) sp->functional[0]));
    PetscCall(PetscQuadratureDestroy(&(sp->allNodes)));
    sp->allNodes = sp->functional[0];
    PetscCall(PetscQuadratureGetData(sp->allNodes, NULL, NULL, &nNodes, NULL, &weights));
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, nDofs, nNodes * Nc, NULL, &allMat));
    PetscCall(MatDenseGetArrayWrite(allMat, &array));
    for (i = 0; i < nNodes * Nc; ++i) array[i] = weights[i];
    PetscCall(MatDenseRestoreArrayWrite(allMat, &array));
    PetscCall(MatAssemblyBegin(allMat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(allMat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatDestroy(&(sp->allMat)));
    sp->allMat = allMat;
    PetscFunctionReturn(0);
  }
  for (f = 0; f < nDofs; f++) {
    PetscInt ncols, c;
    const PetscInt *cols;
    const PetscScalar *vals;
    PetscReal *nodesf;
    PetscReal *weightsf;
    PetscInt nNodesf;
    PetscInt countNodes;

    PetscCall(MatGetRow(allMat, f, &ncols, &cols, &vals));
    PetscCheck(ncols % Nk == 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "all matrix is not laid out as blocks of k-forms");
    for (c = 1, nNodesf = 1; c < ncols; c++) {
      if ((cols[c] / Nc) != (cols[c-1] / Nc)) nNodesf++;
    }
    PetscCall(PetscMalloc1(dim * nNodesf, &nodesf));
    PetscCall(PetscMalloc1(Nc * nNodesf, &weightsf));
    for (c = 0, countNodes = 0; c < ncols; c++) {
      if (!c || ((cols[c] / Nc) != (cols[c-1] / Nc))) {
        PetscInt d;

        for (d = 0; d < Nc; d++) {
          weightsf[countNodes * Nc + d] = 0.;
        }
        for (d = 0; d < dim; d++) {
          nodesf[countNodes * dim + d] = nodes[(cols[c] / Nc) * dim + d];
        }
        countNodes++;
      }
      weightsf[(countNodes - 1) * Nc + (cols[c] % Nc)] = PetscRealPart(vals[c]);
    }
    PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, &(sp->functional[f])));
    PetscCall(PetscQuadratureSetData(sp->functional[f], dim, Nc, nNodesf, nodesf, weightsf));
    PetscCall(MatRestoreRow(allMat, f, &ncols, &cols, &vals));
  }
  PetscFunctionReturn(0);
}

/* take a matrix meant for k-forms and expand it to one for Ncopies */
static PetscErrorCode PetscDualSpaceLagrangeMatrixCreateCopies(Mat A, PetscInt Nk, PetscInt Ncopies, Mat *Abs)
{
  PetscInt       m, n, i, j, k;
  PetscInt       maxnnz, *nnz, *iwork;
  Mat            Ac;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A, &m, &n));
  PetscCheck(n % Nk == 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of columns in A %D is not a multiple of Nk %D", n, Nk);
  PetscCall(PetscMalloc1(m * Ncopies, &nnz));
  for (i = 0, maxnnz = 0; i < m; i++) {
    PetscInt innz;
    PetscCall(MatGetRow(A, i, &innz, NULL, NULL));
    PetscCheck(innz % Nk == 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "A row %D nnzs is not a multiple of Nk %D", innz, Nk);
    for (j = 0; j < Ncopies; j++) nnz[i * Ncopies + j] = innz;
    maxnnz = PetscMax(maxnnz, innz);
  }
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, m * Ncopies, n * Ncopies, 0, nnz, &Ac));
  PetscCall(MatSetOption(Ac, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE));
  PetscCall(PetscFree(nnz));
  PetscCall(PetscMalloc1(maxnnz, &iwork));
  for (i = 0; i < m; i++) {
    PetscInt innz;
    const PetscInt    *cols;
    const PetscScalar *vals;

    PetscCall(MatGetRow(A, i, &innz, &cols, &vals));
    for (j = 0; j < innz; j++) iwork[j] = (cols[j] / Nk) * (Nk * Ncopies) + (cols[j] % Nk);
    for (j = 0; j < Ncopies; j++) {
      PetscInt row = i * Ncopies + j;

      PetscCall(MatSetValues(Ac, 1, &row, innz, iwork, vals, INSERT_VALUES));
      for (k = 0; k < innz; k++) iwork[k] += Nk;
    }
    PetscCall(MatRestoreRow(A, i, &innz, &cols, &vals));
  }
  PetscCall(PetscFree(iwork));
  PetscCall(MatAssemblyBegin(Ac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Ac, MAT_FINAL_ASSEMBLY));
  *Abs = Ac;
  PetscFunctionReturn(0);
}

/* check if a cell is a tensor product of the segment with a facet,
 * specifically checking if f and f2 can be the "endpoints" (like the triangles
 * at either end of a wedge) */
static PetscErrorCode DMPlexPointIsTensor_Internal_Given(DM dm, PetscInt p, PetscInt f, PetscInt f2, PetscBool *isTensor)
{
  PetscInt        coneSize, c;
  const PetscInt *cone;
  const PetscInt *fCone;
  const PetscInt *f2Cone;
  PetscInt        fs[2];
  PetscInt        meetSize, nmeet;
  const PetscInt *meet;

  PetscFunctionBegin;
  fs[0] = f;
  fs[1] = f2;
  PetscCall(DMPlexGetMeet(dm, 2, fs, &meetSize, &meet));
  nmeet = meetSize;
  PetscCall(DMPlexRestoreMeet(dm, 2, fs, &meetSize, &meet));
  /* two points that have a non-empty meet cannot be at opposite ends of a cell */
  if (nmeet) {
    *isTensor = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
  PetscCall(DMPlexGetCone(dm, p, &cone));
  PetscCall(DMPlexGetCone(dm, f, &fCone));
  PetscCall(DMPlexGetCone(dm, f2, &f2Cone));
  for (c = 0; c < coneSize; c++) {
    PetscInt e, ef;
    PetscInt d = -1, d2 = -1;
    PetscInt dcount, d2count;
    PetscInt t = cone[c];
    PetscInt tConeSize;
    PetscBool tIsTensor;
    const PetscInt *tCone;

    if (t == f || t == f2) continue;
    /* for every other facet in the cone, check that is has
     * one ridge in common with each end */
    PetscCall(DMPlexGetConeSize(dm, t, &tConeSize));
    PetscCall(DMPlexGetCone(dm, t, &tCone));

    dcount = 0;
    d2count = 0;
    for (e = 0; e < tConeSize; e++) {
      PetscInt q = tCone[e];
      for (ef = 0; ef < coneSize - 2; ef++) {
        if (fCone[ef] == q) {
          if (dcount) {
            *isTensor = PETSC_FALSE;
            PetscFunctionReturn(0);
          }
          d = q;
          dcount++;
        } else if (f2Cone[ef] == q) {
          if (d2count) {
            *isTensor = PETSC_FALSE;
            PetscFunctionReturn(0);
          }
          d2 = q;
          d2count++;
        }
      }
    }
    /* if the whole cell is a tensor with the segment, then this
     * facet should be a tensor with the segment */
    PetscCall(DMPlexPointIsTensor_Internal_Given(dm, t, d, d2, &tIsTensor));
    if (!tIsTensor) {
      *isTensor = PETSC_FALSE;
      PetscFunctionReturn(0);
    }
  }
  *isTensor = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* determine if a cell is a tensor with a segment by looping over pairs of facets to find a pair
 * that could be the opposite ends */
static PetscErrorCode DMPlexPointIsTensor_Internal(DM dm, PetscInt p, PetscBool *isTensor, PetscInt *endA, PetscInt *endB)
{
  PetscInt        coneSize, c, c2;
  const PetscInt *cone;

  PetscFunctionBegin;
  PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
  if (!coneSize) {
    if (isTensor) *isTensor = PETSC_FALSE;
    if (endA) *endA = -1;
    if (endB) *endB = -1;
  }
  PetscCall(DMPlexGetCone(dm, p, &cone));
  for (c = 0; c < coneSize; c++) {
    PetscInt f = cone[c];
    PetscInt fConeSize;

    PetscCall(DMPlexGetConeSize(dm, f, &fConeSize));
    if (fConeSize != coneSize - 2) continue;

    for (c2 = c + 1; c2 < coneSize; c2++) {
      PetscInt  f2 = cone[c2];
      PetscBool isTensorff2;
      PetscInt f2ConeSize;

      PetscCall(DMPlexGetConeSize(dm, f2, &f2ConeSize));
      if (f2ConeSize != coneSize - 2) continue;

      PetscCall(DMPlexPointIsTensor_Internal_Given(dm, p, f, f2, &isTensorff2));
      if (isTensorff2) {
        if (isTensor) *isTensor = PETSC_TRUE;
        if (endA) *endA = f;
        if (endB) *endB = f2;
        PetscFunctionReturn(0);
      }
    }
  }
  if (isTensor) *isTensor = PETSC_FALSE;
  if (endA) *endA = -1;
  if (endB) *endB = -1;
  PetscFunctionReturn(0);
}

/* determine if a cell is a tensor with a segment by looping over pairs of facets to find a pair
 * that could be the opposite ends */
static PetscErrorCode DMPlexPointIsTensor(DM dm, PetscInt p, PetscBool *isTensor, PetscInt *endA, PetscInt *endB)
{
  DMPlexInterpolatedFlag interpolated;

  PetscFunctionBegin;
  PetscCall(DMPlexIsInterpolated(dm, &interpolated));
  PetscCheck(interpolated == DMPLEX_INTERPOLATED_FULL,PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Only for interpolated DMPlex's");
  PetscCall(DMPlexPointIsTensor_Internal(dm, p, isTensor, endA, endB));
  PetscFunctionReturn(0);
}

/* Let k = formDegree and k' = -sign(k) * dim + k.  Transform a symmetric frame for k-forms on the biunit simplex into
 * a symmetric frame for k'-forms on the biunit simplex.
 *
 * A frame is "symmetric" if the pullback of every symmetry of the biunit simplex is a permutation of the frame.
 *
 * forms in the symmetric frame are used as dofs in the untrimmed simplex spaces.  This way, symmetries of the
 * reference cell result in permutations of dofs grouped by node.
 *
 * Use T to transform dof matrices for k'-forms into dof matrices for k-forms as a block diagonal transformation on
 * the right.
 */
static PetscErrorCode BiunitSimplexSymmetricFormTransformation(PetscInt dim, PetscInt formDegree, PetscReal T[])
{
  PetscInt       k = formDegree;
  PetscInt       kd = k < 0 ? dim + k : k - dim;
  PetscInt       Nk;
  PetscReal      *biToEq, *eqToBi, *biToEqStar, *eqToBiStar;
  PetscInt       fact;

  PetscFunctionBegin;
  PetscCall(PetscDTBinomialInt(dim, PetscAbsInt(k), &Nk));
  PetscCall(PetscCalloc4(dim * dim, &biToEq, dim * dim, &eqToBi, Nk * Nk, &biToEqStar, Nk * Nk, &eqToBiStar));
  /* fill in biToEq: Jacobian of the transformation from the biunit simplex to the equilateral simplex */
  fact = 0;
  for (PetscInt i = 0; i < dim; i++) {
    biToEq[i * dim + i] = PetscSqrtReal(((PetscReal)i + 2.) / (2.*((PetscReal)i+1.)));
    fact += 4*(i+1);
    for (PetscInt j = i+1; j < dim; j++) {
      biToEq[i * dim + j] = PetscSqrtReal(1./(PetscReal)fact);
    }
  }
  /* fill in eqToBi: Jacobian of the transformation from the equilateral simplex to the biunit simplex */
  fact = 0;
  for (PetscInt j = 0; j < dim; j++) {
    eqToBi[j * dim + j] = PetscSqrtReal(2.*((PetscReal)j+1.)/((PetscReal)j+2));
    fact += j+1;
    for (PetscInt i = 0; i < j; i++) {
      eqToBi[i * dim + j] = -PetscSqrtReal(1./(PetscReal)fact);
    }
  }
  PetscCall(PetscDTAltVPullbackMatrix(dim, dim, biToEq, kd, biToEqStar));
  PetscCall(PetscDTAltVPullbackMatrix(dim, dim, eqToBi, k, eqToBiStar));
  /* product of pullbacks simulates the following steps
   *
   * 1. start with frame W = [w_1, w_2, ..., w_m] of k forms that is symmetric on the biunit simplex:
          if J is the Jacobian of a symmetry of the biunit simplex, then J_k* W = [J_k*w_1, ..., J_k*w_m]
          is a permutation of W.
          Even though a k' form --- a (dim - k) form represented by its Hodge star --- has the same geometric
          content as a k form, W is not a symmetric frame of k' forms on the biunit simplex.  That's because,
          for general Jacobian J, J_k* != J_k'*.
   * 2. pullback W to the equilateral triangle using the k pullback, W_eq = eqToBi_k* W.  All symmetries of the
          equilateral simplex have orthonormal Jacobians.  For an orthonormal Jacobian O, J_k* = J_k'*, so W_eq is
          also a symmetric frame for k' forms on the equilateral simplex.
     3. pullback W_eq back to the biunit simplex using the k' pulback, V = biToEq_k'* W_eq = biToEq_k'* eqToBi_k* W.
          V is a symmetric frame for k' forms on the biunit simplex.
   */
  for (PetscInt i = 0; i < Nk; i++) {
    for (PetscInt j = 0; j < Nk; j++) {
      PetscReal val = 0.;
      for (PetscInt k = 0; k < Nk; k++) val += biToEqStar[i * Nk + k] * eqToBiStar[k * Nk + j];
      T[i * Nk + j] = val;
    }
  }
  PetscCall(PetscFree4(biToEq, eqToBi, biToEqStar, eqToBiStar));
  PetscFunctionReturn(0);
}

/* permute a quadrature -> dof matrix so that its rows are in revlex order by nodeIdx */
static PetscErrorCode MatPermuteByNodeIdx(Mat A, PetscLagNodeIndices ni, Mat *Aperm)
{
  PetscInt       m, n, i, j;
  PetscInt       nodeIdxDim = ni->nodeIdxDim;
  PetscInt       nodeVecDim = ni->nodeVecDim;
  PetscInt       *perm;
  IS             permIS;
  IS             id;
  PetscInt       *nIdxPerm;
  PetscReal      *nVecPerm;

  PetscFunctionBegin;
  PetscCall(PetscLagNodeIndicesGetPermutation(ni, &perm));
  PetscCall(MatGetSize(A, &m, &n));
  PetscCall(PetscMalloc1(nodeIdxDim * m, &nIdxPerm));
  PetscCall(PetscMalloc1(nodeVecDim * m, &nVecPerm));
  for (i = 0; i < m; i++) for (j = 0; j < nodeIdxDim; j++) nIdxPerm[i * nodeIdxDim + j] = ni->nodeIdx[perm[i] * nodeIdxDim + j];
  for (i = 0; i < m; i++) for (j = 0; j < nodeVecDim; j++) nVecPerm[i * nodeVecDim + j] = ni->nodeVec[perm[i] * nodeVecDim + j];
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, m, perm, PETSC_USE_POINTER, &permIS));
  PetscCall(ISSetPermutation(permIS));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, n, 0, 1, &id));
  PetscCall(ISSetPermutation(id));
  PetscCall(MatPermute(A, permIS, id, Aperm));
  PetscCall(ISDestroy(&permIS));
  PetscCall(ISDestroy(&id));
  for (i = 0; i < m; i++) perm[i] = i;
  PetscCall(PetscFree(ni->nodeIdx));
  PetscCall(PetscFree(ni->nodeVec));
  ni->nodeIdx = nIdxPerm;
  ni->nodeVec = nVecPerm;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceSetUp_Lagrange(PetscDualSpace sp)
{
  PetscDualSpace_Lag *lag   = (PetscDualSpace_Lag *) sp->data;
  DM                  dm    = sp->dm;
  DM                  dmint = NULL;
  PetscInt            order;
  PetscInt            Nc    = sp->Nc;
  MPI_Comm            comm;
  PetscBool           continuous;
  PetscSection        section;
  PetscInt            depth, dim, pStart, pEnd, cStart, cEnd, p, *pStratStart, *pStratEnd, d;
  PetscInt            formDegree, Nk, Ncopies;
  PetscInt            tensorf = -1, tensorf2 = -1;
  PetscBool           tensorCell, tensorSpace;
  PetscBool           uniform, trimmed;
  Petsc1DNodeFamily   nodeFamily;
  PetscInt            numNodeSkip;
  DMPlexInterpolatedFlag interpolated;
  PetscBool           isbdm;

  PetscFunctionBegin;
  /* step 1: sanitize input */
  PetscCall(PetscObjectGetComm((PetscObject) sp, &comm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscObjectTypeCompare((PetscObject)sp, PETSCDUALSPACEBDM, &isbdm));
  if (isbdm) {
    sp->k = -(dim-1); /* form degree of H-div */
    PetscCall(PetscObjectChangeTypeName((PetscObject)sp, PETSCDUALSPACELAGRANGE));
  }
  PetscCall(PetscDualSpaceGetFormDegree(sp, &formDegree));
  PetscCheck(PetscAbsInt(formDegree) <= dim,comm, PETSC_ERR_ARG_OUTOFRANGE, "Form degree must be bounded by dimension");
  PetscCall(PetscDTBinomialInt(dim,PetscAbsInt(formDegree),&Nk));
  if (sp->Nc <= 0 && lag->numCopies > 0) sp->Nc = Nk * lag->numCopies;
  Nc = sp->Nc;
  PetscCheck(Nc % Nk == 0,comm, PETSC_ERR_ARG_INCOMP, "Number of components is not a multiple of form degree size");
  if (lag->numCopies <= 0) lag->numCopies = Nc / Nk;
  Ncopies = lag->numCopies;
  PetscCheckFalse(Nc / Nk != Ncopies,comm, PETSC_ERR_ARG_INCOMP, "Number of copies * (dim choose k) != Nc");
  if (!dim) sp->order = 0;
  order = sp->order;
  uniform = sp->uniform;
  PetscCheck(uniform,PETSC_COMM_SELF, PETSC_ERR_SUP, "Variable order not supported yet");
  if (lag->trimmed && !formDegree) lag->trimmed = PETSC_FALSE; /* trimmed spaces are the same as full spaces for 0-forms */
  if (lag->nodeType == PETSCDTNODES_DEFAULT) {
    lag->nodeType = PETSCDTNODES_GAUSSJACOBI;
    lag->nodeExponent = 0.;
    /* trimmed spaces don't include corner vertices, so don't use end nodes by default */
    lag->endNodes = lag->trimmed ? PETSC_FALSE : PETSC_TRUE;
  }
  /* If a trimmed space and the user did choose nodes with endpoints, skip them by default */
  if (lag->numNodeSkip < 0) lag->numNodeSkip = (lag->trimmed && lag->endNodes) ? 1 : 0;
  numNodeSkip = lag->numNodeSkip;
  PetscCheck(!lag->trimmed || order,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot have zeroth order trimmed elements");
  if (lag->trimmed && PetscAbsInt(formDegree) == dim) { /* convert trimmed n-forms to untrimmed of one polynomial order less */
    sp->order--;
    order--;
    lag->trimmed = PETSC_FALSE;
  }
  trimmed = lag->trimmed;
  if (!order || PetscAbsInt(formDegree) == dim) lag->continuous = PETSC_FALSE;
  continuous = lag->continuous;
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCheckFalse(pStart != 0 || cStart != 0,PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "Expect DM with chart starting at zero and cells first");
  PetscCheck(cEnd == 1,PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "Use PETSCDUALSPACEREFINED for multi-cell reference meshes");
  PetscCall(DMPlexIsInterpolated(dm, &interpolated));
  if (interpolated != DMPLEX_INTERPOLATED_FULL) {
    PetscCall(DMPlexInterpolate(dm, &dmint));
  } else {
    PetscCall(PetscObjectReference((PetscObject)dm));
    dmint = dm;
  }
  tensorCell = PETSC_FALSE;
  if (dim > 1) {
    PetscCall(DMPlexPointIsTensor(dmint, 0, &tensorCell, &tensorf, &tensorf2));
  }
  lag->tensorCell = tensorCell;
  if (dim < 2 || !lag->tensorCell) lag->tensorSpace = PETSC_FALSE;
  tensorSpace = lag->tensorSpace;
  if (!lag->nodeFamily) {
    PetscCall(Petsc1DNodeFamilyCreate(lag->nodeType, lag->nodeExponent, lag->endNodes, &lag->nodeFamily));
  }
  nodeFamily = lag->nodeFamily;
  PetscCheckFalse(interpolated != DMPLEX_INTERPOLATED_FULL && continuous && (PetscAbsInt(formDegree) > 0 || order > 1),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Reference element won't support all boundary nodes");

  /* step 2: construct the boundary spaces */
  PetscCall(PetscMalloc2(depth+1,&pStratStart,depth+1,&pStratEnd));
  PetscCall(PetscCalloc1(pEnd,&(sp->pointSpaces)));
  for (d = 0; d <= depth; ++d) PetscCall(DMPlexGetDepthStratum(dm, d, &pStratStart[d], &pStratEnd[d]));
  PetscCall(PetscDualSpaceSectionCreate_Internal(sp, &section));
  sp->pointSection = section;
  if (continuous && !(lag->interiorOnly)) {
    PetscInt h;

    for (p = pStratStart[depth - 1]; p < pStratEnd[depth - 1]; p++) { /* calculate the facet dual spaces */
      PetscReal v0[3];
      DMPolytopeType ptype;
      PetscReal J[9], detJ;
      PetscInt  q;

      PetscCall(DMPlexComputeCellGeometryAffineFEM(dm, p, v0, J, NULL, &detJ));
      PetscCall(DMPlexGetCellType(dm, p, &ptype));

      /* compare to previous facets: if computed, reference that dualspace */
      for (q = pStratStart[depth - 1]; q < p; q++) {
        DMPolytopeType qtype;

        PetscCall(DMPlexGetCellType(dm, q, &qtype));
        if (qtype == ptype) break;
      }
      if (q < p) { /* this facet has the same dual space as that one */
        PetscCall(PetscObjectReference((PetscObject)sp->pointSpaces[q]));
        sp->pointSpaces[p] = sp->pointSpaces[q];
        continue;
      }
      /* if not, recursively compute this dual space */
      PetscCall(PetscDualSpaceCreateFacetSubspace_Lagrange(sp,NULL,p,formDegree,Ncopies,PETSC_FALSE,&sp->pointSpaces[p]));
    }
    for (h = 2; h <= depth; h++) { /* get the higher subspaces from the facet subspaces */
      PetscInt hd = depth - h;
      PetscInt hdim = dim - h;

      if (hdim < PetscAbsInt(formDegree)) break;
      for (p = pStratStart[hd]; p < pStratEnd[hd]; p++) {
        PetscInt suppSize, s;
        const PetscInt *supp;

        PetscCall(DMPlexGetSupportSize(dm, p, &suppSize));
        PetscCall(DMPlexGetSupport(dm, p, &supp));
        for (s = 0; s < suppSize; s++) {
          DM             qdm;
          PetscDualSpace qsp, psp;
          PetscInt c, coneSize, q;
          const PetscInt *cone;
          const PetscInt *refCone;

          q = supp[0];
          qsp = sp->pointSpaces[q];
          PetscCall(DMPlexGetConeSize(dm, q, &coneSize));
          PetscCall(DMPlexGetCone(dm, q, &cone));
          for (c = 0; c < coneSize; c++) if (cone[c] == p) break;
          PetscCheck(c != coneSize,PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "cone/support mismatch");
          PetscCall(PetscDualSpaceGetDM(qsp, &qdm));
          PetscCall(DMPlexGetCone(qdm, 0, &refCone));
          /* get the equivalent dual space from the support dual space */
          PetscCall(PetscDualSpaceGetPointSubspace(qsp, refCone[c], &psp));
          if (!s) {
            PetscCall(PetscObjectReference((PetscObject)psp));
            sp->pointSpaces[p] = psp;
          }
        }
      }
    }
    for (p = 1; p < pEnd; p++) {
      PetscInt pspdim;
      if (!sp->pointSpaces[p]) continue;
      PetscCall(PetscDualSpaceGetInteriorDimension(sp->pointSpaces[p], &pspdim));
      PetscCall(PetscSectionSetDof(section, p, pspdim));
    }
  }

  if (Ncopies > 1) {
    Mat intMatScalar, allMatScalar;
    PetscDualSpace scalarsp;
    PetscDualSpace_Lag *scalarlag;

    PetscCall(PetscDualSpaceDuplicate(sp, &scalarsp));
    /* Setting the number of components to Nk is a space with 1 copy of each k-form */
    PetscCall(PetscDualSpaceSetNumComponents(scalarsp, Nk));
    PetscCall(PetscDualSpaceSetUp(scalarsp));
    PetscCall(PetscDualSpaceGetInteriorData(scalarsp, &(sp->intNodes), &intMatScalar));
    PetscCall(PetscObjectReference((PetscObject)(sp->intNodes)));
    if (intMatScalar) PetscCall(PetscDualSpaceLagrangeMatrixCreateCopies(intMatScalar, Nk, Ncopies, &(sp->intMat)));
    PetscCall(PetscDualSpaceGetAllData(scalarsp, &(sp->allNodes), &allMatScalar));
    PetscCall(PetscObjectReference((PetscObject)(sp->allNodes)));
    PetscCall(PetscDualSpaceLagrangeMatrixCreateCopies(allMatScalar, Nk, Ncopies, &(sp->allMat)));
    sp->spdim = scalarsp->spdim * Ncopies;
    sp->spintdim = scalarsp->spintdim * Ncopies;
    scalarlag = (PetscDualSpace_Lag *) scalarsp->data;
    PetscCall(PetscLagNodeIndicesReference(scalarlag->vertIndices));
    lag->vertIndices = scalarlag->vertIndices;
    PetscCall(PetscLagNodeIndicesReference(scalarlag->intNodeIndices));
    lag->intNodeIndices = scalarlag->intNodeIndices;
    PetscCall(PetscLagNodeIndicesReference(scalarlag->allNodeIndices));
    lag->allNodeIndices = scalarlag->allNodeIndices;
    PetscCall(PetscDualSpaceDestroy(&scalarsp));
    PetscCall(PetscSectionSetDof(section, 0, sp->spintdim));
    PetscCall(PetscDualSpaceSectionSetUp_Internal(sp, section));
    PetscCall(PetscDualSpaceComputeFunctionalsFromAllData(sp));
    PetscCall(PetscFree2(pStratStart, pStratEnd));
    PetscCall(DMDestroy(&dmint));
    PetscFunctionReturn(0);
  }

  if (trimmed && !continuous) {
    /* the dofs of a trimmed space don't have a nice tensor/lattice structure:
     * just construct the continuous dual space and copy all of the data over,
     * allocating it all to the cell instead of splitting it up between the boundaries */
    PetscDualSpace  spcont;
    PetscInt        spdim, f;
    PetscQuadrature allNodes;
    PetscDualSpace_Lag *lagc;
    Mat             allMat;

    PetscCall(PetscDualSpaceDuplicate(sp, &spcont));
    PetscCall(PetscDualSpaceLagrangeSetContinuity(spcont, PETSC_TRUE));
    PetscCall(PetscDualSpaceSetUp(spcont));
    PetscCall(PetscDualSpaceGetDimension(spcont, &spdim));
    sp->spdim = sp->spintdim = spdim;
    PetscCall(PetscSectionSetDof(section, 0, spdim));
    PetscCall(PetscDualSpaceSectionSetUp_Internal(sp, section));
    PetscCall(PetscMalloc1(spdim, &(sp->functional)));
    for (f = 0; f < spdim; f++) {
      PetscQuadrature fn;

      PetscCall(PetscDualSpaceGetFunctional(spcont, f, &fn));
      PetscCall(PetscObjectReference((PetscObject)fn));
      sp->functional[f] = fn;
    }
    PetscCall(PetscDualSpaceGetAllData(spcont, &allNodes, &allMat));
    PetscCall(PetscObjectReference((PetscObject) allNodes));
    PetscCall(PetscObjectReference((PetscObject) allNodes));
    sp->allNodes = sp->intNodes = allNodes;
    PetscCall(PetscObjectReference((PetscObject) allMat));
    PetscCall(PetscObjectReference((PetscObject) allMat));
    sp->allMat = sp->intMat = allMat;
    lagc = (PetscDualSpace_Lag *) spcont->data;
    PetscCall(PetscLagNodeIndicesReference(lagc->vertIndices));
    lag->vertIndices = lagc->vertIndices;
    PetscCall(PetscLagNodeIndicesReference(lagc->allNodeIndices));
    PetscCall(PetscLagNodeIndicesReference(lagc->allNodeIndices));
    lag->intNodeIndices = lagc->allNodeIndices;
    lag->allNodeIndices = lagc->allNodeIndices;
    PetscCall(PetscDualSpaceDestroy(&spcont));
    PetscCall(PetscFree2(pStratStart, pStratEnd));
    PetscCall(DMDestroy(&dmint));
    PetscFunctionReturn(0);
  }

  /* step 3: construct intNodes, and intMat, and combine it with boundray data to make allNodes and allMat */
  if (!tensorSpace) {
    if (!tensorCell) PetscCall(PetscLagNodeIndicesCreateSimplexVertices(dm, &(lag->vertIndices)));

    if (trimmed) {
      /* there is one dof in the interior of the a trimmed element for each full polynomial of with degree at most
       * order + k - dim - 1 */
      if (order + PetscAbsInt(formDegree) > dim) {
        PetscInt sum = order + PetscAbsInt(formDegree) - dim - 1;
        PetscInt nDofs;

        PetscCall(PetscDualSpaceLagrangeCreateSimplexNodeMat(nodeFamily, dim, sum, Nk, numNodeSkip, &sp->intNodes, &sp->intMat, &(lag->intNodeIndices)));
        PetscCall(MatGetSize(sp->intMat, &nDofs, NULL));
        PetscCall(PetscSectionSetDof(section, 0, nDofs));
      }
      PetscCall(PetscDualSpaceSectionSetUp_Internal(sp, section));
      PetscCall(PetscDualSpaceCreateAllDataFromInteriorData(sp));
      PetscCall(PetscDualSpaceLagrangeCreateAllNodeIdx(sp));
    } else {
      if (!continuous) {
        /* if discontinuous just construct one node for each set of dofs (a set of dofs is a basis for the k-form
         * space) */
        PetscInt sum = order;
        PetscInt nDofs;

        PetscCall(PetscDualSpaceLagrangeCreateSimplexNodeMat(nodeFamily, dim, sum, Nk, numNodeSkip, &sp->intNodes, &sp->intMat, &(lag->intNodeIndices)));
        PetscCall(MatGetSize(sp->intMat, &nDofs, NULL));
        PetscCall(PetscSectionSetDof(section, 0, nDofs));
        PetscCall(PetscDualSpaceSectionSetUp_Internal(sp, section));
        PetscCall(PetscObjectReference((PetscObject)(sp->intNodes)));
        sp->allNodes = sp->intNodes;
        PetscCall(PetscObjectReference((PetscObject)(sp->intMat)));
        sp->allMat = sp->intMat;
        PetscCall(PetscLagNodeIndicesReference(lag->intNodeIndices));
        lag->allNodeIndices = lag->intNodeIndices;
      } else {
        /* there is one dof in the interior of the a full element for each trimmed polynomial of with degree at most
         * order + k - dim, but with complementary form degree */
        if (order + PetscAbsInt(formDegree) > dim) {
          PetscDualSpace trimmedsp;
          PetscDualSpace_Lag *trimmedlag;
          PetscQuadrature intNodes;
          PetscInt trFormDegree = formDegree >= 0 ? formDegree - dim : dim - PetscAbsInt(formDegree);
          PetscInt nDofs;
          Mat intMat;

          PetscCall(PetscDualSpaceDuplicate(sp, &trimmedsp));
          PetscCall(PetscDualSpaceLagrangeSetTrimmed(trimmedsp, PETSC_TRUE));
          PetscCall(PetscDualSpaceSetOrder(trimmedsp, order + PetscAbsInt(formDegree) - dim));
          PetscCall(PetscDualSpaceSetFormDegree(trimmedsp, trFormDegree));
          trimmedlag = (PetscDualSpace_Lag *) trimmedsp->data;
          trimmedlag->numNodeSkip = numNodeSkip + 1;
          PetscCall(PetscDualSpaceSetUp(trimmedsp));
          PetscCall(PetscDualSpaceGetAllData(trimmedsp, &intNodes, &intMat));
          PetscCall(PetscObjectReference((PetscObject)intNodes));
          sp->intNodes = intNodes;
          PetscCall(PetscLagNodeIndicesReference(trimmedlag->allNodeIndices));
          lag->intNodeIndices = trimmedlag->allNodeIndices;
          PetscCall(PetscObjectReference((PetscObject)intMat));
          if (PetscAbsInt(formDegree) > 0 && PetscAbsInt(formDegree) < dim) {
            PetscReal *T;
            PetscScalar *work;
            PetscInt nCols, nRows;
            Mat intMatT;

            PetscCall(MatDuplicate(intMat, MAT_COPY_VALUES, &intMatT));
            PetscCall(MatGetSize(intMat, &nRows, &nCols));
            PetscCall(PetscMalloc2(Nk * Nk, &T, nCols, &work));
            PetscCall(BiunitSimplexSymmetricFormTransformation(dim, formDegree, T));
            for (PetscInt row = 0; row < nRows; row++) {
              PetscInt nrCols;
              const PetscInt *rCols;
              const PetscScalar *rVals;

              PetscCall(MatGetRow(intMat, row, &nrCols, &rCols, &rVals));
              PetscCheck(nrCols % Nk == 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "nonzeros in intMat matrix are not in k-form size blocks");
              for (PetscInt b = 0; b < nrCols; b += Nk) {
                const PetscScalar *v = &rVals[b];
                PetscScalar *w = &work[b];
                for (PetscInt j = 0; j < Nk; j++) {
                  w[j] = 0.;
                  for (PetscInt i = 0; i < Nk; i++) {
                    w[j] += v[i] * T[i * Nk + j];
                  }
                }
              }
              PetscCall(MatSetValuesBlocked(intMatT, 1, &row, nrCols, rCols, work, INSERT_VALUES));
              PetscCall(MatRestoreRow(intMat, row, &nrCols, &rCols, &rVals));
            }
            PetscCall(MatAssemblyBegin(intMatT, MAT_FINAL_ASSEMBLY));
            PetscCall(MatAssemblyEnd(intMatT, MAT_FINAL_ASSEMBLY));
            PetscCall(MatDestroy(&intMat));
            intMat = intMatT;
            PetscCall(PetscLagNodeIndicesDestroy(&(lag->intNodeIndices)));
            PetscCall(PetscLagNodeIndicesDuplicate(trimmedlag->allNodeIndices, &(lag->intNodeIndices)));
            {
              PetscInt nNodes = lag->intNodeIndices->nNodes;
              PetscReal *newNodeVec = lag->intNodeIndices->nodeVec;
              const PetscReal *oldNodeVec = trimmedlag->allNodeIndices->nodeVec;

              for (PetscInt n = 0; n < nNodes; n++) {
                PetscReal *w = &newNodeVec[n * Nk];
                const PetscReal *v = &oldNodeVec[n * Nk];

                for (PetscInt j = 0; j < Nk; j++) {
                  w[j] = 0.;
                  for (PetscInt i = 0; i < Nk; i++) {
                    w[j] += v[i] * T[i * Nk + j];
                  }
                }
              }
            }
            PetscCall(PetscFree2(T, work));
          }
          sp->intMat = intMat;
          PetscCall(MatGetSize(sp->intMat, &nDofs, NULL));
          PetscCall(PetscDualSpaceDestroy(&trimmedsp));
          PetscCall(PetscSectionSetDof(section, 0, nDofs));
        }
        PetscCall(PetscDualSpaceSectionSetUp_Internal(sp, section));
        PetscCall(PetscDualSpaceCreateAllDataFromInteriorData(sp));
        PetscCall(PetscDualSpaceLagrangeCreateAllNodeIdx(sp));
      }
    }
  } else {
    PetscQuadrature intNodesTrace = NULL;
    PetscQuadrature intNodesFiber = NULL;
    PetscQuadrature intNodes = NULL;
    PetscLagNodeIndices intNodeIndices = NULL;
    Mat             intMat = NULL;

    if (PetscAbsInt(formDegree) < dim) { /* get the trace k-forms on the first facet, and the 0-forms on the edge,
                                            and wedge them together to create some of the k-form dofs */
      PetscDualSpace  trace, fiber;
      PetscDualSpace_Lag *tracel, *fiberl;
      Mat             intMatTrace, intMatFiber;

      if (sp->pointSpaces[tensorf]) {
        PetscCall(PetscObjectReference((PetscObject)(sp->pointSpaces[tensorf])));
        trace = sp->pointSpaces[tensorf];
      } else {
        PetscCall(PetscDualSpaceCreateFacetSubspace_Lagrange(sp,NULL,tensorf,formDegree,Ncopies,PETSC_TRUE,&trace));
      }
      PetscCall(PetscDualSpaceCreateEdgeSubspace_Lagrange(sp,order,0,1,PETSC_TRUE,&fiber));
      tracel = (PetscDualSpace_Lag *) trace->data;
      fiberl = (PetscDualSpace_Lag *) fiber->data;
      PetscCall(PetscLagNodeIndicesCreateTensorVertices(dm, tracel->vertIndices, &(lag->vertIndices)));
      PetscCall(PetscDualSpaceGetInteriorData(trace, &intNodesTrace, &intMatTrace));
      PetscCall(PetscDualSpaceGetInteriorData(fiber, &intNodesFiber, &intMatFiber));
      if (intNodesTrace && intNodesFiber) {
        PetscCall(PetscQuadratureCreateTensor(intNodesTrace, intNodesFiber, &intNodes));
        PetscCall(MatTensorAltV(intMatTrace, intMatFiber, dim-1, formDegree, 1, 0, &intMat));
        PetscCall(PetscLagNodeIndicesTensor(tracel->intNodeIndices, dim - 1, formDegree, fiberl->intNodeIndices, 1, 0, &intNodeIndices));
      }
      PetscCall(PetscObjectReference((PetscObject) intNodesTrace));
      PetscCall(PetscObjectReference((PetscObject) intNodesFiber));
      PetscCall(PetscDualSpaceDestroy(&fiber));
      PetscCall(PetscDualSpaceDestroy(&trace));
    }
    if (PetscAbsInt(formDegree) > 0) { /* get the trace (k-1)-forms on the first facet, and the 1-forms on the edge,
                                          and wedge them together to create the remaining k-form dofs */
      PetscDualSpace  trace, fiber;
      PetscDualSpace_Lag *tracel, *fiberl;
      PetscQuadrature intNodesTrace2, intNodesFiber2, intNodes2;
      PetscLagNodeIndices intNodeIndices2;
      Mat             intMatTrace, intMatFiber, intMat2;
      PetscInt        traceDegree = formDegree > 0 ? formDegree - 1 : formDegree + 1;
      PetscInt        fiberDegree = formDegree > 0 ? 1 : -1;

      PetscCall(PetscDualSpaceCreateFacetSubspace_Lagrange(sp,NULL,tensorf,traceDegree,Ncopies,PETSC_TRUE,&trace));
      PetscCall(PetscDualSpaceCreateEdgeSubspace_Lagrange(sp,order,fiberDegree,1,PETSC_TRUE,&fiber));
      tracel = (PetscDualSpace_Lag *) trace->data;
      fiberl = (PetscDualSpace_Lag *) fiber->data;
      if (!lag->vertIndices) {
        PetscCall(PetscLagNodeIndicesCreateTensorVertices(dm, tracel->vertIndices, &(lag->vertIndices)));
      }
      PetscCall(PetscDualSpaceGetInteriorData(trace, &intNodesTrace2, &intMatTrace));
      PetscCall(PetscDualSpaceGetInteriorData(fiber, &intNodesFiber2, &intMatFiber));
      if (intNodesTrace2 && intNodesFiber2) {
        PetscCall(PetscQuadratureCreateTensor(intNodesTrace2, intNodesFiber2, &intNodes2));
        PetscCall(MatTensorAltV(intMatTrace, intMatFiber, dim-1, traceDegree, 1, fiberDegree, &intMat2));
        PetscCall(PetscLagNodeIndicesTensor(tracel->intNodeIndices, dim - 1, traceDegree, fiberl->intNodeIndices, 1, fiberDegree, &intNodeIndices2));
        if (!intMat) {
          intMat = intMat2;
          intNodes = intNodes2;
          intNodeIndices = intNodeIndices2;
        } else {
          /* merge the matrices, quadrature points, and nodes */
          PetscInt         nM;
          PetscInt         nDof, nDof2;
          PetscInt        *toMerged = NULL, *toMerged2 = NULL;
          PetscQuadrature  merged = NULL;
          PetscLagNodeIndices intNodeIndicesMerged = NULL;
          Mat              matMerged = NULL;

          PetscCall(MatGetSize(intMat, &nDof, NULL));
          PetscCall(MatGetSize(intMat2, &nDof2, NULL));
          PetscCall(PetscQuadraturePointsMerge(intNodes, intNodes2, &merged, &toMerged, &toMerged2));
          PetscCall(PetscQuadratureGetData(merged, NULL, NULL, &nM, NULL, NULL));
          PetscCall(MatricesMerge(intMat, intMat2, dim, formDegree, nM, toMerged, toMerged2, &matMerged));
          PetscCall(PetscLagNodeIndicesMerge(intNodeIndices, intNodeIndices2, &intNodeIndicesMerged));
          PetscCall(PetscFree(toMerged));
          PetscCall(PetscFree(toMerged2));
          PetscCall(MatDestroy(&intMat));
          PetscCall(MatDestroy(&intMat2));
          PetscCall(PetscQuadratureDestroy(&intNodes));
          PetscCall(PetscQuadratureDestroy(&intNodes2));
          PetscCall(PetscLagNodeIndicesDestroy(&intNodeIndices));
          PetscCall(PetscLagNodeIndicesDestroy(&intNodeIndices2));
          intNodes = merged;
          intMat = matMerged;
          intNodeIndices = intNodeIndicesMerged;
          if (!trimmed) {
            /* I think users expect that, when a node has a full basis for the k-forms,
             * they should be consecutive dofs.  That isn't the case for trimmed spaces,
             * but is for some of the nodes in untrimmed spaces, so in that case we
             * sort them to group them by node */
            Mat intMatPerm;

            PetscCall(MatPermuteByNodeIdx(intMat, intNodeIndices, &intMatPerm));
            PetscCall(MatDestroy(&intMat));
            intMat = intMatPerm;
          }
        }
      }
      PetscCall(PetscDualSpaceDestroy(&fiber));
      PetscCall(PetscDualSpaceDestroy(&trace));
    }
    PetscCall(PetscQuadratureDestroy(&intNodesTrace));
    PetscCall(PetscQuadratureDestroy(&intNodesFiber));
    sp->intNodes = intNodes;
    sp->intMat = intMat;
    lag->intNodeIndices = intNodeIndices;
    {
      PetscInt nDofs = 0;

      if (intMat) {
        PetscCall(MatGetSize(intMat, &nDofs, NULL));
      }
      PetscCall(PetscSectionSetDof(section, 0, nDofs));
    }
    PetscCall(PetscDualSpaceSectionSetUp_Internal(sp, section));
    if (continuous) {
      PetscCall(PetscDualSpaceCreateAllDataFromInteriorData(sp));
      PetscCall(PetscDualSpaceLagrangeCreateAllNodeIdx(sp));
    } else {
      PetscCall(PetscObjectReference((PetscObject) intNodes));
      sp->allNodes = intNodes;
      PetscCall(PetscObjectReference((PetscObject) intMat));
      sp->allMat = intMat;
      PetscCall(PetscLagNodeIndicesReference(intNodeIndices));
      lag->allNodeIndices = intNodeIndices;
    }
  }
  PetscCall(PetscSectionGetStorageSize(section, &sp->spdim));
  PetscCall(PetscSectionGetConstrainedStorageSize(section, &sp->spintdim));
  PetscCall(PetscDualSpaceComputeFunctionalsFromAllData(sp));
  PetscCall(PetscFree2(pStratStart, pStratEnd));
  PetscCall(DMDestroy(&dmint));
  PetscFunctionReturn(0);
}

/* Create a matrix that represents the transformation that DMPlexVecGetClosure() would need
 * to get the representation of the dofs for a mesh point if the mesh point had this orientation
 * relative to the cell */
PetscErrorCode PetscDualSpaceCreateInteriorSymmetryMatrix_Lagrange(PetscDualSpace sp, PetscInt ornt, Mat *symMat)
{
  PetscDualSpace_Lag *lag;
  DM dm;
  PetscLagNodeIndices vertIndices, intNodeIndices;
  PetscLagNodeIndices ni;
  PetscInt nodeIdxDim, nodeVecDim, nNodes;
  PetscInt formDegree;
  PetscInt *perm, *permOrnt;
  PetscInt *nnz;
  PetscInt n;
  PetscInt maxGroupSize;
  PetscScalar *V, *W, *work;
  Mat A;

  PetscFunctionBegin;
  if (!sp->spintdim) {
    *symMat = NULL;
    PetscFunctionReturn(0);
  }
  lag = (PetscDualSpace_Lag *) sp->data;
  vertIndices = lag->vertIndices;
  intNodeIndices = lag->intNodeIndices;
  PetscCall(PetscDualSpaceGetDM(sp, &dm));
  PetscCall(PetscDualSpaceGetFormDegree(sp, &formDegree));
  PetscCall(PetscNew(&ni));
  ni->refct = 1;
  ni->nodeIdxDim = nodeIdxDim = intNodeIndices->nodeIdxDim;
  ni->nodeVecDim = nodeVecDim = intNodeIndices->nodeVecDim;
  ni->nNodes = nNodes = intNodeIndices->nNodes;
  PetscCall(PetscMalloc1(nNodes * nodeIdxDim, &(ni->nodeIdx)));
  PetscCall(PetscMalloc1(nNodes * nodeVecDim, &(ni->nodeVec)));
  /* push forward the dofs by the symmetry of the reference element induced by ornt */
  PetscCall(PetscLagNodeIndicesPushForward(dm, vertIndices, 0, vertIndices, intNodeIndices, ornt, formDegree, ni->nodeIdx, ni->nodeVec));
  /* get the revlex order for both the original and transformed dofs */
  PetscCall(PetscLagNodeIndicesGetPermutation(intNodeIndices, &perm));
  PetscCall(PetscLagNodeIndicesGetPermutation(ni, &permOrnt));
  PetscCall(PetscMalloc1(nNodes, &nnz));
  for (n = 0, maxGroupSize = 0; n < nNodes;) { /* incremented in the loop */
    PetscInt *nind = &(ni->nodeIdx[permOrnt[n] * nodeIdxDim]);
    PetscInt m, nEnd;
    PetscInt groupSize;
    /* for each group of dofs that have the same nodeIdx coordinate */
    for (nEnd = n + 1; nEnd < nNodes; nEnd++) {
      PetscInt *mind = &(ni->nodeIdx[permOrnt[nEnd] * nodeIdxDim]);
      PetscInt d;

      /* compare the oriented permutation indices */
      for (d = 0; d < nodeIdxDim; d++) if (mind[d] != nind[d]) break;
      if (d < nodeIdxDim) break;
    }
    /* permOrnt[[n, nEnd)] is a group of dofs that, under the symmetry are at the same location */

    /* the symmetry had better map the group of dofs with the same permuted nodeIdx
     * to a group of dofs with the same size, otherwise we messed up */
    if (PetscDefined(USE_DEBUG)) {
      PetscInt m;
      PetscInt *nind = &(intNodeIndices->nodeIdx[perm[n] * nodeIdxDim]);

      for (m = n + 1; m < nEnd; m++) {
        PetscInt *mind = &(intNodeIndices->nodeIdx[perm[m] * nodeIdxDim]);
        PetscInt d;

        /* compare the oriented permutation indices */
        for (d = 0; d < nodeIdxDim; d++) if (mind[d] != nind[d]) break;
        if (d < nodeIdxDim) break;
      }
      PetscCheck(m >= nEnd,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Dofs with same index after symmetry not same block size");
    }
    groupSize = nEnd - n;
    /* each pushforward dof vector will be expressed in a basis of the unpermuted dofs */
    for (m = n; m < nEnd; m++) nnz[permOrnt[m]] = groupSize;

    maxGroupSize = PetscMax(maxGroupSize, nEnd - n);
    n = nEnd;
  }
  PetscCheck(maxGroupSize <= nodeVecDim,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Dofs are not in blocks that can be solved");
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, nNodes, nNodes, 0, nnz, &A));
  PetscCall(PetscFree(nnz));
  PetscCall(PetscMalloc3(maxGroupSize * nodeVecDim, &V, maxGroupSize * nodeVecDim, &W, nodeVecDim * 2, &work));
  for (n = 0; n < nNodes;) { /* incremented in the loop */
    PetscInt *nind = &(ni->nodeIdx[permOrnt[n] * nodeIdxDim]);
    PetscInt nEnd;
    PetscInt m;
    PetscInt groupSize;
    for (nEnd = n + 1; nEnd < nNodes; nEnd++) {
      PetscInt *mind = &(ni->nodeIdx[permOrnt[nEnd] * nodeIdxDim]);
      PetscInt d;

      /* compare the oriented permutation indices */
      for (d = 0; d < nodeIdxDim; d++) if (mind[d] != nind[d]) break;
      if (d < nodeIdxDim) break;
    }
    groupSize = nEnd - n;
    /* get all of the vectors from the original and all of the pushforward vectors */
    for (m = n; m < nEnd; m++) {
      PetscInt d;

      for (d = 0; d < nodeVecDim; d++) {
        V[(m - n) * nodeVecDim + d] = intNodeIndices->nodeVec[perm[m] * nodeVecDim + d];
        W[(m - n) * nodeVecDim + d] = ni->nodeVec[permOrnt[m] * nodeVecDim + d];
      }
    }
    /* now we have to solve for W in terms of V: the systems isn't always square, but the span
     * of V and W should always be the same, so the solution of the normal equations works */
    {
      char transpose = 'N';
      PetscBLASInt bm = nodeVecDim;
      PetscBLASInt bn = groupSize;
      PetscBLASInt bnrhs = groupSize;
      PetscBLASInt blda = bm;
      PetscBLASInt bldb = bm;
      PetscBLASInt blwork = 2 * nodeVecDim;
      PetscBLASInt info;

      PetscStackCallBLAS("LAPACKgels",LAPACKgels_(&transpose,&bm,&bn,&bnrhs,V,&blda,W,&bldb,work,&blwork, &info));
      PetscCheck(info == 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to GELS");
      /* repack */
      {
        PetscInt i, j;

        for (i = 0; i < groupSize; i++) {
          for (j = 0; j < groupSize; j++) {
            /* notice the different leading dimension */
            V[i * groupSize + j] = W[i * nodeVecDim + j];
          }
        }
      }
      if (PetscDefined(USE_DEBUG)) {
        PetscReal res;

        /* check that the normal error is 0 */
        for (m = n; m < nEnd; m++) {
          PetscInt d;

          for (d = 0; d < nodeVecDim; d++) {
            W[(m - n) * nodeVecDim + d] = ni->nodeVec[permOrnt[m] * nodeVecDim + d];
          }
        }
        res = 0.;
        for (PetscInt i = 0; i < groupSize; i++) {
          for (PetscInt j = 0; j < nodeVecDim; j++) {
            for (PetscInt k = 0; k < groupSize; k++) {
              W[i * nodeVecDim + j] -= V[i * groupSize + k] * intNodeIndices->nodeVec[perm[n+k] * nodeVecDim + j];
            }
            res += PetscAbsScalar(W[i * nodeVecDim + j]);
          }
        }
        PetscCheck(res <= PETSC_SMALL,PETSC_COMM_SELF,PETSC_ERR_LIB,"Dof block did not solve");
      }
    }
    PetscCall(MatSetValues(A, groupSize, &permOrnt[n], groupSize, &perm[n], V, INSERT_VALUES));
    n = nEnd;
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  *symMat = A;
  PetscCall(PetscFree3(V,W,work));
  PetscCall(PetscLagNodeIndicesDestroy(&ni));
  PetscFunctionReturn(0);
}

#define BaryIndex(perEdge,a,b,c) (((b)*(2*perEdge+1-(b)))/2)+(c)

#define CartIndex(perEdge,a,b) (perEdge*(a)+b)

/* the existing interface for symmetries is insufficient for all cases:
 * - it should be sufficient for form degrees that are scalar (0 and n)
 * - it should be sufficient for hypercube dofs
 * - it isn't sufficient for simplex cells with non-scalar form degrees if
 *   there are any dofs in the interior
 *
 * We compute the general transformation matrices, and if they fit, we return them,
 * otherwise we error (but we should probably change the interface to allow for
 * these symmetries)
 */
static PetscErrorCode PetscDualSpaceGetSymmetries_Lagrange(PetscDualSpace sp, const PetscInt ****perms, const PetscScalar ****flips)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscInt           dim, order, Nc;

  PetscFunctionBegin;
  PetscCall(PetscDualSpaceGetOrder(sp,&order));
  PetscCall(PetscDualSpaceGetNumComponents(sp,&Nc));
  PetscCall(DMGetDimension(sp->dm,&dim));
  if (!lag->symComputed) { /* store symmetries */
    PetscInt       pStart, pEnd, p;
    PetscInt       numPoints;
    PetscInt       numFaces;
    PetscInt       spintdim;
    PetscInt       ***symperms;
    PetscScalar    ***symflips;

    PetscCall(DMPlexGetChart(sp->dm, &pStart, &pEnd));
    numPoints = pEnd - pStart;
    {
      DMPolytopeType ct;
      /* The number of arrangements is no longer based on the number of faces */
      PetscCall(DMPlexGetCellType(sp->dm, 0, &ct));
      numFaces = DMPolytopeTypeGetNumArrangments(ct) / 2;
    }
    PetscCall(PetscCalloc1(numPoints,&symperms));
    PetscCall(PetscCalloc1(numPoints,&symflips));
    spintdim = sp->spintdim;
    /* The nodal symmetry behavior is not present when tensorSpace != tensorCell: someone might want this for the "S"
     * family of FEEC spaces.  Most used in particular are discontinuous polynomial L2 spaces in tensor cells, where
     * the symmetries are not necessary for FE assembly.  So for now we assume this is the case and don't return
     * symmetries if tensorSpace != tensorCell */
    if (spintdim && 0 < dim && dim < 3 && (lag->tensorSpace == lag->tensorCell)) { /* compute self symmetries */
      PetscInt **cellSymperms;
      PetscScalar **cellSymflips;
      PetscInt ornt;
      PetscInt nCopies = Nc / lag->intNodeIndices->nodeVecDim;
      PetscInt nNodes = lag->intNodeIndices->nNodes;

      lag->numSelfSym = 2 * numFaces;
      lag->selfSymOff = numFaces;
      PetscCall(PetscCalloc1(2*numFaces,&cellSymperms));
      PetscCall(PetscCalloc1(2*numFaces,&cellSymflips));
      /* we want to be able to index symmetries directly with the orientations, which range from [-numFaces,numFaces) */
      symperms[0] = &cellSymperms[numFaces];
      symflips[0] = &cellSymflips[numFaces];
      PetscCheckFalse(lag->intNodeIndices->nodeVecDim * nCopies != Nc,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Node indices incompatible with dofs");
      PetscCheckFalse(nNodes * nCopies != spintdim,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Node indices incompatible with dofs");
      for (ornt = -numFaces; ornt < numFaces; ornt++) { /* for every symmetry, compute the symmetry matrix, and extract rows to see if it fits in the perm + flip framework */
        Mat symMat;
        PetscInt *perm;
        PetscScalar *flips;
        PetscInt i;

        if (!ornt) continue;
        PetscCall(PetscMalloc1(spintdim, &perm));
        PetscCall(PetscCalloc1(spintdim, &flips));
        for (i = 0; i < spintdim; i++) perm[i] = -1;
        PetscCall(PetscDualSpaceCreateInteriorSymmetryMatrix_Lagrange(sp, ornt, &symMat));
        for (i = 0; i < nNodes; i++) {
          PetscInt ncols;
          PetscInt j, k;
          const PetscInt *cols;
          const PetscScalar *vals;
          PetscBool nz_seen = PETSC_FALSE;

          PetscCall(MatGetRow(symMat, i, &ncols, &cols, &vals));
          for (j = 0; j < ncols; j++) {
            if (PetscAbsScalar(vals[j]) > PETSC_SMALL) {
              PetscCheck(!nz_seen,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "This dual space has symmetries that can't be described as a permutation + sign flips");
              nz_seen = PETSC_TRUE;
              PetscCheckFalse(PetscAbsReal(PetscAbsScalar(vals[j]) - PetscRealConstant(1.)) > PETSC_SMALL,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "This dual space has symmetries that can't be described as a permutation + sign flips");
              PetscCheckFalse(PetscAbsReal(PetscImaginaryPart(vals[j])) > PETSC_SMALL,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "This dual space has symmetries that can't be described as a permutation + sign flips");
              PetscCheckFalse(perm[cols[j] * nCopies] >= 0,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "This dual space has symmetries that can't be described as a permutation + sign flips");
              for (k = 0; k < nCopies; k++) {
                perm[cols[j] * nCopies + k] = i * nCopies + k;
              }
              if (PetscRealPart(vals[j]) < 0.) {
                for (k = 0; k < nCopies; k++) {
                  flips[i * nCopies + k] = -1.;
                }
              } else {
                for (k = 0; k < nCopies; k++) {
                  flips[i * nCopies + k] = 1.;
                }
              }
            }
          }
          PetscCall(MatRestoreRow(symMat, i, &ncols, &cols, &vals));
        }
        PetscCall(MatDestroy(&symMat));
        /* if there were no sign flips, keep NULL */
        for (i = 0; i < spintdim; i++) if (flips[i] != 1.) break;
        if (i == spintdim) {
          PetscCall(PetscFree(flips));
          flips = NULL;
        }
        /* if the permutation is identity, keep NULL */
        for (i = 0; i < spintdim; i++) if (perm[i] != i) break;
        if (i == spintdim) {
          PetscCall(PetscFree(perm));
          perm = NULL;
        }
        symperms[0][ornt] = perm;
        symflips[0][ornt] = flips;
      }
      /* if no orientations produced non-identity permutations, keep NULL */
      for (ornt = -numFaces; ornt < numFaces; ornt++) if (symperms[0][ornt]) break;
      if (ornt == numFaces) {
        PetscCall(PetscFree(cellSymperms));
        symperms[0] = NULL;
      }
      /* if no orientations produced sign flips, keep NULL */
      for (ornt = -numFaces; ornt < numFaces; ornt++) if (symflips[0][ornt]) break;
      if (ornt == numFaces) {
        PetscCall(PetscFree(cellSymflips));
        symflips[0] = NULL;
      }
    }
    { /* get the symmetries of closure points */
      PetscInt closureSize = 0;
      PetscInt *closure = NULL;
      PetscInt r;

      PetscCall(DMPlexGetTransitiveClosure(sp->dm,0,PETSC_TRUE,&closureSize,&closure));
      for (r = 0; r < closureSize; r++) {
        PetscDualSpace psp;
        PetscInt point = closure[2 * r];
        PetscInt pspintdim;
        const PetscInt ***psymperms = NULL;
        const PetscScalar ***psymflips = NULL;

        if (!point) continue;
        PetscCall(PetscDualSpaceGetPointSubspace(sp, point, &psp));
        if (!psp) continue;
        PetscCall(PetscDualSpaceGetInteriorDimension(psp, &pspintdim));
        if (!pspintdim) continue;
        PetscCall(PetscDualSpaceGetSymmetries(psp,&psymperms,&psymflips));
        symperms[r] = (PetscInt **) (psymperms ? psymperms[0] : NULL);
        symflips[r] = (PetscScalar **) (psymflips ? psymflips[0] : NULL);
      }
      PetscCall(DMPlexRestoreTransitiveClosure(sp->dm,0,PETSC_TRUE,&closureSize,&closure));
    }
    for (p = 0; p < pEnd; p++) if (symperms[p]) break;
    if (p == pEnd) {
      PetscCall(PetscFree(symperms));
      symperms = NULL;
    }
    for (p = 0; p < pEnd; p++) if (symflips[p]) break;
    if (p == pEnd) {
      PetscCall(PetscFree(symflips));
      symflips = NULL;
    }
    lag->symperms = symperms;
    lag->symflips = symflips;
    lag->symComputed = PETSC_TRUE;
  }
  if (perms) *perms = (const PetscInt ***) lag->symperms;
  if (flips) *flips = (const PetscScalar ***) lag->symflips;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceLagrangeGetContinuity_Lagrange(PetscDualSpace sp, PetscBool *continuous)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidBoolPointer(continuous, 2);
  *continuous = lag->continuous;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceLagrangeSetContinuity_Lagrange(PetscDualSpace sp, PetscBool continuous)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  lag->continuous = continuous;
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceLagrangeGetContinuity - Retrieves the flag for element continuity

  Not Collective

  Input Parameter:
. sp         - the PetscDualSpace

  Output Parameter:
. continuous - flag for element continuity

  Level: intermediate

.seealso: PetscDualSpaceLagrangeSetContinuity()
@*/
PetscErrorCode PetscDualSpaceLagrangeGetContinuity(PetscDualSpace sp, PetscBool *continuous)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidBoolPointer(continuous, 2);
  PetscTryMethod(sp, "PetscDualSpaceLagrangeGetContinuity_C", (PetscDualSpace,PetscBool*),(sp,continuous));
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceLagrangeSetContinuity - Indicate whether the element is continuous

  Logically Collective on sp

  Input Parameters:
+ sp         - the PetscDualSpace
- continuous - flag for element continuity

  Options Database:
. -petscdualspace_lagrange_continuity <bool> - use a continuous element

  Level: intermediate

.seealso: PetscDualSpaceLagrangeGetContinuity()
@*/
PetscErrorCode PetscDualSpaceLagrangeSetContinuity(PetscDualSpace sp, PetscBool continuous)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidLogicalCollectiveBool(sp, continuous, 2);
  PetscTryMethod(sp, "PetscDualSpaceLagrangeSetContinuity_C", (PetscDualSpace,PetscBool),(sp,continuous));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceLagrangeGetTensor_Lagrange(PetscDualSpace sp, PetscBool *tensor)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *)sp->data;

  PetscFunctionBegin;
  *tensor = lag->tensorSpace;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceLagrangeSetTensor_Lagrange(PetscDualSpace sp, PetscBool tensor)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *)sp->data;

  PetscFunctionBegin;
  lag->tensorSpace = tensor;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceLagrangeGetTrimmed_Lagrange(PetscDualSpace sp, PetscBool *trimmed)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *)sp->data;

  PetscFunctionBegin;
  *trimmed = lag->trimmed;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceLagrangeSetTrimmed_Lagrange(PetscDualSpace sp, PetscBool trimmed)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *)sp->data;

  PetscFunctionBegin;
  lag->trimmed = trimmed;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceLagrangeGetNodeType_Lagrange(PetscDualSpace sp, PetscDTNodeType *nodeType, PetscBool *boundary, PetscReal *exponent)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *)sp->data;

  PetscFunctionBegin;
  if (nodeType) *nodeType = lag->nodeType;
  if (boundary) *boundary = lag->endNodes;
  if (exponent) *exponent = lag->nodeExponent;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceLagrangeSetNodeType_Lagrange(PetscDualSpace sp, PetscDTNodeType nodeType, PetscBool boundary, PetscReal exponent)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *)sp->data;

  PetscFunctionBegin;
  PetscCheckFalse(nodeType == PETSCDTNODES_GAUSSJACOBI && exponent <= -1.,PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_OUTOFRANGE, "Exponent must be > -1");
  lag->nodeType = nodeType;
  lag->endNodes = boundary;
  lag->nodeExponent = exponent;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceLagrangeGetUseMoments_Lagrange(PetscDualSpace sp, PetscBool *useMoments)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *)sp->data;

  PetscFunctionBegin;
  *useMoments = lag->useMoments;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceLagrangeSetUseMoments_Lagrange(PetscDualSpace sp, PetscBool useMoments)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *)sp->data;

  PetscFunctionBegin;
  lag->useMoments = useMoments;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceLagrangeGetMomentOrder_Lagrange(PetscDualSpace sp, PetscInt *momentOrder)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *)sp->data;

  PetscFunctionBegin;
  *momentOrder = lag->momentOrder;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceLagrangeSetMomentOrder_Lagrange(PetscDualSpace sp, PetscInt momentOrder)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *)sp->data;

  PetscFunctionBegin;
  lag->momentOrder = momentOrder;
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceLagrangeGetTensor - Get the tensor nature of the dual space

  Not collective

  Input Parameter:
. sp - The PetscDualSpace

  Output Parameter:
. tensor - Whether the dual space has tensor layout (vs. simplicial)

  Level: intermediate

.seealso: PetscDualSpaceLagrangeSetTensor(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceLagrangeGetTensor(PetscDualSpace sp, PetscBool *tensor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidBoolPointer(tensor, 2);
  PetscTryMethod(sp,"PetscDualSpaceLagrangeGetTensor_C",(PetscDualSpace,PetscBool *),(sp,tensor));
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceLagrangeSetTensor - Set the tensor nature of the dual space

  Not collective

  Input Parameters:
+ sp - The PetscDualSpace
- tensor - Whether the dual space has tensor layout (vs. simplicial)

  Level: intermediate

.seealso: PetscDualSpaceLagrangeGetTensor(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceLagrangeSetTensor(PetscDualSpace sp, PetscBool tensor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscTryMethod(sp,"PetscDualSpaceLagrangeSetTensor_C",(PetscDualSpace,PetscBool),(sp,tensor));
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceLagrangeGetTrimmed - Get the trimmed nature of the dual space

  Not collective

  Input Parameter:
. sp - The PetscDualSpace

  Output Parameter:
. trimmed - Whether the dual space represents to dual basis of a trimmed polynomial space (e.g. Raviart-Thomas and higher order / other form degree variants)

  Level: intermediate

.seealso: PetscDualSpaceLagrangeSetTrimmed(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceLagrangeGetTrimmed(PetscDualSpace sp, PetscBool *trimmed)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidBoolPointer(trimmed, 2);
  PetscTryMethod(sp,"PetscDualSpaceLagrangeGetTrimmed_C",(PetscDualSpace,PetscBool *),(sp,trimmed));
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceLagrangeSetTrimmed - Set the trimmed nature of the dual space

  Not collective

  Input Parameters:
+ sp - The PetscDualSpace
- trimmed - Whether the dual space represents to dual basis of a trimmed polynomial space (e.g. Raviart-Thomas and higher order / other form degree variants)

  Level: intermediate

.seealso: PetscDualSpaceLagrangeGetTrimmed(), PetscDualSpaceCreate()
@*/
PetscErrorCode PetscDualSpaceLagrangeSetTrimmed(PetscDualSpace sp, PetscBool trimmed)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscTryMethod(sp,"PetscDualSpaceLagrangeSetTrimmed_C",(PetscDualSpace,PetscBool),(sp,trimmed));
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceLagrangeGetNodeType - Get a description of how nodes are laid out for Lagrange polynomials in this
  dual space

  Not collective

  Input Parameter:
. sp - The PetscDualSpace

  Output Parameters:
+ nodeType - The type of nodes
. boundary - Whether the node type is one that includes endpoints (if nodeType is PETSCDTNODES_GAUSSJACOBI, nodes that
             include the boundary are Gauss-Lobatto-Jacobi nodes)
- exponent - If nodeType is PETSCDTNODES_GAUSJACOBI, indicates the exponent used for both ends of the 1D Jacobi weight function
             '0' is Gauss-Legendre, '-0.5' is Gauss-Chebyshev of the first type, '0.5' is Gauss-Chebyshev of the second type

  Level: advanced

.seealso: PetscDTNodeType, PetscDualSpaceLagrangeSetNodeType()
@*/
PetscErrorCode PetscDualSpaceLagrangeGetNodeType(PetscDualSpace sp, PetscDTNodeType *nodeType, PetscBool *boundary, PetscReal *exponent)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  if (nodeType) PetscValidPointer(nodeType, 2);
  if (boundary) PetscValidBoolPointer(boundary, 3);
  if (exponent) PetscValidRealPointer(exponent, 4);
  PetscTryMethod(sp,"PetscDualSpaceLagrangeGetNodeType_C",(PetscDualSpace,PetscDTNodeType *,PetscBool *,PetscReal *),(sp,nodeType,boundary,exponent));
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceLagrangeSetNodeType - Set a description of how nodes are laid out for Lagrange polynomials in this
  dual space

  Logically collective

  Input Parameters:
+ sp - The PetscDualSpace
. nodeType - The type of nodes
. boundary - Whether the node type is one that includes endpoints (if nodeType is PETSCDTNODES_GAUSSJACOBI, nodes that
             include the boundary are Gauss-Lobatto-Jacobi nodes)
- exponent - If nodeType is PETSCDTNODES_GAUSJACOBI, indicates the exponent used for both ends of the 1D Jacobi weight function
             '0' is Gauss-Legendre, '-0.5' is Gauss-Chebyshev of the first type, '0.5' is Gauss-Chebyshev of the second type

  Level: advanced

.seealso: PetscDTNodeType, PetscDualSpaceLagrangeGetNodeType()
@*/
PetscErrorCode PetscDualSpaceLagrangeSetNodeType(PetscDualSpace sp, PetscDTNodeType nodeType, PetscBool boundary, PetscReal exponent)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscTryMethod(sp,"PetscDualSpaceLagrangeSetNodeType_C",(PetscDualSpace,PetscDTNodeType,PetscBool,PetscReal),(sp,nodeType,boundary,exponent));
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceLagrangeGetUseMoments - Get the flag for using moment functionals

  Not collective

  Input Parameter:
. sp - The PetscDualSpace

  Output Parameter:
. useMoments - Moment flag

  Level: advanced

.seealso: PetscDualSpaceLagrangeSetUseMoments()
@*/
PetscErrorCode PetscDualSpaceLagrangeGetUseMoments(PetscDualSpace sp, PetscBool *useMoments)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidBoolPointer(useMoments, 2);
  PetscUseMethod(sp,"PetscDualSpaceLagrangeGetUseMoments_C",(PetscDualSpace,PetscBool *),(sp,useMoments));
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceLagrangeSetUseMoments - Set the flag for moment functionals

  Logically collective

  Input Parameters:
+ sp - The PetscDualSpace
- useMoments - The flag for moment functionals

  Level: advanced

.seealso: PetscDualSpaceLagrangeGetUseMoments()
@*/
PetscErrorCode PetscDualSpaceLagrangeSetUseMoments(PetscDualSpace sp, PetscBool useMoments)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscTryMethod(sp,"PetscDualSpaceLagrangeSetUseMoments_C",(PetscDualSpace,PetscBool),(sp,useMoments));
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceLagrangeGetMomentOrder - Get the order for moment integration

  Not collective

  Input Parameter:
. sp - The PetscDualSpace

  Output Parameter:
. order - Moment integration order

  Level: advanced

.seealso: PetscDualSpaceLagrangeSetMomentOrder()
@*/
PetscErrorCode PetscDualSpaceLagrangeGetMomentOrder(PetscDualSpace sp, PetscInt *order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidIntPointer(order, 2);
  PetscUseMethod(sp,"PetscDualSpaceLagrangeGetMomentOrder_C",(PetscDualSpace,PetscInt *),(sp,order));
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceLagrangeSetMomentOrder - Set the order for moment integration

  Logically collective

  Input Parameters:
+ sp - The PetscDualSpace
- order - The order for moment integration

  Level: advanced

.seealso: PetscDualSpaceLagrangeGetMomentOrder()
@*/
PetscErrorCode PetscDualSpaceLagrangeSetMomentOrder(PetscDualSpace sp, PetscInt order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscTryMethod(sp,"PetscDualSpaceLagrangeSetMomentOrder_C",(PetscDualSpace,PetscInt),(sp,order));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceInitialize_Lagrange(PetscDualSpace sp)
{
  PetscFunctionBegin;
  sp->ops->destroy              = PetscDualSpaceDestroy_Lagrange;
  sp->ops->view                 = PetscDualSpaceView_Lagrange;
  sp->ops->setfromoptions       = PetscDualSpaceSetFromOptions_Lagrange;
  sp->ops->duplicate            = PetscDualSpaceDuplicate_Lagrange;
  sp->ops->setup                = PetscDualSpaceSetUp_Lagrange;
  sp->ops->createheightsubspace = NULL;
  sp->ops->createpointsubspace  = NULL;
  sp->ops->getsymmetries        = PetscDualSpaceGetSymmetries_Lagrange;
  sp->ops->apply                = PetscDualSpaceApplyDefault;
  sp->ops->applyall             = PetscDualSpaceApplyAllDefault;
  sp->ops->applyint             = PetscDualSpaceApplyInteriorDefault;
  sp->ops->createalldata        = PetscDualSpaceCreateAllDataDefault;
  sp->ops->createintdata        = PetscDualSpaceCreateInteriorDataDefault;
  PetscFunctionReturn(0);
}

/*MC
  PETSCDUALSPACELAGRANGE = "lagrange" - A PetscDualSpace object that encapsulates a dual space of pointwise evaluation functionals

  Level: intermediate

.seealso: PetscDualSpaceType, PetscDualSpaceCreate(), PetscDualSpaceSetType()
M*/
PETSC_EXTERN PetscErrorCode PetscDualSpaceCreate_Lagrange(PetscDualSpace sp)
{
  PetscDualSpace_Lag *lag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscCall(PetscNewLog(sp,&lag));
  sp->data = lag;

  lag->tensorCell  = PETSC_FALSE;
  lag->tensorSpace = PETSC_FALSE;
  lag->continuous  = PETSC_TRUE;
  lag->numCopies   = PETSC_DEFAULT;
  lag->numNodeSkip = PETSC_DEFAULT;
  lag->nodeType    = PETSCDTNODES_DEFAULT;
  lag->useMoments  = PETSC_FALSE;
  lag->momentOrder = 0;

  PetscCall(PetscDualSpaceInitialize_Lagrange(sp));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetContinuity_C", PetscDualSpaceLagrangeGetContinuity_Lagrange));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetContinuity_C", PetscDualSpaceLagrangeSetContinuity_Lagrange));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetTensor_C", PetscDualSpaceLagrangeGetTensor_Lagrange));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetTensor_C", PetscDualSpaceLagrangeSetTensor_Lagrange));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetTrimmed_C", PetscDualSpaceLagrangeGetTrimmed_Lagrange));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetTrimmed_C", PetscDualSpaceLagrangeSetTrimmed_Lagrange));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetNodeType_C", PetscDualSpaceLagrangeGetNodeType_Lagrange));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetNodeType_C", PetscDualSpaceLagrangeSetNodeType_Lagrange));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetUseMoments_C", PetscDualSpaceLagrangeGetUseMoments_Lagrange));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetUseMoments_C", PetscDualSpaceLagrangeSetUseMoments_Lagrange));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetMomentOrder_C", PetscDualSpaceLagrangeGetMomentOrder_Lagrange));
  PetscCall(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetMomentOrder_C", PetscDualSpaceLagrangeSetMomentOrder_Lagrange));
  PetscFunctionReturn(0);
}
