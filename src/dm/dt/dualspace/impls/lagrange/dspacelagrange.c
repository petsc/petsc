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

static PetscErrorCode Petsc1DNodeFamilyCreate(PetscDTNodeType family, PetscReal gaussJacobiExp, PetscBool endpoints, Petsc1DNodeFamily *nf)
{
  Petsc1DNodeFamily f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&f);CHKERRQ(ierr);
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
    if (gaussJacobiExp <= -1.) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Gauss-Jacobi exponent must be > -1.\n");
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

static PetscErrorCode Petsc1DNodeFamilyDestroy(Petsc1DNodeFamily *nf) {
  PetscInt       i, nc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!(*nf)) PetscFunctionReturn(0);
  if (--(*nf)->refct > 0) {
    *nf = NULL;
    PetscFunctionReturn(0);
  }
  nc = (*nf)->nComputed;
  for (i = 0; i < nc; i++) {
    ierr = PetscFree((*nf)->nodesets[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree((*nf)->nodesets);CHKERRQ(ierr);
  ierr = PetscFree(*nf);CHKERRQ(ierr);
  *nf = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode Petsc1DNodeFamilyGetNodeSets(Petsc1DNodeFamily f, PetscInt degree, PetscReal ***nodesets)
{
  PetscInt       nc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  nc = f->nComputed;
  if (degree >= nc) {
    PetscInt    i, j;
    PetscReal **new_nodesets;
    PetscReal  *w;

    ierr = PetscMalloc1(degree + 1, &new_nodesets);CHKERRQ(ierr);
    ierr = PetscArraycpy(new_nodesets, f->nodesets, nc);CHKERRQ(ierr);
    ierr = PetscFree(f->nodesets);CHKERRQ(ierr);
    f->nodesets = new_nodesets;
    ierr = PetscMalloc1(degree + 1, &w);CHKERRQ(ierr);
    for (i = nc; i < degree + 1; i++) {
      ierr = PetscMalloc1(i + 1, &(f->nodesets[i]));CHKERRQ(ierr);
      if (!i) {
        f->nodesets[i][0] = 0.5;
      } else {
        switch (f->nodeFamily) {
        case PETSCDTNODES_EQUISPACED:
          if (f->endpoints) {
            for (j = 0; j <= i; j++) f->nodesets[i][j] = (PetscReal) j / (PetscReal) i;
          } else {
            for (j = 0; j <= i; j++) f->nodesets[i][j] = ((PetscReal) j + 0.5) / ((PetscReal) i + 1.);
          }
          break;
        case PETSCDTNODES_GAUSSJACOBI:
          if (f->endpoints) {
            ierr = PetscDTGaussLobattoJacobiQuadrature(i + 1, 0., 1., f->gaussJacobiExp, f->gaussJacobiExp, f->nodesets[i], w);CHKERRQ(ierr);
          } else {
            ierr = PetscDTGaussJacobiQuadrature(i + 1, 0., 1., f->gaussJacobiExp, f->gaussJacobiExp, f->nodesets[i], w);CHKERRQ(ierr);
          }
          break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unknown 1D node family");
        }
      }
    }
    ierr = PetscFree(w);CHKERRQ(ierr);
    f->nComputed = degree + 1;
  }
  *nodesets = f->nodesets;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscNodeRecursive_Internal(PetscInt dim, PetscInt degree, PetscReal **nodesets, PetscInt tup[], PetscReal node[])
{
  PetscReal w;
  PetscInt i, j;
  PetscErrorCode ierr;

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
      ierr = PetscNodeRecursive_Internal(dim-1,degree-tup[i],nodesets,&tup[dim+1],&node[dim+1]);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dim < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Must have non-negative dimension\n");
  if (degree < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Must have non-negative degree\n");
  if (!dim) PetscFunctionReturn(0);
  ierr = PetscCalloc1(dim+2, &tup);CHKERRQ(ierr);
  k = 0;
  ierr = PetscDTBinomialInt(degree + dim, dim, &npoints);CHKERRQ(ierr);
  ierr = Petsc1DNodeFamilyGetNodeSets(f, degree, &nodesets);CHKERRQ(ierr);
  worksize = ((dim + 2) * (dim + 3)) / 2;
  ierr = PetscMalloc2(worksize, &nodework, worksize, &tupwork);CHKERRQ(ierr);
  for (k = 0; k < npoints; k++) {
    PetscInt i;

    tup[0] = degree;
    for (i = 0; i < dim; i++) {
      tup[0] -= tup[i+1];
    }
    switch(f->nodeFamily) {
    case PETSCDTNODES_EQUISPACED:
      if (f->endpoints) {
        for (i = 0; i < dim; i++) {
          points[dim*k + i] = (PetscReal) tup[i+1] / (PetscReal) degree;
        }
      } else {
        for (i = 0; i < dim; i++) {
          points[dim*k + i] = ((PetscReal) tup[i+1] + 1./(dim+1.)) / (PetscReal) (degree + 1.);
        }
      }
      break;
    default:
      for (i = 0; i < dim + 1; i++) tupwork[i] = tup[i];
      ierr = PetscNodeRecursive_Internal(dim, degree, nodesets, tupwork, nodework);CHKERRQ(ierr);
      for (i = 0; i < dim; i++) points[dim*k + i] = nodework[i + 1];
      break;
    }
    ierr = PetscDualSpaceLatticePointLexicographic_Internal(dim, degree, &tup[1]);CHKERRQ(ierr);
  }
  /* map from unit simplex to biunit simplex */
  for (k = 0; k < npoints * dim; k++) points[k] = points[k] * 2. - 1.;
  ierr = PetscFree2(nodework, tupwork);CHKERRQ(ierr);
  ierr = PetscFree(tup);
  PetscFunctionReturn(0);
}

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

static PetscErrorCode PetscLagNodeIndicesDestroy(PetscLagNodeIndices *ni) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!(*ni)) PetscFunctionReturn(0);
  if (--(*ni)->refct > 0) {
    *ni = NULL;
    PetscFunctionReturn(0);
  }
  ierr = PetscFree((*ni)->nodeIdx);CHKERRQ(ierr);
  ierr = PetscFree((*ni)->nodeVec);CHKERRQ(ierr);
  ierr = PetscFree((*ni)->perm);CHKERRQ(ierr);
  ierr = PetscFree(*ni);CHKERRQ(ierr);
  *ni = NULL;
  PetscFunctionReturn(0);
}

/* The vertex indices were written as though the vertices were in revlex order
 * wrt coordinates.  To understand the effect of different symmetries, we need
 * them to be in closure order.  We also need a permutation that takes point index
 * to closure number */
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  nVerts = vEnd - vStart;
  ierr = PetscMalloc1(nVerts, &closureOrder);CHKERRQ(ierr);
  ierr = PetscMalloc1(nVerts, &invClosureOrder);CHKERRQ(ierr);
  ierr = PetscMalloc1(nVerts, &revlexOrder);CHKERRQ(ierr);
  if (sortIdx) {
    PetscInt nodeIdxDim = ni->nodeIdxDim;
    PetscInt *idxOrder;

    ierr = PetscMalloc1(nVerts * nodeIdxDim, &newNodeIdx);CHKERRQ(ierr);
    ierr = PetscMalloc1(nVerts, &idxOrder);CHKERRQ(ierr);
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
    ierr = PetscFree(ni->nodeIdx);CHKERRQ(ierr);
    ni->nodeIdx = newNodeIdx;
    newNodeIdx = NULL;
    ierr = PetscFree(idxOrder);CHKERRQ(ierr);
  }
  ierr = DMPlexGetTransitiveClosure(dm, 0, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
  c = closureSize - nVerts;
  for (v = 0; v < nVerts; v++) closureOrder[v] = closure[2 * (c + v)] - vStart;
  for (v = 0; v < nVerts; v++) invClosureOrder[closureOrder[v]] = v;
  ierr = DMPlexRestoreTransitiveClosure(dm, 0, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordVec);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordVec, &coords);CHKERRQ(ierr);
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
  ierr = VecRestoreArrayRead(coordVec, &coords);CHKERRQ(ierr);
  ierr = PetscMalloc1(ni->nodeIdxDim * nVerts, &newNodeIdx);CHKERRQ(ierr);
  /* reorder nodeIdx to be in closure order */
  for (v = 0; v < nVerts; v++) {
    for (d = 0; d < ni->nodeIdxDim; d++) {
      newNodeIdx[revlexOrder[v] * ni->nodeIdxDim + d] = ni->nodeIdx[v * ni->nodeIdxDim + d];
    }
  }
  ierr = PetscFree(ni->nodeIdx);CHKERRQ(ierr);
  ni->nodeIdx = newNodeIdx;
  ni->perm = invClosureOrder;
  ierr = PetscFree(revlexOrder);CHKERRQ(ierr);
  ierr = PetscFree(closureOrder);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLagNodeIndicesCreateSimplexVertices(DM dm, PetscLagNodeIndices *nodeIndices)
{
  PetscLagNodeIndices ni;
  PetscInt       dim, d;

  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&ni);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ni->nodeIdxDim = dim + 1;
  ni->nodeVecDim = 0;
  ni->nNodes = dim + 1;
  ni->refct = 1;
  ierr = PetscCalloc1((dim + 1)*(dim + 1), &(ni->nodeIdx));CHKERRQ(ierr);
  for (d = 0; d < dim + 1; d++) ni->nodeIdx[d*(dim + 2)] = 1;
  ierr = PetscLagNodeIndicesComputeVertexOrder(dm, ni, PETSC_FALSE);CHKERRQ(ierr);
  *nodeIndices = ni;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLagNodeIndicesCreateTensorVertices(DM dm, PetscLagNodeIndices facetni, PetscLagNodeIndices *nodeIndices)
{
  PetscLagNodeIndices ni;
  PetscInt       nodeIdxDim, subNodeIdxDim = facetni->nodeIdxDim;
  PetscInt       nVerts, nSubVerts = facetni->nNodes;
  PetscInt       dim, d, e, f, g;

  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&ni);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ni->nodeIdxDim = nodeIdxDim = subNodeIdxDim + 2;
  ni->nodeVecDim = 0;
  ni->nNodes = nVerts = 2 * nSubVerts;
  ni->refct = 1;
  ierr = PetscCalloc1(nodeIdxDim * nVerts, &(ni->nodeIdx));CHKERRQ(ierr);
  for (f = 0, d = 0; d < 2; d++) {
    for (e = 0; e < nSubVerts; e++, f++) {
      for (g = 0; g < subNodeIdxDim; g++) {
        ni->nodeIdx[f * nodeIdxDim + g] = facetni->nodeIdx[e * subNodeIdxDim + g];
      }
      ni->nodeIdx[f * nodeIdxDim + subNodeIdxDim] = (1 - d);
      ni->nodeIdx[f * nodeIdxDim + subNodeIdxDim + 1] = d;
    }
  }
  ierr = PetscLagNodeIndicesComputeVertexOrder(dm, ni, PETSC_TRUE);CHKERRQ(ierr);
  *nodeIndices = ni;
  PetscFunctionReturn(0);
}

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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordVec);CHKERRQ(ierr);
  ierr = DMPlexGetPointDepth(dm, p, &pdepth);CHKERRQ(ierr);
  pdim = pdepth != depth ? pdepth != 0 ? pdepth : 0 : dim;
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, nSubVert, MPIU_INT, &closureVerts);CHKERRQ(ierr);
  ierr = DMPlexGetTransitiveClosure_Internal(dm, p, ornt, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
  c = closureSize - nSubVert;
  /* we want which cell closure indices the closure of this point corresponds to */
  for (v = 0; v < nSubVert; v++) closureVerts[v] = vert->perm[closure[2 * (c + v)] - vStart];
  ierr = DMPlexRestoreTransitiveClosure(dm, p, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
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
      if (subi < 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Did not find matching coordinate\n");
      for (n = 0; n < nNodes; n++) pfNodeIdx[n * nodeIdxDim + i] = nodeIdx[n * subNodeIdxDim + subi];
    }
  }
  /* push forward vectors */
  ierr = DMGetWorkArray(dm, dim * dim, MPIU_REAL, &J);CHKERRQ(ierr);
  if (ornt != 0) {
    PetscInt        closureSize2 = 0;
    PetscInt       *closure2 = NULL;

    ierr = DMPlexGetTransitiveClosure_Internal(dm, p, 0, PETSC_TRUE, &closureSize2, &closure2);CHKERRQ(ierr);
    ierr = PetscMalloc1(dim * nSubVert, &newCoords);CHKERRQ(ierr);
    ierr = VecGetArrayRead(coordVec, &oldCoords);CHKERRQ(ierr);
    for (v = 0; v < nSubVert; v++) {
      PetscInt d;
      for (d = 0; d < dim; d++) {
        newCoords[(closure2[2 * (c + v)] - vStart) * dim + d] = oldCoords[closureVerts[v] * dim + d];
      }
    }
    ierr = VecRestoreArrayRead(coordVec, &oldCoords);CHKERRQ(ierr);
    ierr = DMPlexRestoreTransitiveClosure(dm, p, PETSC_TRUE, &closureSize2, &closure2);CHKERRQ(ierr);
    ierr = VecPlaceArray(coordVec, newCoords);CHKERRQ(ierr);
  }
  ierr = DMPlexComputeCellGeometryAffineFEM(dm, p, NULL, J, NULL, &detJ);CHKERRQ(ierr);
  if (ornt != 0) {
    ierr = VecResetArray(coordVec);CHKERRQ(ierr);
    ierr = PetscFree(newCoords);CHKERRQ(ierr);
  }
  ierr = DMRestoreWorkArray(dm, nSubVert, MPIU_INT, &closureVerts);CHKERRQ(ierr);
  /* compactify */
  for (i = 0; i < dim; i++) for (j = 0; j < pdim; j++) J[i * pdim + j] = J[i * dim + j];
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(formDegree), &Nk);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(pdim, PetscAbsInt(formDegree), &pNk);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, pNk * Nk, MPIU_REAL, &Jstar);CHKERRQ(ierr);
  ierr = PetscDTAltVPullbackMatrix(pdim, dim, J, formDegree, Jstar);CHKERRQ(ierr);
  for (n = 0; n < nNodes; n++) {
    for (i = 0; i < Nk; i++) {
      PetscReal val = 0.;
      for (j = 0; j < pNk; j++) val += nodeVec[n * pNk + j] * Jstar[j * pNk + i];
      pfNodeVec[n * Nk + i] = val;
    }
  }
  ierr = DMRestoreWorkArray(dm, pNk * Nk, MPIU_REAL, &Jstar);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, dim * dim, MPIU_REAL, &J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(formDegree), &Nk);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dimT, PetscAbsInt(kT), &NkT);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dimF, PetscAbsInt(kF), &NkF);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(kT), &MkT);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(kF), &MkF);CHKERRQ(ierr);
  ierr = PetscNew(&ni);CHKERRQ(ierr);
  ni->nodeIdxDim = nodeIdxDim = tracei->nodeIdxDim + fiberi->nodeIdxDim;
  ni->nodeVecDim = Nk;
  ni->nNodes = nNodes = tracei->nNodes * fiberi->nNodes;
  ni->refct = 1;
  ierr = PetscMalloc1(nNodes * nodeIdxDim, &(ni->nodeIdx));CHKERRQ(ierr);
  /* first concatenate the indices */
  for (l = 0, j = 0; j < fiberi->nNodes; j++) {
    for (i = 0; i < tracei->nNodes; i++, l++) {
      PetscInt m, n = 0;

      for (m = 0; m < tracei->nodeIdxDim; m++) ni->nodeIdx[l * nodeIdxDim + n++] = tracei->nodeIdx[i * tracei->nodeIdxDim + m];
      for (m = 0; m < fiberi->nodeIdxDim; m++) ni->nodeIdx[l * nodeIdxDim + n++] = fiberi->nodeIdx[j * fiberi->nodeIdxDim + m];
    }
  }

  /* now wedge together the push-forward vectors */
  ierr = PetscMalloc1(nNodes * Nk, &(ni->nodeVec));CHKERRQ(ierr);
  ierr = PetscCalloc2(dimT*dim, &projT, dimF*dim, &projF);CHKERRQ(ierr);
  for (i = 0; i < dimT; i++) projT[i * (dim + 1)] = 1.;
  for (i = 0; i < dimF; i++) projF[i * (dim + dimT + 1) + dimT] = 1.;
  ierr = PetscMalloc2(MkT*NkT, &projTstar, MkF*NkF, &projFstar);CHKERRQ(ierr);
  ierr = PetscDTAltVPullbackMatrix(dim, dimT, projT, kT, projTstar);CHKERRQ(ierr);
  ierr = PetscDTAltVPullbackMatrix(dim, dimF, projF, kF, projFstar);CHKERRQ(ierr);
  ierr = PetscMalloc6(MkT, &workT, MkT, &workT2, MkF, &workF, MkF, &workF2, Nk, &work, Nk, &work2);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nk * MkT, &wedgeMat);CHKERRQ(ierr);
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
      ierr = PetscDTAltVStar(dim, PetscAbsInt(kF), 1, workF2, workF);CHKERRQ(ierr);
    }
    /* Compute the matrix that wedges this form with one of the trace k-form */
    ierr = PetscDTAltVWedgeMatrix(dim, PetscAbsInt(kF), PetscAbsInt(kT), workF, wedgeMat);CHKERRQ(ierr);
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
        ierr = PetscDTAltVStar(dim, PetscAbsInt(kT), 1, workT2, workT);CHKERRQ(ierr);
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
        ierr = PetscDTAltVStar(dim, PetscAbsInt(formDegree), -1, work2, work);CHKERRQ(ierr);
      }
      /* insert into the array (adjusting for sign) */
      for (d = 0; d < Nk; d++) ni->nodeVec[l * Nk + d] = sign * work[d];
    }
  }
  ierr = PetscFree(wedgeMat);CHKERRQ(ierr);
  ierr = PetscFree6(workT, workT2, workF, workF2, work, work2);CHKERRQ(ierr);
  ierr = PetscFree2(projTstar, projFstar);CHKERRQ(ierr);
  ierr = PetscFree2(projT, projF);CHKERRQ(ierr);
  *nodeIndices = ni;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscLagNodeIndicesMerge(PetscLagNodeIndices niA, PetscLagNodeIndices niB, PetscLagNodeIndices *nodeIndices)
{
  PetscLagNodeIndices ni;
  PetscInt            nodeIdxDim, nodeVecDim, nNodes;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&ni);CHKERRQ(ierr);
  ni->nodeIdxDim = nodeIdxDim = niA->nodeIdxDim;
  if (niB->nodeIdxDim != nodeIdxDim) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Cannot merge PetscLagNodeIndices with different nodeIdxDim");
  ni->nodeVecDim = nodeVecDim = niA->nodeVecDim;
  if (niB->nodeVecDim != nodeVecDim) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Cannot merge PetscLagNodeIndices with different nodeVecDim");
  ni->nNodes = nNodes = niA->nNodes + niB->nNodes;
  ni->refct = 1;
  ierr = PetscMalloc1(nNodes * nodeIdxDim, &(ni->nodeIdx));CHKERRQ(ierr);
  ierr = PetscMalloc1(nNodes * nodeVecDim, &(ni->nodeVec));CHKERRQ(ierr);
  ierr = PetscArraycpy(ni->nodeIdx, niA->nodeIdx, niA->nNodes * nodeIdxDim);CHKERRQ(ierr);
  ierr = PetscArraycpy(ni->nodeVec, niA->nodeVec, niA->nNodes * nodeVecDim);CHKERRQ(ierr);
  ierr = PetscArraycpy(&(ni->nodeIdx[niA->nNodes * nodeIdxDim]), niB->nodeIdx, niB->nNodes * nodeIdxDim);CHKERRQ(ierr);
  ierr = PetscArraycpy(&(ni->nodeVec[niA->nNodes * nodeVecDim]), niB->nodeVec, niB->nNodes * nodeVecDim);CHKERRQ(ierr);
  *nodeIndices = ni;
  PetscFunctionReturn(0);
}

#define PETSCTUPINTCOMPREVLEX(N)                                   \
static int PetscTupIntCompRevlex_##N(const void *a, const void *b) \
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

static PetscErrorCode PetscLagNodeIndicesGetPermutation(PetscLagNodeIndices ni, PetscInt *perm[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!(ni->perm)) {
    PetscInt *sorter;
    PetscInt m = ni->nNodes;
    PetscInt nodeIdxDim = ni->nodeIdxDim;
    PetscInt i, j, k, l;
    PetscInt *prm;
    int (*comp) (const void *, const void *);

    ierr = PetscMalloc1((nodeIdxDim + 2) * m, &sorter);CHKERRQ(ierr);
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
    ierr = PetscMalloc1(m, &prm);CHKERRQ(ierr);
    for (i = 0; i < m; i++) prm[i] = sorter[(nodeIdxDim + 2) * i + 1];
    ni->perm = prm;
    ierr = PetscFree(sorter);
  }
  *perm = ni->perm;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceDestroy_Lagrange(PetscDualSpace sp)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (lag->symperms) {
    PetscInt **selfSyms = lag->symperms[0];

    if (selfSyms) {
      PetscInt i, **allocated = &selfSyms[-lag->selfSymOff];

      for (i = 0; i < lag->numSelfSym; i++) {
        ierr = PetscFree(allocated[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(allocated);CHKERRQ(ierr);
    }
    ierr = PetscFree(lag->symperms);CHKERRQ(ierr);
  }
  if (lag->symflips) {
    PetscScalar **selfSyms = lag->symflips[0];

    if (selfSyms) {
      PetscInt i;
      PetscScalar **allocated = &selfSyms[-lag->selfSymOff];

      for (i = 0; i < lag->numSelfSym; i++) {
        ierr = PetscFree(allocated[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(allocated);CHKERRQ(ierr);
    }
    ierr = PetscFree(lag->symflips);CHKERRQ(ierr);
  }
  ierr = Petsc1DNodeFamilyDestroy(&(lag->nodeFamily));CHKERRQ(ierr);
  ierr = PetscLagNodeIndicesDestroy(&(lag->vertIndices));CHKERRQ(ierr);
  ierr = PetscLagNodeIndicesDestroy(&(lag->intNodeIndices));CHKERRQ(ierr);
  ierr = PetscLagNodeIndicesDestroy(&(lag->allNodeIndices));CHKERRQ(ierr);
  ierr = PetscFree(lag);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetContinuity_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetContinuity_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetTensor_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetTensor_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetTrimmed_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetTrimmed_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetNodeType_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetNodeType_C", NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceLagrangeView_Ascii(PetscDualSpace sp, PetscViewer viewer)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer, "%s %s%sLagrange dual space\n", lag->continuous ? "Continuous" : "Discontinuous", lag->tensorSpace ? "tensor " : "", lag->trimmed ? "trimmed " : "");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceView_Lagrange(PetscDualSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscDualSpaceLagrangeView_Ascii(sp, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceSetFromOptions_Lagrange(PetscOptionItems *PetscOptionsObject,PetscDualSpace sp)
{
  PetscBool      continuous, tensor, trimmed, flg, flg2, flg3;
  PetscDTNodeType nodeType;
  PetscReal      nodeExponent;
  PetscBool      nodeEndpoints;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceLagrangeGetContinuity(sp, &continuous);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeGetTensor(sp, &tensor);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeGetTrimmed(sp, &trimmed);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeGetNodeType(sp, &nodeType, &nodeEndpoints, &nodeExponent);CHKERRQ(ierr);
  if (nodeType == PETSCDTNODES_DEFAULT) nodeType = PETSCDTNODES_GAUSSJACOBI;
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscDualSpace Lagrange Options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscdualspace_lagrange_continuity", "Flag for continuous element", "PetscDualSpaceLagrangeSetContinuity", continuous, &continuous, &flg);CHKERRQ(ierr);
  if (flg) {ierr = PetscDualSpaceLagrangeSetContinuity(sp, continuous);CHKERRQ(ierr);}
  ierr = PetscOptionsBool("-petscdualspace_lagrange_tensor", "Flag for tensor dual space", "PetscDualSpaceLagrangeSetTensor", tensor, &tensor, &flg);CHKERRQ(ierr);
  if (flg) {ierr = PetscDualSpaceLagrangeSetTensor(sp, tensor);CHKERRQ(ierr);}
  ierr = PetscOptionsBool("-petscdualspace_lagrange_trimmed", "Flag for trimmed dual space", "PetscDualSpaceLagrangeSetTrimmed", trimmed, &trimmed, &flg);CHKERRQ(ierr);
  if (flg) {ierr = PetscDualSpaceLagrangeSetTrimmed(sp, trimmed);CHKERRQ(ierr);}
  ierr = PetscOptionsEnum("-petscdualspace_lagrange_node_type", "Lagrange node location type", "PetscDualSpaceLagrangeSetNodeType", PetscDTNodeTypes, (PetscEnum)nodeType, (PetscEnum *)&nodeType, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscdualspace_lagrange_node_endpoints", "Flag for nodes that include endpoints", "PetscDualSpaceLagrangeSetNodeType", nodeEndpoints, &nodeEndpoints, &flg2);CHKERRQ(ierr);
  flg3 = PETSC_FALSE;
  if (nodeType == PETSCDTNODES_GAUSSJACOBI) {
    ierr = PetscOptionsReal("-petscdualspace_lagrange_node_exponent", "Gauss-Jacobi weight function exponent", "PetscDualSpaceLagrangeSetNodeType", nodeExponent, &nodeExponent, &flg3);CHKERRQ(ierr);
  }
  if (flg || flg2 || flg3) {ierr = PetscDualSpaceLagrangeSetNodeType(sp, nodeType, nodeEndpoints, nodeExponent);CHKERRQ(ierr);}
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceDuplicate_Lagrange(PetscDualSpace sp, PetscDualSpace spNew)
{
  PetscBool           cont, tensor, trimmed, boundary;
  PetscDTNodeType     nodeType;
  PetscReal           exponent;
  PetscDualSpace_Lag *lag    = (PetscDualSpace_Lag *) sp->data;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceLagrangeGetContinuity(sp, &cont);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetContinuity(spNew, cont);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeGetTensor(sp, &tensor);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetTensor(spNew, tensor);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeGetTrimmed(sp, &trimmed);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetTrimmed(spNew, trimmed);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeGetNodeType(sp, &nodeType, &boundary, &exponent);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetNodeType(spNew, nodeType, boundary, exponent);CHKERRQ(ierr);
  if (lag->nodeFamily) {
    PetscDualSpace_Lag *lagnew = (PetscDualSpace_Lag *) spNew->data;

    ierr = Petsc1DNodeFamilyReference(lag->nodeFamily);CHKERRQ(ierr);
    lagnew->nodeFamily = lag->nodeFamily;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceCreateEdgeSubspace_Lagrange(PetscDualSpace sp, PetscInt order, PetscInt k, PetscInt Nc, PetscBool interiorOnly, PetscDualSpace *bdsp)
{
  DM                 K;
  PetscDualSpace_Lag *newlag;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceDuplicate(sp,bdsp);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetFormDegree(*bdsp, k);CHKERRQ(ierr);
  ierr = PetscDualSpaceCreateReferenceCell(*bdsp, 1, PETSC_TRUE, &K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(*bdsp, K);CHKERRQ(ierr);
  ierr = DMDestroy(&K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(*bdsp, order);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetNumComponents(*bdsp, Nc);CHKERRQ(ierr);
  newlag = (PetscDualSpace_Lag *) (*bdsp)->data;
  newlag->interiorOnly = interiorOnly;
  ierr = PetscDualSpaceSetUp(*bdsp);CHKERRQ(ierr);
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
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscQuadratureGetData(trace, &dimTrace, NULL, &numPointsTrace, &pointsTrace, NULL);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(fiber, &dimFiber, NULL, &numPointsFiber, &pointsFiber, NULL);CHKERRQ(ierr);
  dim = dimTrace + dimFiber;
  numPoints = numPointsFiber * numPointsTrace;
  ierr = PetscMalloc1(numPoints * dim, &points);CHKERRQ(ierr);
  for (p = 0, j = 0; j < numPointsFiber; j++) {
    for (i = 0; i < numPointsTrace; i++, p++) {
      for (k = 0; k < dimTrace; k++) points[p * dim +            k] = pointsTrace[i * dimTrace + k];
      for (k = 0; k < dimFiber; k++) points[p * dim + dimTrace + k] = pointsFiber[j * dimFiber + k];
    }
  }
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF, product);CHKERRQ(ierr);
  ierr = PetscQuadratureSetData(*product, dim, 0, numPoints, points, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetSize(trace, &mTrace, &nTrace);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dimTrace, PetscAbsInt(kTrace), &NkTrace);CHKERRQ(ierr);
  if (nTrace % NkTrace) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "point value space of trace matrix is not a multiple of k-form size");
  ierr = MatGetSize(fiber, &mFiber, &nFiber);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dimFiber, PetscAbsInt(kFiber), &NkFiber);CHKERRQ(ierr);
  if (nFiber % NkFiber) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "point value space of fiber matrix is not a multiple of k-form size");
  ierr = PetscMalloc2(mTrace, &nnzTrace, mFiber, &nnzFiber);CHKERRQ(ierr);
  for (i = 0; i < mTrace; i++) {
    ierr = MatGetRow(trace, i, &(nnzTrace[i]), NULL, NULL);CHKERRQ(ierr);
    if (nnzTrace[i] % NkTrace) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "nonzeros in trace matrix are not in k-form size blocks");
  }
  for (i = 0; i < mFiber; i++) {
    ierr = MatGetRow(fiber, i, &(nnzFiber[i]), NULL, NULL);CHKERRQ(ierr);
    if (nnzFiber[i] % NkFiber) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "nonzeros in fiber matrix are not in k-form size blocks");
  }
  dim = dimTrace + dimFiber;
  k = kFiber + kTrace;
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(k), &Nk);CHKERRQ(ierr);
  m = mTrace * mFiber;
  ierr = PetscMalloc1(m, &nnz);CHKERRQ(ierr);
  for (l = 0, j = 0; j < mFiber; j++) for (i = 0; i < mTrace; i++, l++) nnz[l] = (nnzTrace[i] / NkTrace) * (nnzFiber[j] / NkFiber) * Nk;
  n = (nTrace / NkTrace) * (nFiber / NkFiber) * Nk;
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, m, n, 0, nnz, &prod);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);
  ierr = PetscFree2(nnzTrace,nnzFiber);CHKERRQ(ierr);
  /* reasoning about which points each dof needs depends on having zeros computed at points preserved */
  ierr = MatSetOption(prod, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE);CHKERRQ(ierr);
  /* compute pullbacks */
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(kTrace), &dT);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(kFiber), &dF);CHKERRQ(ierr);
  ierr = PetscMalloc4(dimTrace * dim, &projT, dimFiber * dim, &projF, dT * NkTrace, &projTstar, dF * NkFiber, &projFstar);CHKERRQ(ierr);
  ierr = PetscArrayzero(projT, dimTrace * dim);CHKERRQ(ierr);
  for (i = 0; i < dimTrace; i++) projT[i * (dim + 1)] = 1.;
  ierr = PetscArrayzero(projF, dimFiber * dim);CHKERRQ(ierr);
  for (i = 0; i < dimFiber; i++) projF[i * (dim + 1) + dimTrace] = 1.;
  ierr = PetscDTAltVPullbackMatrix(dim, dimTrace, projT, kTrace, projTstar);CHKERRQ(ierr);
  ierr = PetscDTAltVPullbackMatrix(dim, dimFiber, projF, kFiber, projFstar);CHKERRQ(ierr);
  ierr = PetscMalloc5(dT, &workT, dF, &workF, Nk, &work, Nk, &workstar, Nk, &workS);CHKERRQ(ierr);
  ierr = PetscMalloc2(dT, &workT2, dF, &workF2);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nk * dT, &wedgeMat);CHKERRQ(ierr);
  sign = (PetscAbsInt(kTrace * kFiber) & 1) ? -1. : 1.;
  for (i = 0, iF = 0; iF < mFiber; iF++) {
    PetscInt           ncolsF, nformsF;
    const PetscInt    *colsF;
    const PetscScalar *valsF;

    ierr = MatGetRow(fiber, iF, &ncolsF, &colsF, &valsF);CHKERRQ(ierr);
    nformsF = ncolsF / NkFiber;
    for (iT = 0; iT < mTrace; iT++, i++) {
      PetscInt           ncolsT, nformsT;
      const PetscInt    *colsT;
      const PetscScalar *valsT;

      ierr = MatGetRow(trace, iT, &ncolsT, &colsT, &valsT);CHKERRQ(ierr);
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
          ierr = PetscDTAltVStar(dim, PetscAbsInt(kFiber), 1, workF2, workF);CHKERRQ(ierr);
        }
        ierr = PetscDTAltVWedgeMatrix(dim, PetscAbsInt(kFiber), PetscAbsInt(kTrace), workF, wedgeMat);CHKERRQ(ierr);
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
            ierr = PetscDTAltVStar(dim, PetscAbsInt(kTrace), 1, workT2, workT);CHKERRQ(ierr);
          }

          for (il = 0; il < Nk; il++) {
            PetscReal val = 0.;
            for (jl = 0; jl < dT; jl++) val += sign * wedgeMat[il * dT + jl] * workT[jl];
            work[il] = val;
          }
          if (k < 0) {
            ierr = PetscDTAltVStar(dim, PetscAbsInt(k), -1, work, workstar);CHKERRQ(ierr);
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
            ierr = MatSetValue(prod, i, col * Nk + l, vals[l], INSERT_VALUES);CHKERRQ(ierr);
          } /* Nk */
        } /* jT */
      } /* jF */
      ierr = MatRestoreRow(trace, iT, &ncolsT, &colsT, &valsT);CHKERRQ(ierr);
    } /* iT */
    ierr = MatRestoreRow(fiber, iF, &ncolsF, &colsF, &valsF);CHKERRQ(ierr);
  } /* iF */
  ierr = PetscFree(wedgeMat);CHKERRQ(ierr);
  ierr = PetscFree4(projT, projF, projTstar, projFstar);CHKERRQ(ierr);
  ierr = PetscFree2(workT2, workF2);CHKERRQ(ierr);
  ierr = PetscFree5(workT, workF, work, workstar, workS);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(prod, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(prod, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *product = prod;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscQuadraturePointsMerge(PetscQuadrature quadA, PetscQuadrature quadB, PetscQuadrature *quadJoint, PetscInt *aToJoint[], PetscInt *bToJoint[])
{
  PetscInt         dimA, dimB;
  PetscInt         nA, nB, nJoint, i, j, d;
  const PetscReal *pointsA;
  const PetscReal *pointsB;
  PetscReal       *pointsJoint;
  PetscInt        *aToJ, *bToJ;
  PetscQuadrature  qJ;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscQuadratureGetData(quadA, &dimA, NULL, &nA, &pointsA, NULL);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quadB, &dimB, NULL, &nB, &pointsB, NULL);CHKERRQ(ierr);
  if (dimA != dimB) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Quadrature points must be in the same dimension");
  nJoint = nA;
  ierr = PetscMalloc1(nA, &aToJ);CHKERRQ(ierr);
  for (i = 0; i < nA; i++) aToJ[i] = i;
  ierr = PetscMalloc1(nB, &bToJ);CHKERRQ(ierr);
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
  ierr = PetscMalloc1(nJoint * dimA, &pointsJoint);CHKERRQ(ierr);
  ierr = PetscArraycpy(pointsJoint, pointsA, nA * dimA);CHKERRQ(ierr);
  for (i = 0; i < nB; i++) {
    if (bToJ[i] >= nA) {
      for (d = 0; d < dimA; d++) pointsJoint[bToJ[i] * dimA + d] = pointsB[i * dimA + d];
    }
  }
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &qJ);CHKERRQ(ierr);
  ierr = PetscQuadratureSetData(qJ, dimA, 0, nJoint, pointsJoint, NULL);CHKERRQ(ierr);
  *quadJoint = qJ;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatricesMerge(Mat matA, Mat matB, PetscInt dim, PetscInt k, PetscInt numMerged, const PetscInt aToMerged[], const PetscInt bToMerged[], Mat *matMerged)
{
  PetscInt m, n, mA, nA, mB, nB, Nk, i, j, l;
  Mat      M;
  PetscInt *nnz;
  PetscInt maxnnz;
  PetscInt *work;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(k), &Nk);CHKERRQ(ierr);
  ierr = MatGetSize(matA, &mA, &nA);CHKERRQ(ierr);
  if (nA % Nk) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "matA column space not a multiple of k-form size");
  ierr = MatGetSize(matB, &mB, &nB);CHKERRQ(ierr);
  if (nB % Nk) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "matB column space not a multiple of k-form size");
  m = mA + mB;
  n = numMerged * Nk;
  ierr = PetscMalloc1(m, &nnz);CHKERRQ(ierr);
  maxnnz = 0;
  for (i = 0; i < mA; i++) {
    ierr = MatGetRow(matA, i, &(nnz[i]), NULL, NULL);CHKERRQ(ierr);
    if (nnz[i] % Nk) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "nonzeros in matA are not in k-form size blocks");
    maxnnz = PetscMax(maxnnz, nnz[i]);
  }
  for (i = 0; i < mB; i++) {
    ierr = MatGetRow(matB, i, &(nnz[i+mA]), NULL, NULL);CHKERRQ(ierr);
    if (nnz[i+mA] % Nk) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "nonzeros in matB are not in k-form size blocks");
    maxnnz = PetscMax(maxnnz, nnz[i+mA]);
  }
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, m, n, 0, nnz, &M);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);
  /* reasoning about which points each dof needs depends on having zeros computed at points preserved */
  ierr = MatSetOption(M, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxnnz, &work);CHKERRQ(ierr);
  for (i = 0; i < mA; i++) {
    const PetscInt *cols;
    const PetscScalar *vals;
    PetscInt nCols;
    ierr = MatGetRow(matA, i, &nCols, &cols, &vals);CHKERRQ(ierr);
    for (j = 0; j < nCols / Nk; j++) {
      PetscInt newCol = aToMerged[cols[j * Nk] / Nk];
      for (l = 0; l < Nk; l++) work[j * Nk + l] = newCol * Nk + l;
    }
    ierr = MatSetValuesBlocked(M, 1, &i, nCols, work, vals, INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(matA, i, &nCols, &cols, &vals);CHKERRQ(ierr);
  }
  for (i = 0; i < mB; i++) {
    const PetscInt *cols;
    const PetscScalar *vals;

    PetscInt row = i + mA;
    PetscInt nCols;
    ierr = MatGetRow(matB, i, &nCols, &cols, &vals);CHKERRQ(ierr);
    for (j = 0; j < nCols / Nk; j++) {
      PetscInt newCol = bToMerged[cols[j * Nk] / Nk];
      for (l = 0; l < Nk; l++) work[j * Nk + l] = newCol * Nk + l;
    }
    ierr = MatSetValuesBlocked(M, 1, &row, nCols, work, vals, INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(matB, i, &nCols, &cols, &vals);CHKERRQ(ierr);
  }
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *matMerged = M;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceCreateFacetSubspace_Lagrange(PetscDualSpace sp, DM K, PetscInt f, PetscInt k, PetscInt Ncopies, PetscBool interiorOnly, PetscDualSpace *bdsp)
{
  PetscInt           Nknew, Ncnew;
  PetscInt           dim, pointDim = -1;
  PetscInt           depth;
  DM                 dm;
  PetscDualSpace_Lag *newlag;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDM(sp,&dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm,&depth);CHKERRQ(ierr);
  ierr = PetscDualSpaceDuplicate(sp,bdsp);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetFormDegree(*bdsp,k);CHKERRQ(ierr);
  if (!K) {
    PetscBool isSimplex;


    if (depth == dim) {
      PetscInt coneSize;

      pointDim = dim - 1;
      ierr = DMPlexGetConeSize(dm,f,&coneSize);CHKERRQ(ierr);
      isSimplex = (PetscBool) (coneSize == dim);
      ierr = PetscDualSpaceCreateReferenceCell(*bdsp, dim-1, isSimplex, &K);CHKERRQ(ierr);
    } else if (depth == 1) {
      pointDim = 0;
      ierr = PetscDualSpaceCreateReferenceCell(*bdsp, 0, PETSC_TRUE, &K);CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unsupported interpolation state of reference element");
  } else {
    ierr = PetscObjectReference((PetscObject)K);CHKERRQ(ierr);
    ierr = DMGetDimension(K, &pointDim);CHKERRQ(ierr);
  }
  ierr = PetscDualSpaceSetDM(*bdsp, K);CHKERRQ(ierr);
  ierr = DMDestroy(&K);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(pointDim, PetscAbsInt(k), &Nknew);CHKERRQ(ierr);
  Ncnew = Nknew * Ncopies;
  ierr = PetscDualSpaceSetNumComponents(*bdsp, Ncnew);CHKERRQ(ierr);
  newlag = (PetscDualSpace_Lag *) (*bdsp)->data;
  newlag->interiorOnly = interiorOnly;
  ierr = PetscDualSpaceSetUp(*bdsp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceLagrangeCreateSimplexNodeMat(Petsc1DNodeFamily nodeFamily, PetscInt dim, PetscInt sum, PetscInt Nk, PetscInt numNodeSkip, PetscQuadrature *iNodes, Mat *iMat, PetscLagNodeIndices *nodeIndices)
{
  PetscReal *extraNodeCoords, *nodeCoords;
  PetscInt nNodes, nExtraNodes;
  PetscInt i, j, k, extraSum = sum + numNodeSkip * (1 + dim);
  PetscQuadrature intNodes;
  Mat intMat;
  PetscLagNodeIndices ni;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDTBinomialInt(dim + sum, dim, &nNodes);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim + extraSum, dim, &nExtraNodes);CHKERRQ(ierr);

  ierr = PetscMalloc1(dim * nExtraNodes, &extraNodeCoords);CHKERRQ(ierr);
  ierr = PetscNew(&ni);CHKERRQ(ierr);
  ni->nodeIdxDim = dim + 1;
  ni->nodeVecDim = Nk;
  ni->nNodes = nNodes * Nk;
  ni->refct = 1;
  ierr = PetscMalloc1(nNodes * Nk * (dim + 1), &(ni->nodeIdx));CHKERRQ(ierr);
  ierr = PetscMalloc1(nNodes * Nk * Nk, &(ni->nodeVec));CHKERRQ(ierr);
  for (i = 0; i < nNodes; i++) for (j = 0; j < Nk; j++) for (k = 0; k < Nk; k++) ni->nodeVec[(i * Nk + j) * Nk + k] = (j == k) ? 1. : 0.;
  ierr = Petsc1DNodeFamilyComputeSimplexNodes(nodeFamily, dim, extraSum, extraNodeCoords);CHKERRQ(ierr);
  if (numNodeSkip) {
    PetscInt k;
    PetscInt *tup;

    ierr = PetscMalloc1(dim * nNodes, &nodeCoords);CHKERRQ(ierr);
    ierr = PetscMalloc1(dim + 1, &tup);CHKERRQ(ierr);
    for (k = 0; k < nNodes; k++) {
      PetscInt j, c;
      PetscInt index;

      ierr = PetscDTIndexToBary(dim + 1, sum, k, tup);CHKERRQ(ierr);
      for (j = 0; j < dim + 1; j++) tup[j] += numNodeSkip;
      for (c = 0; c < Nk; c++) {
        for (j = 0; j < dim + 1; j++) {
          ni->nodeIdx[(k * Nk + c) * (dim + 1) + j] = tup[j] + 1;
        }
      }
      ierr = PetscDTBaryToIndex(dim + 1, extraSum, tup, &index);CHKERRQ(ierr);
      for (j = 0; j < dim; j++) nodeCoords[k * dim + j] = extraNodeCoords[index * dim + j];
    }
    ierr = PetscFree(tup);CHKERRQ(ierr);
    ierr = PetscFree(extraNodeCoords);CHKERRQ(ierr);
  } else {
    PetscInt k;
    PetscInt *tup;

    nodeCoords = extraNodeCoords;
    ierr = PetscMalloc1(dim + 1, &tup);CHKERRQ(ierr);
    for (k = 0; k < nNodes; k++) {
      PetscInt j, c;

      ierr = PetscDTIndexToBary(dim + 1, sum, k, tup);CHKERRQ(ierr);
      for (c = 0; c < Nk; c++) {
        for (j = 0; j < dim + 1; j++) {
          /* barycentric indices can have zeros, but we don't want to push forward zeros because it makes it harder to
           * determine which nodes correspond to which under symmetries, so we increase by 1 */
          ni->nodeIdx[(k * Nk + c) * (dim + 1) + j] = tup[j] + 1;
        }
      }
    }
    ierr = PetscFree(tup);CHKERRQ(ierr);
  }
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &intNodes);CHKERRQ(ierr);
  ierr = PetscQuadratureSetData(intNodes, dim, 0, nNodes, nodeCoords, NULL);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, nNodes * Nk, nNodes * Nk, Nk, NULL, &intMat);CHKERRQ(ierr);
  ierr = MatSetOption(intMat,MAT_IGNORE_ZERO_ENTRIES,PETSC_FALSE);CHKERRQ(ierr);
  for (j = 0; j < nNodes * Nk; j++) {
    PetscInt rem = j % Nk;
    PetscInt a, aprev = j - rem;
    PetscInt anext = aprev + Nk;

    for (a = aprev; a < anext; a++) {
      ierr = MatSetValue(intMat, j, a, (a == j) ? 1. : 0., INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(intMat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(intMat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *iNodes = intNodes;
  *iMat = intMat;
  *nodeIndices = ni;
  PetscFunctionReturn(0);
}

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  lag = (PetscDualSpace_Lag *) sp->data;
  verti = lag->vertIndices;
  ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetFormDegree(sp, &formDegree);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(formDegree), &Nk);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetSection(sp, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(section, &nDofs);CHKERRQ(ierr);
  ierr = PetscNew(&ni);CHKERRQ(ierr);
  ni->nodeIdxDim = nodeIdxDim = verti->nodeIdxDim;
  ni->nodeVecDim = Nk;
  ni->nNodes = nDofs;
  ni->refct = 1;
  ierr = PetscMalloc1(nodeIdxDim * nDofs, &(ni->nodeIdx));CHKERRQ(ierr);
  ierr = PetscMalloc1(Nk * nDofs, &(ni->nodeVec));CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetDof(section, 0, &spintdim);CHKERRQ(ierr);
  if (spintdim) {
    ierr = PetscArraycpy(ni->nodeIdx, lag->intNodeIndices->nodeIdx, spintdim * nodeIdxDim);CHKERRQ(ierr);
    ierr = PetscArraycpy(ni->nodeVec, lag->intNodeIndices->nodeVec, spintdim * Nk);CHKERRQ(ierr);
  }
  for (p = pStart + 1; p < pEnd; p++) {
    PetscDualSpace psp = sp->pointSpaces[p];
    PetscDualSpace_Lag *plag;
    PetscInt dof, off;

    ierr = PetscSectionGetDof(section, p, &dof);CHKERRQ(ierr);
    if (!dof) continue;
    plag = (PetscDualSpace_Lag *) psp->data;
    ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
    ierr = PetscLagNodeIndicesPushForward(dm, verti, p, plag->vertIndices, plag->intNodeIndices, 0, formDegree, &(ni->nodeIdx[off * nodeIdxDim]), &(ni->nodeVec[off * Nk]));CHKERRQ(ierr);
  }
  lag->allNodeIndices = ni;
  PetscFunctionReturn(0);
}

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
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetSection(sp, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(section, &nDofs);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetFormDegree(sp, &k);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(k), &Nk);CHKERRQ(ierr);
  for (p = pStart, nNodes = 0, maxNzforms = 0; p < pEnd; p++) {
    PetscDualSpace  psp;
    DM              pdm;
    PetscInt        pdim, pNk;
    PetscQuadrature intNodes;
    Mat intMat;

    ierr = PetscDualSpaceGetPointSubspace(sp, p, &psp);CHKERRQ(ierr);
    if (!psp) continue;
    ierr = PetscDualSpaceGetDM(psp, &pdm);CHKERRQ(ierr);
    ierr = DMGetDimension(pdm, &pdim);CHKERRQ(ierr);
    if (pdim < PetscAbsInt(k)) continue;
    ierr = PetscDTBinomialInt(pdim, PetscAbsInt(k), &pNk);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetInteriorData(psp, &intNodes, &intMat);CHKERRQ(ierr);
    if (intNodes) {
      PetscInt nNodesp;

      ierr = PetscQuadratureGetData(intNodes, NULL, NULL, &nNodesp, NULL, NULL);CHKERRQ(ierr);
      nNodes += nNodesp;
    }
    if (intMat) {
      PetscInt maxNzsp;
      PetscInt maxNzformsp;

      ierr = MatSeqAIJGetMaxRowNonzeros(intMat, &maxNzsp);CHKERRQ(ierr);
      if (maxNzsp % pNk) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "interior matrix is not laid out as blocks of k-forms");
      maxNzformsp = maxNzsp / pNk;
      maxNzforms = PetscMax(maxNzforms, maxNzformsp);
    }
  }
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, nDofs, nNodes * Nc, maxNzforms * Nk, NULL, &allMat);CHKERRQ(ierr);
  ierr = MatSetOption(allMat,MAT_IGNORE_ZERO_ENTRIES,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscMalloc7(dim, &v0, dim, &pv0, dim * dim, &J, dim * dim, &Jinv, Nk * Nk, &L, maxNzforms * Nk, &work, maxNzforms * Nk, &iwork);CHKERRQ(ierr);
  for (j = 0; j < dim; j++) pv0[j] = -1.;
  ierr = PetscMalloc1(dim * nNodes, &nodes);CHKERRQ(ierr);
  for (p = pStart, countNodes = 0; p < pEnd; p++) {
    PetscDualSpace  psp;
    PetscQuadrature intNodes;
    DM pdm;
    PetscInt pdim, pNk;
    PetscInt countNodesIn = countNodes;
    PetscReal detJ;
    Mat intMat;

    ierr = PetscDualSpaceGetPointSubspace(sp, p, &psp);CHKERRQ(ierr);
    if (!psp) continue;
    ierr = PetscDualSpaceGetDM(psp, &pdm);CHKERRQ(ierr);
    ierr = DMGetDimension(pdm, &pdim);CHKERRQ(ierr);
    if (pdim < PetscAbsInt(k)) continue;
    ierr = PetscDualSpaceGetInteriorData(psp, &intNodes, &intMat);CHKERRQ(ierr);
    if (intNodes == NULL && intMat == NULL) continue;
    ierr = PetscDTBinomialInt(pdim, PetscAbsInt(k), &pNk);CHKERRQ(ierr);
    if (p) {
      ierr = DMPlexComputeCellGeometryAffineFEM(dm, p, v0, J, Jinv, &detJ);CHKERRQ(ierr);
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
    ierr = PetscDTAltVPullbackMatrix(pdim, dim, J, k, L);CHKERRQ(ierr);
    if (intNodes) { /* "push forward" dof by pulling back a k-form to be evaluated on the point: multiply on the right by L^T */
      PetscInt nNodesp;
      const PetscReal *nodesp;
      PetscInt j;

      ierr = PetscQuadratureGetData(intNodes, NULL, NULL, &nNodesp, &nodesp, NULL);CHKERRQ(ierr);
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

      ierr = PetscSectionGetDof(section, p, &nrows);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(section, p, &off);CHKERRQ(ierr);
      for (j = 0; j < nrows; j++) {
        PetscInt ncols;
        const PetscInt *cols;
        const PetscScalar *vals;
        PetscInt l, d, e;
        PetscInt row = j + off;

        ierr = MatGetRow(intMat, j, &ncols, &cols, &vals);CHKERRQ(ierr);
        if (ncols % pNk) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "interior matrix is not laid out as blocks of k-forms");
        for (l = 0; l < ncols / pNk; l++) {
          PetscInt blockcol;

          for (d = 0; d < pNk; d++) {
            if ((cols[l * pNk + d] % pNk) != d) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "interior matrix is not laid out as blocks of k-forms");
          }
          blockcol = cols[l * pNk] / pNk;
          for (d = 0; d < Nk; d++) {
            iwork[l * Nk + d] = (blockcol + countNodesIn) * Nk + d;
          }
          for (d = 0; d < Nk; d++) work[l * Nk + d] = 0.;
          for (d = 0; d < Nk; d++) {
            for (e = 0; e < pNk; e++) {
              /* "push forward" dof by pulling back a k-form to be evaluated on the point: multiply on the right by L */
              work[l * Nk + d] += vals[l * pNk + e] * L[e * pNk + d];
            }
          }
        }
        ierr = MatSetValues(allMat, 1, &row, (ncols / pNk) * Nk, iwork, work, INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatRestoreRow(intMat, j, &ncols, &cols, &vals);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(allMat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(allMat, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &allNodes);CHKERRQ(ierr);
  ierr = PetscQuadratureSetData(allNodes, dim, 0, nNodes, nodes, NULL);CHKERRQ(ierr);
  ierr = PetscFree7(v0, pv0, J, Jinv, L, work, iwork);CHKERRQ(ierr);
  ierr = MatDestroy(&(sp->allMat));CHKERRQ(ierr);
  sp->allMat = allMat;
  ierr = PetscQuadratureDestroy(&(sp->allNodes));CHKERRQ(ierr);
  sp->allNodes = allNodes;
  PetscFunctionReturn(0);
}

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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetFormDegree(sp, &k);CHKERRQ(ierr);
  ierr = PetscDTBinomialInt(dim, PetscAbsInt(k), &Nk);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetAllData(sp, &allNodes, &allMat);CHKERRQ(ierr);
  nNodes = 0;
  if (allNodes) {
    ierr = PetscQuadratureGetData(allNodes, NULL, NULL, &nNodes, &nodes, NULL);CHKERRQ(ierr);
  }
  ierr = MatGetSize(allMat, &nDofs, NULL);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetSection(sp, &section);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(section, &spdim);CHKERRQ(ierr);
  if (spdim != nDofs) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "incompatible all matrix size");
  ierr = PetscMalloc1(nDofs, &(sp->functional));CHKERRQ(ierr);
  for (f = 0; f < nDofs; f++) {
    PetscInt ncols, c;
    const PetscInt *cols;
    const PetscScalar *vals;
    PetscReal *nodesf;
    PetscReal *weightsf;
    PetscInt nNodesf;
    PetscInt countNodes;

    ierr = MatGetRow(allMat, f, &ncols, &cols, &vals);CHKERRQ(ierr);
    if (ncols % Nk) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "all matrix is not laid out as blocks of k-forms");
    for (c = 1, nNodesf = 1; c < ncols; c++) {
      if ((cols[c] / Nc) != (cols[c-1] / Nc)) nNodesf++;
    }
    ierr = PetscMalloc1(dim * nNodesf, &nodesf);CHKERRQ(ierr);
    ierr = PetscMalloc1(Nc * nNodesf, &weightsf);CHKERRQ(ierr);
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
    ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &(sp->functional[f]));CHKERRQ(ierr);
    ierr = PetscQuadratureSetData(sp->functional[f], dim, Nc, nNodesf, nodesf, weightsf);CHKERRQ(ierr);
    ierr = MatRestoreRow(allMat, f, &ncols, &cols, &vals);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* take a matrix meant for k-forms and expand it to one for Ncopies */
static PetscErrorCode PetscDualSpaceLagrangeMatrixCreateCopies(Mat A, PetscInt Nk, PetscInt Ncopies, Mat *Abs)
{
  PetscInt       m, n, i, j, k;
  PetscInt       maxnnz, *nnz, *iwork;
  Mat            Ac;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetSize(A, &m, &n);CHKERRQ(ierr);
  if (n % Nk) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of columns in A %D is not a multiple of Nk %D", n, Nk);
  ierr = PetscMalloc1(m * Ncopies, &nnz);CHKERRQ(ierr);
  for (i = 0, maxnnz = 0; i < m; i++) {
    PetscInt innz;
    ierr = MatGetRow(A, i, &innz, NULL, NULL);CHKERRQ(ierr);
    if (innz % Nk) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "A row %D nnzs is not a multiple of Nk %D", innz, Nk);
    for (j = 0; j < Ncopies; j++) nnz[i * Ncopies + j] = innz;
    maxnnz = PetscMax(maxnnz, innz);
  }
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, m * Ncopies, n * Ncopies, 0, nnz, &Ac);CHKERRQ(ierr);
  ierr = MatSetOption(Ac, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);
  ierr = PetscMalloc1(maxnnz, &iwork);CHKERRQ(ierr);
  for (i = 0; i < m; i++) {
    PetscInt innz;
    const PetscInt    *cols;
    const PetscScalar *vals;

    ierr = MatGetRow(A, i, &innz, &cols, &vals);CHKERRQ(ierr);
    for (j = 0; j < innz; j++) iwork[j] = (cols[j] / Nk) * (Nk * Ncopies) + (cols[j] % Nk);
    for (j = 0; j < Ncopies; j++) {
      PetscInt row = i * Ncopies + j;

      ierr = MatSetValues(Ac, 1, &row, innz, iwork, vals, INSERT_VALUES);CHKERRQ(ierr);
      for (k = 0; k < innz; k++) iwork[k] += Nk;
    }
    ierr = MatRestoreRow(A, i, &innz, &cols, &vals);CHKERRQ(ierr);
  }
  ierr = PetscFree(iwork);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Ac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Ac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *Abs = Ac;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexPointIsTensor_Internal_Given(DM dm, PetscInt p, PetscInt f, PetscInt f2, PetscBool *isTensor)
{
  PetscInt        coneSize, c;
  const PetscInt *cone;
  const PetscInt *fCone;
  const PetscInt *f2Cone;
  PetscInt        fs[2];
  PetscInt        meetSize, nmeet;
  const PetscInt *meet;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  fs[0] = f;
  fs[1] = f2;
  ierr = DMPlexGetMeet(dm, 2, fs, &meetSize, &meet);CHKERRQ(ierr);
  nmeet = meetSize;
  ierr = DMPlexRestoreMeet(dm, 2, fs, &meetSize, &meet);CHKERRQ(ierr);
  if (nmeet) {
    *isTensor = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, f, &fCone);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, f2, &f2Cone);CHKERRQ(ierr);
  for (c = 0; c < coneSize; c++) {
    PetscInt e, ef;
    PetscInt d = -1, d2 = -1;
    PetscInt dcount, d2count;
    PetscInt t = cone[c];
    PetscInt tConeSize;
    PetscBool tIsTensor;
    const PetscInt *tCone;

    if (t == f || t == f2) continue;
    ierr = DMPlexGetConeSize(dm, t, &tConeSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm, t, &tCone);CHKERRQ(ierr);

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
    ierr = DMPlexPointIsTensor_Internal_Given(dm, t, d, d2, &tIsTensor);CHKERRQ(ierr);
    if (!tIsTensor) {
      *isTensor = PETSC_FALSE;
      PetscFunctionReturn(0);
    }
  }
  *isTensor = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexPointIsTensor_Internal(DM dm, PetscInt p, PetscBool *isTensor, PetscInt *endA, PetscInt *endB)
{
  PetscInt        coneSize, c, c2;
  const PetscInt *cone;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
  if (!coneSize) {
    if (isTensor) *isTensor = PETSC_FALSE;
    if (endA) *endA = -1;
    if (endB) *endB = -1;
  }
  ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
  for (c = 0; c < coneSize; c++) {
    PetscInt f = cone[c];
    PetscInt fConeSize;

    ierr = DMPlexGetConeSize(dm, f, &fConeSize);CHKERRQ(ierr);
    if (fConeSize != coneSize - 2) continue;

    for (c2 = c + 1; c2 < coneSize; c2++) {
      PetscInt  f2 = cone[c2];
      PetscBool isTensorff2;
      PetscInt f2ConeSize;

      ierr = DMPlexGetConeSize(dm, f2, &f2ConeSize);CHKERRQ(ierr);
      if (f2ConeSize != coneSize - 2) continue;

      ierr = DMPlexPointIsTensor_Internal_Given(dm, p, f, f2, &isTensorff2);CHKERRQ(ierr);
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

static PetscErrorCode DMPlexPointIsTensor(DM dm, PetscInt p, PetscBool *isTensor, PetscInt *endA, PetscInt *endB)
{
  DMPlexInterpolatedFlag interpolated;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexIsInterpolated(dm, &interpolated);CHKERRQ(ierr);
  if (interpolated != DMPLEX_INTERPOLATED_FULL) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "Only for interpolated DMPlex's");
  ierr = DMPlexPointIsTensor_Internal(dm, p, isTensor, endA, endB);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLagNodeIndicesGetPermutation(ni, &perm);CHKERRQ(ierr);
  ierr = MatGetSize(A, &m, &n);CHKERRQ(ierr);
  ierr = PetscMalloc1(nodeIdxDim * m, &nIdxPerm);CHKERRQ(ierr);
  ierr = PetscMalloc1(nodeVecDim * m, &nVecPerm);CHKERRQ(ierr);
  for (i = 0; i < m; i++) for (j = 0; j < nodeIdxDim; j++) nIdxPerm[i * nodeIdxDim + j] = ni->nodeIdx[perm[i] * nodeIdxDim + j];
  for (i = 0; i < m; i++) for (j = 0; j < nodeVecDim; j++) nVecPerm[i * nodeVecDim + j] = ni->nodeVec[perm[i] * nodeVecDim + j];
  ierr = ISCreateGeneral(PETSC_COMM_SELF, m, perm, PETSC_USE_POINTER, &permIS);CHKERRQ(ierr);
  ierr = ISSetPermutation(permIS);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF, n, 0, 1, &id);CHKERRQ(ierr);
  ierr = ISSetPermutation(id);CHKERRQ(ierr);
  ierr = MatPermute(A, permIS, id, Aperm);CHKERRQ(ierr);
  ierr = ISDestroy(&permIS);CHKERRQ(ierr);
  ierr = ISDestroy(&id);CHKERRQ(ierr);
  for (i = 0; i < m; i++) perm[i] = i;
  ierr = PetscFree(ni->nodeIdx);CHKERRQ(ierr);
  ierr = PetscFree(ni->nodeVec);CHKERRQ(ierr);
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
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  /* step 1: sanitize input */
  ierr = PetscObjectGetComm((PetscObject) sp, &comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)sp, "bdm", &isbdm);CHKERRQ(ierr);
  if (isbdm) {
    sp->k = -(dim-1); /* form degree of H-div */
    ierr = PetscObjectChangeTypeName((PetscObject)sp, PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
  }
  ierr = PetscDualSpaceGetFormDegree(sp, &formDegree);CHKERRQ(ierr);
  if (PetscAbsInt(formDegree) > dim) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Form degree must be bounded by dimension");
  ierr = PetscDTBinomialInt(dim,PetscAbsInt(formDegree),&Nk);CHKERRQ(ierr);
  if (sp->Nc <= 0 && lag->numCopies > 0) sp->Nc = Nk * lag->numCopies;
  Nc = sp->Nc;
  if (Nc % Nk) SETERRQ(comm, PETSC_ERR_ARG_INCOMP, "Number of components is not a multiple of form degree size");
  if (lag->numCopies <= 0) lag->numCopies = Nc / Nk;
  Ncopies = lag->numCopies;
  if (Nc / Nk != Ncopies) SETERRQ(comm, PETSC_ERR_ARG_INCOMP, "Number of copies * (dim choose k) != Nc");
  if (!dim) sp->order = 0;
  order = sp->order;
  uniform = sp->uniform;
  if (!uniform) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Variable order not supported yet");
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
  if (lag->trimmed && !order) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Cannot have zeroth order trimmed elements");
  if (lag->trimmed && PetscAbsInt(formDegree) == dim) { /* convert trimmed n-forms to untrimmed of one polynomial order less */
    sp->order--;
    order--;
    lag->trimmed = PETSC_FALSE;
  }
  trimmed = lag->trimmed;
  if (!order || PetscAbsInt(formDegree) == dim) lag->continuous = PETSC_FALSE;
  continuous = lag->continuous;
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  if (pStart != 0 || cStart != 0) SETERRQ(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "Expect DM with chart starting at zero and cells first");
  if (cEnd != 1) SETERRQ(PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "Use PETSCDUALSPACEREFINED for multi-cell reference meshes");
  ierr = DMPlexIsInterpolated(dm, &interpolated);CHKERRQ(ierr);
  if (interpolated != DMPLEX_INTERPOLATED_FULL) {
    ierr = DMPlexInterpolate(dm, &dmint);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectReference((PetscObject)dm);CHKERRQ(ierr);
    dmint = dm;
  }
  tensorCell = PETSC_FALSE;
  if (dim > 1) {
    ierr = DMPlexPointIsTensor(dmint, 0, &tensorCell, &tensorf, &tensorf2);CHKERRQ(ierr);
  }
  lag->tensorCell = tensorCell;
  if (dim < 2 || !lag->tensorCell) lag->tensorSpace = PETSC_FALSE;
  tensorSpace = lag->tensorSpace;
  if (!lag->nodeFamily) {
    ierr = Petsc1DNodeFamilyCreate(lag->nodeType, lag->nodeExponent, lag->endNodes, &lag->nodeFamily);CHKERRQ(ierr);
  }
  nodeFamily = lag->nodeFamily;
  if (interpolated != DMPLEX_INTERPOLATED_FULL && continuous && (PetscAbsInt(formDegree) > 0 || order > 1)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Reference element won't support all boundary nodes");

  /* step 2: construct the boundary spaces */
  ierr = PetscMalloc2(depth+1,&pStratStart,depth+1,&pStratEnd);CHKERRQ(ierr);
  ierr = PetscCalloc1(pEnd,&(sp->pointSpaces));CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {ierr = DMPlexGetDepthStratum(dm, d, &pStratStart[d], &pStratEnd[d]);CHKERRQ(ierr);}
  ierr = PetscDualSpaceSectionCreate_Internal(sp, &section);CHKERRQ(ierr);
  sp->pointSection = section;
  if (continuous && !(lag->interiorOnly)) {
    PetscInt h;

    for (p = pStratStart[depth - 1]; p < pStratEnd[depth - 1]; p++) { /* calculate the facet dual spaces */
      PetscReal v0[3];
      DMPolytopeType ptype;
      PetscReal J[9], detJ;
      PetscInt  q;

      ierr = DMPlexComputeCellGeometryAffineFEM(dm, p, v0, J, NULL, &detJ);CHKERRQ(ierr);
      ierr = DMPlexGetCellType(dm, p, &ptype);CHKERRQ(ierr);

      /* compare orders to previous facets: if computed, reference that dualspace */
      for (q = pStratStart[depth - 1]; q < p; q++) {
        DMPolytopeType qtype;

        ierr = DMPlexGetCellType(dm, q, &qtype);CHKERRQ(ierr);
        if (qtype == ptype) break;
      }
      if (q < p) { /* this facet has the same dual space as that one */
        ierr = PetscObjectReference((PetscObject)sp->pointSpaces[q]);CHKERRQ(ierr);
        sp->pointSpaces[p] = sp->pointSpaces[q];
        continue;
      }
      /* if not, recursively compute this dual space */
      ierr = PetscDualSpaceCreateFacetSubspace_Lagrange(sp,NULL,p,formDegree,Ncopies,PETSC_FALSE,&sp->pointSpaces[p]);CHKERRQ(ierr);
    }
    for (h = 2; h <= depth; h++) { /* get the higher subspaces from the facet subspaces */
      PetscInt hd = depth - h;
      PetscInt hdim = dim - h;

      if (hdim < PetscAbsInt(formDegree)) break;
      for (p = pStratStart[hd]; p < pStratEnd[hd]; p++) {
        PetscInt suppSize, s;
        const PetscInt *supp;

        ierr = DMPlexGetSupportSize(dm, p, &suppSize);CHKERRQ(ierr);
        ierr = DMPlexGetSupport(dm, p, &supp);CHKERRQ(ierr);
        for (s = 0; s < suppSize; s++) {
          DM             qdm;
          PetscDualSpace qsp, psp;
          PetscInt c, coneSize, q;
          const PetscInt *cone;
          const PetscInt *refCone;

          q = supp[0];
          qsp = sp->pointSpaces[q];
          ierr = DMPlexGetConeSize(dm, q, &coneSize);CHKERRQ(ierr);
          ierr = DMPlexGetCone(dm, q, &cone);CHKERRQ(ierr);
          for (c = 0; c < coneSize; c++) if (cone[c] == p) break;
          if (c == coneSize) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "cone/suppport mismatch");
          ierr = PetscDualSpaceGetDM(qsp, &qdm);CHKERRQ(ierr);
          ierr = DMPlexGetCone(qdm, 0, &refCone);CHKERRQ(ierr);
          /* get the equivalent dual space from the support dual space */
          ierr = PetscDualSpaceGetPointSubspace(qsp, refCone[c], &psp);CHKERRQ(ierr);
          if (!s) {
            ierr = PetscObjectReference((PetscObject)psp);CHKERRQ(ierr);
            sp->pointSpaces[p] = psp;
          }
        }
      }
    }
    for (p = 1; p < pEnd; p++) {
      PetscInt pspdim;
      if (!sp->pointSpaces[p]) continue;
      ierr = PetscDualSpaceGetInteriorDimension(sp->pointSpaces[p], &pspdim);CHKERRQ(ierr);
      ierr = PetscSectionSetDof(section, p, pspdim);CHKERRQ(ierr);
    }
  }

  if (Ncopies > 1) {
    Mat intMatScalar, allMatScalar;
    PetscDualSpace scalarsp;
    PetscDualSpace_Lag *scalarlag;

    ierr = PetscDualSpaceDuplicate(sp, &scalarsp);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetNumComponents(scalarsp, Nk);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetUp(scalarsp);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetInteriorData(scalarsp, &(sp->intNodes), &intMatScalar);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)(sp->intNodes));CHKERRQ(ierr);
    if (intMatScalar) {ierr = PetscDualSpaceLagrangeMatrixCreateCopies(intMatScalar, Nk, Ncopies, &(sp->intMat));CHKERRQ(ierr);}
    ierr = PetscDualSpaceGetAllData(scalarsp, &(sp->allNodes), &allMatScalar);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)(sp->allNodes));CHKERRQ(ierr);
    ierr = PetscDualSpaceLagrangeMatrixCreateCopies(allMatScalar, Nk, Ncopies, &(sp->allMat));CHKERRQ(ierr);
    sp->spdim = scalarsp->spdim * Ncopies;
    sp->spintdim = scalarsp->spintdim * Ncopies;
    scalarlag = (PetscDualSpace_Lag *) scalarsp->data;
    ierr = PetscLagNodeIndicesReference(scalarlag->vertIndices);CHKERRQ(ierr);
    lag->vertIndices = scalarlag->vertIndices;
    ierr = PetscLagNodeIndicesReference(scalarlag->intNodeIndices);CHKERRQ(ierr);
    lag->intNodeIndices = scalarlag->intNodeIndices;
    ierr = PetscLagNodeIndicesReference(scalarlag->allNodeIndices);CHKERRQ(ierr);
    lag->allNodeIndices = scalarlag->allNodeIndices;
    ierr = PetscDualSpaceDestroy(&scalarsp);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(section, 0, sp->spintdim);CHKERRQ(ierr);
    ierr = PetscDualSpaceSectionSetUp_Internal(sp, section);CHKERRQ(ierr);
    ierr = PetscDualSpaceComputeFunctionalsFromAllData(sp);CHKERRQ(ierr);
    ierr = PetscFree2(pStratStart, pStratEnd);CHKERRQ(ierr);
    ierr = DMDestroy(&dmint);CHKERRQ(ierr);
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

    ierr = PetscDualSpaceDuplicate(sp, &spcont);CHKERRQ(ierr);
    ierr = PetscDualSpaceLagrangeSetContinuity(spcont, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetUp(spcont);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetDimension(spcont, &spdim);CHKERRQ(ierr);
    sp->spdim = sp->spintdim = spdim;
    ierr = PetscSectionSetDof(section, 0, spdim);CHKERRQ(ierr);
    ierr = PetscDualSpaceSectionSetUp_Internal(sp, section);CHKERRQ(ierr);
    ierr = PetscMalloc1(spdim, &(sp->functional));CHKERRQ(ierr);
    for (f = 0; f < spdim; f++) {
      PetscQuadrature fn;

      ierr = PetscDualSpaceGetFunctional(spcont, f, &fn);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)fn);CHKERRQ(ierr);
      sp->functional[f] = fn;
    }
    ierr = PetscDualSpaceGetAllData(spcont, &allNodes, &allMat);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject) allNodes);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject) allNodes);CHKERRQ(ierr);
    sp->allNodes = sp->intNodes = allNodes;
    ierr = PetscObjectReference((PetscObject) allMat);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject) allMat);CHKERRQ(ierr);
    sp->allMat = sp->intMat = allMat;
    /* TODO: copy over symmetries */
    lagc = (PetscDualSpace_Lag *) spcont->data;
    ierr = PetscLagNodeIndicesReference(lagc->vertIndices);CHKERRQ(ierr);
    lag->vertIndices = lagc->vertIndices;
    ierr = PetscLagNodeIndicesReference(lagc->allNodeIndices);CHKERRQ(ierr);
    ierr = PetscLagNodeIndicesReference(lagc->allNodeIndices);CHKERRQ(ierr);
    lag->intNodeIndices = lagc->allNodeIndices;
    lag->allNodeIndices = lagc->allNodeIndices;
    ierr = PetscDualSpaceDestroy(&spcont);CHKERRQ(ierr);
    ierr = PetscFree2(pStratStart, pStratEnd);CHKERRQ(ierr);
    ierr = DMDestroy(&dmint);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* step 3: construct intNodes, and intMat, and combine it with boundray data to make allNodes and allMat */
  if (!tensorSpace) {
    if (!tensorCell) {ierr = PetscLagNodeIndicesCreateSimplexVertices(dm, &(lag->vertIndices));CHKERRQ(ierr);}

    if (trimmed) {
      if (order + PetscAbsInt(formDegree) > dim) {
        PetscInt sum = order + PetscAbsInt(formDegree) - dim - 1;
        PetscInt nDofs;

        ierr = PetscDualSpaceLagrangeCreateSimplexNodeMat(nodeFamily, dim, sum, Nk, numNodeSkip, &sp->intNodes, &sp->intMat, &(lag->intNodeIndices));CHKERRQ(ierr);
        ierr = MatGetSize(sp->intMat, &nDofs, NULL);CHKERRQ(ierr);
        ierr = PetscSectionSetDof(section, 0, nDofs);CHKERRQ(ierr);
      }
      ierr = PetscDualSpaceSectionSetUp_Internal(sp, section);CHKERRQ(ierr);
      ierr = PetscDualSpaceCreateAllDataFromInteriorData(sp);CHKERRQ(ierr);
      ierr = PetscDualSpaceLagrangeCreateAllNodeIdx(sp);CHKERRQ(ierr);
    } else {
      if (!continuous) {
        PetscInt sum = order;
        PetscInt nDofs;

        ierr = PetscDualSpaceLagrangeCreateSimplexNodeMat(nodeFamily, dim, sum, Nk, numNodeSkip, &sp->intNodes, &sp->intMat, &(lag->intNodeIndices));CHKERRQ(ierr);
        ierr = MatGetSize(sp->intMat, &nDofs, NULL);CHKERRQ(ierr);
        ierr = PetscSectionSetDof(section, 0, nDofs);CHKERRQ(ierr);
        ierr = PetscDualSpaceSectionSetUp_Internal(sp, section);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)(sp->intNodes));CHKERRQ(ierr);
        sp->allNodes = sp->intNodes;
        ierr = PetscObjectReference((PetscObject)(sp->intMat));CHKERRQ(ierr);
        sp->allMat = sp->intMat;
        ierr = PetscLagNodeIndicesReference(lag->intNodeIndices);CHKERRQ(ierr);
        lag->allNodeIndices = lag->intNodeIndices;
      } else {
        if (order + PetscAbsInt(formDegree) > dim) {
          PetscDualSpace trimmedsp;
          PetscDualSpace_Lag *trimmedlag;
          PetscQuadrature intNodes;
          PetscInt trFormDegree = formDegree >= 0 ? formDegree - dim : dim - PetscAbsInt(formDegree);
          PetscInt nDofs;
          Mat intMat;

          ierr = PetscDualSpaceDuplicate(sp, &trimmedsp);CHKERRQ(ierr);
          ierr = PetscDualSpaceLagrangeSetTrimmed(trimmedsp, PETSC_TRUE);CHKERRQ(ierr);
          ierr = PetscDualSpaceSetOrder(trimmedsp, order + PetscAbsInt(formDegree) - dim);CHKERRQ(ierr);
          ierr = PetscDualSpaceSetFormDegree(trimmedsp, trFormDegree);CHKERRQ(ierr);
          trimmedlag = (PetscDualSpace_Lag *) trimmedsp->data;
          trimmedlag->numNodeSkip = numNodeSkip + 1;
          ierr = PetscDualSpaceSetUp(trimmedsp);CHKERRQ(ierr);
          ierr = PetscDualSpaceGetAllData(trimmedsp, &intNodes, &intMat);CHKERRQ(ierr);
          ierr = PetscObjectReference((PetscObject)intNodes);CHKERRQ(ierr);
          sp->intNodes = intNodes;
          ierr = PetscObjectReference((PetscObject)intMat);CHKERRQ(ierr);
          sp->intMat = intMat;
          ierr = MatGetSize(sp->intMat, &nDofs, NULL);CHKERRQ(ierr);
          ierr = PetscLagNodeIndicesReference(trimmedlag->allNodeIndices);CHKERRQ(ierr);
          lag->intNodeIndices = trimmedlag->allNodeIndices;
          ierr = PetscDualSpaceDestroy(&trimmedsp);CHKERRQ(ierr);
          ierr = PetscSectionSetDof(section, 0, nDofs);CHKERRQ(ierr);
        }
        ierr = PetscDualSpaceSectionSetUp_Internal(sp, section);CHKERRQ(ierr);
        ierr = PetscDualSpaceCreateAllDataFromInteriorData(sp);CHKERRQ(ierr);
        ierr = PetscDualSpaceLagrangeCreateAllNodeIdx(sp);CHKERRQ(ierr);
      }
    }
  } else {
    /* assume the tensor element has the first facet being the cross-section, having its normal
     * pointing in the last coordinate direction */
    PetscQuadrature intNodesTrace = NULL;
    PetscQuadrature intNodesFiber = NULL;
    PetscQuadrature intNodes = NULL;
    PetscLagNodeIndices intNodeIndices = NULL;
    Mat             intMat = NULL;

    if (PetscAbsInt(formDegree) < dim) { /* get the trace k-forms on the first facet, and the 0-forms on the edge */
      PetscDualSpace  trace, fiber;
      PetscDualSpace_Lag *tracel, *fiberl;
      Mat             intMatTrace, intMatFiber;

      if (sp->pointSpaces[tensorf]) {
        ierr = PetscObjectReference((PetscObject)(sp->pointSpaces[tensorf]));CHKERRQ(ierr);
        trace = sp->pointSpaces[tensorf];
      } else {
        ierr = PetscDualSpaceCreateFacetSubspace_Lagrange(sp,NULL,tensorf,formDegree,Ncopies,PETSC_TRUE,&trace);CHKERRQ(ierr);
      }
      ierr = PetscDualSpaceCreateEdgeSubspace_Lagrange(sp,order,0,1,PETSC_TRUE,&fiber);CHKERRQ(ierr);
      tracel = (PetscDualSpace_Lag *) trace->data;
      fiberl = (PetscDualSpace_Lag *) fiber->data;
      ierr = PetscLagNodeIndicesCreateTensorVertices(dm, tracel->vertIndices, &(lag->vertIndices));CHKERRQ(ierr);
      ierr = PetscDualSpaceGetInteriorData(trace, &intNodesTrace, &intMatTrace);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetInteriorData(fiber, &intNodesFiber, &intMatFiber);CHKERRQ(ierr);
      if (intNodesTrace && intNodesFiber) {
        ierr = PetscQuadratureCreateTensor(intNodesTrace, intNodesFiber, &intNodes);CHKERRQ(ierr);
        ierr = MatTensorAltV(intMatTrace, intMatFiber, dim-1, formDegree, 1, 0, &intMat);CHKERRQ(ierr);
        ierr = PetscLagNodeIndicesTensor(tracel->intNodeIndices, dim - 1, formDegree, fiberl->intNodeIndices, 1, 0, &intNodeIndices);CHKERRQ(ierr);
      }
      ierr = PetscObjectReference((PetscObject) intNodesTrace);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject) intNodesFiber);CHKERRQ(ierr);
      ierr = PetscDualSpaceDestroy(&fiber);CHKERRQ(ierr);
      ierr = PetscDualSpaceDestroy(&trace);CHKERRQ(ierr);
    }
    if (PetscAbsInt(formDegree) > 0) { /* get the trace (k-1)-forms on the first facet, and the 1-forms on the edge */
      PetscDualSpace  trace, fiber;
      PetscDualSpace_Lag *tracel, *fiberl;
      PetscQuadrature intNodesTrace2, intNodesFiber2, intNodes2;
      PetscLagNodeIndices intNodeIndices2;
      Mat             intMatTrace, intMatFiber, intMat2;
      PetscInt        traceDegree = formDegree > 0 ? formDegree - 1 : formDegree + 1;
      PetscInt        fiberDegree = formDegree > 0 ? 1 : -1;

      ierr = PetscDualSpaceCreateFacetSubspace_Lagrange(sp,NULL,tensorf,traceDegree,Ncopies,PETSC_TRUE,&trace);CHKERRQ(ierr);
      ierr = PetscDualSpaceCreateEdgeSubspace_Lagrange(sp,order,fiberDegree,1,PETSC_TRUE,&fiber);CHKERRQ(ierr);
      tracel = (PetscDualSpace_Lag *) trace->data;
      fiberl = (PetscDualSpace_Lag *) fiber->data;
      if (!lag->vertIndices) {
        ierr = PetscLagNodeIndicesCreateTensorVertices(dm, tracel->vertIndices, &(lag->vertIndices));CHKERRQ(ierr);
      }
      ierr = PetscDualSpaceGetInteriorData(trace, &intNodesTrace2, &intMatTrace);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetInteriorData(fiber, &intNodesFiber2, &intMatFiber);CHKERRQ(ierr);
      if (intNodesTrace2 && intNodesFiber2) {
        ierr = PetscQuadratureCreateTensor(intNodesTrace2, intNodesFiber2, &intNodes2);CHKERRQ(ierr);
        ierr = MatTensorAltV(intMatTrace, intMatFiber, dim-1, traceDegree, 1, fiberDegree, &intMat2);CHKERRQ(ierr);
        ierr = PetscLagNodeIndicesTensor(tracel->intNodeIndices, dim - 1, traceDegree, fiberl->intNodeIndices, 1, fiberDegree, &intNodeIndices2);CHKERRQ(ierr);
        if (!intMat) {
          intMat = intMat2;
          intNodes = intNodes2;
          intNodeIndices = intNodeIndices2;
        } else {
          /* merge the two matrices and the two sets of points */
          PetscInt         nM;
          PetscInt         nDof, nDof2;
          PetscInt        *toMerged = NULL, *toMerged2 = NULL;
          PetscQuadrature  merged = NULL;
          PetscLagNodeIndices intNodeIndicesMerged = NULL;
          Mat              matMerged = NULL;

          ierr = MatGetSize(intMat, &nDof, 0);CHKERRQ(ierr);
          ierr = MatGetSize(intMat2, &nDof2, 0);CHKERRQ(ierr);
          ierr = PetscQuadraturePointsMerge(intNodes, intNodes2, &merged, &toMerged, &toMerged2);CHKERRQ(ierr);
          ierr = PetscQuadratureGetData(merged, NULL, NULL, &nM, NULL, NULL);CHKERRQ(ierr);
          ierr = MatricesMerge(intMat, intMat2, dim, formDegree, nM, toMerged, toMerged2, &matMerged);CHKERRQ(ierr);
          ierr = PetscLagNodeIndicesMerge(intNodeIndices, intNodeIndices2, &intNodeIndicesMerged);CHKERRQ(ierr);
          ierr = PetscFree(toMerged);CHKERRQ(ierr);
          ierr = PetscFree(toMerged2);CHKERRQ(ierr);
          ierr = MatDestroy(&intMat);CHKERRQ(ierr);
          ierr = MatDestroy(&intMat2);CHKERRQ(ierr);
          ierr = PetscQuadratureDestroy(&intNodes);CHKERRQ(ierr);
          ierr = PetscQuadratureDestroy(&intNodes2);CHKERRQ(ierr);
          ierr = PetscLagNodeIndicesDestroy(&intNodeIndices);CHKERRQ(ierr);
          ierr = PetscLagNodeIndicesDestroy(&intNodeIndices2);CHKERRQ(ierr);
          intNodes = merged;
          intMat = matMerged;
          intNodeIndices = intNodeIndicesMerged;
          if (!trimmed) {
            Mat intMatPerm;

            ierr = MatPermuteByNodeIdx(intMat, intNodeIndices, &intMatPerm);CHKERRQ(ierr);
            ierr = MatDestroy(&intMat);CHKERRQ(ierr);
            intMat = intMatPerm;
          }
        }
      }
      ierr = PetscDualSpaceDestroy(&fiber);CHKERRQ(ierr);
      ierr = PetscDualSpaceDestroy(&trace);CHKERRQ(ierr);
    }
    ierr = PetscQuadratureDestroy(&intNodesTrace);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&intNodesFiber);CHKERRQ(ierr);
    sp->intNodes = intNodes;
    sp->intMat = intMat;
    lag->intNodeIndices = intNodeIndices;
    {
      PetscInt nDofs = 0;

      if (intMat) {
        ierr = MatGetSize(intMat, &nDofs, NULL);CHKERRQ(ierr);
      }
      ierr = PetscSectionSetDof(section, 0, nDofs);CHKERRQ(ierr);
    }
    ierr = PetscDualSpaceSectionSetUp_Internal(sp, section);CHKERRQ(ierr);
    if (continuous) {
      ierr = PetscDualSpaceCreateAllDataFromInteriorData(sp);CHKERRQ(ierr);
      ierr = PetscDualSpaceLagrangeCreateAllNodeIdx(sp);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectReference((PetscObject) intNodes);CHKERRQ(ierr);
      sp->allNodes = intNodes;
      ierr = PetscObjectReference((PetscObject) intMat);CHKERRQ(ierr);
      sp->allMat = intMat;
      ierr = PetscLagNodeIndicesReference(intNodeIndices);CHKERRQ(ierr);
      lag->allNodeIndices = intNodeIndices;
    }
  }
  ierr = PetscSectionGetStorageSize(section, &sp->spdim);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(section, &sp->spintdim);CHKERRQ(ierr);
  ierr = PetscDualSpaceComputeFunctionalsFromAllData(sp);CHKERRQ(ierr);
  ierr = PetscFree2(pStratStart, pStratEnd);CHKERRQ(ierr);
  ierr = DMDestroy(&dmint);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!sp->spintdim) {
    *symMat = NULL;
    PetscFunctionReturn(0);
  }
  lag = (PetscDualSpace_Lag *) sp->data;
  vertIndices = lag->vertIndices;
  intNodeIndices = lag->intNodeIndices;
  ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetFormDegree(sp, &formDegree);CHKERRQ(ierr);
  ierr = PetscNew(&ni);CHKERRQ(ierr);
  ni->refct = 1;
  ni->nodeIdxDim = nodeIdxDim = intNodeIndices->nodeIdxDim;
  ni->nodeVecDim = nodeVecDim = intNodeIndices->nodeVecDim;
  ni->nNodes = nNodes = intNodeIndices->nNodes;
  ierr = PetscMalloc1(nNodes * nodeIdxDim, &(ni->nodeIdx));CHKERRQ(ierr);
  ierr = PetscMalloc1(nNodes * nodeVecDim, &(ni->nodeVec));CHKERRQ(ierr);
  ierr = PetscLagNodeIndicesPushForward(dm, vertIndices, 0, vertIndices, intNodeIndices, ornt, formDegree, ni->nodeIdx, ni->nodeVec);CHKERRQ(ierr);
  ierr = PetscLagNodeIndicesGetPermutation(intNodeIndices, &perm);CHKERRQ(ierr);
  ierr = PetscLagNodeIndicesGetPermutation(ni, &permOrnt);CHKERRQ(ierr);
  ierr = PetscMalloc1(nNodes, &nnz);CHKERRQ(ierr);
  for (n = 0, maxGroupSize = 0; n < nNodes;) { /* incremented in the loop */
    PetscInt *nind = &(ni->nodeIdx[permOrnt[n] * nodeIdxDim]);
    PetscInt m, nEnd;
    PetscInt groupSize;
    for (nEnd = n + 1; nEnd < nNodes; nEnd++) {
      PetscInt *mind = &(ni->nodeIdx[permOrnt[nEnd] * nodeIdxDim]);
      PetscInt d;

      /* compare the oriented permutation indices */
      for (d = 0; d < nodeIdxDim; d++) if (mind[d] != nind[d]) break;
      if (d < nodeIdxDim) break;
    }
#if defined(PETSC_USE_DEBUG)
    {
      PetscInt m;
      PetscInt *nind = &(intNodeIndices->nodeIdx[perm[n] * nodeIdxDim]);

      for (m = n + 1; m < nEnd; m++) {
        PetscInt *mind = &(intNodeIndices->nodeIdx[perm[m] * nodeIdxDim]);
        PetscInt d;

        /* compare the oriented permutation indices */
        for (d = 0; d < nodeIdxDim; d++) if (mind[d] != nind[d]) break;
        if (d < nodeIdxDim) break;
      }
      if (m < nEnd) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Dofs with same index after symmetry not same block size");
    }
#endif
    groupSize = nEnd - n;
    for (m = n; m < nEnd; m++) nnz[permOrnt[m]] = groupSize;

    maxGroupSize = PetscMax(maxGroupSize, nEnd - n);
    /* permOrnt[[n, nEnd)] is a group of dofs that, under the symmetry are at the same location */
    n = nEnd;
  }
  if (maxGroupSize > nodeVecDim) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Dofs are not in blocks that can be solved");
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, nNodes, nNodes, 0, nnz, &A);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);
  ierr = PetscMalloc3(maxGroupSize * nodeVecDim, &V, maxGroupSize * nodeVecDim, &W, nodeVecDim * 2, &work);CHKERRQ(ierr);
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
    /* get all of the vectors from the original */
    for (m = n; m < nEnd; m++) {
      PetscInt d;

      for (d = 0; d < nodeVecDim; d++) {
        V[(m - n) * nodeVecDim + d] = intNodeIndices->nodeVec[perm[m] * nodeVecDim + d];
        W[(m - n) * nodeVecDim + d] = ni->nodeVec[permOrnt[m] * nodeVecDim + d];
      }
    }
    /* now we have to solve for W in terms of V */
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
      if (info != 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to GELS");
      /* repack */
      {
        PetscInt i, j;

        for (i = 0; i < groupSize; i++) {
          for (j = 0; j < groupSize; j++) {
            V[i * groupSize + j] = W[i * nodeVecDim + j];
          }
        }
      }
    }
    ierr = MatSetValues(A, groupSize, &permOrnt[n], groupSize, &perm[n], V, INSERT_VALUES);CHKERRQ(ierr);
    /* permOrnt[[n, nEnd)] is a group of dofs that, under the symmetry are at the same location */
    n = nEnd;
  }
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *symMat = A;
  ierr = PetscFree3(V,W,work);CHKERRQ(ierr);
  ierr = PetscLagNodeIndicesDestroy(&ni);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define BaryIndex(perEdge,a,b,c) (((b)*(2*perEdge+1-(b)))/2)+(c)

#define CartIndex(perEdge,a,b) (perEdge*(a)+b)

static PetscErrorCode PetscDualSpaceGetSymmetries_Lagrange(PetscDualSpace sp, const PetscInt ****perms, const PetscScalar ****flips)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscInt           dim, order, Nc;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetOrder(sp,&order);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetNumComponents(sp,&Nc);CHKERRQ(ierr);
  ierr = DMGetDimension(sp->dm,&dim);CHKERRQ(ierr);
  if (!lag->symComputed) { /* store symmetries */
    PetscInt       pStart, pEnd, p;
    PetscInt       numPoints;
    PetscInt       numFaces;
    PetscInt       spintdim;
    PetscInt       ***symperms;
    PetscScalar    ***symflips;

    ierr = DMPlexGetChart(sp->dm, &pStart, &pEnd);CHKERRQ(ierr);
    numPoints = pEnd - pStart;
    ierr = DMPlexGetConeSize(sp->dm, 0, &numFaces);CHKERRQ(ierr);
    ierr = PetscCalloc1(numPoints,&symperms);CHKERRQ(ierr);
    ierr = PetscCalloc1(numPoints,&symflips);CHKERRQ(ierr);
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
      ierr = PetscCalloc1(2*numFaces,&cellSymperms);CHKERRQ(ierr);
      ierr = PetscCalloc1(2*numFaces,&cellSymflips);CHKERRQ(ierr);
      /* we want to be able to index symmetries directly with the orientations, which range from [-numFaces,numFaces) */
      symperms[0] = &cellSymperms[numFaces];
      symflips[0] = &cellSymflips[numFaces];
      if (lag->intNodeIndices->nodeVecDim * nCopies != Nc) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Node indices incompatible with dofs");
      if (nNodes * nCopies != spintdim) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Node indices incompatible with dofs");
      for (ornt = -numFaces; ornt < numFaces; ornt++) { /* for every symmetry, compute the symmetry matrix, and extract rows to see if it fits in the perm + flip framework */
        Mat symMat;
        PetscInt *perm;
        PetscScalar *flips;
        PetscInt i;

        if (!ornt) continue;
        ierr = PetscMalloc1(spintdim, &perm);CHKERRQ(ierr);
        ierr = PetscCalloc1(spintdim, &flips);CHKERRQ(ierr);
        for (i = 0; i < spintdim; i++) perm[i] = -1;
        ierr = PetscDualSpaceCreateInteriorSymmetryMatrix_Lagrange(sp, ornt, &symMat);CHKERRQ(ierr);
        for (i = 0; i < nNodes; i++) {
          PetscInt ncols;
          PetscInt j, k;
          const PetscInt *cols;
          const PetscScalar *vals;
          PetscBool nz_seen = PETSC_FALSE;

          ierr = MatGetRow(symMat, i, &ncols, &cols, &vals);CHKERRQ(ierr);
          for (j = 0; j < ncols; j++) {
            if (PetscAbsScalar(vals[j]) > PETSC_SMALL) {
              if (nz_seen) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "This dual space has symmetries that can't be described as a permutation + sign flips");
              nz_seen = PETSC_TRUE;
              if (PetscAbsScalar(PetscAbsScalar(vals[j]) - 1.) > PETSC_SMALL) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "This dual space has symmetries that can't be described as a permutation + sign flips");
              if (PetscAbsReal(PetscImaginaryPart(vals[j])) > PETSC_SMALL) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "This dual space has symmetries that can't be described as a permutation + sign flips");
              if (perm[cols[j] * nCopies] >= 0) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "This dual space has symmetries that can't be described as a permutation + sign flips");
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
          ierr = MatRestoreRow(symMat, i, &ncols, &cols, &vals);CHKERRQ(ierr);
        }
        ierr = MatDestroy(&symMat);CHKERRQ(ierr);
        /* if there were no sign flips, keep NULL */
        for (i = 0; i < spintdim; i++) if (flips[i] != 1.) break;
        if (i == spintdim) {
          ierr = PetscFree(flips);CHKERRQ(ierr);
          flips = NULL;
        }
        /* if the permutation is identity, keep NULL */
        for (i = 0; i < spintdim; i++) if (perm[i] != i) break;
        if (i == spintdim) {
          ierr = PetscFree(perm);CHKERRQ(ierr);
          perm = NULL;
        }
        symperms[0][ornt] = perm;
        symflips[0][ornt] = flips;
      }
      /* if no orientations produced non-identity permutations, keep NULL */
      for (ornt = -numFaces; ornt < numFaces; ornt++) if (symperms[0][ornt]) break;
      if (ornt == numFaces) {
        ierr = PetscFree(cellSymperms);CHKERRQ(ierr);
        symperms[0] = NULL;
      }
      /* if no orientations produced sign flips, keep NULL */
      for (ornt = -numFaces; ornt < numFaces; ornt++) if (symflips[0][ornt]) break;
      if (ornt == numFaces) {
        ierr = PetscFree(cellSymflips);CHKERRQ(ierr);
        symflips[0] = NULL;
      }
    }
    {
      PetscInt closureSize = 0;
      PetscInt *closure = NULL;
      PetscInt r;

      ierr = DMPlexGetTransitiveClosure(sp->dm,0,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      for (r = 0; r < closureSize; r++) {
        PetscDualSpace psp;
        PetscInt point = closure[2 * r];
        PetscInt pspintdim;
        const PetscInt ***psymperms = NULL;
        const PetscScalar ***psymflips = NULL;

        if (!point) continue;
        ierr = PetscDualSpaceGetPointSubspace(sp, point, &psp);CHKERRQ(ierr);
        if (!psp) continue;
        ierr = PetscDualSpaceGetInteriorDimension(psp, &pspintdim);CHKERRQ(ierr);
        if (!pspintdim) continue;
        ierr = PetscDualSpaceGetSymmetries(psp,&psymperms,&psymflips);CHKERRQ(ierr);
        symperms[r] = (PetscInt **) (psymperms ? psymperms[0] : NULL);
        symflips[r] = (PetscScalar **) (psymflips ? psymflips[0] : NULL);
      }
      ierr = DMPlexRestoreTransitiveClosure(sp->dm,0,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
    }
    for (p = 0; p < pEnd; p++) if (symperms[p]) break;
    if (p == pEnd) {
      ierr = PetscFree(symperms);CHKERRQ(ierr);
      symperms = NULL;
    }
    for (p = 0; p < pEnd; p++) if (symflips[p]) break;
    if (p == pEnd) {
      ierr = PetscFree(symflips);CHKERRQ(ierr);
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
  PetscValidPointer(continuous, 2);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(continuous, 2);
  ierr = PetscTryMethod(sp, "PetscDualSpaceLagrangeGetContinuity_C", (PetscDualSpace,PetscBool*),(sp,continuous));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceLagrangeSetContinuity - Indicate whether the element is continuous

  Logically Collective on sp

  Input Parameters:
+ sp         - the PetscDualSpace
- continuous - flag for element continuity

  Options Database:
. -petscdualspace_lagrange_continuity <bool>

  Level: intermediate

.seealso: PetscDualSpaceLagrangeGetContinuity()
@*/
PetscErrorCode PetscDualSpaceLagrangeSetContinuity(PetscDualSpace sp, PetscBool continuous)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidLogicalCollectiveBool(sp, continuous, 2);
  ierr = PetscTryMethod(sp, "PetscDualSpaceLagrangeSetContinuity_C", (PetscDualSpace,PetscBool),(sp,continuous));CHKERRQ(ierr);
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
  if (nodeType == PETSCDTNODES_GAUSSJACOBI && exponent <= -1.) SETERRQ(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_OUTOFRANGE, "Exponent must be > -1");
  lag->nodeType = nodeType;
  lag->endNodes = boundary;
  lag->nodeExponent = exponent;
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(tensor, 2);
  ierr = PetscTryMethod(sp,"PetscDualSpaceLagrangeGetTensor_C",(PetscDualSpace,PetscBool *),(sp,tensor));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  ierr = PetscTryMethod(sp,"PetscDualSpaceLagrangeSetTensor_C",(PetscDualSpace,PetscBool),(sp,tensor));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(trimmed, 2);
  ierr = PetscTryMethod(sp,"PetscDualSpaceLagrangeGetTrimmed_C",(PetscDualSpace,PetscBool *),(sp,trimmed));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  ierr = PetscTryMethod(sp,"PetscDualSpaceLagrangeSetTrimmed_C",(PetscDualSpace,PetscBool),(sp,trimmed));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  if (nodeType) PetscValidPointer(nodeType, 2);
  if (boundary) PetscValidPointer(boundary, 3);
  if (exponent) PetscValidPointer(exponent, 4);
  ierr = PetscTryMethod(sp,"PetscDualSpaceLagrangeGetNodeType_C",(PetscDualSpace,PetscDTNodeType *,PetscBool *,PetscReal *),(sp,nodeType,boundary,exponent));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  ierr = PetscTryMethod(sp,"PetscDualSpaceLagrangeSetNodeType_C",(PetscDualSpace,PetscDTNodeType,PetscBool,PetscReal),(sp,nodeType,boundary,exponent));CHKERRQ(ierr);
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
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp,&lag);CHKERRQ(ierr);
  sp->data = lag;

  lag->tensorCell  = PETSC_FALSE;
  lag->tensorSpace = PETSC_FALSE;
  lag->continuous  = PETSC_TRUE;
  lag->numCopies   = PETSC_DEFAULT;
  lag->numNodeSkip = PETSC_DEFAULT;
  lag->nodeType    = PETSCDTNODES_DEFAULT;

  ierr = PetscDualSpaceInitialize_Lagrange(sp);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetContinuity_C", PetscDualSpaceLagrangeGetContinuity_Lagrange);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetContinuity_C", PetscDualSpaceLagrangeSetContinuity_Lagrange);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetTensor_C", PetscDualSpaceLagrangeGetTensor_Lagrange);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetTensor_C", PetscDualSpaceLagrangeSetTensor_Lagrange);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetTrimmed_C", PetscDualSpaceLagrangeGetTrimmed_Lagrange);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetTrimmed_C", PetscDualSpaceLagrangeSetTrimmed_Lagrange);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetNodeType_C", PetscDualSpaceLagrangeGetNodeType_Lagrange);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetNodeType_C", PetscDualSpaceLagrangeSetNodeType_Lagrange);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

