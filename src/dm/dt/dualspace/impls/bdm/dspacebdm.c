#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/
#include <petscdmplex.h>

/*
Let's work out BDM_1:

The model basis is
  \phi(x, y) = / a + b x + c y \
               \ d + e x + f y /
which is a 6 dimensional space. There are also six dual basis functions,
  \psi_0(v) = \int^1_{-1} dx v(x, -1) \cdot <0, -1> (1-x)/2
  \psi_1(v) = \int^1_{-1} dx v(x, -1) \cdot <0, -1> (1+x)/2
  \psi_2(v) = 1/2 \int^1_{-1} ds v(-s, s) \cdot <1, 1> (1-s)/2 TODO I think the 1/2 is wrong here
  \psi_3(v) = 1/2 \int^1_{-1} ds v(-s, s) \cdot <1, 1> (1+s)/2
  \psi_4(v) = -\int^1_{-1} dy v(-1, y) \cdot <-1, 0> (1+y)/2
  \psi_5(v) = -\int^1_{-1} dy v(-1, y) \cdot <-1, 0> (1-y)/2
So we do the integrals
  \psi_0(\phi) = \int^1_{-1} dx (f - d - e x) (1-x)/2 = (f - d) + e/3
  \psi_1(\phi) = \int^1_{-1} dx (f - d - e x) (1+x)/2 = (f - d) - e/3
  \psi_2(\phi) = \int^1_{-1} ds (a - b s + c s + d - e s + f s) (1-s)/2 = (a + d)/2 - (c + f - b - e)/6
  \psi_3(\phi) = \int^1_{-1} ds (a - b s + c s + d - e s + f s) (1+s)/2 = (a + d)/2 + (c + f - b - e)/6
  \psi_4(\phi) = \int^1_{-1} dy (b - a - c y) (1+y)/2 = (a - b) + c/3
  \psi_5(\phi) = \int^1_{-1} dy (b - a - c y) (1-y)/2 = (a - b) - c/3
so the nodal basis is
  \phi_0 = / -(1+x)/2        \
           \ 1/2 + 3/2 x + y /
  \phi_1 = / 1+x                \
           \ -1 - 3/2 x - 1/2 y /
  \phi_2 = / 1+x      \
           \ -(1+y)/2 /
  \phi_3 = / -(1+x)/2 \
           \ (1+y)    /
  \phi_4 = / -1 - 1/2 x - 3/2 y \
           \ (1+y)             /
  \phi_5 = / 1/2 + x + 3/2 y \
           \ -(1+y)/2        /
*/

static PetscErrorCode PetscDualSpaceDestroy_BDM(PetscDualSpace sp)
{
  PetscDualSpace_BDM *bdm = (PetscDualSpace_BDM *) sp->data;
  PetscInt            i;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (bdm->symmetries) {
    PetscInt    **selfSyms = bdm->symmetries[0];
    PetscScalar **selfFlps = bdm->flips[0];

    if (selfSyms) {
      PetscInt    **fSyms = &selfSyms[-bdm->selfSymOff], i;
      PetscScalar **fFlps = &selfFlps[-bdm->selfSymOff];

      for (i = 0; i < bdm->numSelfSym; i++) {
        ierr = PetscFree(fSyms[i]);CHKERRQ(ierr);
        ierr = PetscFree(fFlps[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(fSyms);CHKERRQ(ierr);
      ierr = PetscFree(fFlps);CHKERRQ(ierr);
    }
    ierr = PetscFree(bdm->symmetries);CHKERRQ(ierr);
    ierr = PetscFree(bdm->flips);CHKERRQ(ierr);
  }
  for (i = 0; i < bdm->height; i++) {
    ierr = PetscDualSpaceDestroy(&bdm->subspaces[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(bdm->subspaces);CHKERRQ(ierr);
  ierr = PetscFree(bdm->numDof);CHKERRQ(ierr);
  ierr = PetscFree(bdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceBDMView_Ascii(PetscDualSpace sp, PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer, "BDM(%D) dual space\n", sp->order);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceView_BDM(PetscDualSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscDualSpaceBDMView_Ascii(sp, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceSetFromOptions_BDM(PetscOptionItems *PetscOptionsObject,PetscDualSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscDualSpace BDM Options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceDuplicate_BDM(PetscDualSpace sp, PetscDualSpace *spNew)
{
  PetscInt       order, Nc;
  const char    *name;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceCreate(PetscObjectComm((PetscObject) sp), spNew);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) sp,     &name);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *spNew,  name);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(*spNew, PETSCDUALSPACEBDM);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetOrder(sp, &order);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(*spNew, order);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetNumComponents(*spNew, Nc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceGetDimension_SingleCell_BDM(PetscDualSpace sp, PetscInt order, PetscInt *dim)
{
  PetscDualSpace_BDM *bdm = (PetscDualSpace_BDM *) sp->data;
  PetscReal           D   = 1.0;
  PetscInt            n, d;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  *dim = -1;
  ierr = DMGetDimension(sp->dm, &n);CHKERRQ(ierr);
  if (!n) {*dim = 0; PetscFunctionReturn(0);}
  if (bdm->simplexCell) {
    for (d = 1; d <= n; ++d) {
      D *= ((PetscReal) (order+d))/d;
    }
    *dim = (PetscInt) (D + 0.5);
  } else {
    *dim = 1;
    for (d = 0; d < n; ++d) *dim *= (order+1);
  }
  if (!bdm->faceSpace) {
    *dim *= sp->Nc;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceCreateHeightSubspace_BDM(PetscDualSpace sp, PetscInt height, PetscDualSpace *bdsp)
{
  PetscDualSpace_BDM *bdm = (PetscDualSpace_BDM *) sp->data;
  PetscInt            order;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetOrder(sp, &order);CHKERRQ(ierr);
  if (!height) {
    ierr = PetscObjectReference((PetscObject) sp);CHKERRQ(ierr);
    *bdsp = sp;
  } else if (!order) {
    *bdsp = NULL;
  } else if (height == 1) {
    DM       dm, K;
    PetscInt dim;

    ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    if (height > dim || height < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Asked for dual space at height %d for dimension %d reference element\n", height, dim);
    ierr = PetscDualSpaceDuplicate(sp, bdsp);CHKERRQ(ierr);
    ierr = PetscDualSpaceCreateReferenceCell(*bdsp, dim-height, bdm->simplexCell, &K);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetDM(*bdsp, K);CHKERRQ(ierr);
    ierr = DMDestroy(&K);CHKERRQ(ierr);
    ((PetscDualSpace_BDM *) (*bdsp)->data)->faceSpace = PETSC_TRUE;
    ierr = PetscDualSpaceSetUp(*bdsp);CHKERRQ(ierr);
  } else {
    *bdsp = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceBDMCreateFaceFE(PetscDualSpace sp, PetscBool tensor, PetscInt faceDim, PetscInt order, PetscFE *fe)
{
  DM              K;
  PetscSpace      P;
  PetscDualSpace  Q;
  PetscQuadrature q;
  const PetscInt  Nc = 1;
  PetscInt        quadPointsPerEdge;
  PetscErrorCode  ierr;

  /* Create space */
  ierr = PetscSpaceCreate(PETSC_COMM_SELF, &P);CHKERRQ(ierr);
  ierr = PetscSpaceSetType(P, PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialSetTensor(P, tensor);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(P, Nc);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(P, faceDim);CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(P, order, order);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(P);CHKERRQ(ierr);
  /* Create dual space */
  ierr = PetscDualSpaceCreate(PETSC_COMM_SELF, &Q);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(Q, PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
  ierr = PetscDualSpaceCreateReferenceCell(Q, faceDim, tensor ? PETSC_FALSE : PETSC_TRUE, &K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(Q, K);CHKERRQ(ierr);
  ierr = DMDestroy(&K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetNumComponents(Q, Nc);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(Q, order);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetTensor(Q, tensor);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetUp(Q);CHKERRQ(ierr);
  /* Create element */
  ierr = PetscFECreate(PETSC_COMM_SELF, fe);CHKERRQ(ierr);
  ierr = PetscFESetType(*fe, PETSCFEBASIC);CHKERRQ(ierr);
  ierr = PetscFESetBasisSpace(*fe, P);CHKERRQ(ierr);
  ierr = PetscFESetDualSpace(*fe, Q);CHKERRQ(ierr);
  ierr = PetscFESetNumComponents(*fe, Nc);CHKERRQ(ierr);
  ierr = PetscFESetUp(*fe);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&P);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&Q);CHKERRQ(ierr);
  /* Create quadrature (with specified order if given) */
  quadPointsPerEdge = PetscMax(order + 1, 1);
  if (tensor) {ierr = PetscDTGaussTensorQuadrature(faceDim, 1, quadPointsPerEdge, -1.0, 1.0, &q);CHKERRQ(ierr);}
  else        {ierr = PetscDTStroudConicalQuadrature(faceDim, 1, quadPointsPerEdge, -1.0, 1.0, &q);CHKERRQ(ierr);}
  ierr = PetscFESetQuadrature(*fe, q);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceBDMCreateCellFE(PetscDualSpace sp, PetscBool tensor, PetscInt dim, PetscInt Nc, PetscInt order, PetscFE *fe)
{
  DM              K;
  PetscSpace      P;
  PetscDualSpace  Q;
  PetscQuadrature q;
  PetscInt        quadPointsPerEdge;
  PetscErrorCode  ierr;

  /* Create space */
  ierr = PetscSpaceCreate(PETSC_COMM_SELF, &P);CHKERRQ(ierr);
  ierr = PetscSpaceSetType(P, PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
  ierr = PetscSpacePolynomialSetTensor(P, tensor);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(P, Nc);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(P, dim);CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(P, order, order);CHKERRQ(ierr);
  ierr = PetscSpaceSetUp(P);CHKERRQ(ierr);
  /* Create dual space */
  /* TODO Needs NED1 dual space */
  ierr = PetscDualSpaceCreate(PETSC_COMM_SELF, &Q);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(Q, PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
  ierr = PetscDualSpaceCreateReferenceCell(Q, dim, tensor ? PETSC_FALSE : PETSC_TRUE, &K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetDM(Q, K);CHKERRQ(ierr);
  ierr = DMDestroy(&K);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetNumComponents(Q, Nc);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetOrder(Q, order);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetTensor(Q, tensor);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetUp(Q);CHKERRQ(ierr);
  /* Create element */
  ierr = PetscFECreate(PETSC_COMM_SELF, fe);CHKERRQ(ierr);
  ierr = PetscFESetBasisSpace(*fe, P);CHKERRQ(ierr);
  ierr = PetscFESetDualSpace(*fe, Q);CHKERRQ(ierr);
  ierr = PetscFESetNumComponents(*fe, Nc);CHKERRQ(ierr);
  ierr = PetscFESetUp(*fe);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&P);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&Q);CHKERRQ(ierr);
  /* Create quadrature (with specified order if given) */
  quadPointsPerEdge = PetscMax(order + 1, 1);
  if (tensor) {ierr = PetscDTGaussTensorQuadrature(dim, 1, quadPointsPerEdge, -1.0, 1.0, &q);CHKERRQ(ierr);}
  else        {ierr = PetscDTStroudConicalQuadrature(dim, 1, quadPointsPerEdge, -1.0, 1.0, &q);CHKERRQ(ierr);}
  ierr = PetscFESetQuadrature(*fe, q);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&q);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceSetUp_BDM(PetscDualSpace sp)
{
  PetscDualSpace_BDM *bdm     = (PetscDualSpace_BDM *) sp->data;
  DM                  dm      = sp->dm;
  PetscInt            order   = sp->order;
  PetscInt            Nc      = sp->Nc;
  PetscBool           faceSp  = bdm->faceSpace;
  MPI_Comm            comm;
  PetscSection        csection;
  Vec                 coordinates;
  PetscInt            depth, dim, cdim, pdimMax, pStart, pEnd, p, *pStratStart, *pStratEnd, coneSize, d, f = 0, c;
  PetscBool           simplex;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) sp, &comm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  if (depth != dim) SETERRQ2(comm, PETSC_ERR_ARG_WRONG, "BDM element requires interpolated meshes, but depth %D != topological dimension %D", depth, dim);
  if (!order)       SETERRQ(comm, PETSC_ERR_ARG_WRONG, "BDM elements not defined for order 0");
  if (!faceSp && Nc != cdim) SETERRQ2(comm, PETSC_ERR_ARG_WRONG, "BDM element has %D components != %D space dimension", Nc, cdim);
  ierr = PetscCalloc1(dim+1, &bdm->numDof);CHKERRQ(ierr);
  ierr = PetscMalloc2(depth+1, &pStratStart, depth+1, &pStratEnd);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {ierr = DMPlexGetDepthStratum(dm, d, &pStratStart[d], &pStratEnd[d]);CHKERRQ(ierr);}
  ierr = DMPlexGetConeSize(dm, pStratStart[depth], &coneSize);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &csection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  if      (coneSize == dim+1)   simplex = PETSC_TRUE;
  else if (coneSize == 2 * dim) simplex = PETSC_FALSE;
  else SETERRQ(comm, PETSC_ERR_SUP, "Only support simplices and tensor product cells");
  bdm->simplexCell = simplex;
  bdm->height      = 0;
  bdm->subspaces   = NULL;
  /* Create the height 1 subspace for every dimension */
  if (order > 0 && dim > 0) {
    PetscInt i;

    bdm->height = dim;
    ierr = PetscMalloc1(dim, &bdm->subspaces);CHKERRQ(ierr);
    ierr = PetscDualSpaceCreateHeightSubspace_BDM(sp, 1, &bdm->subspaces[0]);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetUp(bdm->subspaces[0]);CHKERRQ(ierr);
    for (i = 1; i < dim; i++) {
      ierr = PetscDualSpaceGetHeightSubspace(bdm->subspaces[i-1], 1, &bdm->subspaces[i]);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject) bdm->subspaces[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscDualSpaceGetDimension_SingleCell_BDM(sp, sp->order, &pdimMax);CHKERRQ(ierr);
  pdimMax *= (pStratEnd[depth] - pStratStart[depth]);
  ierr = PetscMalloc1(pdimMax, &sp->functional);CHKERRQ(ierr);
  if (!dim) {
    bdm->numDof[0] = 0;
  } else {
    PetscFE      faceFE, cellFE;
    PetscSection section;
    CellRefiner  cellRefiner;
    PetscInt     faceDim = PetscMax(dim-1, 1), faceNum = 0;
    PetscReal   *v0 = NULL, *J = NULL, *detJ = NULL;

    ierr = PetscSectionCreate(PETSC_COMM_SELF, &section);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(section, pStart, pEnd);CHKERRQ(ierr);
    if (!faceSp) {
      ierr = DMPlexGetCellRefiner_Internal(dm, &cellRefiner);CHKERRQ(ierr);
      ierr = CellRefinerGetAffineFaceTransforms_Internal(cellRefiner, NULL, &v0, &J, NULL, &detJ);CHKERRQ(ierr);
    }
    /* Create P_q(f) */
    ierr = PetscDualSpaceBDMCreateFaceFE(sp, simplex ? PETSC_FALSE : PETSC_TRUE, faceDim, order, &faceFE);CHKERRQ(ierr);
    /* Create NED^1_{q-1}(T) = P^d_{q-2} + S_{q-1}(T) */
    ierr = PetscDualSpaceBDMCreateCellFE(sp, simplex ? PETSC_FALSE : PETSC_TRUE, faceDim, Nc, order-1, &cellFE);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; p++) {
      PetscBool isFace, isCell;
      PetscInt  d;

      for (d = 0; d < depth; d++) {if (p >= pStratStart[d] && p < pStratEnd[d]) break;}
      isFace = ((d == dim) &&  faceSp) || ((d == dim-1) && !faceSp) ? PETSC_TRUE : PETSC_FALSE;
      isCell = ((d == dim) && !faceSp) ? PETSC_TRUE : PETSC_FALSE;
      if (isFace) {
        PetscQuadrature  fq;
        PetscTabulation  T;
        PetscReal       *B, n[3];
        const PetscReal *fqpoints, *fqweights;
        PetscInt         faceDim = PetscMax(dim-1, 1), Nq, q, fdim, fb;

        if (cdim == 1) {n[0] = 0.; n[1] = 1.;}
        else           {ierr = DMPlexComputeCellGeometryFVM(dm, p, NULL, NULL, n);CHKERRQ(ierr);}
        ierr = PetscFEGetCellTabulation(faceFE, &T);CHKERRQ(ierr);
        B = T->T[0];
        ierr = PetscFEGetQuadrature(faceFE, &fq);CHKERRQ(ierr);
        ierr = PetscQuadratureGetData(fq, NULL, NULL, &Nq, &fqpoints, &fqweights);CHKERRQ(ierr);
        /* Create a dual basis vector for each basis function */
        ierr = PetscFEGetDimension(faceFE, &fdim);CHKERRQ(ierr);
        for (fb = 0; fb < fdim; ++fb, ++f) {
          PetscReal *qpoints, *qweights;

          ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
          ierr = PetscMalloc1(Nq*dim, &qpoints);CHKERRQ(ierr);
          ierr = PetscCalloc1(Nq*Nc,  &qweights);CHKERRQ(ierr);
          ierr = PetscQuadratureSetOrder(sp->functional[f], order);CHKERRQ(ierr);
          ierr = PetscQuadratureSetData(sp->functional[f], dim, Nc, Nq, qpoints, qweights);CHKERRQ(ierr);
          for (q = 0; q < Nq; ++q) {
            PetscInt g, h;

            if (faceDim < dim) {
              /* Transform quadrature points from face coordinates to cell coordinates */
              for (g = 0; g < dim; ++g) {
                qpoints[q*dim+g] = v0[faceNum*dim+g];
                for (h = 0; h < faceDim; ++h) qpoints[q*dim+g] += J[faceNum*dim*faceDim+g*faceDim+h] * fqpoints[q*faceDim+h];
              }
            } else {
              for (g = 0; g < dim; ++g) qpoints[q*dim+g] = fqpoints[q*faceDim+g];
            }
            /* Make Radon measure for integral \hat n p ds */
            for (c = 0; c < Nc; ++c) qweights[q*Nc+c] = B[q*fdim+fb]*n[c]*fqweights[q]*(detJ ? detJ[faceNum] : 1.0);
          }
        }
        bdm->numDof[d] = fdim;
        ierr = PetscSectionSetDof(section, p, bdm->numDof[d]);CHKERRQ(ierr);
        ++faceNum;
      }
      if (order < 2) continue;
      if (isCell) {
        PetscSpace       csp;
        PetscQuadrature  cq;
        PetscReal       *B;
        const PetscReal *cqpoints, *cqweights;
        PetscInt         Nq, q, cdim, cb;

        ierr = PetscFEGetBasisSpace(cellFE, &csp);CHKERRQ(ierr);
        ierr = PetscFEGetQuadrature(cellFE, &cq);CHKERRQ(ierr);
        ierr = PetscQuadratureGetData(cq, NULL, NULL, &Nq, &cqpoints, &cqweights);CHKERRQ(ierr);
        /* Create a dual basis vector for each basis function */
        ierr = PetscSpaceGetDimension(csp, &cdim);CHKERRQ(ierr);
        ierr = PetscMalloc1(Nq*cdim*Nc, &B);CHKERRQ(ierr);
        ierr = PetscSpaceEvaluate(csp, Nq, cqpoints, B, NULL, NULL);CHKERRQ(ierr);
        for (cb = 0; cb < cdim; ++cb, ++f) {
          PetscReal *qpoints, *qweights;

          ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
          ierr = PetscMalloc1(Nq*dim, &qpoints);CHKERRQ(ierr);
          ierr = PetscCalloc1(Nq*Nc,  &qweights);CHKERRQ(ierr);
          ierr = PetscQuadratureSetOrder(sp->functional[f], order);CHKERRQ(ierr);
          ierr = PetscQuadratureSetData(sp->functional[f], dim, Nc, Nq, qpoints, qweights);CHKERRQ(ierr);
          ierr = PetscArraycpy(qpoints, cqpoints, Nq*dim);CHKERRQ(ierr);
          for (q = 0; q < Nq; ++q) {
            /* Make Radon measure for integral p dx */
            for (c = 0; c < Nc; ++c) qweights[q*Nc+c] = B[(q*cdim+cb)*Nc+c]*cqweights[q*Nc+c];
          }
        }
        ierr = PetscFree(B);CHKERRQ(ierr);
        bdm->numDof[d] = cdim;
        ierr = PetscSectionSetDof(section, p, bdm->numDof[d]);CHKERRQ(ierr);
      }
    }
    ierr = PetscFEDestroy(&faceFE);CHKERRQ(ierr);
    ierr = PetscFEDestroy(&cellFE);CHKERRQ(ierr);
    ierr = PetscFree(v0);CHKERRQ(ierr);
    ierr = PetscFree(J);CHKERRQ(ierr);
    ierr = PetscFree(detJ);CHKERRQ(ierr);
    ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
    { /* reorder to closure order */
      PetscQuadrature *reorder = NULL;
      PetscInt        *key, count;

      ierr = PetscCalloc1(f, &key);CHKERRQ(ierr);
      ierr = PetscMalloc1(f, &reorder);CHKERRQ(ierr);
      for (p = pStratStart[depth], count = 0; p < pStratEnd[depth]; p++) {
        PetscInt *closure = NULL, closureSize, c;

        ierr = DMPlexGetTransitiveClosure(dm, p, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
        for (c = 0; c < closureSize*2; c += 2) {
          PetscInt point = closure[c], dof, off, i;

          ierr = PetscSectionGetDof(section, point, &dof);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(section, point, &off);CHKERRQ(ierr);
          for (i = 0; i < dof; ++i) {
            PetscInt fi = off + i;

            if (!key[fi]) {
              key[fi] = 1;
              reorder[count++] = sp->functional[fi];
            }
          }
        }
        ierr = DMPlexRestoreTransitiveClosure(dm, p, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      }
      ierr = PetscFree(key);CHKERRQ(ierr);
      ierr = PetscFree(sp->functional);CHKERRQ(ierr);
      sp->functional = reorder;
    }
    ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  }
  if (pStratEnd[depth] == 1 && f != pdimMax) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of dual basis vectors %D not equal to dimension %D", f, pdimMax);
  if (f > pdimMax) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of dual basis vectors %D is greater than max size %D", f, pdimMax);
  ierr = PetscFree2(pStratStart, pStratEnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceGetDimension_BDM(PetscDualSpace sp, PetscInt *dim)
{
  DM              K;
  const PetscInt *numDof;
  PetscInt        spatialDim, cEnd, size = 0, d;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDM(sp, &K);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetNumDof(sp, &numDof);CHKERRQ(ierr);
  ierr = DMGetDimension(K, &spatialDim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(K, 0, NULL, &cEnd);CHKERRQ(ierr);
  if (cEnd == 1) {ierr = PetscDualSpaceGetDimension_SingleCell_BDM(sp, sp->order, dim);CHKERRQ(ierr); PetscFunctionReturn(0);}
  for (d = 0; d <= spatialDim; ++d) {
    PetscInt pStart, pEnd;

    ierr = DMPlexGetDepthStratum(K, d, &pStart, &pEnd);CHKERRQ(ierr);
    size += (pEnd-pStart)*numDof[d];
  }
  *dim = size;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceGetNumDof_BDM(PetscDualSpace sp, const PetscInt **numDof)
{
  PetscDualSpace_BDM *bdm = (PetscDualSpace_BDM *) sp->data;

  PetscFunctionBegin;
  *numDof = bdm->numDof;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceGetHeightSubspace_BDM(PetscDualSpace sp, PetscInt height, PetscDualSpace *bdsp)
{
  PetscDualSpace_BDM *bdm = (PetscDualSpace_BDM *) sp->data;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidPointer(bdsp, 2);
  if (height == 0) {
    *bdsp = sp;
  } else {
    DM       dm;
    PetscInt dim;

    ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    if (height > dim || height < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Asked for dual space at height %D for dimension %D reference element\n", height, dim);
    if (height <= bdm->height) {*bdsp = bdm->subspaces[height-1];}
    else                       {*bdsp = NULL;}
  }
  PetscFunctionReturn(0);
}

#define BaryIndex(perEdge,a,b,c) (((b)*(2*perEdge+1-(b)))/2)+(c)

#define CartIndex(perEdge,a,b) (perEdge*(a)+b)

static PetscErrorCode PetscDualSpaceGetSymmetries_BDM(PetscDualSpace sp, const PetscInt ****perms, const PetscScalar ****rots)
{

  PetscDualSpace_BDM *bdm = (PetscDualSpace_BDM *) sp->data;
  PetscInt            dim, order, p, Nc;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetOrder(sp, &order);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  ierr = DMGetDimension(sp->dm, &dim);CHKERRQ(ierr);
  if (dim < 1) PetscFunctionReturn(0);
  if (dim > 3) SETERRQ1(PetscObjectComm((PetscObject) sp), PETSC_ERR_SUP, "BDM symmetries not implemented for dim = %D > 3", dim);
  if (!bdm->symmetries) { /* store symmetries */
    PetscInt    ***symmetries, **cellSymmetries;
    PetscScalar ***flips,      **cellFlips;
    PetscInt       numPoints, numFaces, d;

    if (bdm->simplexCell) {
      numPoints = 1;
      for (d = 0; d < dim; d++) numPoints = numPoints * 2 + 1;
      numFaces  = 1 + dim;
    } else {
      numPoints = PetscPowInt(3,dim);
      numFaces  = 2 * dim;
    }
    ierr = PetscCalloc1(numPoints, &symmetries);CHKERRQ(ierr);
    ierr = PetscCalloc1(numPoints, &flips);CHKERRQ(ierr);
    /* compute self symmetries */
    bdm->numSelfSym = 2 * numFaces;
    bdm->selfSymOff = numFaces;
    ierr = PetscCalloc1(2*numFaces, &cellSymmetries);CHKERRQ(ierr);
    ierr = PetscCalloc1(2*numFaces, &cellFlips);CHKERRQ(ierr);
    /* we want to be able to index symmetries directly with the orientations, which range from [-numFaces,numFaces) */
    symmetries[0] = &cellSymmetries[bdm->selfSymOff];
    flips[0]      = &cellFlips[bdm->selfSymOff];
    switch (dim) {
    case 1: /* Edge symmetries */
    {
      PetscScalar *invert;
      PetscInt    *reverse;
      PetscInt     dofPerEdge = order+1, eNc = 1 /* ??? */, i, j;

      ierr = PetscMalloc1(dofPerEdge*eNc, &reverse);CHKERRQ(ierr);
      ierr = PetscMalloc1(dofPerEdge*eNc, &invert);CHKERRQ(ierr);
      for (i = 0; i < dofPerEdge; ++i) {
        for (j = 0; j < eNc; ++j) {
          reverse[i*eNc + j] = eNc * (dofPerEdge - 1 - i) + j;
          invert[i*eNc + j]  = -1.0;
        }
      }
      symmetries[0][-2] = reverse;
      flips[0][-2]      = invert;

      /* yes, this is redundant, but it makes it easier to cleanup if I don't have to worry about what not to free */
      ierr = PetscMalloc1(dofPerEdge*eNc, &reverse);CHKERRQ(ierr);
      ierr = PetscMalloc1(dofPerEdge*eNc, &invert);CHKERRQ(ierr);
      for (i = 0; i < dofPerEdge; i++) {
        for (j = 0; j < eNc; j++) {
          reverse[i*eNc + j] = eNc * (dofPerEdge - 1 - i) + j;
          invert[i*eNc + j]  = -1.0;
        }
      }
      symmetries[0][1] = reverse;
      flips[0][1]      = invert;
      break;
    }
    case 2: /* Face symmetries  */
    {
      PetscInt dofPerEdge = bdm->simplexCell ? (order - 2) : (order - 1), s;
      PetscInt dofPerFace;

      for (s = -numFaces; s < numFaces; s++) {
        PetscScalar *flp;
        PetscInt    *sym, fNc = 1, i, j, k, l;

        if (!s) continue;
        if (bdm->simplexCell) {
          dofPerFace = (dofPerEdge * (dofPerEdge + 1))/2;
          ierr = PetscMalloc1(fNc*dofPerFace,&sym);CHKERRQ(ierr);
          ierr = PetscMalloc1(fNc*dofPerFace,&flp);CHKERRQ(ierr);
          for (j = 0, l = 0; j < dofPerEdge; j++) {
            for (k = 0; k < dofPerEdge - j; k++, l++) {
              i = dofPerEdge - 1 - j - k;
              switch (s) {
              case -3:
                sym[fNc*l] = BaryIndex(dofPerEdge,i,k,j);
                break;
              case -2:
                sym[fNc*l] = BaryIndex(dofPerEdge,j,i,k);
                break;
              case -1:
                sym[fNc*l] = BaryIndex(dofPerEdge,k,j,i);
                break;
              case 1:
                sym[fNc*l] = BaryIndex(dofPerEdge,k,i,j);
                break;
              case 2:
                sym[fNc*l] = BaryIndex(dofPerEdge,j,k,i);
                break;
              }
              flp[fNc*l] = s < 0 ? -1.0 : 1.0;
            }
          }
        } else {
          dofPerFace = dofPerEdge * dofPerEdge;
          ierr = PetscMalloc1(fNc*dofPerFace,&sym);CHKERRQ(ierr);
          ierr = PetscMalloc1(fNc*dofPerFace,&flp);CHKERRQ(ierr);
          for (j = 0, l = 0; j < dofPerEdge; j++) {
            for (k = 0; k < dofPerEdge; k++, l++) {
              switch (s) {
              case -4:
                sym[fNc*l] = CartIndex(dofPerEdge,k,j);
                break;
              case -3:
                sym[fNc*l] = CartIndex(dofPerEdge,(dofPerEdge - 1 - j),k);
                break;
              case -2:
                sym[fNc*l] = CartIndex(dofPerEdge,(dofPerEdge - 1 - k),(dofPerEdge - 1 - j));
                break;
              case -1:
                sym[fNc*l] = CartIndex(dofPerEdge,j,(dofPerEdge - 1 - k));
                break;
              case 1:
                sym[fNc*l] = CartIndex(dofPerEdge,(dofPerEdge - 1 - k),j);
                break;
              case 2:
                sym[fNc*l] = CartIndex(dofPerEdge,(dofPerEdge - 1 - j),(dofPerEdge - 1 - k));
                break;
              case 3:
                sym[fNc*l] = CartIndex(dofPerEdge,k,(dofPerEdge - 1 - j));
                break;
              }
              flp[fNc*l] = s < 0 ? -1.0 : 1.0;
            }
          }
        }
        for (i = 0; i < dofPerFace; i++) {
          sym[fNc*i] *= fNc;
          for (j = 1; j < fNc; j++) {
            sym[fNc*i+j] = sym[fNc*i] + j;
            flp[fNc*i+j] = flp[fNc*i];
          }
        }
        symmetries[0][s] = sym;
        flips[0][s]      = flp;
      }
      break;
    }
    default: SETERRQ1(PetscObjectComm((PetscObject) sp), PETSC_ERR_SUP, "No symmetries for point of dimension %D", dim);
    }
    /* Copy subspace symmetries */
    {
      PetscDualSpace       hsp;
      DM                   K;
      const PetscInt    ***hsymmetries;
      const PetscScalar ***hflips;

      ierr = PetscDualSpaceGetHeightSubspace(sp, 1, &hsp);CHKERRQ(ierr);
      ierr = PetscDualSpaceGetSymmetries(hsp, &hsymmetries, &hflips);CHKERRQ(ierr);
      if (hsymmetries || hflips) {
        PetscBool      *seen;
        const PetscInt *cone;
        PetscInt        KclosureSize, *Kclosure = NULL;

        ierr = PetscDualSpaceGetDM(sp, &K);CHKERRQ(ierr);
        ierr = PetscCalloc1(numPoints, &seen);CHKERRQ(ierr);
        ierr = DMPlexGetCone(K, 0, &cone);CHKERRQ(ierr);
        ierr = DMPlexGetTransitiveClosure(K, 0, PETSC_TRUE, &KclosureSize, &Kclosure);CHKERRQ(ierr);
        for (p = 0; p < numFaces; ++p) {
          PetscInt closureSize, *closure = NULL, q;

          ierr = DMPlexGetTransitiveClosure(K, cone[p], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
          for (q = 0; q < closureSize; ++q) {
            PetscInt point = closure[q*2], r;

            if (!seen[point]) {
              for (r = 0; r < KclosureSize; ++r) {
                if (Kclosure[r*2] == point) break;
              }
              seen[point] = PETSC_TRUE;
              symmetries[r] = (PetscInt **)    hsymmetries[q];
              flips[r]      = (PetscScalar **) hflips[q];
            }
          }
          ierr = DMPlexRestoreTransitiveClosure(K, cone[p], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
        }
        ierr = DMPlexRestoreTransitiveClosure(K, 0, PETSC_TRUE, &KclosureSize, &Kclosure);CHKERRQ(ierr);
        ierr = PetscFree(seen);CHKERRQ(ierr);
      }
    }
    bdm->symmetries = symmetries;
    bdm->flips      = flips;
  }
  if (perms) *perms = (const PetscInt ***)    bdm->symmetries;
  if (rots)  *rots  = (const PetscScalar ***) bdm->flips;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceInitialize_BDM(PetscDualSpace sp)
{
  PetscFunctionBegin;
  sp->ops->destroy           = PetscDualSpaceDestroy_BDM;
  sp->ops->view              = PetscDualSpaceView_BDM;
  sp->ops->setfromoptions    = PetscDualSpaceSetFromOptions_BDM;
  sp->ops->duplicate         = PetscDualSpaceDuplicate_BDM;
  sp->ops->setup             = PetscDualSpaceSetUp_BDM;
  sp->ops->getdimension      = PetscDualSpaceGetDimension_BDM;
  sp->ops->getnumdof         = PetscDualSpaceGetNumDof_BDM;
  sp->ops->getheightsubspace = PetscDualSpaceGetHeightSubspace_BDM;
  sp->ops->getsymmetries     = PetscDualSpaceGetSymmetries_BDM;
  sp->ops->apply             = PetscDualSpaceApplyDefault;
  sp->ops->applyall          = PetscDualSpaceApplyAllDefault;
  sp->ops->createallpoints   = PetscDualSpaceCreateAllPointsDefault;
  PetscFunctionReturn(0);
}
/*MC
  PETSCDUALSPACEBDM = "bdm" - A PetscDualSpace object that encapsulates a dual space for Brezzi-Douglas-Marini elements

  Level: intermediate

.seealso: PetscDualSpaceType, PetscDualSpaceCreate(), PetscDualSpaceSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscDualSpaceCreate_BDM(PetscDualSpace sp)
{
  PetscDualSpace_BDM *bdm;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp, &bdm);CHKERRQ(ierr);
  sp->data = bdm;
  sp->k    = 3;

  bdm->numDof      = NULL;
  bdm->simplexCell = PETSC_TRUE;

  ierr = PetscDualSpaceInitialize_BDM(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
