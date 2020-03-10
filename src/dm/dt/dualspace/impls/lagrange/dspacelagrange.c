#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/
#include <petscdmplex.h>

static PetscErrorCode PetscDualSpaceDestroy_Lagrange(PetscDualSpace sp)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscInt            i;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  if (lag->symmetries) {
    PetscInt **selfSyms = lag->symmetries[0];

    if (selfSyms) {
      PetscInt i, **allocated = &selfSyms[-lag->selfSymOff];

      for (i = 0; i < lag->numSelfSym; i++) {
        ierr = PetscFree(allocated[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree(allocated);CHKERRQ(ierr);
    }
    ierr = PetscFree(lag->symmetries);CHKERRQ(ierr);
  }
  for (i = 0; i < lag->height; i++) {
    ierr = PetscDualSpaceDestroy(&lag->subspaces[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(lag->subspaces);CHKERRQ(ierr);
  ierr = PetscFree(lag);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetContinuity_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetContinuity_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetTensor_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetTensor_C", NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceLagrangeView_Ascii(PetscDualSpace sp, PetscViewer viewer)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer, "%s %sLagrange dual space\n", lag->continuous ? "Continuous" : "Discontinuous", lag->tensorSpace ? "tensor " : "");CHKERRQ(ierr);
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
  PetscBool      continuous, tensor, flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceLagrangeGetContinuity(sp, &continuous);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeGetTensor(sp, &tensor);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"PetscDualSpace Lagrange Options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-petscdualspace_lagrange_continuity", "Flag for continuous element", "PetscDualSpaceLagrangeSetContinuity", continuous, &continuous, &flg);CHKERRQ(ierr);
  if (flg) {ierr = PetscDualSpaceLagrangeSetContinuity(sp, continuous);CHKERRQ(ierr);}
  ierr = PetscOptionsBool("-petscdualspace_lagrange_tensor", "Flag for tensor dual space", "PetscDualSpaceLagrangeSetContinuity", tensor, &tensor, &flg);CHKERRQ(ierr);
  if (flg) {ierr = PetscDualSpaceLagrangeSetTensor(sp, tensor);CHKERRQ(ierr);}
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceDuplicate_Lagrange(PetscDualSpace sp, PetscDualSpace spNew)
{
  PetscBool      cont, tensor;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceLagrangeGetContinuity(sp, &cont);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetContinuity(spNew, cont);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeGetTensor(sp, &tensor);CHKERRQ(ierr);
  ierr = PetscDualSpaceLagrangeSetTensor(spNew, tensor);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceGetDimension_SingleCell_Lagrange(PetscDualSpace sp, PetscInt order, PetscInt *dim)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscReal           D   = 1.0;
  PetscInt            n, d;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  *dim = -1;
  ierr = DMGetDimension(sp->dm, &n);CHKERRQ(ierr);
  if (!lag->tensorSpace) {
    for (d = 1; d <= n; ++d) {
      D *= ((PetscReal) (order+d))/d;
    }
    *dim = (PetscInt) (D + 0.5);
  } else {
    *dim = 1;
    for (d = 0; d < n; ++d) *dim *= (order+1);
  }
  *dim *= sp->Nc;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceCreateHeightSubspace_Lagrange(PetscDualSpace sp, PetscInt height, PetscDualSpace *bdsp)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscBool          continuous, tensor;
  PetscInt           order;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceLagrangeGetContinuity(sp,&continuous);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetOrder(sp,&order);CHKERRQ(ierr);
  if (height == 0) {
    ierr = PetscObjectReference((PetscObject)sp);CHKERRQ(ierr);
    *bdsp = sp;
  } else if (continuous == PETSC_FALSE || !order) {
    *bdsp = NULL;
  } else {
    DM dm, K;
    PetscInt dim;

    ierr = PetscDualSpaceGetDM(sp,&dm);CHKERRQ(ierr);
    ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
    if (height > dim || height < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Asked for dual space at height %d for dimension %d reference element\n",height,dim);
    ierr = PetscDualSpaceDuplicate(sp,bdsp);CHKERRQ(ierr);
    ierr = PetscDualSpaceCreateReferenceCell(*bdsp, dim-height, lag->simplexCell, &K);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetDM(*bdsp, K);CHKERRQ(ierr);
    ierr = DMDestroy(&K);CHKERRQ(ierr);
    ierr = PetscDualSpaceLagrangeGetTensor(sp,&tensor);CHKERRQ(ierr);
    ierr = PetscDualSpaceLagrangeSetTensor(*bdsp,tensor);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetUp(*bdsp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceSetUp_Lagrange(PetscDualSpace sp)
{
  PetscDualSpace_Lag *lag   = (PetscDualSpace_Lag *) sp->data;
  DM                  dm    = sp->dm;
  PetscInt            order = sp->order;
  PetscInt            Nc    = sp->Nc;
  MPI_Comm            comm;
  PetscBool           continuous;
  PetscSection        csection;
  Vec                 coordinates;
  PetscReal          *qpoints, *qweights;
  PetscInt            depth, dim, pdimMax, pStart, pEnd, p, *pStratStart, *pStratEnd, coneSize, d, f = 0, c;
  PetscBool           simplex, tensorSpace;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) sp, &comm);CHKERRQ(ierr);
  if (!order) lag->continuous = PETSC_FALSE;
  continuous = lag->continuous;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscMalloc2(depth+1,&pStratStart,depth+1,&pStratEnd);CHKERRQ(ierr);
  for (d = 0; d <= depth; ++d) {ierr = DMPlexGetDepthStratum(dm, d, &pStratStart[d], &pStratEnd[d]);CHKERRQ(ierr);}
  ierr = DMPlexGetConeSize(dm, pStratStart[depth], &coneSize);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &csection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  if (depth == 1) {
    if      (coneSize == dim+1)    simplex = PETSC_TRUE;
    else if (coneSize == 1 << dim) simplex = PETSC_FALSE;
    else SETERRQ(comm, PETSC_ERR_SUP, "Only support simplices and tensor product cells");
  } else if (depth == dim) {
    if      (coneSize == dim+1)   simplex = PETSC_TRUE;
    else if (coneSize == 2 * dim) simplex = PETSC_FALSE;
    else SETERRQ(comm, PETSC_ERR_SUP, "Only support simplices and tensor product cells");
  } else SETERRQ(comm, PETSC_ERR_SUP, "Only support cell-vertex meshes or fully interpolated meshes");
  lag->simplexCell = simplex;
  if (dim > 1 && continuous && lag->simplexCell == lag->tensorSpace) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP, "Mismatching simplex/tensor cells and spaces only allowed for discontinuous elements");
  tensorSpace    = lag->tensorSpace;
  lag->height    = 0;
  lag->subspaces = NULL;
  if (continuous && order > 0 && dim > 0) {
    PetscInt i;

    lag->height = dim;
    ierr = PetscMalloc1(dim,&lag->subspaces);CHKERRQ(ierr);
    ierr = PetscDualSpaceCreateHeightSubspace_Lagrange(sp,1,&lag->subspaces[0]);CHKERRQ(ierr);
    ierr = PetscDualSpaceSetUp(lag->subspaces[0]);CHKERRQ(ierr);
    for (i = 1; i < dim; i++) {
      ierr = PetscDualSpaceGetHeightSubspace(lag->subspaces[i-1],1,&lag->subspaces[i]);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)(lag->subspaces[i]));CHKERRQ(ierr);
    }
  }
  ierr = PetscDualSpaceGetDimension_SingleCell_Lagrange(sp, sp->order, &pdimMax);CHKERRQ(ierr);
  pdimMax *= (pStratEnd[depth] - pStratStart[depth]);
  ierr = PetscMalloc1(pdimMax, &sp->functional);CHKERRQ(ierr);
  if (!dim) {
    for (c = 0; c < Nc; ++c) {
      ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
      ierr = PetscCalloc1(Nc, &qweights);CHKERRQ(ierr);
      ierr = PetscQuadratureSetOrder(sp->functional[f], 0);CHKERRQ(ierr);
      ierr = PetscQuadratureSetData(sp->functional[f], 0, Nc, 1, NULL, qweights);CHKERRQ(ierr);
      qweights[c] = 1.0;
      ++f;
    }
  } else {
    PetscSection section;
    PetscReal    *v0, *hv0, *J, *invJ, detJ, hdetJ;
    PetscInt     *tup;

    ierr = PetscSectionCreate(PETSC_COMM_SELF,&section);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(section,pStart,pEnd);CHKERRQ(ierr);
    ierr = PetscCalloc5(dim+1,&tup,dim,&v0,dim,&hv0,dim*dim,&J,dim*dim,&invJ);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; p++) {
      PetscInt       pointDim, d, nFunc = 0;
      PetscDualSpace hsp;

      ierr = DMPlexComputeCellGeometryFEM(dm, p, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
      for (d = 0; d < depth; d++) {if (p >= pStratStart[d] && p < pStratEnd[d]) break;}
      pointDim = (depth == 1 && d == 1) ? dim : d;
      hsp = ((pointDim < dim) && lag->subspaces) ? lag->subspaces[dim - pointDim - 1] : NULL;
      if (hsp) {
        DM                 hdm;

        ierr = PetscDualSpaceGetDM(hsp,&hdm);CHKERRQ(ierr);
        ierr = DMPlexComputeCellGeometryFEM(hdm, 0, NULL, hv0, NULL, NULL, &hdetJ);CHKERRQ(ierr);
        nFunc = sp->numDof[pointDim];
      }
      if (pointDim == dim) {
        /* Cells, create for self */
        PetscInt     orderEff = continuous ? (!tensorSpace ? order-1-dim : order-2) : order;
        PetscReal    denom    = continuous ? order : (!tensorSpace ? order+1+dim : order+2);
        PetscReal    numer    = (!simplex || !tensorSpace) ? 2. : (2./dim);
        PetscReal    dx = numer/denom;
        PetscInt     cdim, d, d2;

        if (orderEff < 0) continue;
        ierr = PetscDualSpaceGetDimension_SingleCell_Lagrange(sp, orderEff, &cdim);CHKERRQ(ierr);
        ierr = PetscArrayzero(tup,dim+1);CHKERRQ(ierr);
        if (!tensorSpace) {
          while (!tup[dim]) {
            for (c = 0; c < Nc; ++c) {
              ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
              ierr = PetscMalloc1(dim, &qpoints);CHKERRQ(ierr);
              ierr = PetscCalloc1(Nc,  &qweights);CHKERRQ(ierr);
              ierr = PetscQuadratureSetOrder(sp->functional[f], 0);CHKERRQ(ierr);
              ierr = PetscQuadratureSetData(sp->functional[f], dim, Nc, 1, qpoints, qweights);CHKERRQ(ierr);
              for (d = 0; d < dim; ++d) {
                qpoints[d] = v0[d];
                for (d2 = 0; d2 < dim; ++d2) qpoints[d] += J[d*dim+d2]*((tup[d2]+1)*dx);
              }
              qweights[c] = 1.0;
              ++f;
            }
            ierr = PetscDualSpaceLatticePointLexicographic_Internal(dim, orderEff, tup);CHKERRQ(ierr);
          }
        } else {
          while (!tup[dim]) {
            for (c = 0; c < Nc; ++c) {
              ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
              ierr = PetscMalloc1(dim, &qpoints);CHKERRQ(ierr);
              ierr = PetscCalloc1(Nc,  &qweights);CHKERRQ(ierr);
              ierr = PetscQuadratureSetOrder(sp->functional[f], 0);CHKERRQ(ierr);
              ierr = PetscQuadratureSetData(sp->functional[f], dim, Nc, 1, qpoints, qweights);CHKERRQ(ierr);
              for (d = 0; d < dim; ++d) {
                qpoints[d] = v0[d];
                for (d2 = 0; d2 < dim; ++d2) qpoints[d] += J[d*dim+d2]*((tup[d2]+1)*dx);
              }
              qweights[c] = 1.0;
              ++f;
            }
            ierr = PetscDualSpaceTensorPointLexicographic_Internal(dim, orderEff, tup);CHKERRQ(ierr);
          }
        }
      } else { /* transform functionals from subspaces */
        PetscInt q;

        for (q = 0; q < nFunc; q++, f++) {
          PetscQuadrature fn;
          PetscInt        fdim, Nc, c, nPoints, i;
          const PetscReal *points;
          const PetscReal *weights;
          PetscReal       *qpoints;
          PetscReal       *qweights;

          ierr = PetscDualSpaceGetFunctional(hsp, q, &fn);CHKERRQ(ierr);
          ierr = PetscQuadratureGetData(fn,&fdim,&Nc,&nPoints,&points,&weights);CHKERRQ(ierr);
          if (fdim != pointDim) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Expected height dual space dim %D, got %D",pointDim,fdim);
          ierr = PetscMalloc1(nPoints * dim, &qpoints);CHKERRQ(ierr);
          ierr = PetscCalloc1(nPoints * Nc,  &qweights);CHKERRQ(ierr);
          for (i = 0; i < nPoints; i++) {
            PetscInt  j, k;
            PetscReal *qp = &qpoints[i * dim];

            for (c = 0; c < Nc; ++c) qweights[i*Nc+c] = weights[i*Nc+c];
            for (j = 0; j < dim; ++j) qp[j] = v0[j];
            for (j = 0; j < dim; ++j) {
              for (k = 0; k < pointDim; k++) qp[j] += J[dim * j + k] * (points[pointDim * i + k] - hv0[k]);
            }
          }
          ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &sp->functional[f]);CHKERRQ(ierr);
          ierr = PetscQuadratureSetOrder(sp->functional[f],0);CHKERRQ(ierr);
          ierr = PetscQuadratureSetData(sp->functional[f],dim,Nc,nPoints,qpoints,qweights);CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscFree5(tup,v0,hv0,J,invJ);CHKERRQ(ierr);
    ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
    { /* reorder to closure order */
      PetscInt *key, count;
      PetscQuadrature *reorder = NULL;

      ierr = PetscCalloc1(f,&key);CHKERRQ(ierr);
      ierr = PetscMalloc1(f*sp->Nc,&reorder);CHKERRQ(ierr);

      for (p = pStratStart[depth], count = 0; p < pStratEnd[depth]; p++) {
        PetscInt *closure = NULL, closureSize, c;

        ierr = DMPlexGetTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
        for (c = 0; c < closureSize; c++) {
          PetscInt point = closure[2 * c], dof, off, i;

          ierr = PetscSectionGetDof(section,point,&dof);CHKERRQ(ierr);
          ierr = PetscSectionGetOffset(section,point,&off);CHKERRQ(ierr);
          for (i = 0; i < dof; i++) {
            PetscInt fi = i + off;
            if (!key[fi]) {
              key[fi] = 1;
              reorder[count++] = sp->functional[fi];
            }
          }
        }
        ierr = DMPlexRestoreTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      }
      ierr = PetscFree(sp->functional);CHKERRQ(ierr);
      sp->functional = reorder;
      ierr = PetscFree(key);CHKERRQ(ierr);
    }
    ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  }
  if (pStratEnd[depth] == 1 && f != pdimMax) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of dual basis vectors %D not equal to dimension %D", f, pdimMax);
  if (f > pdimMax) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Number of dual basis vectors %D is greater than max size %D", f, pdimMax);
  ierr = PetscFree2(pStratStart, pStratEnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceGetHeightSubspace_Lagrange(PetscDualSpace sp, PetscInt height, PetscDualSpace *bdsp)
{
  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (height == 0) {
    *bdsp = sp;
  } else {
    DM       dm;
    PetscInt dim;

    ierr = PetscDualSpaceGetDM(sp,&dm);CHKERRQ(ierr);
    ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
    if (height > dim || height < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Asked for dual space at height %D for dimension %D reference element\n",height,dim);
    if (height <= lag->height) {*bdsp = lag->subspaces[height-1];}
    else                       {*bdsp = NULL;}
  }
  PetscFunctionReturn(0);
}

#define BaryIndex(perEdge,a,b,c) (((b)*(2*perEdge+1-(b)))/2)+(c)

#define CartIndex(perEdge,a,b) (perEdge*(a)+b)

static PetscErrorCode PetscDualSpaceGetSymmetries_Lagrange(PetscDualSpace sp, const PetscInt ****perms, const PetscScalar ****flips)
{

  PetscDualSpace_Lag *lag = (PetscDualSpace_Lag *) sp->data;
  PetscInt           dim, order, p, Nc;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetOrder(sp,&order);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetNumComponents(sp,&Nc);CHKERRQ(ierr);
  ierr = DMGetDimension(sp->dm,&dim);CHKERRQ(ierr);
  if (!dim || !lag->continuous || order < 3) PetscFunctionReturn(0);
  if (dim > 3) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Lagrange symmetries not implemented for dim = %D > 3",dim);
  if (!lag->symmetries) { /* store symmetries */
    PetscDualSpace hsp;
    DM             K;
    PetscInt       numPoints = 1, d;
    PetscInt       numFaces;
    PetscInt       ***symmetries;
    const PetscInt ***hsymmetries;

    if (lag->simplexCell) {
      numFaces = 1 + dim;
      for (d = 0; d < dim; d++) numPoints = numPoints * 2 + 1;
    } else {
      numPoints = PetscPowInt(3,dim);
      numFaces  = 2 * dim;
    }
    ierr = PetscCalloc1(numPoints,&symmetries);CHKERRQ(ierr);
    if (0 < dim && dim < 3) { /* compute self symmetries */
      PetscInt **cellSymmetries;

      lag->numSelfSym = 2 * numFaces;
      lag->selfSymOff = numFaces;
      ierr = PetscCalloc1(2*numFaces,&cellSymmetries);CHKERRQ(ierr);
      /* we want to be able to index symmetries directly with the orientations, which range from [-numFaces,numFaces) */
      symmetries[0] = &cellSymmetries[numFaces];
      if (dim == 1) {
        PetscInt dofPerEdge = order - 1;

        if (dofPerEdge > 1) {
          PetscInt i, j, *reverse;

          ierr = PetscMalloc1(dofPerEdge*Nc,&reverse);CHKERRQ(ierr);
          for (i = 0; i < dofPerEdge; i++) {
            for (j = 0; j < Nc; j++) {
              reverse[i*Nc + j] = Nc * (dofPerEdge - 1 - i) + j;
            }
          }
          symmetries[0][-2] = reverse;

          /* yes, this is redundant, but it makes it easier to cleanup if I don't have to worry about what not to free */
          ierr = PetscMalloc1(dofPerEdge*Nc,&reverse);CHKERRQ(ierr);
          for (i = 0; i < dofPerEdge; i++) {
            for (j = 0; j < Nc; j++) {
              reverse[i*Nc + j] = Nc * (dofPerEdge - 1 - i) + j;
            }
          }
          symmetries[0][1] = reverse;
        }
      } else {
        PetscInt dofPerEdge = lag->simplexCell ? (order - 2) : (order - 1), s;
        PetscInt dofPerFace;

        if (dofPerEdge > 1) {
          for (s = -numFaces; s < numFaces; s++) {
            PetscInt *sym, i, j, k, l;

            if (!s) continue;
            if (lag->simplexCell) {
              dofPerFace = (dofPerEdge * (dofPerEdge + 1))/2;
              ierr = PetscMalloc1(Nc*dofPerFace,&sym);CHKERRQ(ierr);
              for (j = 0, l = 0; j < dofPerEdge; j++) {
                for (k = 0; k < dofPerEdge - j; k++, l++) {
                  i = dofPerEdge - 1 - j - k;
                  switch (s) {
                  case -3:
                    sym[Nc*l] = BaryIndex(dofPerEdge,i,k,j);
                    break;
                  case -2:
                    sym[Nc*l] = BaryIndex(dofPerEdge,j,i,k);
                    break;
                  case -1:
                    sym[Nc*l] = BaryIndex(dofPerEdge,k,j,i);
                    break;
                  case 1:
                    sym[Nc*l] = BaryIndex(dofPerEdge,k,i,j);
                    break;
                  case 2:
                    sym[Nc*l] = BaryIndex(dofPerEdge,j,k,i);
                    break;
                  }
                }
              }
            } else {
              dofPerFace = dofPerEdge * dofPerEdge;
              ierr = PetscMalloc1(Nc*dofPerFace,&sym);CHKERRQ(ierr);
              for (j = 0, l = 0; j < dofPerEdge; j++) {
                for (k = 0; k < dofPerEdge; k++, l++) {
                  switch (s) {
                  case -4:
                    sym[Nc*l] = CartIndex(dofPerEdge,k,j);
                    break;
                  case -3:
                    sym[Nc*l] = CartIndex(dofPerEdge,(dofPerEdge - 1 - j),k);
                    break;
                  case -2:
                    sym[Nc*l] = CartIndex(dofPerEdge,(dofPerEdge - 1 - k),(dofPerEdge - 1 - j));
                    break;
                  case -1:
                    sym[Nc*l] = CartIndex(dofPerEdge,j,(dofPerEdge - 1 - k));
                    break;
                  case 1:
                    sym[Nc*l] = CartIndex(dofPerEdge,(dofPerEdge - 1 - k),j);
                    break;
                  case 2:
                    sym[Nc*l] = CartIndex(dofPerEdge,(dofPerEdge - 1 - j),(dofPerEdge - 1 - k));
                    break;
                  case 3:
                    sym[Nc*l] = CartIndex(dofPerEdge,k,(dofPerEdge - 1 - j));
                    break;
                  }
                }
              }
            }
            for (i = 0; i < dofPerFace; i++) {
              sym[Nc*i] *= Nc;
              for (j = 1; j < Nc; j++) {
                sym[Nc*i+j] = sym[Nc*i] + j;
              }
            }
            symmetries[0][s] = sym;
          }
        }
      }
    }
    ierr = PetscDualSpaceGetHeightSubspace(sp,1,&hsp);CHKERRQ(ierr);
    ierr = PetscDualSpaceGetSymmetries(hsp,&hsymmetries,NULL);CHKERRQ(ierr);
    if (hsymmetries) {
      PetscBool      *seen;
      const PetscInt *cone;
      PetscInt       KclosureSize, *Kclosure = NULL;

      ierr = PetscDualSpaceGetDM(sp,&K);CHKERRQ(ierr);
      ierr = PetscCalloc1(numPoints,&seen);CHKERRQ(ierr);
      ierr = DMPlexGetCone(K,0,&cone);CHKERRQ(ierr);
      ierr = DMPlexGetTransitiveClosure(K,0,PETSC_TRUE,&KclosureSize,&Kclosure);CHKERRQ(ierr);
      for (p = 0; p < numFaces; p++) {
        PetscInt closureSize, *closure = NULL, q;

        ierr = DMPlexGetTransitiveClosure(K,cone[p],PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
        for (q = 0; q < closureSize; q++) {
          PetscInt point = closure[2*q], r;

          if(!seen[point]) {
            for (r = 0; r < KclosureSize; r++) {
              if (Kclosure[2 * r] == point) break;
            }
            seen[point] = PETSC_TRUE;
            symmetries[r] = (PetscInt **) hsymmetries[q];
          }
        }
        ierr = DMPlexRestoreTransitiveClosure(K,cone[p],PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      }
      ierr = DMPlexRestoreTransitiveClosure(K,0,PETSC_TRUE,&KclosureSize,&Kclosure);CHKERRQ(ierr);
      ierr = PetscFree(seen);CHKERRQ(ierr);
    }
    lag->symmetries = symmetries;
  }
  if (perms) *perms = (const PetscInt ***) lag->symmetries;
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

static PetscErrorCode PetscDualSpaceInitialize_Lagrange(PetscDualSpace sp)
{
  PetscFunctionBegin;
  sp->ops->destroy           = PetscDualSpaceDestroy_Lagrange;
  sp->ops->view              = PetscDualSpaceView_Lagrange;
  sp->ops->setfromoptions    = PetscDualSpaceSetFromOptions_Lagrange;
  sp->ops->duplicate         = PetscDualSpaceDuplicate_Lagrange;
  sp->ops->setup             = PetscDualSpaceSetUp_Lagrange;
  sp->ops->createheightsubspace = PetscDualSpaceGetHeightSubspace_Lagrange;
  sp->ops->getsymmetries     = PetscDualSpaceGetSymmetries_Lagrange;
  sp->ops->apply             = PetscDualSpaceApplyDefault;
  sp->ops->applyall          = PetscDualSpaceApplyAllDefault;
  sp->ops->createalldata     = PetscDualSpaceCreateAllDataDefault;
  sp->ops->applyint          = PetscDualSpaceApplyInteriorDefault;
  sp->ops->createintdata     = PetscDualSpaceCreateInteriorDataDefault;
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

  lag->simplexCell = PETSC_TRUE;
  lag->tensorSpace = PETSC_FALSE;
  lag->continuous  = PETSC_TRUE;

  ierr = PetscDualSpaceInitialize_Lagrange(sp);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetContinuity_C", PetscDualSpaceLagrangeGetContinuity_Lagrange);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetContinuity_C", PetscDualSpaceLagrangeSetContinuity_Lagrange);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeGetTensor_C", PetscDualSpaceLagrangeGetTensor_Lagrange);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceLagrangeSetTensor_C", PetscDualSpaceLagrangeSetTensor_Lagrange);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

