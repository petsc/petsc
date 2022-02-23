#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/
#include <petscblaslapack.h>

static PetscErrorCode PetscFEDestroy_Basic(PetscFE fem)
{
  PetscFE_Basic *b = (PetscFE_Basic *) fem->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFEView_Basic_Ascii(PetscFE fe, PetscViewer v)
{
  PetscInt          dim, Nc;
  PetscSpace        basis = NULL;
  PetscDualSpace    dual = NULL;
  PetscQuadrature   quad = NULL;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fe, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
  ierr = PetscFEGetBasisSpace(fe, &basis);CHKERRQ(ierr);
  ierr = PetscFEGetDualSpace(fe, &dual);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(v);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(v, "Basic Finite Element in %D dimensions with %D components\n",dim,Nc);CHKERRQ(ierr);
  if (basis) {ierr = PetscSpaceView(basis, v);CHKERRQ(ierr);}
  if (dual)  {ierr = PetscDualSpaceView(dual, v);CHKERRQ(ierr);}
  if (quad)  {ierr = PetscQuadratureView(quad, v);CHKERRQ(ierr);}
  ierr = PetscViewerASCIIPopTab(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFEView_Basic(PetscFE fe, PetscViewer v)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) v, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscFEView_Basic_Ascii(fe, v);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* Construct the change of basis from prime basis to nodal basis */
PETSC_INTERN PetscErrorCode PetscFESetUp_Basic(PetscFE fem)
{
  PetscReal     *work;
  PetscBLASInt  *pivots;
  PetscBLASInt   n, info;
  PetscInt       pdim, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDimension(fem->dualSpace, &pdim);CHKERRQ(ierr);
  ierr = PetscMalloc1(pdim*pdim,&fem->invV);CHKERRQ(ierr);
  for (j = 0; j < pdim; ++j) {
    PetscReal       *Bf;
    PetscQuadrature  f;
    const PetscReal *points, *weights;
    PetscInt         Nc, Nq, q, k, c;

    ierr = PetscDualSpaceGetFunctional(fem->dualSpace, j, &f);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(f, NULL, &Nc, &Nq, &points, &weights);CHKERRQ(ierr);
    ierr = PetscMalloc1(Nc*Nq*pdim,&Bf);CHKERRQ(ierr);
    ierr = PetscSpaceEvaluate(fem->basisSpace, Nq, points, Bf, NULL, NULL);CHKERRQ(ierr);
    for (k = 0; k < pdim; ++k) {
      /* V_{jk} = n_j(\phi_k) = \int \phi_k(x) n_j(x) dx */
      fem->invV[j*pdim+k] = 0.0;

      for (q = 0; q < Nq; ++q) {
        for (c = 0; c < Nc; ++c) fem->invV[j*pdim+k] += Bf[(q*pdim + k)*Nc + c]*weights[q*Nc + c];
      }
    }
    ierr = PetscFree(Bf);CHKERRQ(ierr);
  }

  ierr = PetscMalloc2(pdim,&pivots,pdim,&work);CHKERRQ(ierr);
  n = pdim;
  PetscStackCallBLAS("LAPACKgetrf", LAPACKREALgetrf_(&n, &n, fem->invV, &n, pivots, &info));
  PetscCheckFalse(info,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error returned from LAPACKgetrf %D",(PetscInt)info);
  PetscStackCallBLAS("LAPACKgetri", LAPACKREALgetri_(&n, fem->invV, &n, pivots, work, &n, &info));
  PetscCheckFalse(info,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error returned from LAPACKgetri %D",(PetscInt)info);
  ierr = PetscFree2(pivots,work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEGetDimension_Basic(PetscFE fem, PetscInt *dim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDimension(fem->dualSpace, dim);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Tensor contraction on the middle index,
 *    C[m,n,p] = A[m,k,p] * B[k,n]
 * where all matrices use C-style ordering.
 */
static PetscErrorCode TensorContract_Private(PetscInt m,PetscInt n,PetscInt p,PetscInt k,const PetscReal *A,const PetscReal *B,PetscReal *C)
{
  PetscErrorCode ierr;
  PetscInt i;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    PetscBLASInt n_,p_,k_,lda,ldb,ldc;
    PetscReal one = 1, zero = 0;
    /* Taking contiguous submatrices, we wish to comput c[n,p] = a[k,p] * B[k,n]
     * or, in Fortran ordering, c(p,n) = a(p,k) * B(n,k)
     */
    ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(p,&p_);CHKERRQ(ierr);
    ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr);
    lda = p_;
    ldb = n_;
    ldc = p_;
    PetscStackCallBLAS("BLASgemm",BLASREALgemm_("N","T",&p_,&n_,&k_,&one,A+i*k*p,&lda,B,&ldb,&zero,C+i*n*p,&ldc));
  }
  ierr = PetscLogFlops(2.*m*n*p*k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscFECreateTabulation_Basic(PetscFE fem, PetscInt npoints, const PetscReal points[], PetscInt K, PetscTabulation T)
{
  DM               dm;
  PetscInt         pdim; /* Dimension of FE space P */
  PetscInt         dim;  /* Spatial dimension */
  PetscInt         Nc;   /* Field components */
  PetscReal       *B = K >= 0 ? T->T[0] : NULL;
  PetscReal       *D = K >= 1 ? T->T[1] : NULL;
  PetscReal       *H = K >= 2 ? T->T[2] : NULL;
  PetscReal       *tmpB = NULL, *tmpD = NULL, *tmpH = NULL;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDM(fem->dualSpace, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDimension(fem->dualSpace, &pdim);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fem, &Nc);CHKERRQ(ierr);
  /* Evaluate the prime basis functions at all points */
  if (K >= 0) {ierr = DMGetWorkArray(dm, npoints*pdim*Nc, MPIU_REAL, &tmpB);CHKERRQ(ierr);}
  if (K >= 1) {ierr = DMGetWorkArray(dm, npoints*pdim*Nc*dim, MPIU_REAL, &tmpD);CHKERRQ(ierr);}
  if (K >= 2) {ierr = DMGetWorkArray(dm, npoints*pdim*Nc*dim*dim, MPIU_REAL, &tmpH);CHKERRQ(ierr);}
  ierr = PetscSpaceEvaluate(fem->basisSpace, npoints, points, tmpB, tmpD, tmpH);CHKERRQ(ierr);
  /* Translate from prime to nodal basis */
  if (B) {
    /* B[npoints, nodes, Nc] = tmpB[npoints, prime, Nc] * invV[prime, nodes] */
    ierr = TensorContract_Private(npoints, pdim, Nc, pdim, tmpB, fem->invV, B);CHKERRQ(ierr);
  }
  if (D) {
    /* D[npoints, nodes, Nc, dim] = tmpD[npoints, prime, Nc, dim] * invV[prime, nodes] */
    ierr = TensorContract_Private(npoints, pdim, Nc*dim, pdim, tmpD, fem->invV, D);CHKERRQ(ierr);
  }
  if (H) {
    /* H[npoints, nodes, Nc, dim, dim] = tmpH[npoints, prime, Nc, dim, dim] * invV[prime, nodes] */
    ierr = TensorContract_Private(npoints, pdim, Nc*dim*dim, pdim, tmpH, fem->invV, H);CHKERRQ(ierr);
  }
  if (K >= 0) {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nc, MPIU_REAL, &tmpB);CHKERRQ(ierr);}
  if (K >= 1) {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nc*dim, MPIU_REAL, &tmpD);CHKERRQ(ierr);}
  if (K >= 2) {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nc*dim*dim, MPIU_REAL, &tmpH);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFEIntegrate_Basic(PetscDS ds, PetscInt field, PetscInt Ne, PetscFEGeom *cgeom,
                                             const PetscScalar coefficients[], PetscDS dsAux, const PetscScalar coefficientsAux[], PetscScalar integral[])
{
  const PetscInt     debug = 0;
  PetscFE            fe;
  PetscPointFunc     obj_func;
  PetscQuadrature    quad;
  PetscTabulation   *T, *TAux = NULL;
  PetscScalar       *u, *u_x, *a, *a_x;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL;
  PetscInt           dim, dE, Np, numConstants, Nf, NfAux = 0, totDim, totDimAux = 0, cOffset = 0, cOffsetAux = 0, e;
  PetscBool          isAffine;
  const PetscReal   *quadPoints, *quadWeights;
  PetscInt           qNc, Nq, q;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDSGetObjective(ds, field, &obj_func);CHKERRQ(ierr);
  if (!obj_func) PetscFunctionReturn(0);
  ierr = PetscDSGetDiscretization(ds, field, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetSpatialDimension(fe, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(ds, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(ds, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(ds, &T);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(ds, &u, NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetWorkspace(ds, &x, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(ds, &numConstants, &constants);CHKERRQ(ierr);
  if (dsAux) {
    ierr = PetscDSGetNumFields(dsAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(dsAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(dsAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(dsAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetTabulation(dsAux, &TAux);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(dsAux, &a, NULL, &a_x);CHKERRQ(ierr);
    PetscCheckFalse(T[0]->Np != TAux[0]->Np,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of tabulation points %D != %D number of auxiliary tabulation points", T[0]->Np, TAux[0]->Np);
  }
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  PetscCheckFalse(qNc != 1,PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components", qNc);
  Np = cgeom->numPoints;
  dE = cgeom->dimEmbed;
  isAffine = cgeom->isAffine;
  for (e = 0; e < Ne; ++e) {
    PetscFEGeom fegeom;

    fegeom.dim      = cgeom->dim;
    fegeom.dimEmbed = cgeom->dimEmbed;
    if (isAffine) {
      fegeom.v    = x;
      fegeom.xi   = cgeom->xi;
      fegeom.J    = &cgeom->J[e*Np*dE*dE];
      fegeom.invJ = &cgeom->invJ[e*Np*dE*dE];
      fegeom.detJ = &cgeom->detJ[e*Np];
    }
    for (q = 0; q < Nq; ++q) {
      PetscScalar integrand;
      PetscReal   w;

      if (isAffine) {
        CoordinatesRefToReal(dE, dim, fegeom.xi, &cgeom->v[e*Np*dE], fegeom.J, &quadPoints[q*dim], x);
      } else {
        fegeom.v    = &cgeom->v[(e*Np+q)*dE];
        fegeom.J    = &cgeom->J[(e*Np+q)*dE*dE];
        fegeom.invJ = &cgeom->invJ[(e*Np+q)*dE*dE];
        fegeom.detJ = &cgeom->detJ[e*Np+q];
      }
      w = fegeom.detJ[0]*quadWeights[q];
      if (debug > 1 && q < Np) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", fegeom.detJ[0]);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
        ierr = DMPrintCellMatrix(e, "invJ", dim, dim, fegeom.invJ);CHKERRQ(ierr);
#endif
      }
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      ierr = PetscFEEvaluateFieldJets_Internal(ds, Nf, 0, q, T, &fegeom, &coefficients[cOffset], NULL, u, u_x, NULL);CHKERRQ(ierr);
      if (dsAux) {ierr = PetscFEEvaluateFieldJets_Internal(dsAux, NfAux, 0, q, TAux, &fegeom, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);CHKERRQ(ierr);}
      obj_func(dim, Nf, NfAux, uOff, uOff_x, u, NULL, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, fegeom.v, numConstants, constants, &integrand);
      integrand *= w;
      integral[e*Nf+field] += integrand;
      if (debug > 1) {ierr = PetscPrintf(PETSC_COMM_SELF, "    int: %g %g\n", (double) PetscRealPart(integrand), (double) PetscRealPart(integral[field]));CHKERRQ(ierr);}
    }
    cOffset    += totDim;
    cOffsetAux += totDimAux;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFEIntegrateBd_Basic(PetscDS ds, PetscInt field,
                                               PetscBdPointFunc obj_func,
                                               PetscInt Ne, PetscFEGeom *fgeom, const PetscScalar coefficients[], PetscDS dsAux, const PetscScalar coefficientsAux[], PetscScalar integral[])
{
  const PetscInt     debug = 0;
  PetscFE            fe;
  PetscQuadrature    quad;
  PetscTabulation   *Tf, *TfAux = NULL;
  PetscScalar       *u, *u_x, *a, *a_x, *basisReal, *basisDerReal;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL;
  PetscBool          isAffine, auxOnBd;
  const PetscReal   *quadPoints, *quadWeights;
  PetscInt           qNc, Nq, q, Np, dE;
  PetscInt           dim, dimAux, numConstants, Nf, NfAux = 0, totDim, totDimAux = 0, cOffset = 0, cOffsetAux = 0, e;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!obj_func) PetscFunctionReturn(0);
  ierr = PetscDSGetDiscretization(ds, field, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetSpatialDimension(fe, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetFaceQuadrature(fe, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(ds, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(ds, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(ds, &u, NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetWorkspace(ds, &x, &basisReal, &basisDerReal, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetFaceTabulation(ds, &Tf);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(ds, &numConstants, &constants);CHKERRQ(ierr);
  if (dsAux) {
    ierr = PetscDSGetSpatialDimension(dsAux, &dimAux);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(dsAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(dsAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(dsAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(dsAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(dsAux, &a, NULL, &a_x);CHKERRQ(ierr);
    auxOnBd = dimAux < dim ? PETSC_TRUE : PETSC_FALSE;
    if (auxOnBd) {ierr = PetscDSGetTabulation(dsAux, &TfAux);CHKERRQ(ierr);}
    else         {ierr = PetscDSGetFaceTabulation(dsAux, &TfAux);CHKERRQ(ierr);}
    PetscCheckFalse(Tf[0]->Np != TfAux[0]->Np,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of tabulation points %D != %D number of auxiliary tabulation points", Tf[0]->Np, TfAux[0]->Np);
  }
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  PetscCheckFalse(qNc != 1,PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components", qNc);
  Np = fgeom->numPoints;
  dE = fgeom->dimEmbed;
  isAffine = fgeom->isAffine;
  for (e = 0; e < Ne; ++e) {
    PetscFEGeom    fegeom, cgeom;
    const PetscInt face = fgeom->face[e][0]; /* Local face number in cell */
    fegeom.n = NULL;
    fegeom.v = NULL;
    fegeom.J = NULL;
    fegeom.detJ = NULL;
    fegeom.dim      = fgeom->dim;
    fegeom.dimEmbed = fgeom->dimEmbed;
    cgeom.dim       = fgeom->dim;
    cgeom.dimEmbed  = fgeom->dimEmbed;
    if (isAffine) {
      fegeom.v    = x;
      fegeom.xi   = fgeom->xi;
      fegeom.J    = &fgeom->J[e*Np*dE*dE];
      fegeom.invJ = &fgeom->invJ[e*Np*dE*dE];
      fegeom.detJ = &fgeom->detJ[e*Np];
      fegeom.n    = &fgeom->n[e*Np*dE];

      cgeom.J     = &fgeom->suppJ[0][e*Np*dE*dE];
      cgeom.invJ  = &fgeom->suppInvJ[0][e*Np*dE*dE];
      cgeom.detJ  = &fgeom->suppDetJ[0][e*Np];
    }
    for (q = 0; q < Nq; ++q) {
      PetscScalar integrand;
      PetscReal   w;

      if (isAffine) {
        CoordinatesRefToReal(dE, dim-1, fegeom.xi, &fgeom->v[e*Np*dE], fegeom.J, &quadPoints[q*(dim-1)], x);
      } else {
        fegeom.v    = &fgeom->v[(e*Np+q)*dE];
        fegeom.J    = &fgeom->J[(e*Np+q)*dE*dE];
        fegeom.invJ = &fgeom->invJ[(e*Np+q)*dE*dE];
        fegeom.detJ = &fgeom->detJ[e*Np+q];
        fegeom.n    = &fgeom->n[(e*Np+q)*dE];

        cgeom.J     = &fgeom->suppJ[0][(e*Np+q)*dE*dE];
        cgeom.invJ  = &fgeom->suppInvJ[0][(e*Np+q)*dE*dE];
        cgeom.detJ  = &fgeom->suppDetJ[0][e*Np+q];
      }
      w = fegeom.detJ[0]*quadWeights[q];
      if (debug > 1 && q < Np) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", fegeom.detJ[0]);CHKERRQ(ierr);
#ifndef PETSC_USE_COMPLEX
        ierr = DMPrintCellMatrix(e, "invJ", dim, dim, fegeom.invJ);CHKERRQ(ierr);
#endif
      }
      if (debug > 1) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      ierr = PetscFEEvaluateFieldJets_Internal(ds, Nf, face, q, Tf, &cgeom, &coefficients[cOffset], NULL, u, u_x, NULL);CHKERRQ(ierr);
      if (dsAux) {ierr = PetscFEEvaluateFieldJets_Internal(dsAux, NfAux, face, q, TfAux, &cgeom, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);CHKERRQ(ierr);}
      obj_func(dim, Nf, NfAux, uOff, uOff_x, u, NULL, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, fegeom.v, fegeom.n, numConstants, constants, &integrand);
      integrand *= w;
      integral[e*Nf+field] += integrand;
      if (debug > 1) {ierr = PetscPrintf(PETSC_COMM_SELF, "    int: %g %g\n", (double) PetscRealPart(integrand), (double) PetscRealPart(integral[e*Nf+field]));CHKERRQ(ierr);}
    }
    cOffset    += totDim;
    cOffsetAux += totDimAux;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEIntegrateResidual_Basic(PetscDS ds, PetscFormKey key, PetscInt Ne, PetscFEGeom *cgeom,
                                              const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS dsAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  const PetscInt     debug = 0;
  const PetscInt     field = key.field;
  PetscFE            fe;
  PetscWeakForm      wf;
  PetscInt           n0,       n1, i;
  PetscPointFunc    *f0_func, *f1_func;
  PetscQuadrature    quad;
  PetscTabulation   *T, *TAux = NULL;
  PetscScalar       *f0, *f1, *u, *u_t = NULL, *u_x, *a, *a_x, *basisReal, *basisDerReal;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL;
  PetscInt           dim, numConstants, Nf, NfAux = 0, totDim, totDimAux = 0, cOffset = 0, cOffsetAux = 0, fOffset, e;
  const PetscReal   *quadPoints, *quadWeights;
  PetscInt           qdim, qNc, Nq, q, dE;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDSGetDiscretization(ds, field, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetSpatialDimension(fe, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(ds, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(ds, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(ds, field, &fOffset);CHKERRQ(ierr);
  ierr = PetscDSGetWeakForm(ds, &wf);CHKERRQ(ierr);
  ierr = PetscWeakFormGetResidual(wf, key.label, key.value, key.field, key.part, &n0, &f0_func, &n1, &f1_func);CHKERRQ(ierr);
  if (!n0 && !n1) PetscFunctionReturn(0);
  ierr = PetscDSGetEvaluationArrays(ds, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetWorkspace(ds, &x, &basisReal, &basisDerReal, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(ds, &f0, &f1, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(ds, &T);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(ds, &numConstants, &constants);CHKERRQ(ierr);
  if (dsAux) {
    ierr = PetscDSGetNumFields(dsAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(dsAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(dsAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(dsAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(dsAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetTabulation(dsAux, &TAux);CHKERRQ(ierr);
    PetscCheckFalse(T[0]->Np != TAux[0]->Np,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of tabulation points %D != %D number of auxiliary tabulation points", T[0]->Np, TAux[0]->Np);
  }
  ierr = PetscQuadratureGetData(quad, &qdim, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  PetscCheckFalse(qNc != 1,PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components", qNc);
  dE = cgeom->dimEmbed;
  PetscCheckFalse(cgeom->dim != qdim,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "FEGeom dim %D != %D quadrature dim", cgeom->dim, qdim);
  for (e = 0; e < Ne; ++e) {
    PetscFEGeom fegeom;

    fegeom.v = x; /* workspace */
    ierr = PetscArrayzero(f0, Nq*T[field]->Nc);CHKERRQ(ierr);
    ierr = PetscArrayzero(f1, Nq*T[field]->Nc*dE);CHKERRQ(ierr);
    for (q = 0; q < Nq; ++q) {
      PetscReal w;
      PetscInt  c, d;

      ierr = PetscFEGeomGetPoint(cgeom, e, q, &quadPoints[q*cgeom->dim], &fegeom);CHKERRQ(ierr);
      w = fegeom.detJ[0]*quadWeights[q];
      if (debug > 1 && q < cgeom->numPoints) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", fegeom.detJ[0]);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
        ierr = DMPrintCellMatrix(e, "invJ", dim, dim, fegeom.invJ);CHKERRQ(ierr);
#endif
      }
      ierr = PetscFEEvaluateFieldJets_Internal(ds, Nf, 0, q, T, &fegeom, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);CHKERRQ(ierr);
      if (dsAux) {ierr = PetscFEEvaluateFieldJets_Internal(dsAux, NfAux, 0, q, TAux, &fegeom, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);CHKERRQ(ierr);}
      for (i = 0; i < n0; ++i) f0_func[i](dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, fegeom.v, numConstants, constants, &f0[q*T[field]->Nc]);
      for (c = 0; c < T[field]->Nc; ++c) f0[q*T[field]->Nc+c] *= w;
      for (i = 0; i < n1; ++i) f1_func[i](dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, fegeom.v, numConstants, constants, &f1[q*T[field]->Nc*dim]);
      for (c = 0; c < T[field]->Nc; ++c) for (d = 0; d < dim; ++d) f1[(q*T[field]->Nc+c)*dim+d] *= w;
      if (debug) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d wt %g\n", q, quadWeights[q]);CHKERRQ(ierr);
        if (debug > 2) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "  field %d:", field);CHKERRQ(ierr);
          for (c = 0; c < T[field]->Nc; ++c) {ierr = PetscPrintf(PETSC_COMM_SELF, " %g", u[uOff[field]+c]);CHKERRQ(ierr);}
          ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
          ierr = PetscPrintf(PETSC_COMM_SELF, "  resid %d:", field);CHKERRQ(ierr);
          for (c = 0; c < T[field]->Nc; ++c) {ierr = PetscPrintf(PETSC_COMM_SELF, " %g", f0[q*T[field]->Nc+c]);CHKERRQ(ierr);}
          ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscFEUpdateElementVec_Internal(fe, T[field], 0, basisReal, basisDerReal, e, cgeom, f0, f1, &elemVec[cOffset+fOffset]);CHKERRQ(ierr);
    cOffset    += totDim;
    cOffsetAux += totDimAux;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEIntegrateBdResidual_Basic(PetscDS ds, PetscWeakForm wf, PetscFormKey key, PetscInt Ne, PetscFEGeom *fgeom,
                                                const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS dsAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  const PetscInt     debug = 0;
  const PetscInt     field = key.field;
  PetscFE            fe;
  PetscInt           n0,       n1, i;
  PetscBdPointFunc  *f0_func, *f1_func;
  PetscQuadrature    quad;
  PetscTabulation   *Tf, *TfAux = NULL;
  PetscScalar       *f0, *f1, *u, *u_t = NULL, *u_x, *a, *a_x, *basisReal, *basisDerReal;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL;
  PetscInt           dim, dimAux, numConstants, Nf, NfAux = 0, totDim, totDimAux = 0, cOffset = 0, cOffsetAux = 0, fOffset, e, NcI;
  PetscBool          auxOnBd = PETSC_FALSE;
  const PetscReal   *quadPoints, *quadWeights;
  PetscInt           qdim, qNc, Nq, q, dE;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDSGetDiscretization(ds, field, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetSpatialDimension(fe, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetFaceQuadrature(fe, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(ds, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(ds, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(ds, field, &fOffset);CHKERRQ(ierr);
  ierr = PetscWeakFormGetBdResidual(wf, key.label, key.value, key.field, key.part, &n0, &f0_func, &n1, &f1_func);CHKERRQ(ierr);
  if (!n0 && !n1) PetscFunctionReturn(0);
  ierr = PetscDSGetEvaluationArrays(ds, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetWorkspace(ds, &x, &basisReal, &basisDerReal, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(ds, &f0, &f1, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetFaceTabulation(ds, &Tf);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(ds, &numConstants, &constants);CHKERRQ(ierr);
  if (dsAux) {
    ierr = PetscDSGetSpatialDimension(dsAux, &dimAux);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(dsAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(dsAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(dsAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(dsAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(dsAux, &a, NULL, &a_x);CHKERRQ(ierr);
    auxOnBd = dimAux < dim ? PETSC_TRUE : PETSC_FALSE;
    if (auxOnBd) {ierr = PetscDSGetTabulation(dsAux, &TfAux);CHKERRQ(ierr);}
    else         {ierr = PetscDSGetFaceTabulation(dsAux, &TfAux);CHKERRQ(ierr);}
    PetscCheckFalse(Tf[0]->Np != TfAux[0]->Np,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of tabulation points %D != %D number of auxiliary tabulation points", Tf[0]->Np, TfAux[0]->Np);
  }
  NcI = Tf[field]->Nc;
  ierr = PetscQuadratureGetData(quad, &qdim, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  PetscCheckFalse(qNc != 1,PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components", qNc);
  dE = fgeom->dimEmbed;
  /* TODO FIX THIS */
  fgeom->dim = dim-1;
  PetscCheckFalse(fgeom->dim != qdim,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "FEGeom dim %D != %D quadrature dim", fgeom->dim, qdim);
  for (e = 0; e < Ne; ++e) {
    PetscFEGeom    fegeom, cgeom;
    const PetscInt face = fgeom->face[e][0];

    fegeom.v = x; /* Workspace */
    ierr = PetscArrayzero(f0, Nq*NcI);CHKERRQ(ierr);
    ierr = PetscArrayzero(f1, Nq*NcI*dE);CHKERRQ(ierr);
    for (q = 0; q < Nq; ++q) {
      PetscReal w;
      PetscInt  c, d;

      ierr = PetscFEGeomGetPoint(fgeom, e, q, &quadPoints[q*fgeom->dim], &fegeom);CHKERRQ(ierr);
      ierr = PetscFEGeomGetCellPoint(fgeom, e, q, &cgeom);CHKERRQ(ierr);
      w = fegeom.detJ[0]*quadWeights[q];
      if (debug > 1) {
        if ((fgeom->isAffine && q == 0) || (!fgeom->isAffine)) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", fegeom.detJ[0]);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
          ierr = DMPrintCellMatrix(e, "invJ", dim, dim, fegeom.invJ);CHKERRQ(ierr);
          ierr = DMPrintCellVector(e, "n", dim, fegeom.n);CHKERRQ(ierr);
#endif
        }
      }
      ierr = PetscFEEvaluateFieldJets_Internal(ds, Nf, face, q, Tf, &cgeom, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);CHKERRQ(ierr);
      if (dsAux) {ierr = PetscFEEvaluateFieldJets_Internal(dsAux, NfAux, auxOnBd ? 0 : face, q, TfAux, &cgeom, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);CHKERRQ(ierr);}
      for (i = 0; i < n0; ++i) f0_func[i](dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, fegeom.v, fegeom.n, numConstants, constants, &f0[q*NcI]);
      for (c = 0; c < NcI; ++c) f0[q*NcI+c] *= w;
      for (i = 0; i < n1; ++i) f1_func[i](dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, fegeom.v, fegeom.n, numConstants, constants, &f1[q*NcI*dim]);
      for (c = 0; c < NcI; ++c) for (d = 0; d < dim; ++d) f1[(q*NcI+c)*dim+d] *= w;
      if (debug) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "  elem %D quad point %d\n", e, q);CHKERRQ(ierr);
        for (c = 0; c < NcI; ++c) {
          if (n0) {ierr = PetscPrintf(PETSC_COMM_SELF, "  f0[%D] %g\n", c, f0[q*NcI+c]);CHKERRQ(ierr);}
          if (n1) {
            for (d = 0; d < dim; ++d) {ierr = PetscPrintf(PETSC_COMM_SELF, "  f1[%D,%D] %g", c, d, f1[(q*NcI + c)*dim + d]);CHKERRQ(ierr);}
            ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
          }
        }
      }
    }
    ierr = PetscFEUpdateElementVec_Internal(fe, Tf[field], face, basisReal, basisDerReal, e, fgeom, f0, f1, &elemVec[cOffset+fOffset]);CHKERRQ(ierr);
    cOffset    += totDim;
    cOffsetAux += totDimAux;
  }
  PetscFunctionReturn(0);
}

/*
  BdIntegral: Operates completely in the embedding dimension. The trick is to have special "face quadrature" so we only integrate over the face, but
              all transforms operate in the full space and are square.

  HybridIntegral: The discretization is lower dimensional. That means the transforms are non-square.
    1) DMPlexGetCellFields() retrieves from the hybrid cell, so it gets fields from both faces
    2) We need to assume that the orientation is 0 for both
    3) TODO We need to use a non-square Jacobian for the derivative maps, meaning the embedding dimension has to go to EvaluateFieldJets() and UpdateElementVec()
*/
static PetscErrorCode PetscFEIntegrateHybridResidual_Basic(PetscDS ds, PetscFormKey key, PetscInt s, PetscInt Ne, PetscFEGeom *fgeom,
                                                           const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS dsAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  const PetscInt     debug = 0;
  const PetscInt     field = key.field;
  PetscFE            fe;
  PetscWeakForm      wf;
  PetscInt           n0,      n1, i;
  PetscBdPointFunc  *f0_func, *f1_func;
  PetscQuadrature    quad;
  PetscTabulation   *Tf, *TfAux = NULL;
  PetscScalar       *f0, *f1, *u, *u_t = NULL, *u_x, *a, *a_x, *basisReal, *basisDerReal;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL;
  PetscInt           dim, dimAux, numConstants, Nf, NfAux = 0, totDim, totDimAux = 0, cOffset = 0, cOffsetAux = 0, fOffset, e, NcI, NcS;
  PetscBool          isCohesiveField, auxOnBd = PETSC_FALSE;
  const PetscReal   *quadPoints, *quadWeights;
  PetscInt           qdim, qNc, Nq, q, dE;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* Hybrid discretization is posed directly on faces */
  ierr = PetscDSGetDiscretization(ds, field, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetSpatialDimension(fe, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsetsCohesive(ds, s, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsetsCohesive(ds, s, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffsetCohesive(ds, field, &fOffset);CHKERRQ(ierr);
  ierr = PetscDSGetWeakForm(ds, &wf);CHKERRQ(ierr);
  ierr = PetscWeakFormGetBdResidual(wf, key.label, key.value, key.field, key.part, &n0, &f0_func, &n1, &f1_func);CHKERRQ(ierr);
  if (!n0 && !n1) PetscFunctionReturn(0);
  ierr = PetscDSGetEvaluationArrays(ds, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetWorkspace(ds, &x, &basisReal, &basisDerReal, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(ds, &f0, &f1, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  /* NOTE This is a bulk tabulation because the DS is a face discretization */
  ierr = PetscDSGetTabulation(ds, &Tf);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(ds, &numConstants, &constants);CHKERRQ(ierr);
  if (dsAux) {
    ierr = PetscDSGetSpatialDimension(dsAux, &dimAux);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(dsAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(dsAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(dsAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(dsAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(dsAux, &a, NULL, &a_x);CHKERRQ(ierr);
    auxOnBd = dimAux == dim ? PETSC_TRUE : PETSC_FALSE;
    if (auxOnBd) {ierr = PetscDSGetTabulation(dsAux, &TfAux);CHKERRQ(ierr);}
    else         {ierr = PetscDSGetFaceTabulation(dsAux, &TfAux);CHKERRQ(ierr);}
    PetscCheckFalse(Tf[0]->Np != TfAux[0]->Np,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of tabulation points %D != %D number of auxiliary tabulation points", Tf[0]->Np, TfAux[0]->Np);
  }
  ierr = PetscDSGetCohesive(ds, field, &isCohesiveField);CHKERRQ(ierr);
  NcI = Tf[field]->Nc;
  NcS = NcI;
  ierr = PetscQuadratureGetData(quad, &qdim, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  PetscCheckFalse(qNc != 1,PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components", qNc);
  dE = fgeom->dimEmbed;
  PetscCheckFalse(fgeom->dim != qdim,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "FEGeom dim %D != %D quadrature dim", fgeom->dim, qdim);
  for (e = 0; e < Ne; ++e) {
    PetscFEGeom    fegeom;
    const PetscInt face = fgeom->face[e][0];

    fegeom.v = x; /* Workspace */
    ierr = PetscArrayzero(f0, Nq*NcS);CHKERRQ(ierr);
    ierr = PetscArrayzero(f1, Nq*NcS*dE);CHKERRQ(ierr);
    for (q = 0; q < Nq; ++q) {
      PetscReal w;
      PetscInt  c, d;

      ierr = PetscFEGeomGetPoint(fgeom, e, q, &quadPoints[q*fgeom->dim], &fegeom);CHKERRQ(ierr);
      w = fegeom.detJ[0]*quadWeights[q];
      if (debug > 1 && q < fgeom->numPoints) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", fegeom.detJ[0]);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
        ierr = DMPrintCellMatrix(e, "invJ", dim, dE, fegeom.invJ);CHKERRQ(ierr);
#endif
      }
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      /* TODO Is this cell or face quadrature, meaning should we use 'q' or 'face*Nq+q' */
      ierr = PetscFEEvaluateFieldJets_Hybrid_Internal(ds, Nf, 0, q, Tf, &fegeom, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);CHKERRQ(ierr);
      if (dsAux) {ierr = PetscFEEvaluateFieldJets_Internal(dsAux, NfAux, 0, auxOnBd ? q : face*Nq+q, TfAux, &fegeom, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);CHKERRQ(ierr);}
      for (i = 0; i < n0; ++i) f0_func[i](dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, fegeom.v, fegeom.n, numConstants, constants, &f0[q*NcS]);
      for (c = 0; c < NcS; ++c) f0[q*NcS+c] *= w;
      for (i = 0; i < n1; ++i) f1_func[i](dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, fegeom.v, fegeom.n, numConstants, constants, &f1[q*NcS*dE]);
      for (c = 0; c < NcS; ++c) for (d = 0; d < dE; ++d) f1[(q*NcS+c)*dE+d] *= w;
    }
    if (isCohesiveField) {PetscFEUpdateElementVec_Internal(fe, Tf[field], 0, basisReal, basisDerReal, e, fgeom, f0, f1, &elemVec[cOffset+fOffset]);}
    else                 {PetscFEUpdateElementVec_Hybrid_Internal(fe, Tf[field], 0, s, basisReal, basisDerReal, fgeom, f0, f1, &elemVec[cOffset+fOffset]);}
    cOffset    += totDim;
    cOffsetAux += totDimAux;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEIntegrateJacobian_Basic(PetscDS ds, PetscFEJacobianType jtype, PetscFormKey key, PetscInt Ne, PetscFEGeom *cgeom,
                                              const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS dsAux, const PetscScalar coefficientsAux[], PetscReal t, PetscReal u_tshift, PetscScalar elemMat[])
{
  const PetscInt     debug      = 0;
  PetscFE            feI, feJ;
  PetscWeakForm      wf;
  PetscPointJac     *g0_func, *g1_func, *g2_func, *g3_func;
  PetscInt           n0, n1, n2, n3, i;
  PetscInt           cOffset    = 0; /* Offset into coefficients[] for element e */
  PetscInt           cOffsetAux = 0; /* Offset into coefficientsAux[] for element e */
  PetscInt           eOffset    = 0; /* Offset into elemMat[] for element e */
  PetscInt           offsetI    = 0; /* Offset into an element vector for fieldI */
  PetscInt           offsetJ    = 0; /* Offset into an element vector for fieldJ */
  PetscQuadrature    quad;
  PetscTabulation   *T, *TAux = NULL;
  PetscScalar       *g0, *g1, *g2, *g3, *u, *u_t = NULL, *u_x, *a, *a_x, *basisReal, *basisDerReal, *testReal, *testDerReal;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL;
  PetscInt           NcI = 0, NcJ = 0;
  PetscInt           dim, numConstants, Nf, fieldI, fieldJ, NfAux = 0, totDim, totDimAux = 0, e;
  PetscInt           dE, Np;
  PetscBool          isAffine;
  const PetscReal   *quadPoints, *quadWeights;
  PetscInt           qNc, Nq, q;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  fieldI = key.field / Nf;
  fieldJ = key.field % Nf;
  ierr = PetscDSGetDiscretization(ds, fieldI, (PetscObject *) &feI);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(ds, fieldJ, (PetscObject *) &feJ);CHKERRQ(ierr);
  ierr = PetscFEGetSpatialDimension(feI, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(feI, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(ds, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(ds, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetWeakForm(ds, &wf);CHKERRQ(ierr);
  switch(jtype) {
  case PETSCFE_JACOBIAN_DYN: ierr = PetscWeakFormGetDynamicJacobian(wf, key.label, key.value, fieldI, fieldJ, key.part, &n0, &g0_func, &n1, &g1_func, &n2, &g2_func, &n3, &g3_func);CHKERRQ(ierr);break;
  case PETSCFE_JACOBIAN_PRE: ierr = PetscWeakFormGetJacobianPreconditioner(wf, key.label, key.value, fieldI, fieldJ, key.part, &n0, &g0_func, &n1, &g1_func, &n2, &g2_func, &n3, &g3_func);CHKERRQ(ierr);break;
  case PETSCFE_JACOBIAN:     ierr = PetscWeakFormGetJacobian(wf, key.label, key.value, fieldI, fieldJ, key.part, &n0, &g0_func, &n1, &g1_func, &n2, &g2_func, &n3, &g3_func);CHKERRQ(ierr);break;
  }
  if (!n0 && !n1 && !n2 && !n3) PetscFunctionReturn(0);
  ierr = PetscDSGetEvaluationArrays(ds, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetWorkspace(ds, &x, &basisReal, &basisDerReal, &testReal, &testDerReal);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(ds, NULL, NULL, &g0, &g1, &g2, &g3);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(ds, &T);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(ds, fieldI, &offsetI);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(ds, fieldJ, &offsetJ);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(ds, &numConstants, &constants);CHKERRQ(ierr);
  if (dsAux) {
    ierr = PetscDSGetNumFields(dsAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(dsAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(dsAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(dsAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(dsAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetTabulation(dsAux, &TAux);CHKERRQ(ierr);
    PetscCheckFalse(T[0]->Np != TAux[0]->Np,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of tabulation points %D != %D number of auxiliary tabulation points", T[0]->Np, TAux[0]->Np);
  }
  NcI = T[fieldI]->Nc;
  NcJ = T[fieldJ]->Nc;
  Np  = cgeom->numPoints;
  dE  = cgeom->dimEmbed;
  isAffine = cgeom->isAffine;
  /* Initialize here in case the function is not defined */
  ierr = PetscArrayzero(g0, NcI*NcJ);CHKERRQ(ierr);
  ierr = PetscArrayzero(g1, NcI*NcJ*dE);CHKERRQ(ierr);
  ierr = PetscArrayzero(g2, NcI*NcJ*dE);CHKERRQ(ierr);
  ierr = PetscArrayzero(g3, NcI*NcJ*dE*dE);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  PetscCheckFalse(qNc != 1,PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components", qNc);
  for (e = 0; e < Ne; ++e) {
    PetscFEGeom fegeom;

    fegeom.dim      = cgeom->dim;
    fegeom.dimEmbed = cgeom->dimEmbed;
    if (isAffine) {
      fegeom.v    = x;
      fegeom.xi   = cgeom->xi;
      fegeom.J    = &cgeom->J[e*Np*dE*dE];
      fegeom.invJ = &cgeom->invJ[e*Np*dE*dE];
      fegeom.detJ = &cgeom->detJ[e*Np];
    }
    for (q = 0; q < Nq; ++q) {
      PetscReal w;
      PetscInt  c;

      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      if (isAffine) {
        CoordinatesRefToReal(dE, dim, fegeom.xi, &cgeom->v[e*Np*dE], fegeom.J, &quadPoints[q*dim], x);
      } else {
        fegeom.v    = &cgeom->v[(e*Np+q)*dE];
        fegeom.J    = &cgeom->J[(e*Np+q)*dE*dE];
        fegeom.invJ = &cgeom->invJ[(e*Np+q)*dE*dE];
        fegeom.detJ = &cgeom->detJ[e*Np+q];
      }
      w = fegeom.detJ[0]*quadWeights[q];
      if (coefficients) {ierr = PetscFEEvaluateFieldJets_Internal(ds, Nf, 0, q, T, &fegeom, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);CHKERRQ(ierr);}
      if (dsAux)        {ierr = PetscFEEvaluateFieldJets_Internal(dsAux, NfAux, 0, q, TAux, &fegeom, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);CHKERRQ(ierr);}
      if (n0) {
        ierr = PetscArrayzero(g0, NcI*NcJ);CHKERRQ(ierr);
        for (i = 0; i < n0; ++i) g0_func[i](dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, fegeom.v, numConstants, constants, g0);
        for (c = 0; c < NcI*NcJ; ++c) g0[c] *= w;
      }
      if (n1) {
        ierr = PetscArrayzero(g1, NcI*NcJ*dE);CHKERRQ(ierr);
        for (i = 0; i < n1; ++i) g1_func[i](dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, fegeom.v, numConstants, constants, g1);
        for (c = 0; c < NcI*NcJ*dim; ++c) g1[c] *= w;
      }
      if (n2) {
        ierr = PetscArrayzero(g2, NcI*NcJ*dE);CHKERRQ(ierr);
        for (i = 0; i < n2; ++i) g2_func[i](dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, fegeom.v, numConstants, constants, g2);
        for (c = 0; c < NcI*NcJ*dim; ++c) g2[c] *= w;
      }
      if (n3) {
        ierr = PetscArrayzero(g3, NcI*NcJ*dE*dE);CHKERRQ(ierr);
        for (i = 0; i < n3; ++i) g3_func[i](dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, fegeom.v, numConstants, constants, g3);
        for (c = 0; c < NcI*NcJ*dim*dim; ++c) g3[c] *= w;
      }

      ierr = PetscFEUpdateElementMat_Internal(feI, feJ, 0, q, T[fieldI], basisReal, basisDerReal, T[fieldJ], testReal, testDerReal, &fegeom, g0, g1, g2, g3, eOffset, totDim, offsetI, offsetJ, elemMat);CHKERRQ(ierr);
    }
    if (debug > 1) {
      PetscInt fc, f, gc, g;

      ierr = PetscPrintf(PETSC_COMM_SELF, "Element matrix for fields %d and %d\n", fieldI, fieldJ);CHKERRQ(ierr);
      for (fc = 0; fc < T[fieldI]->Nc; ++fc) {
        for (f = 0; f < T[fieldI]->Nb; ++f) {
          const PetscInt i = offsetI + f*T[fieldI]->Nc+fc;
          for (gc = 0; gc < T[fieldJ]->Nc; ++gc) {
            for (g = 0; g < T[fieldJ]->Nb; ++g) {
              const PetscInt j = offsetJ + g*T[fieldJ]->Nc+gc;
              ierr = PetscPrintf(PETSC_COMM_SELF, "    elemMat[%d,%d,%d,%d]: %g\n", f, fc, g, gc, PetscRealPart(elemMat[eOffset+i*totDim+j]));CHKERRQ(ierr);
            }
          }
          ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
        }
      }
    }
    cOffset    += totDim;
    cOffsetAux += totDimAux;
    eOffset    += PetscSqr(totDim);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFEIntegrateBdJacobian_Basic(PetscDS ds, PetscWeakForm wf, PetscFormKey key, PetscInt Ne, PetscFEGeom *fgeom,
                                                       const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS dsAux, const PetscScalar coefficientsAux[], PetscReal t, PetscReal u_tshift, PetscScalar elemMat[])
{
  const PetscInt     debug      = 0;
  PetscFE            feI, feJ;
  PetscBdPointJac   *g0_func, *g1_func, *g2_func, *g3_func;
  PetscInt           n0,       n1,       n2,       n3, i;
  PetscInt           cOffset    = 0; /* Offset into coefficients[] for element e */
  PetscInt           cOffsetAux = 0; /* Offset into coefficientsAux[] for element e */
  PetscInt           eOffset    = 0; /* Offset into elemMat[] for element e */
  PetscInt           offsetI    = 0; /* Offset into an element vector for fieldI */
  PetscInt           offsetJ    = 0; /* Offset into an element vector for fieldJ */
  PetscQuadrature    quad;
  PetscTabulation   *T, *TAux = NULL;
  PetscScalar       *g0, *g1, *g2, *g3, *u, *u_t = NULL, *u_x, *a, *a_x, *basisReal, *basisDerReal, *testReal, *testDerReal;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL;
  PetscInt           NcI = 0, NcJ = 0;
  PetscInt           dim, numConstants, Nf, fieldI, fieldJ, NfAux = 0, totDim, totDimAux = 0, e;
  PetscBool          isAffine;
  const PetscReal   *quadPoints, *quadWeights;
  PetscInt           qNc, Nq, q, Np, dE;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  fieldI = key.field / Nf;
  fieldJ = key.field % Nf;
  ierr = PetscDSGetDiscretization(ds, fieldI, (PetscObject *) &feI);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(ds, fieldJ, (PetscObject *) &feJ);CHKERRQ(ierr);
  ierr = PetscFEGetSpatialDimension(feI, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetFaceQuadrature(feI, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(ds, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(ds, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(ds, fieldI, &offsetI);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(ds, fieldJ, &offsetJ);CHKERRQ(ierr);
  ierr = PetscWeakFormGetBdJacobian(wf, key.label, key.value, fieldI, fieldJ, key.part, &n0, &g0_func, &n1, &g1_func, &n2, &g2_func, &n3, &g3_func);CHKERRQ(ierr);
  if (!n0 && !n1 && !n2 && !n3) PetscFunctionReturn(0);
  ierr = PetscDSGetEvaluationArrays(ds, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetWorkspace(ds, &x, &basisReal, &basisDerReal, &testReal, &testDerReal);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(ds, NULL, NULL, &g0, &g1, &g2, &g3);CHKERRQ(ierr);
  ierr = PetscDSGetFaceTabulation(ds, &T);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(ds, &numConstants, &constants);CHKERRQ(ierr);
  if (dsAux) {
    ierr = PetscDSGetNumFields(dsAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(dsAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(dsAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(dsAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(dsAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetFaceTabulation(dsAux, &TAux);CHKERRQ(ierr);
  }
  NcI = T[fieldI]->Nc, NcJ = T[fieldJ]->Nc;
  Np = fgeom->numPoints;
  dE = fgeom->dimEmbed;
  isAffine = fgeom->isAffine;
  /* Initialize here in case the function is not defined */
  ierr = PetscArrayzero(g0, NcI*NcJ);CHKERRQ(ierr);
  ierr = PetscArrayzero(g1, NcI*NcJ*dE);CHKERRQ(ierr);
  ierr = PetscArrayzero(g2, NcI*NcJ*dE);CHKERRQ(ierr);
  ierr = PetscArrayzero(g3, NcI*NcJ*dE*dE);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  PetscCheckFalse(qNc != 1,PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components", qNc);
  for (e = 0; e < Ne; ++e) {
    PetscFEGeom    fegeom, cgeom;
    const PetscInt face = fgeom->face[e][0];
    fegeom.n = NULL;
    fegeom.v = NULL;
    fegeom.J = NULL;
    fegeom.detJ = NULL;
    fegeom.dim      = fgeom->dim;
    fegeom.dimEmbed = fgeom->dimEmbed;
    cgeom.dim       = fgeom->dim;
    cgeom.dimEmbed  = fgeom->dimEmbed;
    if (isAffine) {
      fegeom.v    = x;
      fegeom.xi   = fgeom->xi;
      fegeom.J    = &fgeom->J[e*Np*dE*dE];
      fegeom.invJ = &fgeom->invJ[e*Np*dE*dE];
      fegeom.detJ = &fgeom->detJ[e*Np];
      fegeom.n    = &fgeom->n[e*Np*dE];

      cgeom.J     = &fgeom->suppJ[0][e*Np*dE*dE];
      cgeom.invJ  = &fgeom->suppInvJ[0][e*Np*dE*dE];
      cgeom.detJ  = &fgeom->suppDetJ[0][e*Np];
    }
    for (q = 0; q < Nq; ++q) {
      PetscReal w;
      PetscInt  c;

      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      if (isAffine) {
        CoordinatesRefToReal(dE, dim-1, fegeom.xi, &fgeom->v[e*Np*dE], fegeom.J, &quadPoints[q*(dim-1)], x);
      } else {
        fegeom.v    = &fgeom->v[(e*Np+q)*dE];
        fegeom.J    = &fgeom->J[(e*Np+q)*dE*dE];
        fegeom.invJ = &fgeom->invJ[(e*Np+q)*dE*dE];
        fegeom.detJ = &fgeom->detJ[e*Np+q];
        fegeom.n    = &fgeom->n[(e*Np+q)*dE];

        cgeom.J     = &fgeom->suppJ[0][(e*Np+q)*dE*dE];
        cgeom.invJ  = &fgeom->suppInvJ[0][(e*Np+q)*dE*dE];
        cgeom.detJ  = &fgeom->suppDetJ[0][e*Np+q];
      }
      w = fegeom.detJ[0]*quadWeights[q];
      if (coefficients) {ierr = PetscFEEvaluateFieldJets_Internal(ds, Nf, face, q, T, &cgeom, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);CHKERRQ(ierr);}
      if (dsAux)        {ierr = PetscFEEvaluateFieldJets_Internal(dsAux, NfAux, face, q, TAux, &cgeom, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);CHKERRQ(ierr);}
      if (n0) {
        ierr = PetscArrayzero(g0, NcI*NcJ);CHKERRQ(ierr);
        for (i = 0; i < n0; ++i) g0_func[i](dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, fegeom.v, fegeom.n, numConstants, constants, g0);
        for (c = 0; c < NcI*NcJ; ++c) g0[c] *= w;
      }
      if (n1) {
        ierr = PetscArrayzero(g1, NcI*NcJ*dE);CHKERRQ(ierr);
        for (i = 0; i < n1; ++i) g1_func[i](dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, fegeom.v, fegeom.n, numConstants, constants, g1);
        for (c = 0; c < NcI*NcJ*dim; ++c) g1[c] *= w;
      }
      if (n2) {
        ierr = PetscArrayzero(g2, NcI*NcJ*dE);CHKERRQ(ierr);
        for (i = 0; i < n2; ++i) g2_func[i](dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, fegeom.v, fegeom.n, numConstants, constants, g2);
        for (c = 0; c < NcI*NcJ*dim; ++c) g2[c] *= w;
      }
      if (n3) {
        ierr = PetscArrayzero(g3, NcI*NcJ*dE*dE);CHKERRQ(ierr);
        for (i = 0; i < n3; ++i) g3_func[i](dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, fegeom.v, fegeom.n, numConstants, constants, g3);
        for (c = 0; c < NcI*NcJ*dim*dim; ++c) g3[c] *= w;
      }

      ierr = PetscFEUpdateElementMat_Internal(feI, feJ, face, q, T[fieldI], basisReal, basisDerReal, T[fieldJ], testReal, testDerReal, &cgeom, g0, g1, g2, g3, eOffset, totDim, offsetI, offsetJ, elemMat);CHKERRQ(ierr);
    }
    if (debug > 1) {
      PetscInt fc, f, gc, g;

      ierr = PetscPrintf(PETSC_COMM_SELF, "Element matrix for fields %d and %d\n", fieldI, fieldJ);CHKERRQ(ierr);
      for (fc = 0; fc < T[fieldI]->Nc; ++fc) {
        for (f = 0; f < T[fieldI]->Nb; ++f) {
          const PetscInt i = offsetI + f*T[fieldI]->Nc+fc;
          for (gc = 0; gc < T[fieldJ]->Nc; ++gc) {
            for (g = 0; g < T[fieldJ]->Nb; ++g) {
              const PetscInt j = offsetJ + g*T[fieldJ]->Nc+gc;
              ierr = PetscPrintf(PETSC_COMM_SELF, "    elemMat[%d,%d,%d,%d]: %g\n", f, fc, g, gc, PetscRealPart(elemMat[eOffset+i*totDim+j]));CHKERRQ(ierr);
            }
          }
          ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
        }
      }
    }
    cOffset    += totDim;
    cOffsetAux += totDimAux;
    eOffset    += PetscSqr(totDim);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEIntegrateHybridJacobian_Basic(PetscDS ds, PetscFEJacobianType jtype, PetscFormKey key, PetscInt s, PetscInt Ne, PetscFEGeom *fgeom,
                                              const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS dsAux, const PetscScalar coefficientsAux[], PetscReal t, PetscReal u_tshift, PetscScalar elemMat[])
{
  const PetscInt     debug      = 0;
  PetscFE            feI, feJ;
  PetscWeakForm      wf;
  PetscBdPointJac   *g0_func, *g1_func, *g2_func, *g3_func;
  PetscInt           n0,       n1,       n2,       n3, i;
  PetscInt           cOffset    = 0; /* Offset into coefficients[] for element e */
  PetscInt           cOffsetAux = 0; /* Offset into coefficientsAux[] for element e */
  PetscInt           eOffset    = 0; /* Offset into elemMat[] for element e */
  PetscInt           offsetI    = 0; /* Offset into an element vector for fieldI */
  PetscInt           offsetJ    = 0; /* Offset into an element vector for fieldJ */
  PetscQuadrature    quad;
  PetscTabulation   *T, *TAux = NULL;
  PetscScalar       *g0, *g1, *g2, *g3, *u, *u_t = NULL, *u_x, *a, *a_x, *basisReal, *basisDerReal, *testReal, *testDerReal;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL;
  PetscInt           NcI = 0, NcJ = 0, NcS, NcT;
  PetscInt           dim, dimAux, numConstants, Nf, fieldI, fieldJ, NfAux = 0, totDim, totDimAux = 0, e;
  PetscBool          isCohesiveFieldI, isCohesiveFieldJ, isAffine, auxOnBd = PETSC_FALSE;
  const PetscReal   *quadPoints, *quadWeights;
  PetscInt           qNc, Nq, q, Np, dE;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  fieldI = key.field / Nf;
  fieldJ = key.field % Nf;
  /* Hybrid discretization is posed directly on faces */
  ierr = PetscDSGetDiscretization(ds, fieldI, (PetscObject *) &feI);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(ds, fieldJ, (PetscObject *) &feJ);CHKERRQ(ierr);
  ierr = PetscFEGetSpatialDimension(feI, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(feI, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsetsCohesive(ds, s, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsetsCohesive(ds, s, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetWeakForm(ds, &wf);CHKERRQ(ierr);
  switch(jtype) {
  case PETSCFE_JACOBIAN_PRE: ierr = PetscWeakFormGetBdJacobianPreconditioner(wf, key.label, key.value, fieldI, fieldJ, key.part, &n0, &g0_func, &n1, &g1_func, &n2, &g2_func, &n3, &g3_func);CHKERRQ(ierr);break;
  case PETSCFE_JACOBIAN:     ierr = PetscWeakFormGetBdJacobian(wf, key.label, key.value, fieldI, fieldJ, key.part, &n0, &g0_func, &n1, &g1_func, &n2, &g2_func, &n3, &g3_func);CHKERRQ(ierr);break;
  case PETSCFE_JACOBIAN_DYN: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No boundary hybrid Jacobians :)");
  }
  if (!n0 && !n1 && !n2 && !n3) PetscFunctionReturn(0);
  ierr = PetscDSGetEvaluationArrays(ds, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetWorkspace(ds, &x, &basisReal, &basisDerReal, &testReal, &testDerReal);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(ds, NULL, NULL, &g0, &g1, &g2, &g3);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(ds, &T);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffsetCohesive(ds, fieldI, &offsetI);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffsetCohesive(ds, fieldJ, &offsetJ);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(ds, &numConstants, &constants);CHKERRQ(ierr);
  if (dsAux) {
    ierr = PetscDSGetSpatialDimension(dsAux, &dimAux);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(dsAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(dsAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(dsAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(dsAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(dsAux, &a, NULL, &a_x);CHKERRQ(ierr);
    auxOnBd = dimAux == dim ? PETSC_TRUE : PETSC_FALSE;
    if (auxOnBd) {ierr = PetscDSGetTabulation(dsAux, &TAux);CHKERRQ(ierr);}
    else         {ierr = PetscDSGetFaceTabulation(dsAux, &TAux);CHKERRQ(ierr);}
    PetscCheckFalse(T[0]->Np != TAux[0]->Np,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of tabulation points %D != %D number of auxiliary tabulation points", T[0]->Np, TAux[0]->Np);
  }
  ierr = PetscDSGetCohesive(ds, fieldI, &isCohesiveFieldI);CHKERRQ(ierr);
  ierr = PetscDSGetCohesive(ds, fieldJ, &isCohesiveFieldJ);CHKERRQ(ierr);
  NcI = T[fieldI]->Nc;
  NcJ = T[fieldJ]->Nc;
  NcS = isCohesiveFieldI ? NcI : 2*NcI;
  NcT = isCohesiveFieldJ ? NcJ : 2*NcJ;
  Np = fgeom->numPoints;
  dE = fgeom->dimEmbed;
  isAffine = fgeom->isAffine;
  ierr = PetscArrayzero(g0, NcS*NcT);CHKERRQ(ierr);
  ierr = PetscArrayzero(g1, NcS*NcT*dE);CHKERRQ(ierr);
  ierr = PetscArrayzero(g2, NcS*NcT*dE);CHKERRQ(ierr);
  ierr = PetscArrayzero(g3, NcS*NcT*dE*dE);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  PetscCheckFalse(qNc != 1,PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components", qNc);
  for (e = 0; e < Ne; ++e) {
    PetscFEGeom    fegeom;
    const PetscInt face = fgeom->face[e][0];

    fegeom.dim      = fgeom->dim;
    fegeom.dimEmbed = fgeom->dimEmbed;
    if (isAffine) {
      fegeom.v    = x;
      fegeom.xi   = fgeom->xi;
      fegeom.J    = &fgeom->J[e*dE*dE];
      fegeom.invJ = &fgeom->invJ[e*dE*dE];
      fegeom.detJ = &fgeom->detJ[e];
      fegeom.n    = &fgeom->n[e*dE];
    }
    for (q = 0; q < Nq; ++q) {
      PetscReal w;
      PetscInt  c;

      if (isAffine) {
        /* TODO Is it correct to have 'dim' here, or should it be 'dim-1'? */
        CoordinatesRefToReal(dE, dim, fegeom.xi, &fgeom->v[e*dE], fegeom.J, &quadPoints[q*dim], x);
      } else {
        fegeom.v    = &fegeom.v[(e*Np+q)*dE];
        fegeom.J    = &fgeom->J[(e*Np+q)*dE*dE];
        fegeom.invJ = &fgeom->invJ[(e*Np+q)*dE*dE];
        fegeom.detJ = &fgeom->detJ[e*Np+q];
        fegeom.n    = &fgeom->n[(e*Np+q)*dE];
      }
      w = fegeom.detJ[0]*quadWeights[q];
      if (debug > 1 && q < Np) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", fegeom.detJ[0]);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
        ierr = DMPrintCellMatrix(e, "invJ", dim, dim, fegeom.invJ);CHKERRQ(ierr);
#endif
      }
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      if (coefficients) {ierr = PetscFEEvaluateFieldJets_Hybrid_Internal(ds, Nf, 0, q, T, &fegeom, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);CHKERRQ(ierr);}
      if (dsAux) {ierr = PetscFEEvaluateFieldJets_Internal(dsAux, NfAux, 0, auxOnBd ? q : face*Nq+q, TAux, &fegeom, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);CHKERRQ(ierr);}
      if (n0) {
        ierr = PetscArrayzero(g0, NcS*NcT);CHKERRQ(ierr);
        for (i = 0; i < n0; ++i) g0_func[i](dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, fegeom.v, fegeom.n, numConstants, constants, g0);
        for (c = 0; c < NcS*NcT; ++c) g0[c] *= w;
      }
      if (n1) {
        ierr = PetscArrayzero(g1, NcS*NcT*dE);CHKERRQ(ierr);
        for (i = 0; i < n1; ++i) g1_func[i](dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, fegeom.v, fegeom.n, numConstants, constants, g1);
        for (c = 0; c < NcS*NcT*dE; ++c) g1[c] *= w;
      }
      if (n2) {
        ierr = PetscArrayzero(g2, NcS*NcT*dE);CHKERRQ(ierr);
        for (i = 0; i < n2; ++i) g2_func[i](dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, fegeom.v, fegeom.n, numConstants, constants, g2);
        for (c = 0; c < NcS*NcT*dE; ++c) g2[c] *= w;
      }
      if (n3) {
        ierr = PetscArrayzero(g3, NcS*NcT*dE*dE);CHKERRQ(ierr);
        for (i = 0; i < n3; ++i) g3_func[i](dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, fegeom.v, fegeom.n, numConstants, constants, g3);
        for (c = 0; c < NcS*NcT*dE*dE; ++c) g3[c] *= w;
      }

      if (isCohesiveFieldI) {
        if (isCohesiveFieldJ) {
          ierr = PetscFEUpdateElementMat_Internal(feI, feJ, 0, q, T[fieldI], basisReal, basisDerReal, T[fieldJ], testReal, testDerReal, &fegeom, g0, g1, g2, g3, eOffset, totDim, offsetI, offsetJ, elemMat);CHKERRQ(ierr);
        } else {
          ierr = PetscFEUpdateElementMat_Hybrid_Internal(feI, isCohesiveFieldI, feJ, isCohesiveFieldJ, 0, 0, q, T[fieldI], basisReal, basisDerReal, T[fieldJ], testReal, testDerReal, &fegeom, g0, g1, g2, g3, eOffset, totDim, offsetI, offsetJ, elemMat);CHKERRQ(ierr);
          ierr = PetscFEUpdateElementMat_Hybrid_Internal(feI, isCohesiveFieldI, feJ, isCohesiveFieldJ, 0, 1, q, T[fieldI], basisReal, basisDerReal, T[fieldJ], testReal, testDerReal, &fegeom, &g0[NcI*NcJ], &g1[NcI*NcJ*dim], &g2[NcI*NcJ*dim], &g3[NcI*NcJ*dim*dim], eOffset, totDim, offsetI, offsetJ, elemMat);CHKERRQ(ierr);
        }
      } else {
        ierr = PetscFEUpdateElementMat_Hybrid_Internal(feI, isCohesiveFieldI, feJ, isCohesiveFieldJ, 0, s, q, T[fieldI], basisReal, basisDerReal, T[fieldJ], testReal, testDerReal, &fegeom, g0, g1, g2, g3, eOffset, totDim, offsetI, offsetJ, elemMat);CHKERRQ(ierr);
      }
    }
    if (debug > 1) {
      PetscInt fc, f, gc, g;

      ierr = PetscPrintf(PETSC_COMM_SELF, "Element matrix for fields %d and %d\n", fieldI, fieldJ);CHKERRQ(ierr);
      for (fc = 0; fc < NcI; ++fc) {
        for (f = 0; f < T[fieldI]->Nb; ++f) {
          const PetscInt i = offsetI + f*NcI+fc;
          for (gc = 0; gc < NcJ; ++gc) {
            for (g = 0; g < T[fieldJ]->Nb; ++g) {
              const PetscInt j = offsetJ + g*NcJ+gc;
              ierr = PetscPrintf(PETSC_COMM_SELF, "    elemMat[%d,%d,%d,%d]: %g\n", f, fc, g, gc, PetscRealPart(elemMat[eOffset+i*totDim+j]));CHKERRQ(ierr);
            }
          }
          ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
        }
      }
    }
    cOffset    += totDim;
    cOffsetAux += totDimAux;
    eOffset    += PetscSqr(totDim);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFEInitialize_Basic(PetscFE fem)
{
  PetscFunctionBegin;
  fem->ops->setfromoptions          = NULL;
  fem->ops->setup                   = PetscFESetUp_Basic;
  fem->ops->view                    = PetscFEView_Basic;
  fem->ops->destroy                 = PetscFEDestroy_Basic;
  fem->ops->getdimension            = PetscFEGetDimension_Basic;
  fem->ops->createtabulation        = PetscFECreateTabulation_Basic;
  fem->ops->integrate               = PetscFEIntegrate_Basic;
  fem->ops->integratebd             = PetscFEIntegrateBd_Basic;
  fem->ops->integrateresidual       = PetscFEIntegrateResidual_Basic;
  fem->ops->integratebdresidual     = PetscFEIntegrateBdResidual_Basic;
  fem->ops->integratehybridresidual = PetscFEIntegrateHybridResidual_Basic;
  fem->ops->integratejacobianaction = NULL/* PetscFEIntegrateJacobianAction_Basic */;
  fem->ops->integratejacobian       = PetscFEIntegrateJacobian_Basic;
  fem->ops->integratebdjacobian     = PetscFEIntegrateBdJacobian_Basic;
  fem->ops->integratehybridjacobian = PetscFEIntegrateHybridJacobian_Basic;
  PetscFunctionReturn(0);
}

/*MC
  PETSCFEBASIC = "basic" - A PetscFE object that integrates with basic tiling and no vectorization

  Level: intermediate

.seealso: PetscFEType, PetscFECreate(), PetscFESetType()
M*/

PETSC_EXTERN PetscErrorCode PetscFECreate_Basic(PetscFE fem)
{
  PetscFE_Basic *b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fem, PETSCFE_CLASSID, 1);
  ierr      = PetscNewLog(fem,&b);CHKERRQ(ierr);
  fem->data = b;

  ierr = PetscFEInitialize_Basic(fem);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
