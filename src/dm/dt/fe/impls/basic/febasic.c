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
PetscErrorCode PetscFESetUp_Basic(PetscFE fem)
{
  PetscScalar   *work, *invVscalar;
  PetscBLASInt  *pivots;
  PetscBLASInt   n, info;
  PetscInt       pdim, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDimension(fem->dualSpace, &pdim);CHKERRQ(ierr);
  ierr = PetscMalloc1(pdim*pdim,&fem->invV);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc1(pdim*pdim,&invVscalar);CHKERRQ(ierr);
#else
  invVscalar = fem->invV;
#endif
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
      invVscalar[j*pdim+k] = 0.0;

      for (q = 0; q < Nq; ++q) {
        for (c = 0; c < Nc; ++c) invVscalar[j*pdim+k] += Bf[(q*pdim + k)*Nc + c]*weights[q*Nc + c];
      }
    }
    ierr = PetscFree(Bf);CHKERRQ(ierr);
  }

  ierr = PetscMalloc2(pdim,&pivots,pdim,&work);CHKERRQ(ierr);
  n = pdim;
  PetscStackCallBLAS("LAPACKgetrf", LAPACKgetrf_(&n, &n, invVscalar, &n, pivots, &info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error returned from LAPACKgetrf %D",(PetscInt)info);
  PetscStackCallBLAS("LAPACKgetri", LAPACKgetri_(&n, invVscalar, &n, pivots, work, &pdim, &info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error returned from LAPACKgetri %D",(PetscInt)info);
#if defined(PETSC_USE_COMPLEX)
  for (j = 0; j < pdim*pdim; j++) fem->invV[j] = PetscRealPart(invVscalar[j]);
  ierr = PetscFree(invVscalar);CHKERRQ(ierr);
#endif
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

PetscErrorCode PetscFEGetTabulation_Basic(PetscFE fem, PetscInt npoints, const PetscReal points[], PetscReal *B, PetscReal *D, PetscReal *H)
{
  DM               dm;
  PetscInt         pdim; /* Dimension of FE space P */
  PetscInt         dim;  /* Spatial dimension */
  PetscInt         Nc;   /* Field components */
  PetscReal       *tmpB, *tmpD, *tmpH;
  PetscInt         p, d, j, k, c;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceGetDM(fem->dualSpace, &dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDimension(fem->dualSpace, &pdim);CHKERRQ(ierr);
  ierr = PetscFEGetNumComponents(fem, &Nc);CHKERRQ(ierr);
  /* Evaluate the prime basis functions at all points */
  if (B) {ierr = DMGetWorkArray(dm, npoints*pdim*Nc, MPIU_REAL, &tmpB);CHKERRQ(ierr);}
  if (D) {ierr = DMGetWorkArray(dm, npoints*pdim*Nc*dim, MPIU_REAL, &tmpD);CHKERRQ(ierr);}
  if (H) {ierr = DMGetWorkArray(dm, npoints*pdim*Nc*dim*dim, MPIU_REAL, &tmpH);CHKERRQ(ierr);}
  ierr = PetscSpaceEvaluate(fem->basisSpace, npoints, points, B ? tmpB : NULL, D ? tmpD : NULL, H ? tmpH : NULL);CHKERRQ(ierr);
  /* Translate to the nodal basis */
  for (p = 0; p < npoints; ++p) {
    if (B) {
      /* Multiply by V^{-1} (pdim x pdim) */
      for (j = 0; j < pdim; ++j) {
        const PetscInt i = (p*pdim + j)*Nc;

        for (c = 0; c < Nc; ++c) {
          B[i+c] = 0.0;
          for (k = 0; k < pdim; ++k) {
            B[i+c] += fem->invV[k*pdim+j] * tmpB[(p*pdim + k)*Nc+c];
          }
        }
      }
    }
    if (D) {
      /* Multiply by V^{-1} (pdim x pdim) */
      for (j = 0; j < pdim; ++j) {
        for (c = 0; c < Nc; ++c) {
          for (d = 0; d < dim; ++d) {
            const PetscInt i = ((p*pdim + j)*Nc + c)*dim + d;

            D[i] = 0.0;
            for (k = 0; k < pdim; ++k) {
              D[i] += fem->invV[k*pdim+j] * tmpD[((p*pdim + k)*Nc + c)*dim + d];
            }
          }
        }
      }
    }
    if (H) {
      /* Multiply by V^{-1} (pdim x pdim) */
      for (j = 0; j < pdim; ++j) {
        for (c = 0; c < Nc; ++c) {
          for (d = 0; d < dim*dim; ++d) {
            const PetscInt i = ((p*pdim + j)*Nc + c)*dim*dim + d;

            H[i] = 0.0;
            for (k = 0; k < pdim; ++k) {
              H[i] += fem->invV[k*pdim+j] * tmpH[((p*pdim + k)*Nc + c)*dim*dim + d];
            }
          }
        }
      }
    }
  }
  if (B) {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nc, MPIU_REAL, &tmpB);CHKERRQ(ierr);}
  if (D) {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nc*dim, MPIU_REAL, &tmpD);CHKERRQ(ierr);}
  if (H) {ierr = DMRestoreWorkArray(dm, npoints*pdim*Nc*dim*dim, MPIU_REAL, &tmpH);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscFEIntegrate_Basic(PetscDS ds, PetscInt field, PetscInt Ne, PetscFEGeom *cgeom,
                                             const PetscScalar coefficients[], PetscDS dsAux, const PetscScalar coefficientsAux[], PetscScalar integral[])
{
  const PetscInt     debug = 0;
  PetscFE            fe;
  PetscPointFunc     obj_func;
  PetscQuadrature    quad;
  PetscScalar       *u, *u_x, *a, *a_x;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscReal        **B, **D, **BAux = NULL, **DAux = NULL;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL, *Nb, *Nc, *NbAux = NULL, *NcAux = NULL;
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
  ierr = PetscDSGetDimensions(ds, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(ds, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(ds, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(ds, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(ds, &u, NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetWorkspace(ds, &x, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(ds, &B, &D);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(ds, &numConstants, &constants);CHKERRQ(ierr);
  if (dsAux) {
    ierr = PetscDSGetNumFields(dsAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(dsAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(dsAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(dsAux, &NcAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(dsAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(dsAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(dsAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetTabulation(dsAux, &BAux, &DAux);CHKERRQ(ierr);
  }
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
  Np = cgeom->numPoints;
  dE = cgeom->dimEmbed;
  isAffine = cgeom->isAffine;
  for (e = 0; e < Ne; ++e) {
    PetscFEGeom fegeom;

    if (isAffine) {
      fegeom.v    = x;
      fegeom.xi   = cgeom->xi;
      fegeom.J    = &cgeom->J[e*dE*dE];
      fegeom.invJ = &cgeom->invJ[e*dE*dE];
      fegeom.detJ = &cgeom->detJ[e];
    }
    for (q = 0; q < Nq; ++q) {
      PetscScalar integrand;
      PetscReal   w;

      if (isAffine) {
        CoordinatesRefToReal(dE, dim, fegeom.xi, &cgeom->v[e*dE], fegeom.J, &quadPoints[q*dim], x);
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
      ierr = PetscFEEvaluateFieldJets_Internal(ds, dim, Nf, Nb, Nc, q, B, D, &fegeom, &coefficients[cOffset], NULL, u, u_x, NULL);CHKERRQ(ierr);
      if (dsAux) {ierr = PetscFEEvaluateFieldJets_Internal(dsAux, dim, NfAux, NbAux, NcAux, q, BAux, DAux, &fegeom, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);CHKERRQ(ierr);}
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
  PetscScalar       *u, *u_x, *a, *a_x, *basisReal, *basisDerReal;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscReal        **B, **D, **BAux = NULL, **DAux = NULL;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL, *Nb, *Nc, *NbAux = NULL, *NcAux = NULL;
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
  ierr = PetscDSGetDimensions(ds, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(ds, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(ds, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(ds, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(ds, &u, NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetWorkspace(ds, &x, &basisReal, &basisDerReal, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetFaceTabulation(ds, &B, &D);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(ds, &numConstants, &constants);CHKERRQ(ierr);
  if (dsAux) {
    ierr = PetscDSGetSpatialDimension(dsAux, &dimAux);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(dsAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(dsAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(dsAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(dsAux, &NcAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(dsAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(dsAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(dsAux, &a, NULL, &a_x);CHKERRQ(ierr);
    auxOnBd = dimAux < dim ? PETSC_TRUE : PETSC_FALSE;
    if (auxOnBd) {ierr = PetscDSGetTabulation(dsAux, &BAux, &DAux);CHKERRQ(ierr);}
    else         {ierr = PetscDSGetFaceTabulation(dsAux, &BAux, &DAux);CHKERRQ(ierr);}
  }
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
  Np = fgeom->numPoints;
  dE = fgeom->dimEmbed;
  isAffine = fgeom->isAffine;
  for (e = 0; e < Ne; ++e) {
    PetscFEGeom    fegeom, cgeom;
    const PetscInt face = fgeom->face[e][0]; /* Local face number in cell */
    fegeom.n = 0;
    fegeom.v = 0;
    fegeom.J = 0;
    fegeom.detJ = 0;
    if (isAffine) {
      fegeom.v    = x;
      fegeom.xi   = fgeom->xi;
      fegeom.J    = &fgeom->J[e*dE*dE];
      fegeom.invJ = &fgeom->invJ[e*dE*dE];
      fegeom.detJ = &fgeom->detJ[e];
      fegeom.n    = &fgeom->n[e*dE];

      cgeom.J     = &fgeom->suppJ[0][e*dE*dE];
      cgeom.invJ  = &fgeom->suppInvJ[0][e*dE*dE];
      cgeom.detJ  = &fgeom->suppDetJ[0][e];
    }
    for (q = 0; q < Nq; ++q) {
      PetscScalar integrand;
      PetscReal   w;

      if (isAffine) {
        CoordinatesRefToReal(dE, dim-1, fegeom.xi, &fgeom->v[e*dE], fegeom.J, &quadPoints[q*(dim-1)], x);
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
      ierr = PetscFEEvaluateFieldJets_Internal(ds, dim, Nf, Nb, Nc, face*Nq+q, B, D, &cgeom, &coefficients[cOffset], NULL, u, u_x, NULL);CHKERRQ(ierr);
      if (dsAux) {ierr = PetscFEEvaluateFieldJets_Internal(dsAux, dimAux, NfAux, NbAux, NcAux, face*Nq+q, BAux, DAux, &cgeom, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);CHKERRQ(ierr);}
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

PetscErrorCode PetscFEIntegrateResidual_Basic(PetscDS ds, PetscInt field, PetscInt Ne, PetscFEGeom *cgeom,
                                              const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS dsAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  const PetscInt     debug = 0;
  PetscFE            fe;
  PetscPointFunc     f0_func;
  PetscPointFunc     f1_func;
  PetscQuadrature    quad;
  PetscScalar       *f0, *f1, *u, *u_t = NULL, *u_x, *a, *a_x, *basisReal, *basisDerReal;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscReal        **B, **D, **BAux = NULL, **DAux = NULL, *BI, *DI;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL, *Nb, *Nc, *NbAux = NULL, *NcAux = NULL;
  PetscInt           dim, numConstants, Nf, NfAux = 0, totDim, totDimAux = 0, cOffset = 0, cOffsetAux = 0, fOffset, e, NbI, NcI;
  PetscBool          isAffine;
  const PetscReal   *quadPoints, *quadWeights;
  PetscInt           qNc, Nq, q, Np, dE;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDSGetDiscretization(ds, field, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetSpatialDimension(fe, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(ds, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(ds, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(ds, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(ds, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(ds, field, &fOffset);CHKERRQ(ierr);
  ierr = PetscDSGetResidual(ds, field, &f0_func, &f1_func);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(ds, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetWorkspace(ds, &x, &basisReal, &basisDerReal, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(ds, &f0, &f1, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  if (!f0_func && !f1_func) PetscFunctionReturn(0);
  ierr = PetscDSGetTabulation(ds, &B, &D);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(ds, &numConstants, &constants);CHKERRQ(ierr);
  if (dsAux) {
    ierr = PetscDSGetNumFields(dsAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(dsAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(dsAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(dsAux, &NcAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(dsAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(dsAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(dsAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetTabulation(dsAux, &BAux, &DAux);CHKERRQ(ierr);
  }
  NbI = Nb[field];
  NcI = Nc[field];
  BI  = B[field];
  DI  = D[field];
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
  Np = cgeom->numPoints;
  dE = cgeom->dimEmbed;
  isAffine = cgeom->isAffine;
  for (e = 0; e < Ne; ++e) {
    PetscFEGeom fegeom;

    if (isAffine) {
      fegeom.v    = x;
      fegeom.xi   = cgeom->xi;
      fegeom.J    = &cgeom->J[e*dE*dE];
      fegeom.invJ = &cgeom->invJ[e*dE*dE];
      fegeom.detJ = &cgeom->detJ[e];
    }
    ierr = PetscArrayzero(f0, Nq*NcI);CHKERRQ(ierr);
    ierr = PetscArrayzero(f1, Nq*NcI*dim);CHKERRQ(ierr);
    for (q = 0; q < Nq; ++q) {
      PetscReal w;
      PetscInt  c, d;

      if (isAffine) {
        CoordinatesRefToReal(dE, dim, fegeom.xi, &cgeom->v[e*dE], fegeom.J, &quadPoints[q*dim], x);
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
      ierr = PetscFEEvaluateFieldJets_Internal(ds, dim, Nf, Nb, Nc, q, B, D, &fegeom, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);CHKERRQ(ierr);
      if (dsAux) {ierr = PetscFEEvaluateFieldJets_Internal(dsAux, dim, NfAux, NbAux, NcAux, q, BAux, DAux, &fegeom, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);CHKERRQ(ierr);}
      if (f0_func) {
        f0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, fegeom.v, numConstants, constants, &f0[q*NcI]);
        for (c = 0; c < NcI; ++c) f0[q*NcI+c] *= w;
      }
      if (f1_func) {
        f1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, fegeom.v, numConstants, constants, &f1[q*NcI*dim]);
        for (c = 0; c < NcI; ++c) for (d = 0; d < dim; ++d) f1[(q*NcI+c)*dim+d] *= w;
      }
    }
    ierr = PetscFEUpdateElementVec_Internal(fe, dim, Nq, NbI, NcI, BI, DI, basisReal, basisDerReal, &fegeom, f0, f1, &elemVec[cOffset+fOffset]);CHKERRQ(ierr);
    cOffset    += totDim;
    cOffsetAux += totDimAux;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEIntegrateBdResidual_Basic(PetscDS ds, PetscInt field, PetscInt Ne, PetscFEGeom *fgeom,
                                                const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS dsAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  const PetscInt     debug = 0;
  PetscFE            fe;
  PetscBdPointFunc   f0_func;
  PetscBdPointFunc   f1_func;
  PetscQuadrature    quad;
  PetscScalar       *f0, *f1, *u, *u_t = NULL, *u_x, *a, *a_x, *basisReal, *basisDerReal;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscReal        **B, **D, **BAux = NULL, **DAux = NULL, *BI, *DI;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL, *Nb, *Nc, *NbAux = NULL, *NcAux = NULL;
  PetscInt           dim, dimAux, numConstants, Nf, NfAux = 0, totDim, totDimAux = 0, cOffset = 0, cOffsetAux = 0, fOffset, e, NbI, NcI;
  PetscBool          isAffine, auxOnBd = PETSC_FALSE;
  const PetscReal   *quadPoints, *quadWeights;
  PetscInt           qNc, Nq, q, Np, dE;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDSGetDiscretization(ds, field, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetSpatialDimension(fe, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetFaceQuadrature(fe, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(ds, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(ds, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(ds, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(ds, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(ds, field, &fOffset);CHKERRQ(ierr);
  ierr = PetscDSGetBdResidual(ds, field, &f0_func, &f1_func);CHKERRQ(ierr);
  if (!f0_func && !f1_func) PetscFunctionReturn(0);
  ierr = PetscDSGetEvaluationArrays(ds, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetWorkspace(ds, &x, &basisReal, &basisDerReal, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(ds, &f0, &f1, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetFaceTabulation(ds, &B, &D);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(ds, &numConstants, &constants);CHKERRQ(ierr);
  if (dsAux) {
    ierr = PetscDSGetSpatialDimension(dsAux, &dimAux);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(dsAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(dsAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(dsAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(dsAux, &NcAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(dsAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(dsAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(dsAux, &a, NULL, &a_x);CHKERRQ(ierr);
    auxOnBd = dimAux < dim ? PETSC_TRUE : PETSC_FALSE;
    if (auxOnBd) {ierr = PetscDSGetTabulation(dsAux, &BAux, &DAux);CHKERRQ(ierr);}
    else         {ierr = PetscDSGetFaceTabulation(dsAux, &BAux, &DAux);CHKERRQ(ierr);}
  }
  NbI = Nb[field];
  NcI = Nc[field];
  BI  = B[field];
  DI  = D[field];
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
  Np = fgeom->numPoints;
  dE = fgeom->dimEmbed;
  isAffine = fgeom->isAffine;
  for (e = 0; e < Ne; ++e) {
    PetscFEGeom    fegeom, cgeom;
    const PetscInt face = fgeom->face[e][0];
    fegeom.n = 0;
    fegeom.v = 0;
    fegeom.J = 0;
    fegeom.detJ = 0;
    if (isAffine) {
      fegeom.v    = x;
      fegeom.xi   = fgeom->xi;
      fegeom.J    = &fgeom->J[e*dE*dE];
      fegeom.invJ = &fgeom->invJ[e*dE*dE];
      fegeom.detJ = &fgeom->detJ[e];
      fegeom.n    = &fgeom->n[e*dE];

      cgeom.J     = &fgeom->suppJ[0][e*dE*dE];
      cgeom.invJ  = &fgeom->suppInvJ[0][e*dE*dE];
      cgeom.detJ  = &fgeom->suppDetJ[0][e];
    }
    ierr = PetscArrayzero(f0, Nq*NcI);CHKERRQ(ierr);
    ierr = PetscArrayzero(f1, Nq*NcI*dim);CHKERRQ(ierr);
    for (q = 0; q < Nq; ++q) {
      PetscReal w;
      PetscInt  c, d;

      if (isAffine) {
        CoordinatesRefToReal(dE, dim-1, fegeom.xi, &fgeom->v[e*dE], fegeom.J, &quadPoints[q*(dim-1)], x);
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
#if !defined(PETSC_USE_COMPLEX)
        ierr = DMPrintCellMatrix(e, "invJ", dim, dim, fegeom.invJ);CHKERRQ(ierr);
#endif
      }
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      ierr = PetscFEEvaluateFieldJets_Internal(ds, dim, Nf, Nb, Nc, face*Nq+q, B, D, &cgeom, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);CHKERRQ(ierr);
      if (dsAux) {ierr = PetscFEEvaluateFieldJets_Internal(dsAux, dimAux, NfAux, NbAux, NcAux, auxOnBd ? q : face*Nq+q, BAux, DAux, &cgeom, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);CHKERRQ(ierr);}
      if (f0_func) {
        f0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, fegeom.v, fegeom.n, numConstants, constants, &f0[q*NcI]);
        for (c = 0; c < NcI; ++c) f0[q*NcI+c] *= w;
      }
      if (f1_func) {
        f1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, fegeom.v, fegeom.n, numConstants, constants, &f1[q*NcI*dim]);
        for (c = 0; c < NcI; ++c) for (d = 0; d < dim; ++d) f1[(q*NcI+c)*dim+d] *= w;
      }
    }
    ierr = PetscFEUpdateElementVec_Internal(fe, dim, Nq, NbI, NcI, &BI[face*Nq*NbI*NcI], &DI[face*Nq*NbI*NcI*dim], basisReal, basisDerReal, &cgeom, f0, f1, &elemVec[cOffset+fOffset]);CHKERRQ(ierr);
    cOffset    += totDim;
    cOffsetAux += totDimAux;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEIntegrateJacobian_Basic(PetscDS ds, PetscFEJacobianType jtype, PetscInt fieldI, PetscInt fieldJ, PetscInt Ne, PetscFEGeom *cgeom,
                                              const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS dsAux, const PetscScalar coefficientsAux[], PetscReal t, PetscReal u_tshift, PetscScalar elemMat[])
{
  const PetscInt     debug      = 0;
  PetscFE            feI, feJ;
  PetscPointJac      g0_func, g1_func, g2_func, g3_func;
  PetscInt           cOffset    = 0; /* Offset into coefficients[] for element e */
  PetscInt           cOffsetAux = 0; /* Offset into coefficientsAux[] for element e */
  PetscInt           eOffset    = 0; /* Offset into elemMat[] for element e */
  PetscInt           offsetI    = 0; /* Offset into an element vector for fieldI */
  PetscInt           offsetJ    = 0; /* Offset into an element vector for fieldJ */
  PetscQuadrature    quad;
  PetscScalar       *g0, *g1, *g2, *g3, *u, *u_t = NULL, *u_x, *a, *a_x, *basisReal, *basisDerReal, *testReal, *testDerReal;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscReal        **B, **D, **BAux = NULL, **DAux = NULL, *BI, *DI, *BJ, *DJ;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL, *Nb, *Nc, *NbAux = NULL, *NcAux = NULL;
  PetscInt           NbI = 0, NcI = 0, NbJ = 0, NcJ = 0;
  PetscInt           dim, numConstants, Nf, NfAux = 0, totDim, totDimAux = 0, e;
  PetscInt           dE, Np;
  PetscBool          isAffine;
  const PetscReal   *quadPoints, *quadWeights;
  PetscInt           qNc, Nq, q;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDSGetDiscretization(ds, fieldI, (PetscObject *) &feI);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(ds, fieldJ, (PetscObject *) &feJ);CHKERRQ(ierr);
  ierr = PetscFEGetSpatialDimension(feI, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(feI, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(ds, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(ds, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(ds, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(ds, &uOff_x);CHKERRQ(ierr);
  switch(jtype) {
  case PETSCFE_JACOBIAN_DYN: ierr = PetscDSGetDynamicJacobian(ds, fieldI, fieldJ, &g0_func, &g1_func, &g2_func, &g3_func);CHKERRQ(ierr);break;
  case PETSCFE_JACOBIAN_PRE: ierr = PetscDSGetJacobianPreconditioner(ds, fieldI, fieldJ, &g0_func, &g1_func, &g2_func, &g3_func);CHKERRQ(ierr);break;
  case PETSCFE_JACOBIAN:     ierr = PetscDSGetJacobian(ds, fieldI, fieldJ, &g0_func, &g1_func, &g2_func, &g3_func);CHKERRQ(ierr);break;
  }
  if (!g0_func && !g1_func && !g2_func && !g3_func) PetscFunctionReturn(0);
  ierr = PetscDSGetEvaluationArrays(ds, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetWorkspace(ds, &x, &basisReal, &basisDerReal, &testReal, &testDerReal);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(ds, NULL, NULL, &g0, &g1, &g2, &g3);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(ds, &B, &D);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(ds, fieldI, &offsetI);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(ds, fieldJ, &offsetJ);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(ds, &numConstants, &constants);CHKERRQ(ierr);
  if (dsAux) {
    ierr = PetscDSGetNumFields(dsAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(dsAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(dsAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(dsAux, &NcAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(dsAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(dsAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(dsAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetTabulation(dsAux, &BAux, &DAux);CHKERRQ(ierr);
  }
  NbI = Nb[fieldI], NbJ = Nb[fieldJ];
  NcI = Nc[fieldI], NcJ = Nc[fieldJ];
  BI  = B[fieldI],  BJ  = B[fieldJ];
  DI  = D[fieldI],  DJ  = D[fieldJ];
  /* Initialize here in case the function is not defined */
  ierr = PetscArrayzero(g0, NcI*NcJ);CHKERRQ(ierr);
  ierr = PetscArrayzero(g1, NcI*NcJ*dim);CHKERRQ(ierr);
  ierr = PetscArrayzero(g2, NcI*NcJ*dim);CHKERRQ(ierr);
  ierr = PetscArrayzero(g3, NcI*NcJ*dim*dim);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
  Np = cgeom->numPoints;
  dE = cgeom->dimEmbed;
  isAffine = cgeom->isAffine;
  for (e = 0; e < Ne; ++e) {
    PetscFEGeom fegeom;

    if (isAffine) {
      fegeom.v    = x;
      fegeom.xi   = cgeom->xi;
      fegeom.J    = &cgeom->J[e*dE*dE];
      fegeom.invJ = &cgeom->invJ[e*dE*dE];
      fegeom.detJ = &cgeom->detJ[e];
    }
    for (q = 0; q < Nq; ++q) {
      const PetscReal *BIq = &BI[q*NbI*NcI],     *BJq = &BJ[q*NbJ*NcJ];
      const PetscReal *DIq = &DI[q*NbI*NcI*dim], *DJq = &DJ[q*NbJ*NcJ*dim];
      PetscReal        w;
      PetscInt         c;

      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      if (isAffine) {
        CoordinatesRefToReal(dE, dim, fegeom.xi, &cgeom->v[e*dE], fegeom.J, &quadPoints[q*dim], x);
      } else {
        fegeom.v    = &cgeom->v[(e*Np+q)*dE];
        fegeom.J    = &cgeom->J[(e*Np+q)*dE*dE];
        fegeom.invJ = &cgeom->invJ[(e*Np+q)*dE*dE];
        fegeom.detJ = &cgeom->detJ[e*Np+q];
      }
      w = fegeom.detJ[0]*quadWeights[q];
      if (coefficients) {ierr = PetscFEEvaluateFieldJets_Internal(ds, dim, Nf, Nb, Nc, q, B, D, &fegeom, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);CHKERRQ(ierr);}
      if (dsAux)      {ierr = PetscFEEvaluateFieldJets_Internal(dsAux, dim, NfAux, NbAux, NcAux, q, BAux, DAux, &fegeom, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);CHKERRQ(ierr);}
      if (g0_func) {
        ierr = PetscArrayzero(g0, NcI*NcJ);CHKERRQ(ierr);
        g0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, fegeom.v, numConstants, constants, g0);
        for (c = 0; c < NcI*NcJ; ++c) g0[c] *= w;
      }
      if (g1_func) {
        ierr = PetscArrayzero(g1, NcI*NcJ*dim);CHKERRQ(ierr);
        g1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, fegeom.v, numConstants, constants, g1);
        for (c = 0; c < NcI*NcJ*dim; ++c) g1[c] *= w;
      }
      if (g2_func) {
        ierr = PetscArrayzero(g2, NcI*NcJ*dim);CHKERRQ(ierr);
        g2_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, fegeom.v, numConstants, constants, g2);
        for (c = 0; c < NcI*NcJ*dim; ++c) g2[c] *= w;
      }
      if (g3_func) {
        ierr = PetscArrayzero(g3, NcI*NcJ*dim*dim);CHKERRQ(ierr);
        g3_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, fegeom.v, numConstants, constants, g3);
        for (c = 0; c < NcI*NcJ*dim*dim; ++c) g3[c] *= w;
      }

      ierr = PetscFEUpdateElementMat_Internal(feI, feJ, dim, NbI, NcI, BIq, DIq, basisReal, basisDerReal, NbJ, NcJ, BJq, DJq, testReal, testDerReal, &fegeom, g0, g1, g2, g3, eOffset, totDim, offsetI, offsetJ, elemMat);CHKERRQ(ierr);
    }
    if (debug > 1) {
      PetscInt fc, f, gc, g;

      ierr = PetscPrintf(PETSC_COMM_SELF, "Element matrix for fields %d and %d\n", fieldI, fieldJ);CHKERRQ(ierr);
      for (fc = 0; fc < NcI; ++fc) {
        for (f = 0; f < NbI; ++f) {
          const PetscInt i = offsetI + f*NcI+fc;
          for (gc = 0; gc < NcJ; ++gc) {
            for (g = 0; g < NbJ; ++g) {
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

static PetscErrorCode PetscFEIntegrateBdJacobian_Basic(PetscDS ds, PetscInt fieldI, PetscInt fieldJ, PetscInt Ne, PetscFEGeom *fgeom,
                                                       const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS dsAux, const PetscScalar coefficientsAux[], PetscReal t, PetscReal u_tshift, PetscScalar elemMat[])
{
  const PetscInt     debug      = 0;
  PetscFE            feI, feJ;
  PetscBdPointJac    g0_func, g1_func, g2_func, g3_func;
  PetscInt           cOffset    = 0; /* Offset into coefficients[] for element e */
  PetscInt           cOffsetAux = 0; /* Offset into coefficientsAux[] for element e */
  PetscInt           eOffset    = 0; /* Offset into elemMat[] for element e */
  PetscInt           offsetI    = 0; /* Offset into an element vector for fieldI */
  PetscInt           offsetJ    = 0; /* Offset into an element vector for fieldJ */
  PetscQuadrature    quad;
  PetscScalar       *g0, *g1, *g2, *g3, *u, *u_t = NULL, *u_x, *a, *a_x, *basisReal, *basisDerReal, *testReal, *testDerReal;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscReal        **B, **D, **BAux = NULL, **DAux = NULL, *BI, *DI, *BJ, *DJ;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL, *Nb, *Nc, *NbAux = NULL, *NcAux = NULL;
  PetscInt           NbI = 0, NcI = 0, NbJ = 0, NcJ = 0;
  PetscInt           dim, numConstants, Nf, NfAux = 0, totDim, totDimAux = 0, e;
  PetscBool          isAffine;
  const PetscReal   *quadPoints, *quadWeights;
  PetscInt           qNc, Nq, q, Np, dE;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscDSGetDiscretization(ds, fieldI, (PetscObject *) &feI);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(ds, fieldJ, (PetscObject *) &feJ);CHKERRQ(ierr);
  ierr = PetscFEGetSpatialDimension(feI, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetFaceQuadrature(feI, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(ds, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(ds, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(ds, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(ds, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(ds, fieldI, &offsetI);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(ds, fieldJ, &offsetJ);CHKERRQ(ierr);
  ierr = PetscDSGetBdJacobian(ds, fieldI, fieldJ, &g0_func, &g1_func, &g2_func, &g3_func);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(ds, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetWorkspace(ds, &x, &basisReal, &basisDerReal, &testReal, &testDerReal);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(ds, NULL, NULL, &g0, &g1, &g2, &g3);CHKERRQ(ierr);
  ierr = PetscDSGetFaceTabulation(ds, &B, &D);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(ds, &numConstants, &constants);CHKERRQ(ierr);
  if (dsAux) {
    ierr = PetscDSGetNumFields(dsAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(dsAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(dsAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(dsAux, &NcAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(dsAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(dsAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(dsAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetFaceTabulation(dsAux, &BAux, &DAux);CHKERRQ(ierr);
  }
  NbI = Nb[fieldI], NbJ = Nb[fieldJ];
  NcI = Nc[fieldI], NcJ = Nc[fieldJ];
  BI  = B[fieldI],  BJ  = B[fieldJ];
  DI  = D[fieldI],  DJ  = D[fieldJ];
  /* Initialize here in case the function is not defined */
  ierr = PetscArrayzero(g0, NcI*NcJ);CHKERRQ(ierr);
  ierr = PetscArrayzero(g1, NcI*NcJ*dim);CHKERRQ(ierr);
  ierr = PetscArrayzero(g2, NcI*NcJ*dim);CHKERRQ(ierr);
  ierr = PetscArrayzero(g3, NcI*NcJ*dim*dim);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
  Np = fgeom->numPoints;
  dE = fgeom->dimEmbed;
  isAffine = fgeom->isAffine;
  for (e = 0; e < Ne; ++e) {
    PetscFEGeom    fegeom, cgeom;
    const PetscInt face = fgeom->face[e][0];
    fegeom.n = 0;
    fegeom.v = 0;
    fegeom.J = 0;
    fegeom.detJ = 0;
    if (isAffine) {
      fegeom.v    = x;
      fegeom.xi   = fgeom->xi;
      fegeom.J    = &fgeom->J[e*dE*dE];
      fegeom.invJ = &fgeom->invJ[e*dE*dE];
      fegeom.detJ = &fgeom->detJ[e];
      fegeom.n    = &fgeom->n[e*dE];

      cgeom.J     = &fgeom->suppJ[0][e*dE*dE];
      cgeom.invJ  = &fgeom->suppInvJ[0][e*dE*dE];
      cgeom.detJ  = &fgeom->suppDetJ[0][e];
    }
    for (q = 0; q < Nq; ++q) {
      const PetscReal *BIq = &BI[(face*Nq+q)*NbI*NcI],     *BJq = &BJ[(face*Nq+q)*NbJ*NcJ];
      const PetscReal *DIq = &DI[(face*Nq+q)*NbI*NcI*dim], *DJq = &DJ[(face*Nq+q)*NbJ*NcJ*dim];
      PetscReal  w;
      PetscInt   c;

      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      if (isAffine) {
        CoordinatesRefToReal(dE, dim-1, fegeom.xi, &fgeom->v[e*dE], fegeom.J, &quadPoints[q*(dim-1)], x);
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
      if (coefficients) {ierr = PetscFEEvaluateFieldJets_Internal(ds, dim, Nf, Nb, Nc, face*Nq+q, B, D, &cgeom, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);CHKERRQ(ierr);}
      if (dsAux)      {ierr = PetscFEEvaluateFieldJets_Internal(dsAux, dim, NfAux, NbAux, NcAux, face*Nq+q, BAux, DAux, &cgeom, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);CHKERRQ(ierr);}
      if (g0_func) {
        ierr = PetscArrayzero(g0, NcI*NcJ);CHKERRQ(ierr);
        g0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, fegeom.v, fegeom.n, numConstants, constants, g0);
        for (c = 0; c < NcI*NcJ; ++c) g0[c] *= w;
      }
      if (g1_func) {
        ierr = PetscArrayzero(g1, NcI*NcJ*dim);CHKERRQ(ierr);
        g1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, fegeom.v, fegeom.n, numConstants, constants, g1);
        for (c = 0; c < NcI*NcJ*dim; ++c) g1[c] *= w;
      }
      if (g2_func) {
        ierr = PetscArrayzero(g2, NcI*NcJ*dim);CHKERRQ(ierr);
        g2_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, fegeom.v, fegeom.n, numConstants, constants, g2);
        for (c = 0; c < NcI*NcJ*dim; ++c) g2[c] *= w;
      }
      if (g3_func) {
        ierr = PetscArrayzero(g3, NcI*NcJ*dim*dim);CHKERRQ(ierr);
        g3_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, fegeom.v, fegeom.n, numConstants, constants, g3);
        for (c = 0; c < NcI*NcJ*dim*dim; ++c) g3[c] *= w;
      }

      ierr = PetscFEUpdateElementMat_Internal(feI, feJ, dim, NbI, NcI, BIq, DIq, basisReal, basisDerReal, NbJ, NcJ, BJq, DJq, testReal, testDerReal, &cgeom, g0, g1, g2, g3, eOffset, totDim, offsetI, offsetJ, elemMat);CHKERRQ(ierr);
    }
    if (debug > 1) {
      PetscInt fc, f, gc, g;

      ierr = PetscPrintf(PETSC_COMM_SELF, "Element matrix for fields %d and %d\n", fieldI, fieldJ);CHKERRQ(ierr);
      for (fc = 0; fc < NcI; ++fc) {
        for (f = 0; f < NbI; ++f) {
          const PetscInt i = offsetI + f*NcI+fc;
          for (gc = 0; gc < NcJ; ++gc) {
            for (g = 0; g < NbJ; ++g) {
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
  fem->ops->gettabulation           = PetscFEGetTabulation_Basic;
  fem->ops->integrate               = PetscFEIntegrate_Basic;
  fem->ops->integratebd             = PetscFEIntegrateBd_Basic;
  fem->ops->integrateresidual       = PetscFEIntegrateResidual_Basic;
  fem->ops->integratebdresidual     = PetscFEIntegrateBdResidual_Basic;
  fem->ops->integratejacobianaction = NULL/* PetscFEIntegrateJacobianAction_Basic */;
  fem->ops->integratejacobian       = PetscFEIntegrateJacobian_Basic;
  fem->ops->integratebdjacobian     = PetscFEIntegrateBdJacobian_Basic;
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
