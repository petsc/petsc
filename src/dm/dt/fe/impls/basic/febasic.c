#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/
#include <petscblaslapack.h>

PetscErrorCode PetscFEDestroy_Basic(PetscFE fem)
{
  PetscFE_Basic *b = (PetscFE_Basic *) fem->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEView_Basic_Ascii(PetscFE fe, PetscViewer v)
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

PetscErrorCode PetscFEView_Basic(PetscFE fe, PetscViewer v)
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
  PetscStackCallBLAS("LAPACKgetri", LAPACKgetri_(&n, invVscalar, &n, pivots, work, &n, &info));
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

PetscErrorCode PetscFEIntegrate_Basic(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFEGeom *cgeom,
                                      const PetscScalar coefficients[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscScalar integral[])
{
  const PetscInt     debug = 0;
  PetscPointFunc     obj_func;
  PetscQuadrature    quad;
  PetscScalar       *u, *u_x, *a, *a_x, *refSpaceDer, *refSpaceDerAux;
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
  ierr = PetscDSGetObjective(prob, field, &obj_func);CHKERRQ(ierr);
  if (!obj_func) PetscFunctionReturn(0);
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(prob, &u, NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &B, &D);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(prob, &numConstants, &constants);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(probAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(probAux, &NcAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetRefCoordArrays(probAux, NULL, &refSpaceDerAux);CHKERRQ(ierr);
    ierr = PetscDSGetTabulation(probAux, &BAux, &DAux);CHKERRQ(ierr);
  }
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  Np = cgeom->numPoints;
  dE = cgeom->dimEmbed;
  isAffine = cgeom->isAffine;
  for (e = 0; e < Ne; ++e) {
    const PetscReal *v0   = &cgeom->v[e*Np*dE];
    const PetscReal *J    = &cgeom->J[e*Np*dE*dE];

    if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
    for (q = 0; q < Nq; ++q) {
      PetscScalar integrand;
      const PetscReal *v;
      const PetscReal *invJ;
      PetscReal detJ;

      if (isAffine) {
        CoordinatesRefToReal(dE, dim, cgeom->xi, v0, J, &quadPoints[q*dim], x);
        v = x;
        invJ = &cgeom->invJ[e*dE*dE];
        detJ = cgeom->detJ[e];
      } else {
        v = &v0[q*dE];
        invJ = &cgeom->invJ[(e*Np+q)*dE*dE];
        detJ = cgeom->detJ[e*Np + q];
      }
      if (debug > 1 && q < Np) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", detJ);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
        ierr = DMPrintCellMatrix(e, "invJ", dim, dim, invJ);CHKERRQ(ierr);
#endif
      }
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      EvaluateFieldJets(dim, Nf, Nb, Nc, q, B, D, refSpaceDer, invJ, &coefficients[cOffset], NULL, u, u_x, NULL);
      if (probAux) EvaluateFieldJets(dim, NfAux, NbAux, NcAux, q, BAux, DAux, refSpaceDerAux, invJ, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);
      obj_func(dim, Nf, NfAux, uOff, uOff_x, u, NULL, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, v, numConstants, constants, &integrand);
      integrand *= detJ*quadWeights[q];
      integral[e*Nf+field] += integrand;
      if (debug > 1) {ierr = PetscPrintf(PETSC_COMM_SELF, "    int: %g %g\n", (double) PetscRealPart(integrand), (double) PetscRealPart(integral[field]));CHKERRQ(ierr);}
    }
    cOffset    += totDim;
    cOffsetAux += totDimAux;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEIntegrateBd_Basic(PetscFE fem, PetscDS prob, PetscInt field,
                                        PetscBdPointFunc obj_func,
                                        PetscInt Ne, PetscFEGeom *fgeom, const PetscScalar coefficients[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscScalar integral[])
{
  const PetscInt     debug = 0;
  PetscQuadrature    quad;
  PetscScalar       *u, *u_x, *a, *a_x, *refSpaceDer, *refSpaceDerAux;
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
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetFaceQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(prob, &u, NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = PetscDSGetFaceTabulation(prob, &B, &D);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(prob, &numConstants, &constants);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetSpatialDimension(probAux, &dimAux);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(probAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(probAux, &NcAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetRefCoordArrays(probAux, NULL, &refSpaceDerAux);CHKERRQ(ierr);
    auxOnBd = dimAux < dim ? PETSC_TRUE : PETSC_FALSE;
    if (auxOnBd) {ierr = PetscDSGetTabulation(probAux, &BAux, &DAux);CHKERRQ(ierr);}
    else         {ierr = PetscDSGetFaceTabulation(probAux, &BAux, &DAux);CHKERRQ(ierr);}
  }
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
  Np = fgeom->numPoints;
  dE = fgeom->dimEmbed;
  isAffine = fgeom->isAffine;
  for (e = 0; e < Ne; ++e) {
    const PetscReal *v0   = &fgeom->v[e*Np*dE];
    const PetscReal *J    = &fgeom->J[e*Np*dE*dE];
    const PetscInt   face = fgeom->face[e][0]; /* Local face number in cell */

    for (q = 0; q < Nq; ++q) {
      const PetscReal *v;
      const PetscReal *invJ;
      const PetscReal *n;
      PetscReal        detJ;
      PetscScalar      integrand;

      if (isAffine) {
        CoordinatesRefToReal(dE, dim-1, fgeom->xi, v0, J, &quadPoints[q*(dim-1)], x);
        v = x;
        invJ = &fgeom->suppInvJ[0][e*dE*dE];
        detJ = fgeom->detJ[e];
        n    = &fgeom->n[e*dE];
      } else {
        v = &v0[q*dE];
        invJ = &fgeom->suppInvJ[0][(e*Np+q)*dE*dE];
        detJ = fgeom->detJ[e*Np + q];
        n    = &fgeom->n[(e*Np+q)*dE];
      }
      if (debug > 1 && q < Np) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", detJ);CHKERRQ(ierr);
#ifndef PETSC_USE_COMPLEX
        ierr = DMPrintCellMatrix(e, "invJ", dim, dim, invJ);CHKERRQ(ierr);
#endif
      }
      if (debug > 1) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      EvaluateFieldJets(dim, Nf, Nb, Nc, face*Nq+q, B, D, refSpaceDer, invJ, &coefficients[cOffset], NULL, u, u_x, NULL);
      if (probAux) EvaluateFieldJets(dimAux, NfAux, NbAux, NcAux, face*Nq+q, BAux, DAux, refSpaceDerAux, invJ, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);
      obj_func(dim, Nf, NfAux, uOff, uOff_x, u, NULL, u_x, aOff, aOff_x, a, NULL, a_x, 0.0, v, n, numConstants, constants, &integrand);
      integrand *= detJ*quadWeights[q];
      integral[e*Nf+field] += integrand;
      if (debug > 1) {ierr = PetscPrintf(PETSC_COMM_SELF, "    int: %g %g\n", (double) PetscRealPart(integrand), (double) PetscRealPart(integral[e*Nf+field]));CHKERRQ(ierr);}
    }
    cOffset    += totDim;
    cOffsetAux += totDimAux;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEIntegrateResidual_Basic(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFEGeom *cgeom,
                                              const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  const PetscInt     debug = 0;
  PetscPointFunc     f0_func;
  PetscPointFunc     f1_func;
  PetscQuadrature    quad;
  PetscScalar       *f0, *f1, *u, *u_t = NULL, *u_x, *a, *a_x, *refSpaceDer, *refSpaceDerAux;
  const PetscScalar *constants;
  PetscReal         *x;
  PetscReal        **B, **D, **BAux = NULL, **DAux = NULL, *BI, *DI;
  PetscInt          *uOff, *uOff_x, *aOff = NULL, *aOff_x = NULL, *Nb, *Nc, *NbAux = NULL, *NcAux = NULL;
  PetscInt           dim, numConstants, Nf, NfAux = 0, totDim, totDimAux = 0, cOffset = 0, cOffsetAux = 0, fOffset, e, NbI, NcI;
  PetscInt           dE, Np;
  PetscBool          isAffine;
  const PetscReal   *quadPoints, *quadWeights;
  PetscInt           qNc, Nq, q;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, field, &fOffset);CHKERRQ(ierr);
  ierr = PetscDSGetResidual(prob, field, &f0_func, &f1_func);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(prob, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(prob, &f0, &f1, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &B, &D);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(prob, &numConstants, &constants);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(probAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(probAux, &NcAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetRefCoordArrays(probAux, NULL, &refSpaceDerAux);CHKERRQ(ierr);
    ierr = PetscDSGetTabulation(probAux, &BAux, &DAux);CHKERRQ(ierr);
  }
  NbI = Nb[field];
  NcI = Nc[field];
  BI  = B[field];
  DI  = D[field];
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  Np = cgeom->numPoints;
  dE = cgeom->dimEmbed;
  isAffine = cgeom->isAffine;
  for (e = 0; e < Ne; ++e) {
    const PetscReal *v0   = &cgeom->v[e*Np*dE];
    const PetscReal *J    = &cgeom->J[e*Np*dE*dE];

    if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
    ierr = PetscMemzero(f0, Nq*NcI* sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(f1, Nq*NcI*dim * sizeof(PetscScalar));CHKERRQ(ierr);
    for (q = 0; q < Nq; ++q) {
      const PetscReal *v;
      const PetscReal *invJ;
      PetscReal detJ;

      if (isAffine) {
        CoordinatesRefToReal(dE, dim, cgeom->xi, v0, J, &quadPoints[q*dim], x);
        v = x;
        invJ = &cgeom->invJ[e*dE*dE];
        detJ = cgeom->detJ[e];
      } else {
        v = &v0[q*dE];
        invJ = &cgeom->invJ[(e*Np+q)*dE*dE];
        detJ = cgeom->detJ[e*Np + q];
      }
      if (debug > 1 && q < Np) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", detJ);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
        ierr = DMPrintCellMatrix(e, "invJ", dim, dim, invJ);CHKERRQ(ierr);
#endif
      }
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      EvaluateFieldJets(dim, Nf, Nb, Nc, q, B, D, refSpaceDer, invJ, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);
      if (probAux) EvaluateFieldJets(dim, NfAux, NbAux, NcAux, q, BAux, DAux, refSpaceDerAux, invJ, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);
      if (f0_func) f0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, v, numConstants, constants, &f0[q*NcI]);
      if (f1_func) {
        ierr = PetscMemzero(refSpaceDer, NcI*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        f1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, v, numConstants, constants, refSpaceDer);
      }
      TransformF(dim, NcI, q, invJ, detJ, quadWeights, refSpaceDer, f0_func ? f0 : NULL, f1_func ? f1 : NULL);
    }
    UpdateElementVec(dim, Nq, NbI, NcI, BI, DI, f0, f1, &elemVec[cOffset+fOffset]);
    cOffset    += totDim;
    cOffsetAux += totDimAux;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEIntegrateBdResidual_Basic(PetscFE fem, PetscDS prob, PetscInt field, PetscInt Ne, PetscFEGeom *fgeom,
                                                const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscScalar elemVec[])
{
  const PetscInt     debug = 0;
  PetscBdPointFunc   f0_func;
  PetscBdPointFunc   f1_func;
  PetscQuadrature    quad;
  PetscScalar       *f0, *f1, *u, *u_t = NULL, *u_x, *a, *a_x, *refSpaceDer, *refSpaceDerAux;
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
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetFaceQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, field, &fOffset);CHKERRQ(ierr);
  ierr = PetscDSGetBdResidual(prob, field, &f0_func, &f1_func);CHKERRQ(ierr);
  if (!f0_func && !f1_func) PetscFunctionReturn(0);
  ierr = PetscDSGetEvaluationArrays(prob, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(prob, &f0, &f1, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscDSGetFaceTabulation(prob, &B, &D);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(prob, &numConstants, &constants);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetSpatialDimension(probAux, &dimAux);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(probAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(probAux, &NcAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetRefCoordArrays(probAux, NULL, &refSpaceDerAux);CHKERRQ(ierr);
    auxOnBd = dimAux < dim ? PETSC_TRUE : PETSC_FALSE;
    if (auxOnBd) {ierr = PetscDSGetTabulation(probAux, &BAux, &DAux);CHKERRQ(ierr);}
    else         {ierr = PetscDSGetFaceTabulation(probAux, &BAux, &DAux);CHKERRQ(ierr);}
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
    const PetscReal *v0   = &fgeom->v[e*Np*dE];
    const PetscReal *J    = &fgeom->J[e*Np*dE*dE];
    const PetscInt   face = fgeom->face[e][0];

    ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
    if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
    ierr = PetscMemzero(f0, Nq*NcI* sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscMemzero(f1, Nq*NcI*dim * sizeof(PetscScalar));CHKERRQ(ierr);
    for (q = 0; q < Nq; ++q) {
      const PetscReal *v;
      const PetscReal *invJ;
      const PetscReal *n;
      PetscReal detJ;
      if (isAffine) {
        CoordinatesRefToReal(dE, dim-1, fgeom->xi, v0, J, &quadPoints[q*(dim-1)], x);
        v = x;
        invJ = &fgeom->suppInvJ[0][e*dE*dE];
        detJ = fgeom->detJ[e];
        n    = &fgeom->n[e*dE];
      } else {
        v = &v0[q*dE];
        invJ = &fgeom->suppInvJ[0][(e*Np+q)*dE*dE];
        detJ = fgeom->detJ[e*Np + q];
        n    = &fgeom->n[(e*Np+q)*dE];
      }
      if (debug > 1 && q < Np) {
        ierr = PetscPrintf(PETSC_COMM_SELF, "  detJ: %g\n", detJ);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
        ierr = DMPrintCellMatrix(e, "invJ", dim, dim, invJ);CHKERRQ(ierr);
#endif
      }
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      EvaluateFieldJets(dim, Nf, Nb, Nc, face*Nq+q, B, D, refSpaceDer, invJ, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);
      if (probAux) EvaluateFieldJets(dimAux, NfAux, NbAux, NcAux, auxOnBd ? q : face*Nq+q, BAux, DAux, refSpaceDerAux, invJ, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);
      if (f0_func) f0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, v, n, numConstants, constants, &f0[q*NcI]);
      if (f1_func) {
        ierr = PetscMemzero(refSpaceDer, NcI*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        f1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, v, n, numConstants, constants, refSpaceDer);
      }
      TransformF(dim, NcI, q, invJ, detJ, quadWeights, refSpaceDer, f0_func ? f0 : NULL, f1_func ? f1 : NULL);
    }
    UpdateElementVec(dim, Nq, NbI, NcI, &BI[face*Nq*NbI*NcI], &DI[face*Nq*NbI*NcI*dim], f0, f1, &elemVec[cOffset+fOffset]);
    cOffset    += totDim;
    cOffsetAux += totDimAux;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscFEIntegrateJacobian_Basic(PetscFE fem, PetscDS prob, PetscFEJacobianType jtype, PetscInt fieldI, PetscInt fieldJ, PetscInt Ne, PetscFEGeom *geom,
                                              const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscReal u_tshift, PetscScalar elemMat[])
{
  const PetscInt     debug      = 0;
  PetscPointJac      g0_func;
  PetscPointJac      g1_func;
  PetscPointJac      g2_func;
  PetscPointJac      g3_func;
  PetscInt           cOffset    = 0; /* Offset into coefficients[] for element e */
  PetscInt           cOffsetAux = 0; /* Offset into coefficientsAux[] for element e */
  PetscInt           eOffset    = 0; /* Offset into elemMat[] for element e */
  PetscInt           offsetI    = 0; /* Offset into an element vector for fieldI */
  PetscInt           offsetJ    = 0; /* Offset into an element vector for fieldJ */
  PetscQuadrature    quad;
  PetscScalar       *g0, *g1, *g2, *g3, *u, *u_t = NULL, *u_x, *a, *a_x, *refSpaceDer, *refSpaceDerAux;
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
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  switch(jtype) {
  case PETSCFE_JACOBIAN_DYN: ierr = PetscDSGetDynamicJacobian(prob, fieldI, fieldJ, &g0_func, &g1_func, &g2_func, &g3_func);CHKERRQ(ierr);break;
  case PETSCFE_JACOBIAN_PRE: ierr = PetscDSGetJacobianPreconditioner(prob, fieldI, fieldJ, &g0_func, &g1_func, &g2_func, &g3_func);CHKERRQ(ierr);break;
  case PETSCFE_JACOBIAN:     ierr = PetscDSGetJacobian(prob, fieldI, fieldJ, &g0_func, &g1_func, &g2_func, &g3_func);CHKERRQ(ierr);break;
  }
  if (!g0_func && !g1_func && !g2_func && !g3_func) PetscFunctionReturn(0);
  ierr = PetscDSGetEvaluationArrays(prob, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(prob, NULL, NULL, &g0, &g1, &g2, &g3);CHKERRQ(ierr);
  ierr = PetscDSGetTabulation(prob, &B, &D);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, fieldI, &offsetI);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, fieldJ, &offsetJ);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(prob, &numConstants, &constants);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(probAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(probAux, &NcAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetRefCoordArrays(probAux, NULL, &refSpaceDerAux);CHKERRQ(ierr);
    ierr = PetscDSGetTabulation(probAux, &BAux, &DAux);CHKERRQ(ierr);
  }
  NbI = Nb[fieldI], NbJ = Nb[fieldJ];
  NcI = Nc[fieldI], NcJ = Nc[fieldJ];
  BI  = B[fieldI],  BJ  = B[fieldJ];
  DI  = D[fieldI],  DJ  = D[fieldJ];
  /* Initialize here in case the function is not defined */
  ierr = PetscMemzero(g0, NcI*NcJ * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g1, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g2, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g3, NcI*NcJ*dim*dim * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  Np = geom->numPoints;
  dE = geom->dimEmbed;
  isAffine = geom->isAffine;
  for (e = 0; e < Ne; ++e) {
    const PetscReal *v0   = &geom->v[e*Np*dE];
    const PetscReal *J    = &geom->J[e*Np*dE*dE];

    ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
    if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
    for (q = 0; q < Nq; ++q) {
      const PetscReal *v;
      const PetscReal *invJ;
      PetscReal detJ;
      const PetscReal *BIq = &BI[q*NbI*NcI], *BJq = &BJ[q*NbJ*NcJ];
      const PetscReal *DIq = &DI[q*NbI*NcI*dim], *DJq = &DJ[q*NbJ*NcJ*dim];
      PetscReal  w;
      PetscInt f, g, fc, gc, c;

      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      if (isAffine) {
        CoordinatesRefToReal(dE, dim, geom->xi, v0, J, &quadPoints[q*dim], x);
        v = x;
        invJ = &geom->invJ[e*dE*dE];
        detJ = geom->detJ[e];
      } else {
        v = &v0[q*dE];
        invJ = &geom->invJ[(e*Np+q)*dE*dE];
        detJ = geom->detJ[e*Np + q];
      }
      w = detJ*quadWeights[q];
      if (coefficients) EvaluateFieldJets(dim, Nf, Nb, Nc, q, B, D, refSpaceDer, invJ, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);
      if (probAux)      EvaluateFieldJets(dim, NfAux, NbAux, NcAux, q, BAux, DAux, refSpaceDerAux, invJ, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);
      if (g0_func) {
        ierr = PetscMemzero(g0, NcI*NcJ * sizeof(PetscScalar));CHKERRQ(ierr);
        g0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, v, numConstants, constants, g0);
        for (c = 0; c < NcI*NcJ; ++c) g0[c] *= w;
      }
      if (g1_func) {
        PetscInt d, d2;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, v, numConstants, constants, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < dim; ++d) {
              g1[(fc*NcJ+gc)*dim+d] = 0.0;
              for (d2 = 0; d2 < dim; ++d2) g1[(fc*NcJ+gc)*dim+d] += invJ[d*dim+d2]*refSpaceDer[(fc*NcJ+gc)*dim+d2];
              g1[(fc*NcJ+gc)*dim+d] *= w;
            }
          }
        }
      }
      if (g2_func) {
        PetscInt d, d2;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g2_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, v, numConstants, constants, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < dim; ++d) {
              g2[(fc*NcJ+gc)*dim+d] = 0.0;
              for (d2 = 0; d2 < dim; ++d2) g2[(fc*NcJ+gc)*dim+d] += invJ[d*dim+d2]*refSpaceDer[(fc*NcJ+gc)*dim+d2];
              g2[(fc*NcJ+gc)*dim+d] *= w;
            }
          }
        }
      }
      if (g3_func) {
        PetscInt d, d2, dp, d3;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g3_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, v, numConstants, constants, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < dim; ++d) {
              for (dp = 0; dp < dim; ++dp) {
                g3[((fc*NcJ+gc)*dim+d)*dim+dp] = 0.0;
                for (d2 = 0; d2 < dim; ++d2) {
                  for (d3 = 0; d3 < dim; ++d3) {
                    g3[((fc*NcJ+gc)*dim+d)*dim+dp] += invJ[d*dim+d2]*refSpaceDer[((fc*NcJ+gc)*dim+d2)*dim+d3]*invJ[dp*dim+d3];
                  }
                }
                g3[((fc*NcJ+gc)*dim+d)*dim+dp] *= w;
              }
            }
          }
        }
      }

      for (f = 0; f < NbI; ++f) {
        for (fc = 0; fc < NcI; ++fc) {
          const PetscInt fidx = f*NcI+fc; /* Test function basis index */
          const PetscInt i    = offsetI+f; /* Element matrix row */
          for (g = 0; g < NbJ; ++g) {
            for (gc = 0; gc < NcJ; ++gc) {
              const PetscInt gidx = g*NcJ+gc; /* Trial function basis index */
              const PetscInt j    = offsetJ+g; /* Element matrix column */
              const PetscInt fOff = eOffset+i*totDim+j;
              PetscInt       d, d2;

              elemMat[fOff] += BIq[fidx]*g0[fc*NcJ+gc]*BJq[gidx];
              for (d = 0; d < dim; ++d) {
                elemMat[fOff] += BIq[fidx]*g1[(fc*NcJ+gc)*dim+d]*DJq[gidx*dim+d];
                elemMat[fOff] += DIq[fidx*dim+d]*g2[(fc*NcJ+gc)*dim+d]*BJq[gidx];
                for (d2 = 0; d2 < dim; ++d2) {
                  elemMat[fOff] += DIq[fidx*dim+d]*g3[((fc*NcJ+gc)*dim+d)*dim+d2]*DJq[gidx*dim+d2];
                }
              }
            }
          }
        }
      }
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

PetscErrorCode PetscFEIntegrateBdJacobian_Basic(PetscFE fem, PetscDS prob, PetscInt fieldI, PetscInt fieldJ, PetscInt Ne, PetscFEGeom *fgeom,
                                                const PetscScalar coefficients[], const PetscScalar coefficients_t[], PetscDS probAux, const PetscScalar coefficientsAux[], PetscReal t, PetscReal u_tshift, PetscScalar elemMat[])
{
  const PetscInt     debug      = 0;
  PetscBdPointJac    g0_func;
  PetscBdPointJac    g1_func;
  PetscBdPointJac    g2_func;
  PetscBdPointJac    g3_func;
  PetscInt           cOffset    = 0; /* Offset into coefficients[] for element e */
  PetscInt           cOffsetAux = 0; /* Offset into coefficientsAux[] for element e */
  PetscInt           eOffset    = 0; /* Offset into elemMat[] for element e */
  PetscInt           offsetI    = 0; /* Offset into an element vector for fieldI */
  PetscInt           offsetJ    = 0; /* Offset into an element vector for fieldJ */
  PetscQuadrature    quad;
  PetscScalar       *g0, *g1, *g2, *g3, *u, *u_t = NULL, *u_x, *a, *a_x, *refSpaceDer, *refSpaceDerAux;
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
  ierr = PetscFEGetSpatialDimension(fem, &dim);CHKERRQ(ierr);
  ierr = PetscFEGetFaceQuadrature(fem, &quad);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetDimensions(prob, &Nb);CHKERRQ(ierr);
  ierr = PetscDSGetComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscDSGetComponentOffsets(prob, &uOff);CHKERRQ(ierr);
  ierr = PetscDSGetComponentDerivativeOffsets(prob, &uOff_x);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, fieldI, &offsetI);CHKERRQ(ierr);
  ierr = PetscDSGetFieldOffset(prob, fieldJ, &offsetJ);CHKERRQ(ierr);
  ierr = PetscDSGetBdJacobian(prob, fieldI, fieldJ, &g0_func, &g1_func, &g2_func, &g3_func);CHKERRQ(ierr);
  ierr = PetscDSGetEvaluationArrays(prob, &u, coefficients_t ? &u_t : NULL, &u_x);CHKERRQ(ierr);
  ierr = PetscDSGetRefCoordArrays(prob, &x, &refSpaceDer);CHKERRQ(ierr);
  ierr = PetscDSGetWeakFormArrays(prob, NULL, NULL, &g0, &g1, &g2, &g3);CHKERRQ(ierr);
  ierr = PetscDSGetFaceTabulation(prob, &B, &D);CHKERRQ(ierr);
  ierr = PetscDSGetConstants(prob, &numConstants, &constants);CHKERRQ(ierr);
  if (probAux) {
    ierr = PetscDSGetNumFields(probAux, &NfAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = PetscDSGetDimensions(probAux, &NbAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponents(probAux, &NcAux);CHKERRQ(ierr);
    ierr = PetscDSGetComponentOffsets(probAux, &aOff);CHKERRQ(ierr);
    ierr = PetscDSGetComponentDerivativeOffsets(probAux, &aOff_x);CHKERRQ(ierr);
    ierr = PetscDSGetEvaluationArrays(probAux, &a, NULL, &a_x);CHKERRQ(ierr);
    ierr = PetscDSGetRefCoordArrays(probAux, NULL, &refSpaceDerAux);CHKERRQ(ierr);
    ierr = PetscDSGetFaceTabulation(probAux, &BAux, &DAux);CHKERRQ(ierr);
  }
  NbI = Nb[fieldI], NbJ = Nb[fieldJ];
  NcI = Nc[fieldI], NcJ = Nc[fieldJ];
  BI  = B[fieldI],  BJ  = B[fieldJ];
  DI  = D[fieldI],  DJ  = D[fieldJ];
  /* Initialize here in case the function is not defined */
  ierr = PetscMemzero(g0, NcI*NcJ * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g1, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g2, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(g3, NcI*NcJ*dim*dim * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
  Np = fgeom->numPoints;
  dE = fgeom->dimEmbed;
  isAffine = fgeom->isAffine;
  for (e = 0; e < Ne; ++e) {
    const PetscReal *v0   = &fgeom->v[e*Np*dE];
    const PetscReal *J    = &fgeom->J[e*Np*dE*dE];
    const PetscInt   face = fgeom->face[e][0];

    ierr = PetscQuadratureGetData(quad, NULL, &qNc, &Nq, &quadPoints, &quadWeights);CHKERRQ(ierr);
    if (qNc != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only supports scalar quadrature, not %D components\n", qNc);
    for (q = 0; q < Nq; ++q) {
      const PetscReal *BIq = &BI[(face*Nq+q)*NbI*NcI], *BJq = &BJ[(face*Nq+q)*NbJ*NcJ];
      const PetscReal *DIq = &DI[(face*Nq+q)*NbI*NcI*dim], *DJq = &DJ[(face*Nq+q)*NbJ*NcJ*dim];
      PetscReal  w;
      PetscInt f, g, fc, gc, c;
      const PetscReal *v;
      const PetscReal *invJ;
      const PetscReal *n;
      PetscReal detJ;

      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  quad point %d\n", q);CHKERRQ(ierr);}
      if (isAffine) {
        CoordinatesRefToReal(dE, dim-1, fgeom->xi, v0, J, &quadPoints[q*(dim-1)], x);
        v = x;
        invJ = &fgeom->suppInvJ[0][e*dE*dE];
        detJ = fgeom->detJ[e];
        n    = &fgeom->n[e*dE];
      } else {
        v = &v0[q*dE];
        invJ = &fgeom->suppInvJ[0][(e*Np+q)*dE*dE];
        detJ = fgeom->detJ[e*Np + q];
        n    = &fgeom->n[(e*Np+q)*dE];
      }
      w = detJ*quadWeights[q];

      if (coefficients) EvaluateFieldJets(dim, Nf, Nb, Nc, face*Nq+q, B, D, refSpaceDer, invJ, &coefficients[cOffset], &coefficients_t[cOffset], u, u_x, u_t);
      if (probAux)      EvaluateFieldJets(dim, NfAux, NbAux, NcAux, face*Nq+q, BAux, DAux, refSpaceDerAux, invJ, &coefficientsAux[cOffsetAux], NULL, a, a_x, NULL);
      if (g0_func) {
        ierr = PetscMemzero(g0, NcI*NcJ * sizeof(PetscScalar));CHKERRQ(ierr);
        g0_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, v, n, numConstants, constants, g0);
        for (c = 0; c < NcI*NcJ; ++c) g0[c] *= w;
      }
      if (g1_func) {
        PetscInt d, d2;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g1_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, v, n, numConstants, constants, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < dim; ++d) {
              g1[(fc*NcJ+gc)*dim+d] = 0.0;
              for (d2 = 0; d2 < dim; ++d2) g1[(fc*NcJ+gc)*dim+d] += invJ[d*dim+d2]*refSpaceDer[(fc*NcJ+gc)*dim+d2];
              g1[(fc*NcJ+gc)*dim+d] *= w;
            }
          }
        }
      }
      if (g2_func) {
        PetscInt d, d2;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g2_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, v, n, numConstants, constants, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < dim; ++d) {
              g2[(fc*NcJ+gc)*dim+d] = 0.0;
              for (d2 = 0; d2 < dim; ++d2) g2[(fc*NcJ+gc)*dim+d] += invJ[d*dim+d2]*refSpaceDer[(fc*NcJ+gc)*dim+d2];
              g2[(fc*NcJ+gc)*dim+d] *= w;
            }
          }
        }
      }
      if (g3_func) {
        PetscInt d, d2, dp, d3;
        ierr = PetscMemzero(refSpaceDer, NcI*NcJ*dim*dim * sizeof(PetscScalar));CHKERRQ(ierr);
        g3_func(dim, Nf, NfAux, uOff, uOff_x, u, u_t, u_x, aOff, aOff_x, a, NULL, a_x, t, u_tshift, v, n, numConstants, constants, refSpaceDer);
        for (fc = 0; fc < NcI; ++fc) {
          for (gc = 0; gc < NcJ; ++gc) {
            for (d = 0; d < dim; ++d) {
              for (dp = 0; dp < dim; ++dp) {
                g3[((fc*NcJ+gc)*dim+d)*dim+dp] = 0.0;
                for (d2 = 0; d2 < dim; ++d2) {
                  for (d3 = 0; d3 < dim; ++d3) {
                    g3[((fc*NcJ+gc)*dim+d)*dim+dp] += invJ[d*dim+d2]*refSpaceDer[((fc*NcJ+gc)*dim+d2)*dim+d3]*invJ[dp*dim+d3];
                  }
                }
                g3[((fc*NcJ+gc)*dim+d)*dim+dp] *= w;
              }
            }
          }
        }
      }

      for (f = 0; f < NbI; ++f) {
        for (fc = 0; fc < NcI; ++fc) {
          const PetscInt fidx = f*NcI+fc; /* Test function basis index */
          const PetscInt i    = offsetI+f; /* Element matrix row */
          for (g = 0; g < NbJ; ++g) {
            for (gc = 0; gc < NcJ; ++gc) {
              const PetscInt gidx = g*NcJ+gc; /* Trial function basis index */
              const PetscInt j    = offsetJ+g; /* Element matrix column */
              const PetscInt fOff = eOffset+i*totDim+j;
              PetscInt       d, d2;

              elemMat[fOff] += BIq[fidx]*g0[fc*NcJ+gc]*BJq[gidx];
              for (d = 0; d < dim; ++d) {
                elemMat[fOff] += BIq[fidx]*g1[(fc*NcJ+gc)*dim+d]*DJq[gidx*dim+d];
                elemMat[fOff] += DIq[fidx*dim+d]*g2[(fc*NcJ+gc)*dim+d]*BJq[gidx];
                for (d2 = 0; d2 < dim; ++d2) {
                  elemMat[fOff] += DIq[fidx*dim+d]*g3[((fc*NcJ+gc)*dim+d)*dim+d2]*DJq[gidx*dim+d2];
                }
              }
            }
          }
        }
      }
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

PetscErrorCode PetscFEInitialize_Basic(PetscFE fem)
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


