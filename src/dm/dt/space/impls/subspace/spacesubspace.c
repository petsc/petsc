#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

typedef struct {
  PetscDualSpace dualSubspace;
  PetscSpace     origSpace;
  PetscReal      *x;
  PetscReal      *x_alloc;
  PetscReal      *Jx;
  PetscReal      *Jx_alloc;
  PetscReal      *u;
  PetscReal      *u_alloc;
  PetscReal      *Ju;
  PetscReal      *Ju_alloc;
  PetscReal      *Q;
  PetscInt       Nb;
} PetscSpace_Subspace;

static PetscErrorCode PetscSpaceDestroy_Subspace(PetscSpace sp)
{
  PetscSpace_Subspace *subsp;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  subsp = (PetscSpace_Subspace *) sp->data;
  subsp->x = NULL;
  ierr = PetscFree(subsp->x_alloc);CHKERRQ(ierr);
  subsp->Jx = NULL;
  ierr = PetscFree(subsp->Jx_alloc);CHKERRQ(ierr);
  subsp->u = NULL;
  ierr = PetscFree(subsp->u_alloc);CHKERRQ(ierr);
  subsp->Ju = NULL;
  ierr = PetscFree(subsp->Ju_alloc);CHKERRQ(ierr);
  ierr = PetscFree(subsp->Q);CHKERRQ(ierr);
  ierr = PetscSpaceDestroy(&subsp->origSpace);CHKERRQ(ierr);
  ierr = PetscDualSpaceDestroy(&subsp->dualSubspace);CHKERRQ(ierr);
  ierr = PetscFree(subsp);CHKERRQ(ierr);
  sp->data = NULL;
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePolynomialGetTensor_C", NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceView_Subspace(PetscSpace sp, PetscViewer viewer)
{
  PetscBool           iascii;
  PetscSpace_Subspace *subsp;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  subsp = (PetscSpace_Subspace *) sp->data;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscInt origDim, subDim, origNc, subNc, o, s;

    ierr = PetscSpaceGetNumVariables(subsp->origSpace,&origDim);CHKERRQ(ierr);
    ierr = PetscSpaceGetNumComponents(subsp->origSpace,&origNc);CHKERRQ(ierr);
    ierr = PetscSpaceGetNumVariables(sp,&subDim);CHKERRQ(ierr);
    ierr = PetscSpaceGetNumComponents(sp,&subNc);CHKERRQ(ierr);
    if (subsp->x) {
      ierr = PetscViewerASCIIPrintf(viewer,"Subspace-to-space domain shift:\n\n");CHKERRQ(ierr);
      for (o = 0; o < origDim; o++) {
        ierr = PetscViewerASCIIPrintf(viewer," %g\n", (double)subsp->x[o]);CHKERRQ(ierr);
      }
    }
    if (subsp->Jx) {
      ierr = PetscViewerASCIIPrintf(viewer,"Subspace-to-space domain transform:\n\n");CHKERRQ(ierr);
      for (o = 0; o < origDim; o++) {
        ierr = PetscViewerASCIIPrintf(viewer," %g", (double)subsp->Jx[o * subDim + 0]);CHKERRQ(ierr);
        ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
        for (s = 1; s < subDim; s++) {
          ierr = PetscViewerASCIIPrintf(viewer," %g", (double)subsp->Jx[o * subDim + s]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
        ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
      }
    }
    if (subsp->u) {
      ierr = PetscViewerASCIIPrintf(viewer,"Space-to-subspace range shift:\n\n");CHKERRQ(ierr);
      for (o = 0; o < origNc; o++) {
        ierr = PetscViewerASCIIPrintf(viewer," %d\n", subsp->u[o]);CHKERRQ(ierr);
      }
    }
    if (subsp->Ju) {
      ierr = PetscViewerASCIIPrintf(viewer,"Space-to-subsace domain transform:\n");CHKERRQ(ierr);
      for (o = 0; o < origNc; o++) {
        ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
        for (s = 0; s < subNc; s++) {
          ierr = PetscViewerASCIIPrintf(viewer," %d", subsp->Ju[o * subNc + s]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"Original space:\n");CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscSpaceView(subsp->origSpace,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceEvaluate_Subspace(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscSpace_Subspace *subsp = (PetscSpace_Subspace *) sp->data;
  PetscSpace          origsp;
  PetscInt            origDim, subDim, origNc, subNc, subNb, origNb, i, j, k, l, m, n, o;
  PetscReal           *inpoints, *inB = NULL, *inD = NULL, *inH = NULL;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  origsp = subsp->origSpace;
  ierr = PetscSpaceGetNumVariables(sp,&subDim);CHKERRQ(ierr);
  ierr = PetscSpaceGetNumVariables(origsp,&origDim);CHKERRQ(ierr);
  ierr = PetscSpaceGetNumComponents(sp,&subNc);CHKERRQ(ierr);
  ierr = PetscSpaceGetNumComponents(origsp,&origNc);CHKERRQ(ierr);
  ierr = PetscSpaceGetDimension(sp,&subNb);CHKERRQ(ierr);
  ierr = PetscSpaceGetDimension(origsp,&origNb);CHKERRQ(ierr);
  ierr = DMGetWorkArray(sp->dm,npoints*origDim,MPIU_REAL,&inpoints);CHKERRQ(ierr);
  for (i = 0; i < npoints; i++) {
    if (subsp->x) {
      for (j = 0; j < origDim; j++) inpoints[i * origDim + j] = subsp->x[j];
    } else {
      for (j = 0; j < origDim; j++) inpoints[i * origDim + j] = 0.0;
    }
    if (subsp->Jx) {
      for (j = 0; j < origDim; j++) {
        for (k = 0; k < subDim; k++) {
          inpoints[i * origDim + j] += subsp->Jx[j * subDim + k] * points[i * subDim + k];
        }
      }
    } else {
      for (j = 0; j < PetscMin(subDim, origDim); j++) {
        inpoints[i * origDim + j] += points[i * subDim + j];
      }
    }
  }
  if (B) {
    ierr = DMGetWorkArray(sp->dm,npoints*origNb*origNc,MPIU_REAL,&inB);CHKERRQ(ierr);
  }
  if (D) {
    ierr = DMGetWorkArray(sp->dm,npoints*origNb*origNc*origDim,MPIU_REAL,&inD);CHKERRQ(ierr);
  }
  if (H) {
    ierr = DMGetWorkArray(sp->dm,npoints*origNb*origNc*origDim*origDim,MPIU_REAL,&inH);CHKERRQ(ierr);
  }
  ierr = PetscSpaceEvaluate(origsp,npoints,inpoints,inB,inD,inH);CHKERRQ(ierr);
  if (H) {
    PetscReal *phi, *psi;

    ierr = DMGetWorkArray(sp->dm,origNc*origDim*origDim,MPIU_REAL,&phi);CHKERRQ(ierr);
    ierr = DMGetWorkArray(sp->dm,origNc*subDim*subDim,MPIU_REAL,&psi);CHKERRQ(ierr);
    for (i = 0; i < npoints * subNb * subNc * subDim; i++) D[i] = 0.0;
    for (i = 0; i < subNb; i++) {
      const PetscReal *subq = &subsp->Q[i * origNb];

      for (j = 0; j < npoints; j++) {
        for (k = 0; k < origNc * origDim; k++) phi[k] = 0.;
        for (k = 0; k < origNc * subDim; k++) psi[k] = 0.;
        for (k = 0; k < origNb; k++) {
          for (l = 0; l < origNc * origDim * origDim; l++) {
            phi[l] += inH[(j * origNb + k) * origNc * origDim * origDim + l] * subq[k];
          }
        }
        if (subsp->Jx) {
          for (k = 0; k < subNc; k++) {
            for (l = 0; l < subDim; l++) {
              for (m = 0; m < origDim; m++) {
                for (n = 0; n < subDim; n++) {
                  for (o = 0; o < origDim; o++) {
                    psi[(k * subDim + l) * subDim + n] += subsp->Jx[m * subDim + l] * subsp->Jx[o * subDim + n] * phi[(k * origDim + m) * origDim + o];
                  }
                }
              }
            }
          }
        } else {
          for (k = 0; k < subNc; k++) {
            for (l = 0; l < PetscMin(subDim, origDim); l++) {
              for (m = 0; m < PetscMin(subDim, origDim); m++) {
                psi[(k * subDim + l) * subDim + m] += phi[(k * origDim + l) * origDim + m];
              }
            }
          }
        }
        if (subsp->Ju) {
          for (k = 0; k < subNc; k++) {
            for (l = 0; l < origNc; l++) {
              for (m = 0; m < subDim * subDim; m++) {
                H[((j * subNb + i) * subNc + k) * subDim * subDim + m] += subsp->Ju[k * origNc + l] * psi[l * subDim * subDim + m];
              }
            }
          }
        }
        else {
          for (k = 0; k < PetscMin(subNc, origNc); k++) {
            for (l = 0; l < subDim * subDim; l++) {
              H[((j * subNb + i) * subNc + k) * subDim * subDim + l] += psi[k * subDim * subDim + l];
            }
          }
        }
      }
    }
    ierr = DMRestoreWorkArray(sp->dm,subNc*origDim,MPIU_REAL,&psi);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(sp->dm,origNc*origDim,MPIU_REAL,&phi);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(sp->dm,npoints*origNb*origNc*origDim,MPIU_REAL,&inH);CHKERRQ(ierr);
  }
  if (D) {
    PetscReal *phi, *psi;

    ierr = DMGetWorkArray(sp->dm,origNc*origDim,MPIU_REAL,&phi);CHKERRQ(ierr);
    ierr = DMGetWorkArray(sp->dm,origNc*subDim,MPIU_REAL,&psi);CHKERRQ(ierr);
    for (i = 0; i < npoints * subNb * subNc * subDim; i++) D[i] = 0.0;
    for (i = 0; i < subNb; i++) {
      const PetscReal *subq = &subsp->Q[i * origNb];

      for (j = 0; j < npoints; j++) {
        for (k = 0; k < origNc * origDim; k++) phi[k] = 0.;
        for (k = 0; k < origNc * subDim; k++) psi[k] = 0.;
        for (k = 0; k < origNb; k++) {
          for (l = 0; l < origNc * origDim; l++) {
            phi[l] += inD[(j * origNb + k) * origNc * origDim + l] * subq[k];
          }
        }
        if (subsp->Jx) {
          for (k = 0; k < subNc; k++) {
            for (l = 0; l < subDim; l++) {
              for (m = 0; m < origDim; m++) {
                psi[k * subDim + l] += subsp->Jx[m * subDim + l] * phi[k * origDim + m];
              }
            }
          }
        } else {
          for (k = 0; k < subNc; k++) {
            for (l = 0; l < PetscMin(subDim, origDim); l++) {
              psi[k * subDim + l] += phi[k * origDim + l];
            }
          }
        }
        if (subsp->Ju) {
          for (k = 0; k < subNc; k++) {
            for (l = 0; l < origNc; l++) {
              for (m = 0; m < subDim; m++) {
                D[((j * subNb + i) * subNc + k) * subDim + m] += subsp->Ju[k * origNc + l] * psi[l * subDim + m];
              }
            }
          }
        }
        else {
          for (k = 0; k < PetscMin(subNc, origNc); k++) {
            for (l = 0; l < subDim; l++) {
              D[((j * subNb + i) * subNc + k) * subDim + l] += psi[k * subDim + l];
            }
          }
        }
      }
    }
    ierr = DMRestoreWorkArray(sp->dm,subNc*origDim,MPIU_REAL,&psi);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(sp->dm,origNc*origDim,MPIU_REAL,&phi);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(sp->dm,npoints*origNb*origNc*origDim,MPIU_REAL,&inD);CHKERRQ(ierr);
  }
  if (B) {
    PetscReal *phi;

    ierr = DMGetWorkArray(sp->dm,origNc,MPIU_REAL,&phi);CHKERRQ(ierr);
    if (subsp->u) {
      for (i = 0; i < npoints * subNb; i++) {
        for (j = 0; j < subNc; j++) B[i * subNc + j] = subsp->u[j];
      }
    } else {
      for (i = 0; i < npoints * subNb * subNc; i++) B[i] = 0.0;
    }
    for (i = 0; i < subNb; i++) {
      const PetscReal *subq = &subsp->Q[i * origNb];

      for (j = 0; j < npoints; j++) {
        for (k = 0; k < origNc; k++) phi[k] = 0.;
        for (k = 0; k < origNb; k++) {
          for (l = 0; l < origNc; l++) {
            phi[l] += inB[(j * origNb + k) * origNc + l] * subq[k];
          }
        }
        if (subsp->Ju) {
          for (k = 0; k < subNc; k++) {
            for (l = 0; l < origNc; l++) {
              B[(j * subNb + i) * subNc + k] += subsp->Ju[k * origNc + l] * phi[l];
            }
          }
        }
        else {
          for (k = 0; k < PetscMin(subNc, origNc); k++) {
            B[(j * subNb + i) * subNc + k] += phi[k];
          }
        }
      }
    }
    ierr = DMRestoreWorkArray(sp->dm,origNc,MPIU_REAL,&phi);CHKERRQ(ierr);
    ierr = DMRestoreWorkArray(sp->dm,npoints*origNb*origNc,MPIU_REAL,&inB);CHKERRQ(ierr);
  }
  ierr = DMRestoreWorkArray(sp->dm,npoints*origDim,MPIU_REAL,&inpoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Subspace(PetscSpace sp)
{
  PetscSpace_Subspace *subsp;

  PetscErrorCode ierr;
  ierr = PetscNewLog(sp,&subsp);CHKERRQ(ierr);
  sp->data = (void *) subsp;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetDimension_Subspace(PetscSpace sp, PetscInt *dim)
{
  PetscSpace_Subspace *subsp;

  PetscFunctionBegin;
  subsp = (PetscSpace_Subspace *) sp->data;
  *dim = subsp->Nb;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSetUp_Subspace(PetscSpace sp)
{
  const PetscReal     *x;
  const PetscReal     *Jx;
  const PetscReal     *u;
  const PetscReal     *Ju;
  PetscDualSpace      dualSubspace;
  PetscSpace          origSpace;
  PetscInt            origDim, subDim, origNc, subNc, origNb, subNb, f, i, j, numPoints, offset;
  PetscReal           *allPoints, *allWeights, *B, *V;
  DM                  dm;
  PetscSpace_Subspace *subsp;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  subsp = (PetscSpace_Subspace *) sp->data;
  x            = subsp->x;
  Jx           = subsp->Jx;
  u            = subsp->u;
  Ju           = subsp->Ju;
  origSpace    = subsp->origSpace;
  dualSubspace = subsp->dualSubspace;
  ierr = PetscSpaceGetNumComponents(origSpace,&origNc);CHKERRQ(ierr);
  ierr = PetscSpaceGetNumVariables(origSpace,&origDim);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDM(dualSubspace,&dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&subDim);CHKERRQ(ierr);
  ierr = PetscSpaceGetDimension(origSpace,&origNb);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDimension(dualSubspace,&subNb);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetNumComponents(dualSubspace,&subNc);CHKERRQ(ierr);

  for (f = 0, numPoints = 0; f < subNb; f++) {
    PetscQuadrature q;
    PetscInt        qNp;

    ierr = PetscDualSpaceGetFunctional(dualSubspace,f,&q);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(q,NULL,NULL,&qNp,NULL,NULL);CHKERRQ(ierr);
    numPoints += qNp;
  }
  ierr = PetscMalloc1(subNb*origNb,&V);CHKERRQ(ierr);
  ierr = PetscMalloc3(numPoints*origDim,&allPoints,numPoints*origNc,&allWeights,numPoints*origNb*origNc,&B);CHKERRQ(ierr);
  for (f = 0, offset = 0; f < subNb; f++) {
    PetscQuadrature q;
    PetscInt        qNp, p;
    const PetscReal *qp;
    const PetscReal *qw;

    ierr = PetscDualSpaceGetFunctional(dualSubspace,f,&q);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(q,NULL,NULL,&qNp,&qp,&qw);CHKERRQ(ierr);
    for (p = 0; p < qNp; p++, offset++) {
      if (x) {
        for (i = 0; i < origDim; i++) allPoints[origDim * offset + i] = x[i];
      } else {
        for (i = 0; i < origDim; i++) allPoints[origDim * offset + i] = 0.0;
      }
      if (Jx) {
        for (i = 0; i < origDim; i++) {
          for (j = 0; j < subDim; j++) {
            allPoints[origDim * offset + i] += Jx[i * subDim + j] * qp[j];
          }
        }
      } else {
        for (i = 0; i < PetscMin(subDim, origDim); i++) allPoints[origDim * offset + i] += qp[i];
      }
      for (i = 0; i < origNc; i++) allWeights[origNc * offset + i] = 0.0;
      if (Ju) {
        for (i = 0; i < origNc; i++) {
          for (j = 0; j < subNc; j++) {
            allWeights[offset * origNc + i] += qw[j] * Ju[j * origNc + i];
          }
        }
      } else {
        for (i = 0; i < PetscMin(subNc, origNc); i++) allWeights[offset * origNc + i] += qw[i];
      }
    }
  }
  ierr = PetscSpaceEvaluate(origSpace,numPoints,allPoints,B,NULL,NULL);CHKERRQ(ierr);
  for (f = 0, offset = 0; f < subNb; f++) {
    PetscInt b, p, s, qNp;
    PetscQuadrature q;
    const PetscReal *qw;

    ierr = PetscDualSpaceGetFunctional(dualSubspace,f,&q);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(q,NULL,NULL,&qNp,NULL,&qw);CHKERRQ(ierr);
    if (u) {
      for (b = 0; b < origNb; b++) {
        for (s = 0; s < subNc; s++) {
          V[f * origNb + b] += qw[s] * u[s];
        }
      }
    } else {
      for (b = 0; b < origNb; b++) V[f * origNb + b] = 0.0;
    }
    for (p = 0; p < qNp; p++, offset++) {
      for (b = 0; b < origNb; b++) {
        for (s = 0; s < origNc; s++) {
          V[f * origNb + b] += B[(offset * origNb + b) * origNc + s] * allWeights[offset * origNc + s];
        }
      }
    }
  }
  /* orthnormalize rows of V */
  for (f = 0; f < subNb; f++) {
    PetscReal rho = 0.0, scal;

    for (i = 0; i < origNb; i++) rho += PetscSqr(V[f * origNb + i]);

    scal = 1. / PetscSqrtReal(rho);

    for (i = 0; i < origNb; i++) V[f * origNb + i] *= scal;
    for (j = f + 1; j < subNb; j++) {
      for (i = 0, scal = 0.; i < origNb; i++) scal += V[f * origNb + i] * V[j * origNb + i];
      for (i = 0; i < origNb; i++) V[j * origNb + i] -= V[f * origNb + i] * scal;
    }
  }
  ierr = PetscFree3(allPoints,allWeights,B);CHKERRQ(ierr);
  subsp->Q = V;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpacePolynomialGetTensor_Subspace(PetscSpace sp, PetscBool *poly)
{
  PetscSpace_Subspace *subsp = (PetscSpace_Subspace *) sp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *poly = PETSC_FALSE;
  ierr = PetscSpacePolynomialGetTensor(subsp->origSpace,poly);CHKERRQ(ierr);
  if (*poly) {
    if (subsp->Jx) {
      PetscInt subDim, origDim, i, j;
      PetscInt maxnnz;

      ierr = PetscSpaceGetNumVariables(subsp->origSpace,&origDim);CHKERRQ(ierr);
      ierr = PetscSpaceGetNumVariables(sp,&subDim);CHKERRQ(ierr);
      maxnnz = 0;
      for (i = 0; i < origDim; i++) {
        PetscInt nnz = 0;

        for (j = 0; j < subDim; j++) nnz += (subsp->Jx[i * subDim + j] != 0.);
        maxnnz = PetscMax(maxnnz,nnz);
      }
      for (j = 0; j < subDim; j++) {
        PetscInt nnz = 0;

        for (i = 0; i < origDim; i++) nnz += (subsp->Jx[i * subDim + j] != 0.);
        maxnnz = PetscMax(maxnnz,nnz);
      }
      if (maxnnz > 1) *poly = PETSC_FALSE;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceInitialize_Subspace(PetscSpace sp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sp->ops->setup = PetscSpaceSetUp_Subspace;
  sp->ops->view  = PetscSpaceView_Subspace;
  sp->ops->destroy  = PetscSpaceDestroy_Subspace;
  sp->ops->getdimension  = PetscSpaceGetDimension_Subspace;
  sp->ops->evaluate = PetscSpaceEvaluate_Subspace;
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscSpacePolynomialGetTensor_C", PetscSpacePolynomialGetTensor_Subspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscSpaceCreateSubspace(PetscSpace origSpace, PetscDualSpace dualSubspace, PetscReal *x, PetscReal *Jx, PetscReal *u, PetscReal *Ju, PetscCopyMode copymode, PetscSpace *subspace)
{
  PetscSpace_Subspace *subsp;
  PetscInt            origDim, subDim, origNc, subNc, subNb;
  PetscInt            order;
  DM                  dm;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(origSpace,PETSCSPACE_CLASSID,1);
  PetscValidHeaderSpecific(dualSubspace,PETSCDUALSPACE_CLASSID,2);
  if (x) PetscValidRealPointer(x,3);
  if (Jx) PetscValidRealPointer(Jx,4);
  if (u) PetscValidRealPointer(u,5);
  if (Ju) PetscValidRealPointer(Ju,6);
  PetscValidPointer(subspace,8);
  ierr = PetscSpaceGetNumComponents(origSpace,&origNc);CHKERRQ(ierr);
  ierr = PetscSpaceGetNumVariables(origSpace,&origDim);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDM(dualSubspace,&dm);CHKERRQ(ierr);
  ierr = DMGetDimension(dm,&subDim);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDimension(dualSubspace,&subNb);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetNumComponents(dualSubspace,&subNc);CHKERRQ(ierr);
  ierr = PetscSpaceCreate(PetscObjectComm((PetscObject)origSpace),subspace);CHKERRQ(ierr);
  ierr = PetscSpaceSetType(*subspace,PETSCSPACESUBSPACE);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumVariables(*subspace,subDim);CHKERRQ(ierr);
  ierr = PetscSpaceSetNumComponents(*subspace,subNc);CHKERRQ(ierr);
  ierr = PetscSpaceGetDegree(origSpace,&order,NULL);CHKERRQ(ierr);
  ierr = PetscSpaceSetDegree(*subspace,order,PETSC_DETERMINE);CHKERRQ(ierr);
  subsp = (PetscSpace_Subspace *) (*subspace)->data;
  subsp->Nb = subNb;
  switch (copymode) {
  case PETSC_OWN_POINTER:
    if (x) subsp->x_alloc = x;
    if (Jx) subsp->Jx_alloc = Jx;
    if (u) subsp->u_alloc = u;
    if (Ju) subsp->Ju_alloc = Ju;
  case PETSC_USE_POINTER:
    if (x) subsp->x = x;
    if (Jx) subsp->Jx = Jx;
    if (u) subsp->u = u;
    if (Ju) subsp->Ju = Ju;
    break;
  case PETSC_COPY_VALUES:
    if (x) {
      ierr = PetscMalloc1(origDim,&subsp->x_alloc);CHKERRQ(ierr);
      ierr = PetscArraycpy(subsp->x_alloc,x,origDim);CHKERRQ(ierr);
      subsp->x = subsp->x_alloc;
    }
    if (Jx) {
      ierr = PetscMalloc1(origDim * subDim,&subsp->Jx_alloc);CHKERRQ(ierr);
      ierr = PetscArraycpy(subsp->Jx_alloc,Jx,origDim * subDim);CHKERRQ(ierr);
      subsp->Jx = subsp->Jx_alloc;
    }
    if (u) {
      ierr = PetscMalloc1(subNc,&subsp->u_alloc);CHKERRQ(ierr);
      ierr = PetscArraycpy(subsp->u_alloc,u,subNc);CHKERRQ(ierr);
      subsp->u = subsp->u_alloc;
    }
    if (Ju) {
      ierr = PetscMalloc1(origNc * subNc,&subsp->Ju_alloc);CHKERRQ(ierr);
      ierr = PetscArraycpy(subsp->Ju_alloc,Ju,origNc * subNc);CHKERRQ(ierr);
      subsp->Ju = subsp->Ju_alloc;
    }
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)origSpace),PETSC_ERR_ARG_OUTOFRANGE,"Unknown copy mode");
  }
  ierr = PetscObjectReference((PetscObject)origSpace);CHKERRQ(ierr);
  subsp->origSpace = origSpace;
  ierr = PetscObjectReference((PetscObject)dualSubspace);CHKERRQ(ierr);
  subsp->dualSubspace = dualSubspace;
  ierr = PetscSpaceInitialize_Subspace(*subspace);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

