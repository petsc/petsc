#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

typedef struct {
  PetscDualSpace dualSubspace;
  PetscSpace     origSpace;
  PetscReal     *x;
  PetscReal     *x_alloc;
  PetscReal     *Jx;
  PetscReal     *Jx_alloc;
  PetscReal     *u;
  PetscReal     *u_alloc;
  PetscReal     *Ju;
  PetscReal     *Ju_alloc;
  PetscReal     *Q;
  PetscInt       Nb;
} PetscSpace_Subspace;

static PetscErrorCode PetscSpaceDestroy_Subspace(PetscSpace sp)
{
  PetscSpace_Subspace *subsp;

  PetscFunctionBegin;
  subsp    = (PetscSpace_Subspace *)sp->data;
  subsp->x = NULL;
  PetscCall(PetscFree(subsp->x_alloc));
  subsp->Jx = NULL;
  PetscCall(PetscFree(subsp->Jx_alloc));
  subsp->u = NULL;
  PetscCall(PetscFree(subsp->u_alloc));
  subsp->Ju = NULL;
  PetscCall(PetscFree(subsp->Ju_alloc));
  PetscCall(PetscFree(subsp->Q));
  PetscCall(PetscSpaceDestroy(&subsp->origSpace));
  PetscCall(PetscDualSpaceDestroy(&subsp->dualSubspace));
  PetscCall(PetscFree(subsp));
  sp->data = NULL;
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscSpacePolynomialGetTensor_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceView_Subspace(PetscSpace sp, PetscViewer viewer)
{
  PetscBool            iascii;
  PetscSpace_Subspace *subsp;

  PetscFunctionBegin;
  subsp = (PetscSpace_Subspace *)sp->data;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscInt origDim, subDim, origNc, subNc, o, s;

    PetscCall(PetscSpaceGetNumVariables(subsp->origSpace, &origDim));
    PetscCall(PetscSpaceGetNumComponents(subsp->origSpace, &origNc));
    PetscCall(PetscSpaceGetNumVariables(sp, &subDim));
    PetscCall(PetscSpaceGetNumComponents(sp, &subNc));
    if (subsp->x) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Subspace-to-space domain shift:\n\n"));
      for (o = 0; o < origDim; o++) PetscCall(PetscViewerASCIIPrintf(viewer, " %g\n", (double)subsp->x[o]));
    }
    if (subsp->Jx) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Subspace-to-space domain transform:\n\n"));
      for (o = 0; o < origDim; o++) {
        PetscCall(PetscViewerASCIIPrintf(viewer, " %g", (double)subsp->Jx[o * subDim + 0]));
        PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
        for (s = 1; s < subDim; s++) PetscCall(PetscViewerASCIIPrintf(viewer, " %g", (double)subsp->Jx[o * subDim + s]));
        PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
        PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
      }
    }
    if (subsp->u) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Space-to-subspace range shift:\n\n"));
      for (o = 0; o < origNc; o++) PetscCall(PetscViewerASCIIPrintf(viewer, " %g\n", (double)subsp->u[o]));
    }
    if (subsp->Ju) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "Space-to-subsace domain transform:\n"));
      for (o = 0; o < origNc; o++) {
        PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
        for (s = 0; s < subNc; s++) PetscCall(PetscViewerASCIIPrintf(viewer, " %g", (double)subsp->Ju[o * subNc + s]));
        PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
      }
      PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "Original space:\n"));
  }
  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscSpaceView(subsp->origSpace, viewer));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceEvaluate_Subspace(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscSpace_Subspace *subsp = (PetscSpace_Subspace *)sp->data;
  PetscSpace           origsp;
  PetscInt             origDim, subDim, origNc, subNc, subNb, origNb, i, j, k, l, m, n, o;
  PetscReal           *inpoints, *inB = NULL, *inD = NULL, *inH = NULL;

  PetscFunctionBegin;
  origsp = subsp->origSpace;
  PetscCall(PetscSpaceGetNumVariables(sp, &subDim));
  PetscCall(PetscSpaceGetNumVariables(origsp, &origDim));
  PetscCall(PetscSpaceGetNumComponents(sp, &subNc));
  PetscCall(PetscSpaceGetNumComponents(origsp, &origNc));
  PetscCall(PetscSpaceGetDimension(sp, &subNb));
  PetscCall(PetscSpaceGetDimension(origsp, &origNb));
  PetscCall(DMGetWorkArray(sp->dm, npoints * origDim, MPIU_REAL, &inpoints));
  for (i = 0; i < npoints; i++) {
    if (subsp->x) {
      for (j = 0; j < origDim; j++) inpoints[i * origDim + j] = subsp->x[j];
    } else {
      for (j = 0; j < origDim; j++) inpoints[i * origDim + j] = 0.0;
    }
    if (subsp->Jx) {
      for (j = 0; j < origDim; j++) {
        for (k = 0; k < subDim; k++) inpoints[i * origDim + j] += subsp->Jx[j * subDim + k] * points[i * subDim + k];
      }
    } else {
      for (j = 0; j < PetscMin(subDim, origDim); j++) inpoints[i * origDim + j] += points[i * subDim + j];
    }
  }
  if (B) PetscCall(DMGetWorkArray(sp->dm, npoints * origNb * origNc, MPIU_REAL, &inB));
  if (D) PetscCall(DMGetWorkArray(sp->dm, npoints * origNb * origNc * origDim, MPIU_REAL, &inD));
  if (H) PetscCall(DMGetWorkArray(sp->dm, npoints * origNb * origNc * origDim * origDim, MPIU_REAL, &inH));
  PetscCall(PetscSpaceEvaluate(origsp, npoints, inpoints, inB, inD, inH));
  if (H) {
    PetscReal *phi, *psi;

    PetscCall(DMGetWorkArray(sp->dm, origNc * origDim * origDim, MPIU_REAL, &phi));
    PetscCall(DMGetWorkArray(sp->dm, origNc * subDim * subDim, MPIU_REAL, &psi));
    for (i = 0; i < npoints * subNb * subNc * subDim; i++) D[i] = 0.0;
    for (i = 0; i < subNb; i++) {
      const PetscReal *subq = &subsp->Q[i * origNb];

      for (j = 0; j < npoints; j++) {
        for (k = 0; k < origNc * origDim; k++) phi[k] = 0.;
        for (k = 0; k < origNc * subDim; k++) psi[k] = 0.;
        for (k = 0; k < origNb; k++) {
          for (l = 0; l < origNc * origDim * origDim; l++) phi[l] += inH[(j * origNb + k) * origNc * origDim * origDim + l] * subq[k];
        }
        if (subsp->Jx) {
          for (k = 0; k < subNc; k++) {
            for (l = 0; l < subDim; l++) {
              for (m = 0; m < origDim; m++) {
                for (n = 0; n < subDim; n++) {
                  for (o = 0; o < origDim; o++) psi[(k * subDim + l) * subDim + n] += subsp->Jx[m * subDim + l] * subsp->Jx[o * subDim + n] * phi[(k * origDim + m) * origDim + o];
                }
              }
            }
          }
        } else {
          for (k = 0; k < subNc; k++) {
            for (l = 0; l < PetscMin(subDim, origDim); l++) {
              for (m = 0; m < PetscMin(subDim, origDim); m++) psi[(k * subDim + l) * subDim + m] += phi[(k * origDim + l) * origDim + m];
            }
          }
        }
        if (subsp->Ju) {
          for (k = 0; k < subNc; k++) {
            for (l = 0; l < origNc; l++) {
              for (m = 0; m < subDim * subDim; m++) H[((j * subNb + i) * subNc + k) * subDim * subDim + m] += subsp->Ju[k * origNc + l] * psi[l * subDim * subDim + m];
            }
          }
        } else {
          for (k = 0; k < PetscMin(subNc, origNc); k++) {
            for (l = 0; l < subDim * subDim; l++) H[((j * subNb + i) * subNc + k) * subDim * subDim + l] += psi[k * subDim * subDim + l];
          }
        }
      }
    }
    PetscCall(DMRestoreWorkArray(sp->dm, subNc * origDim, MPIU_REAL, &psi));
    PetscCall(DMRestoreWorkArray(sp->dm, origNc * origDim, MPIU_REAL, &phi));
    PetscCall(DMRestoreWorkArray(sp->dm, npoints * origNb * origNc * origDim, MPIU_REAL, &inH));
  }
  if (D) {
    PetscReal *phi, *psi;

    PetscCall(DMGetWorkArray(sp->dm, origNc * origDim, MPIU_REAL, &phi));
    PetscCall(DMGetWorkArray(sp->dm, origNc * subDim, MPIU_REAL, &psi));
    for (i = 0; i < npoints * subNb * subNc * subDim; i++) D[i] = 0.0;
    for (i = 0; i < subNb; i++) {
      const PetscReal *subq = &subsp->Q[i * origNb];

      for (j = 0; j < npoints; j++) {
        for (k = 0; k < origNc * origDim; k++) phi[k] = 0.;
        for (k = 0; k < origNc * subDim; k++) psi[k] = 0.;
        for (k = 0; k < origNb; k++) {
          for (l = 0; l < origNc * origDim; l++) phi[l] += inD[(j * origNb + k) * origNc * origDim + l] * subq[k];
        }
        if (subsp->Jx) {
          for (k = 0; k < subNc; k++) {
            for (l = 0; l < subDim; l++) {
              for (m = 0; m < origDim; m++) psi[k * subDim + l] += subsp->Jx[m * subDim + l] * phi[k * origDim + m];
            }
          }
        } else {
          for (k = 0; k < subNc; k++) {
            for (l = 0; l < PetscMin(subDim, origDim); l++) psi[k * subDim + l] += phi[k * origDim + l];
          }
        }
        if (subsp->Ju) {
          for (k = 0; k < subNc; k++) {
            for (l = 0; l < origNc; l++) {
              for (m = 0; m < subDim; m++) D[((j * subNb + i) * subNc + k) * subDim + m] += subsp->Ju[k * origNc + l] * psi[l * subDim + m];
            }
          }
        } else {
          for (k = 0; k < PetscMin(subNc, origNc); k++) {
            for (l = 0; l < subDim; l++) D[((j * subNb + i) * subNc + k) * subDim + l] += psi[k * subDim + l];
          }
        }
      }
    }
    PetscCall(DMRestoreWorkArray(sp->dm, subNc * origDim, MPIU_REAL, &psi));
    PetscCall(DMRestoreWorkArray(sp->dm, origNc * origDim, MPIU_REAL, &phi));
    PetscCall(DMRestoreWorkArray(sp->dm, npoints * origNb * origNc * origDim, MPIU_REAL, &inD));
  }
  if (B) {
    PetscReal *phi;

    PetscCall(DMGetWorkArray(sp->dm, origNc, MPIU_REAL, &phi));
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
          for (l = 0; l < origNc; l++) phi[l] += inB[(j * origNb + k) * origNc + l] * subq[k];
        }
        if (subsp->Ju) {
          for (k = 0; k < subNc; k++) {
            for (l = 0; l < origNc; l++) B[(j * subNb + i) * subNc + k] += subsp->Ju[k * origNc + l] * phi[l];
          }
        } else {
          for (k = 0; k < PetscMin(subNc, origNc); k++) B[(j * subNb + i) * subNc + k] += phi[k];
        }
      }
    }
    PetscCall(DMRestoreWorkArray(sp->dm, origNc, MPIU_REAL, &phi));
    PetscCall(DMRestoreWorkArray(sp->dm, npoints * origNb * origNc, MPIU_REAL, &inB));
  }
  PetscCall(DMRestoreWorkArray(sp->dm, npoints * origDim, MPIU_REAL, &inpoints));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Subspace(PetscSpace sp)
{
  PetscSpace_Subspace *subsp;

  PetscFunctionBegin;
  PetscCall(PetscNew(&subsp));
  sp->data = (void *)subsp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceGetDimension_Subspace(PetscSpace sp, PetscInt *dim)
{
  PetscSpace_Subspace *subsp;

  PetscFunctionBegin;
  subsp = (PetscSpace_Subspace *)sp->data;
  *dim  = subsp->Nb;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceSetUp_Subspace(PetscSpace sp)
{
  const PetscReal     *x;
  const PetscReal     *Jx;
  const PetscReal     *u;
  const PetscReal     *Ju;
  PetscDualSpace       dualSubspace;
  PetscSpace           origSpace;
  PetscInt             origDim, subDim, origNc, subNc, origNb, subNb, f, i, j, numPoints, offset;
  PetscReal           *allPoints, *allWeights, *B, *V;
  DM                   dm;
  PetscSpace_Subspace *subsp;

  PetscFunctionBegin;
  subsp        = (PetscSpace_Subspace *)sp->data;
  x            = subsp->x;
  Jx           = subsp->Jx;
  u            = subsp->u;
  Ju           = subsp->Ju;
  origSpace    = subsp->origSpace;
  dualSubspace = subsp->dualSubspace;
  PetscCall(PetscSpaceGetNumComponents(origSpace, &origNc));
  PetscCall(PetscSpaceGetNumVariables(origSpace, &origDim));
  PetscCall(PetscDualSpaceGetDM(dualSubspace, &dm));
  PetscCall(DMGetDimension(dm, &subDim));
  PetscCall(PetscSpaceGetDimension(origSpace, &origNb));
  PetscCall(PetscDualSpaceGetDimension(dualSubspace, &subNb));
  PetscCall(PetscDualSpaceGetNumComponents(dualSubspace, &subNc));

  for (f = 0, numPoints = 0; f < subNb; f++) {
    PetscQuadrature q;
    PetscInt        qNp;

    PetscCall(PetscDualSpaceGetFunctional(dualSubspace, f, &q));
    PetscCall(PetscQuadratureGetData(q, NULL, NULL, &qNp, NULL, NULL));
    numPoints += qNp;
  }
  PetscCall(PetscMalloc1(subNb * origNb, &V));
  PetscCall(PetscMalloc3(numPoints * origDim, &allPoints, numPoints * origNc, &allWeights, numPoints * origNb * origNc, &B));
  for (f = 0, offset = 0; f < subNb; f++) {
    PetscQuadrature  q;
    PetscInt         qNp, p;
    const PetscReal *qp;
    const PetscReal *qw;

    PetscCall(PetscDualSpaceGetFunctional(dualSubspace, f, &q));
    PetscCall(PetscQuadratureGetData(q, NULL, NULL, &qNp, &qp, &qw));
    for (p = 0; p < qNp; p++, offset++) {
      if (x) {
        for (i = 0; i < origDim; i++) allPoints[origDim * offset + i] = x[i];
      } else {
        for (i = 0; i < origDim; i++) allPoints[origDim * offset + i] = 0.0;
      }
      if (Jx) {
        for (i = 0; i < origDim; i++) {
          for (j = 0; j < subDim; j++) allPoints[origDim * offset + i] += Jx[i * subDim + j] * qp[j];
        }
      } else {
        for (i = 0; i < PetscMin(subDim, origDim); i++) allPoints[origDim * offset + i] += qp[i];
      }
      for (i = 0; i < origNc; i++) allWeights[origNc * offset + i] = 0.0;
      if (Ju) {
        for (i = 0; i < origNc; i++) {
          for (j = 0; j < subNc; j++) allWeights[offset * origNc + i] += qw[j] * Ju[j * origNc + i];
        }
      } else {
        for (i = 0; i < PetscMin(subNc, origNc); i++) allWeights[offset * origNc + i] += qw[i];
      }
    }
  }
  PetscCall(PetscSpaceEvaluate(origSpace, numPoints, allPoints, B, NULL, NULL));
  for (f = 0, offset = 0; f < subNb; f++) {
    PetscInt         b, p, s, qNp;
    PetscQuadrature  q;
    const PetscReal *qw;

    PetscCall(PetscDualSpaceGetFunctional(dualSubspace, f, &q));
    PetscCall(PetscQuadratureGetData(q, NULL, NULL, &qNp, NULL, &qw));
    if (u) {
      for (b = 0; b < origNb; b++) {
        for (s = 0; s < subNc; s++) V[f * origNb + b] += qw[s] * u[s];
      }
    } else {
      for (b = 0; b < origNb; b++) V[f * origNb + b] = 0.0;
    }
    for (p = 0; p < qNp; p++, offset++) {
      for (b = 0; b < origNb; b++) {
        for (s = 0; s < origNc; s++) V[f * origNb + b] += B[(offset * origNb + b) * origNc + s] * allWeights[offset * origNc + s];
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
  PetscCall(PetscFree3(allPoints, allWeights, B));
  subsp->Q = V;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpacePolynomialGetTensor_Subspace(PetscSpace sp, PetscBool *poly)
{
  PetscSpace_Subspace *subsp = (PetscSpace_Subspace *)sp->data;

  PetscFunctionBegin;
  *poly = PETSC_FALSE;
  PetscCall(PetscSpacePolynomialGetTensor(subsp->origSpace, poly));
  if (*poly) {
    if (subsp->Jx) {
      PetscInt subDim, origDim, i, j;
      PetscInt maxnnz;

      PetscCall(PetscSpaceGetNumVariables(subsp->origSpace, &origDim));
      PetscCall(PetscSpaceGetNumVariables(sp, &subDim));
      maxnnz = 0;
      for (i = 0; i < origDim; i++) {
        PetscInt nnz = 0;

        for (j = 0; j < subDim; j++) nnz += (subsp->Jx[i * subDim + j] != 0.);
        maxnnz = PetscMax(maxnnz, nnz);
      }
      for (j = 0; j < subDim; j++) {
        PetscInt nnz = 0;

        for (i = 0; i < origDim; i++) nnz += (subsp->Jx[i * subDim + j] != 0.);
        maxnnz = PetscMax(maxnnz, nnz);
      }
      if (maxnnz > 1) *poly = PETSC_FALSE;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscSpaceInitialize_Subspace(PetscSpace sp)
{
  PetscFunctionBegin;
  sp->ops->setup        = PetscSpaceSetUp_Subspace;
  sp->ops->view         = PetscSpaceView_Subspace;
  sp->ops->destroy      = PetscSpaceDestroy_Subspace;
  sp->ops->getdimension = PetscSpaceGetDimension_Subspace;
  sp->ops->evaluate     = PetscSpaceEvaluate_Subspace;
  PetscCall(PetscObjectComposeFunction((PetscObject)sp, "PetscSpacePolynomialGetTensor_C", PetscSpacePolynomialGetTensor_Subspace));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
    PetscSpaceCreateSubspace - creates a subspace from a an `origSpace` and its dual `dualSubspace`

    Input Parameters:
+   origSpace - the original `PetscSpace`
.   dualSubspace - no idea
.   x - no idea
.   Jx - no idea
.   u - no idea
.   Ju - no idea
-   copymode - whether to copy, borrow, or own some of the input arrays I guess

    Output Parameter:
.   subspace - the subspace

    Level: advanced

.seealso: `PetscSpace`, `PetscDualSpace`, `PetscCopyMode`, `PetscSpaceType`
@*/
PetscErrorCode PetscSpaceCreateSubspace(PetscSpace origSpace, PetscDualSpace dualSubspace, PetscReal *x, PetscReal *Jx, PetscReal *u, PetscReal *Ju, PetscCopyMode copymode, PetscSpace *subspace)
{
  PetscSpace_Subspace *subsp;
  PetscInt             origDim, subDim, origNc, subNc, subNb;
  PetscInt             order;
  DM                   dm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(origSpace, PETSCSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(dualSubspace, PETSCDUALSPACE_CLASSID, 2);
  if (x) PetscValidRealPointer(x, 3);
  if (Jx) PetscValidRealPointer(Jx, 4);
  if (u) PetscValidRealPointer(u, 5);
  if (Ju) PetscValidRealPointer(Ju, 6);
  PetscValidPointer(subspace, 8);
  PetscCall(PetscSpaceGetNumComponents(origSpace, &origNc));
  PetscCall(PetscSpaceGetNumVariables(origSpace, &origDim));
  PetscCall(PetscDualSpaceGetDM(dualSubspace, &dm));
  PetscCall(DMGetDimension(dm, &subDim));
  PetscCall(PetscDualSpaceGetDimension(dualSubspace, &subNb));
  PetscCall(PetscDualSpaceGetNumComponents(dualSubspace, &subNc));
  PetscCall(PetscSpaceCreate(PetscObjectComm((PetscObject)origSpace), subspace));
  PetscCall(PetscSpaceSetType(*subspace, PETSCSPACESUBSPACE));
  PetscCall(PetscSpaceSetNumVariables(*subspace, subDim));
  PetscCall(PetscSpaceSetNumComponents(*subspace, subNc));
  PetscCall(PetscSpaceGetDegree(origSpace, &order, NULL));
  PetscCall(PetscSpaceSetDegree(*subspace, order, PETSC_DETERMINE));
  subsp     = (PetscSpace_Subspace *)(*subspace)->data;
  subsp->Nb = subNb;
  switch (copymode) {
  case PETSC_OWN_POINTER:
    if (x) subsp->x_alloc = x;
    if (Jx) subsp->Jx_alloc = Jx;
    if (u) subsp->u_alloc = u;
    if (Ju) subsp->Ju_alloc = Ju;
    /* fall through */
  case PETSC_USE_POINTER:
    if (x) subsp->x = x;
    if (Jx) subsp->Jx = Jx;
    if (u) subsp->u = u;
    if (Ju) subsp->Ju = Ju;
    break;
  case PETSC_COPY_VALUES:
    if (x) {
      PetscCall(PetscMalloc1(origDim, &subsp->x_alloc));
      PetscCall(PetscArraycpy(subsp->x_alloc, x, origDim));
      subsp->x = subsp->x_alloc;
    }
    if (Jx) {
      PetscCall(PetscMalloc1(origDim * subDim, &subsp->Jx_alloc));
      PetscCall(PetscArraycpy(subsp->Jx_alloc, Jx, origDim * subDim));
      subsp->Jx = subsp->Jx_alloc;
    }
    if (u) {
      PetscCall(PetscMalloc1(subNc, &subsp->u_alloc));
      PetscCall(PetscArraycpy(subsp->u_alloc, u, subNc));
      subsp->u = subsp->u_alloc;
    }
    if (Ju) {
      PetscCall(PetscMalloc1(origNc * subNc, &subsp->Ju_alloc));
      PetscCall(PetscArraycpy(subsp->Ju_alloc, Ju, origNc * subNc));
      subsp->Ju = subsp->Ju_alloc;
    }
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)origSpace), PETSC_ERR_ARG_OUTOFRANGE, "Unknown copy mode");
  }
  PetscCall(PetscObjectReference((PetscObject)origSpace));
  subsp->origSpace = origSpace;
  PetscCall(PetscObjectReference((PetscObject)dualSubspace));
  subsp->dualSubspace = dualSubspace;
  PetscCall(PetscSpaceInitialize_Subspace(*subspace));
  PetscFunctionReturn(PETSC_SUCCESS);
}
