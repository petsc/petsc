/* Internal and DMStag-specific functions related to multigrid */
#include <petsc/private/dmstagimpl.h> /*I  "petscdmstag.h"   I*/

/*@
  DMStagRestrictSimple - restricts data from a fine to a coarse `DMSTAG`, in the simplest way

  Values on coarse cells are averages of all fine cells that they cover.
  Thus, values on vertices are injected, values on edges are averages
  of the underlying two fine edges, and values on elements in
  d dimensions are averages of $2^d$ underlying elements.

  Input Parameters:
+ dmf - fine `DM`
. xf  - data on fine `DM`
- dmc - coarse `DM`

  Output Parameter:
. xc - data on coarse `DM`

  Level: advanced

.seealso: [](ch_stag), `DMSTAG`, `DM`, `DMRestrict()`, `DMCoarsen()`, `DMCreateInjection()`
@*/
PetscErrorCode DMStagRestrictSimple(DM dmf, Vec xf, DM dmc, Vec xc)
{
  PetscInt dim;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dmf, &dim));
  if (PetscDefined(USE_DEBUG)) {
    PetscInt d, nf[DMSTAG_MAX_DIM], nc[DMSTAG_MAX_DIM], doff[DMSTAG_MAX_STRATA], dofc[DMSTAG_MAX_STRATA];

    PetscCall(DMStagGetLocalSizes(dmf, &nf[0], &nf[1], &nf[2]));
    PetscCall(DMStagGetLocalSizes(dmc, &nc[0], &nc[1], &nc[2]));
    for (d = 0; d < dim; ++d) PetscCheck(nf[d] % nc[d] == 0, PetscObjectComm((PetscObject)dmf), PETSC_ERR_PLIB, "Not implemented for non-integer refinement factor");
    PetscCall(DMStagGetDOF(dmf, &doff[0], &doff[1], &doff[2], &doff[3]));
    PetscCall(DMStagGetDOF(dmc, &dofc[0], &dofc[1], &dofc[2], &dofc[3]));
    for (d = 0; d < dim + 1; ++d) PetscCheck(doff[d] == dofc[d], PetscObjectComm((PetscObject)dmf), PETSC_ERR_PLIB, "Cannot transfer between DMStag objects with different dof on each stratum");
    {
      PetscInt size_local, entries_local;

      PetscCall(DMStagGetEntriesLocal(dmf, &entries_local));
      PetscCall(VecGetLocalSize(xf, &size_local));
      PetscCheck(entries_local == size_local, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "Fine vector must be a local vector of size %" PetscInt_FMT ", but a vector of size %" PetscInt_FMT " was supplied", entries_local, size_local);
    }
    {
      PetscInt size_local, entries_local;

      PetscCall(DMStagGetEntriesLocal(dmc, &entries_local));
      PetscCall(VecGetLocalSize(xc, &size_local));
      PetscCheck(entries_local == size_local, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "Coarse vector must be a local vector of size %" PetscInt_FMT ", but a vector of size %" PetscInt_FMT " was supplied", entries_local, size_local);
    }
  }
  switch (dim) {
  case 1:
    PetscCall(DMStagRestrictSimple_1d(dmf, xf, dmc, xc));
    break;
  case 2:
    PetscCall(DMStagRestrictSimple_2d(dmf, xf, dmc, xc));
    break;
  case 3:
    PetscCall(DMStagRestrictSimple_3d(dmf, xf, dmc, xc));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dmf), PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension %" PetscInt_FMT, dim);
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetInterpolationCoefficientVertex_Private(PetscInt index, PetscInt factor, PetscScalar *a)
{
  PetscFunctionBegin;
  *a = (index % factor) / (PetscScalar)factor;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetInterpolationCoefficientCenter_Private(PetscBool belowHalf, PetscInt index, PetscInt factor, PetscScalar *a)
{
  PetscFunctionBegin;
  if (belowHalf) *a = 0.5 + ((index % factor) + 0.5) / (PetscScalar)factor;
  else *a = ((index % factor) + 0.5) / (PetscScalar)factor - 0.5;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetInterpolationWeight1d_Private(PetscScalar ax, PetscScalar weight[])
{
  PetscFunctionBegin;
  weight[0] = 1.0 - ax;
  weight[1] = ax;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetInterpolationWeight2d_Private(PetscScalar ax, PetscScalar ay, PetscScalar weight[])
{
  PetscFunctionBegin;
  weight[0] = (1.0 - ax) * (1.0 - ay);
  weight[1] = ax * (1.0 - ay);
  weight[2] = (1.0 - ax) * ay;
  weight[3] = ax * ay;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeInterpolationWeightToLeft2d_Private(PetscScalar weight[])
{
  PetscFunctionBegin;
  weight[0] += weight[1];
  weight[2] += weight[3];
  weight[1] = weight[3] = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeInterpolationWeightToRight2d_Private(PetscScalar weight[])
{
  PetscFunctionBegin;
  weight[1] += weight[0];
  weight[3] += weight[2];
  weight[0] = weight[2] = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeInterpolationWeightToBottom2d_Private(PetscScalar weight[])
{
  PetscFunctionBegin;
  weight[0] += weight[2];
  weight[1] += weight[3];
  weight[2] = weight[3] = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeInterpolationWeightToTop2d_Private(PetscScalar weight[])
{
  PetscFunctionBegin;
  weight[2] += weight[0];
  weight[3] += weight[1];
  weight[0] = weight[1] = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetInterpolationWeight3d_Private(PetscScalar ax, PetscScalar ay, PetscScalar az, PetscScalar weight[])
{
  PetscFunctionBegin;
  weight[0] = (1.0 - ax) * (1.0 - ay) * (1.0 - az);
  weight[1] = ax * (1.0 - ay) * (1.0 - az);
  weight[2] = (1.0 - ax) * ay * (1.0 - az);
  weight[3] = ax * ay * (1.0 - az);
  weight[4] = (1.0 - ax) * (1.0 - ay) * az;
  weight[5] = ax * (1.0 - ay) * az;
  weight[6] = (1.0 - ax) * ay * az;
  weight[7] = ax * ay * az;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeInterpolationWeightToLeft3d_Private(PetscScalar weight[])
{
  PetscFunctionBegin;
  weight[0] += weight[1];
  weight[2] += weight[3];
  weight[4] += weight[5];
  weight[6] += weight[7];
  weight[1] = weight[3] = weight[5] = weight[7] = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeInterpolationWeightToRight3d_Private(PetscScalar weight[])
{
  PetscFunctionBegin;
  weight[1] += weight[0];
  weight[3] += weight[2];
  weight[5] += weight[4];
  weight[7] += weight[6];
  weight[0] = weight[2] = weight[4] = weight[6] = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeInterplationWeightToBottom3d_Private(PetscScalar weight[])
{
  PetscFunctionBegin;
  weight[0] += weight[2];
  weight[1] += weight[3];
  weight[4] += weight[6];
  weight[5] += weight[7];
  weight[2] = weight[3] = weight[6] = weight[7] = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeInterpolationWeightToTop3d_Private(PetscScalar weight[])
{
  PetscFunctionBegin;
  weight[2] += weight[0];
  weight[3] += weight[1];
  weight[6] += weight[4];
  weight[7] += weight[5];
  weight[0] = weight[1] = weight[4] = weight[5] = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeInterpolationWeightToBack3d_Private(PetscScalar weight[])
{
  PetscFunctionBegin;
  weight[0] += weight[4];
  weight[1] += weight[5];
  weight[2] += weight[6];
  weight[3] += weight[7];
  weight[4] = weight[5] = weight[6] = weight[7] = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeInterpolationWeightToFront3d_Private(PetscScalar weight[])
{
  PetscFunctionBegin;
  weight[4] += weight[0];
  weight[5] += weight[1];
  weight[6] += weight[2];
  weight[7] += weight[3];
  weight[0] = weight[1] = weight[2] = weight[3] = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetRestrictionCoefficientVertex_Private(PetscInt index, PetscInt factor, PetscScalar *a)
{
  PetscFunctionBegin;
  *a = (factor - PetscAbsInt(index)) / (PetscScalar)(factor * factor);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetRestrictionCoefficientCenter_Private(PetscInt index, PetscInt factor, PetscScalar *a)
{
  PetscFunctionBegin;
  if (2 * index + 1 < factor) *a = 0.5 + (index + 0.5) / (PetscScalar)factor;
  else *a = 1.5 - (index + 0.5) / (PetscScalar)factor;
  /* Normalization depends on whether the restriction factor is even or odd */
  if (factor % 2 == 0) *a /= 0.75 * factor;
  else *a /= (3 * factor * factor + 1) / (PetscScalar)(4 * factor);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeRestrictionWeightToLeft1d_Private(PetscScalar weight[], PetscInt m)
{
  PetscInt       i;
  const PetscInt dst = m / 2;

  PetscFunctionBegin;
  PetscCheck(m % 2 == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "dimension must be odd");
  for (i = m / 2 + 1; i < m; ++i) {
    weight[dst] += weight[i];
    weight[i] = 0.0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeRestrictionWeightToRight1d_Private(PetscScalar weight[], PetscInt m)
{
  PetscInt       i;
  const PetscInt dst = m / 2;

  PetscFunctionBegin;
  PetscCheck(m % 2 == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "dimension must be odd");
  for (i = 0; i < m / 2; ++i) {
    weight[dst] += weight[i];
    weight[i] = 0.0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeRestrictionWeightToLeft2d_Private(PetscScalar weight[], PetscInt m, PetscInt n)
{
  PetscInt i, j, src, dst;

  PetscFunctionBegin;
  PetscCheck(m % 2 == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "dimension must be odd");
  for (j = 0; j < n; ++j)
    for (i = m / 2 + 1; i < m; ++i) {
      src = m * j + i;
      dst = m * j + m / 2;
      weight[dst] += weight[src];
      weight[src] = 0.0;
    }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeRestrictionWeightToRight2d_Private(PetscScalar weight[], PetscInt m, PetscInt n)
{
  PetscInt i, j, src, dst;

  PetscFunctionBegin;
  PetscCheck(m % 2 == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "dimension must be odd");
  for (j = 0; j < n; ++j)
    for (i = 0; i < m / 2; ++i) {
      src = m * j + i;
      dst = m * j + m / 2;
      weight[dst] += weight[src];
      weight[src] = 0.0;
    }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeRestrictionWeightToBottom2d_Private(PetscScalar weight[], PetscInt m, PetscInt n)
{
  PetscInt i, j, src, dst;

  PetscFunctionBegin;
  PetscCheck(n % 2 == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "dimension must be odd");
  for (j = n / 2 + 1; j < n; ++j)
    for (i = 0; i < m; ++i) {
      src = m * j + i;
      dst = m * (n / 2) + i;
      weight[dst] += weight[src];
      weight[src] = 0.0;
    }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeRestrictionWeightToTop2d_Private(PetscScalar weight[], PetscInt m, PetscInt n)
{
  PetscInt i, j, src, dst;

  PetscFunctionBegin;
  PetscCheck(n % 2 == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "dimension must be odd");
  for (j = 0; j < n / 2; ++j)
    for (i = 0; i < m; ++i) {
      src = m * j + i;
      dst = m * (n / 2) + i;
      weight[dst] += weight[src];
      weight[src] = 0.0;
    }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeRestrictionWeightToLeft3d_Private(PetscScalar weight[], PetscInt m, PetscInt n, PetscInt p)
{
  PetscInt i, j, k, src, dst;

  PetscFunctionBegin;
  PetscCheck(m % 2 == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "dimension must be odd");
  for (k = 0; k < p; ++k)
    for (j = 0; j < n; ++j)
      for (i = m / 2 + 1; i < m; ++i) {
        src = m * n * k + m * j + i;
        dst = m * n * k + m * j + m / 2;
        weight[dst] += weight[src];
        weight[src] = 0.0;
      }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeRestrictionWeightToRight3d_Private(PetscScalar weight[], PetscInt m, PetscInt n, PetscInt p)
{
  PetscInt i, j, k, src, dst;

  PetscFunctionBegin;
  PetscCheck(m % 2 == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "dimension must be odd");
  for (k = 0; k < p; ++k)
    for (j = 0; j < n; ++j)
      for (i = 0; i < m / 2; ++i) {
        src = m * n * k + m * j + i;
        dst = m * n * k + m * j + m / 2;
        weight[dst] += weight[src];
        weight[src] = 0.0;
      }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeRestrictionWeightToBottom3d_Private(PetscScalar weight[], PetscInt m, PetscInt n, PetscInt p)
{
  PetscInt i, j, k, src, dst;

  PetscFunctionBegin;
  PetscCheck(n % 2 == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "dimension must be odd");
  for (k = 0; k < p; ++k)
    for (j = n / 2 + 1; j < n; ++j)
      for (i = 0; i < m; ++i) {
        src = m * n * k + m * j + i;
        dst = m * n * k + m * (n / 2) + i;
        weight[dst] += weight[src];
        weight[src] = 0.0;
      }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeRestrictionWeightToTop3d_Private(PetscScalar weight[], PetscInt m, PetscInt n, PetscInt p)
{
  PetscInt i, j, k, src, dst;

  PetscFunctionBegin;
  PetscCheck(n % 2 == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "dimension must be odd");
  for (k = 0; k < p; ++k)
    for (j = 0; j < n / 2; ++j)
      for (i = 0; i < m; ++i) {
        src = m * n * k + m * j + i;
        dst = m * n * k + m * (n / 2) + i;
        weight[dst] += weight[src];
        weight[src] = 0.0;
      }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeRestrictionWeightToBack3d_Private(PetscScalar weight[], PetscInt m, PetscInt n, PetscInt p)
{
  PetscInt i, j, k, src, dst;

  PetscFunctionBegin;
  PetscCheck(p % 2 == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "dimension must be odd");
  for (k = p / 2 + 1; k < p; ++k)
    for (j = 0; j < n; ++j)
      for (i = 0; i < m; ++i) {
        src = m * n * k + m * j + i;
        dst = m * n * (p / 2) + m * j + i;
        weight[dst] += weight[src];
        weight[src] = 0.0;
      }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MergeRestrictionWeightToFront3d_Private(PetscScalar weight[], PetscInt m, PetscInt n, PetscInt p)
{
  PetscInt i, j, k, src, dst;

  PetscFunctionBegin;
  PetscCheck(p % 2 == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "dimension must be odd");
  for (k = 0; k < p / 2; ++k)
    for (j = 0; j < n; ++j)
      for (i = 0; i < m; ++i) {
        src = m * n * k + m * j + i;
        dst = m * n * (p / 2) + m * j + i;
        weight[dst] += weight[src];
        weight[src] = 0.0;
      }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RemoveZeroWeights_Private(PetscInt n, DMStagStencil colc[], PetscScalar weight[], PetscInt *count)
{
  PetscInt i;

  PetscFunctionBegin;
  *count = 0;
  for (i = 0; i < n; ++i)
    if (weight[i] != 0.0) {
      colc[*count]   = colc[i];
      weight[*count] = weight[i];
      ++(*count);
    }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Code duplication note: the next two functions are nearly identical, save the inclusion of the element terms */
PETSC_INTERN PetscErrorCode DMStagPopulateInterpolation1d_Internal(DM dmc, DM dmf, Mat A)
{
  PetscInt       Mc, Mf, factorx, dof[2];
  PetscInt       xf, mf, nExtraxf, i, d, count;
  DMStagStencil  rowf, colc[2];
  PetscScalar    ax, weight[2];
  PetscInt       ir, ic[2];
  const PetscInt dim = 1;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dmc, &Mc, NULL, NULL));
  PetscCall(DMStagGetGlobalSizes(dmf, &Mf, NULL, NULL));
  factorx = Mf / Mc;
  PetscCall(DMStagGetDOF(dmc, &dof[0], &dof[1], NULL, NULL));

  /* In 1D, each fine point can receive data from at most 2 coarse points, at most one of which could be off-process */
  PetscCall(MatSeqAIJSetPreallocation(A, 2, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 2, NULL, 1, NULL));

  PetscCall(DMStagGetCorners(dmf, &xf, NULL, NULL, &mf, NULL, NULL, &nExtraxf, NULL, NULL));

  /* Linear interpolation for vertices */
  for (d = 0; d < dof[0]; ++d)
    for (i = xf; i < xf + mf + nExtraxf; ++i) {
      rowf.i   = i;
      rowf.c   = d;
      rowf.loc = DMSTAG_LEFT;
      for (count = 0; count < 2; ++count) {
        colc[count].i = i / factorx;
        colc[count].c = d;
      }
      colc[0].loc = DMSTAG_LEFT;
      colc[1].loc = DMSTAG_RIGHT;
      PetscCall(SetInterpolationCoefficientVertex_Private(i, factorx, &ax));
      PetscCall(SetInterpolationWeight1d_Private(ax, weight));

      PetscCall(RemoveZeroWeights_Private(2, colc, weight, &count));
      PetscCall(DMStagStencilToIndexLocal(dmf, dim, 1, &rowf, &ir));
      PetscCall(DMStagStencilToIndexLocal(dmc, dim, count, colc, ic));
      PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
    }

  /* Nearest neighbor for elements */
  for (d = 0; d < dof[1]; ++d)
    for (i = xf; i < xf + mf; ++i) {
      rowf.i      = i;
      rowf.c      = d;
      rowf.loc    = DMSTAG_ELEMENT;
      colc[0].i   = i / factorx;
      colc[0].c   = d;
      colc[0].loc = DMSTAG_ELEMENT;
      weight[0]   = 1.0;
      PetscCall(DMStagStencilToIndexLocal(dmf, dim, 1, &rowf, &ir));
      PetscCall(DMStagStencilToIndexLocal(dmc, dim, 1, colc, ic));
      PetscCall(MatSetValuesLocal(A, 1, &ir, 1, ic, weight, INSERT_VALUES));
    }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode DMStagPopulateInterpolation2d_Internal(DM dmc, DM dmf, Mat A)
{
  PetscInt       Mc, Nc, Mf, Nf, factorx, factory, dof[3];
  PetscInt       xf, yf, mf, nf, nExtraxf, nExtrayf, i, j, d, count;
  DMStagStencil  rowf, colc[4];
  PetscScalar    ax, ay, weight[4];
  PetscInt       ir, ic[4];
  const PetscInt dim = 2;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dmc, &Mc, &Nc, NULL));
  PetscCall(DMStagGetGlobalSizes(dmf, &Mf, &Nf, NULL));
  factorx = Mf / Mc;
  factory = Nf / Nc;
  PetscCall(DMStagGetDOF(dmc, &dof[0], &dof[1], &dof[2], NULL));

  /* In 2D, each fine point can receive data from at most 4 coarse points, at most 3 of which could be off-process */
  PetscCall(MatSeqAIJSetPreallocation(A, 4, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 4, NULL, 3, NULL));

  PetscCall(DMStagGetCorners(dmf, &xf, &yf, NULL, &mf, &nf, NULL, &nExtraxf, &nExtrayf, NULL));

  /* Linear interpolation for vertices */
  for (d = 0; d < dof[0]; ++d)
    for (j = yf; j < yf + nf + nExtrayf; ++j)
      for (i = xf; i < xf + mf + nExtraxf; ++i) {
        rowf.i   = i;
        rowf.j   = j;
        rowf.c   = d;
        rowf.loc = DMSTAG_DOWN_LEFT;
        for (count = 0; count < 4; ++count) {
          colc[count].i = i / factorx;
          colc[count].j = j / factory;
          colc[count].c = d;
        }
        colc[0].loc = DMSTAG_DOWN_LEFT;
        colc[1].loc = DMSTAG_DOWN_RIGHT;
        colc[2].loc = DMSTAG_UP_LEFT;
        colc[3].loc = DMSTAG_UP_RIGHT;
        PetscCall(SetInterpolationCoefficientVertex_Private(i, factorx, &ax));
        PetscCall(SetInterpolationCoefficientVertex_Private(j, factory, &ay));
        PetscCall(SetInterpolationWeight2d_Private(ax, ay, weight));

        PetscCall(RemoveZeroWeights_Private(4, colc, weight, &count));
        PetscCall(DMStagStencilToIndexLocal(dmf, dim, 1, &rowf, &ir));
        PetscCall(DMStagStencilToIndexLocal(dmc, dim, count, colc, ic));
        PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
      }

  /* Linear interpolation for left edges */
  for (d = 0; d < dof[1]; ++d)
    for (j = yf; j < yf + nf; ++j)
      for (i = xf; i < xf + mf + nExtraxf; ++i) {
        const PetscBool belowHalfy = (PetscBool)(2 * (j % factory) + 1 < factory);

        rowf.i   = i;
        rowf.j   = j;
        rowf.c   = d;
        rowf.loc = DMSTAG_LEFT;
        for (count = 0; count < 4; ++count) {
          colc[count].i = i / factorx;
          colc[count].j = j / factory;
          if (belowHalfy) colc[count].j -= 1;
          if (count / 2 == 1) colc[count].j += 1;
          colc[count].c = d;
        }
        colc[0].loc = DMSTAG_LEFT;
        colc[1].loc = DMSTAG_RIGHT;
        colc[2].loc = DMSTAG_LEFT;
        colc[3].loc = DMSTAG_RIGHT;
        PetscCall(SetInterpolationCoefficientVertex_Private(i, factorx, &ax));
        PetscCall(SetInterpolationCoefficientCenter_Private(belowHalfy, j, factory, &ay));
        PetscCall(SetInterpolationWeight2d_Private(ax, ay, weight));
        /* Assume Neumann boundary condition */
        if (j / factory == 0 && belowHalfy) PetscCall(MergeInterpolationWeightToTop2d_Private(weight));
        else if (j / factory == Nc - 1 && !belowHalfy) PetscCall(MergeInterpolationWeightToBottom2d_Private(weight));

        PetscCall(RemoveZeroWeights_Private(4, colc, weight, &count));
        PetscCall(DMStagStencilToIndexLocal(dmf, dim, 1, &rowf, &ir));
        PetscCall(DMStagStencilToIndexLocal(dmc, dim, count, colc, ic));
        PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
      }

  /* Linear interpolation for down edges */
  for (d = 0; d < dof[1]; ++d)
    for (j = yf; j < yf + nf + nExtrayf; ++j)
      for (i = xf; i < xf + mf; ++i) {
        const PetscBool belowHalfx = (PetscBool)(2 * (i % factorx) + 1 < factorx);

        rowf.i   = i;
        rowf.j   = j;
        rowf.c   = d;
        rowf.loc = DMSTAG_DOWN;
        for (count = 0; count < 4; ++count) {
          colc[count].i = i / factorx;
          colc[count].j = j / factory;
          if (belowHalfx) colc[count].i -= 1;
          if (count % 2 == 1) colc[count].i += 1;
          colc[count].c = d;
        }
        colc[0].loc = DMSTAG_DOWN;
        colc[1].loc = DMSTAG_DOWN;
        colc[2].loc = DMSTAG_UP;
        colc[3].loc = DMSTAG_UP;
        PetscCall(SetInterpolationCoefficientCenter_Private(belowHalfx, i, factorx, &ax));
        PetscCall(SetInterpolationCoefficientVertex_Private(j, factory, &ay));
        PetscCall(SetInterpolationWeight2d_Private(ax, ay, weight));
        /* Assume Neumann boundary condition */
        if (i / factorx == 0 && belowHalfx) PetscCall(MergeInterpolationWeightToRight2d_Private(weight));
        else if (i / factorx == Mc - 1 && !belowHalfx) PetscCall(MergeInterpolationWeightToLeft2d_Private(weight));

        PetscCall(RemoveZeroWeights_Private(4, colc, weight, &count));
        PetscCall(DMStagStencilToIndexLocal(dmf, dim, 1, &rowf, &ir));
        PetscCall(DMStagStencilToIndexLocal(dmc, dim, count, colc, ic));
        PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
      }

  /* Nearest neighbor for elements */
  for (d = 0; d < dof[2]; ++d)
    for (j = yf; j < yf + nf; ++j)
      for (i = xf; i < xf + mf; ++i) {
        rowf.i      = i;
        rowf.j      = j;
        rowf.c      = d;
        rowf.loc    = DMSTAG_ELEMENT;
        colc[0].i   = i / factorx;
        colc[0].j   = j / factory;
        colc[0].c   = d;
        colc[0].loc = DMSTAG_ELEMENT;
        weight[0]   = 1.0;
        PetscCall(DMStagStencilToIndexLocal(dmf, dim, 1, &rowf, &ir));
        PetscCall(DMStagStencilToIndexLocal(dmc, dim, 1, colc, ic));
        PetscCall(MatSetValuesLocal(A, 1, &ir, 1, ic, weight, INSERT_VALUES));
      }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode DMStagPopulateInterpolation3d_Internal(DM dmc, DM dmf, Mat A)
{
  PetscInt       Mc, Nc, Pc, Mf, Nf, Pf, factorx, factory, factorz, dof[4];
  PetscInt       xf, yf, zf, mf, nf, pf, nExtraxf, nExtrayf, nExtrazf, i, j, k, d, count;
  DMStagStencil  rowf, colc[8];
  PetscScalar    ax, ay, az, weight[8];
  PetscInt       ir, ic[8];
  const PetscInt dim = 3;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dmc, &Mc, &Nc, &Pc));
  PetscCall(DMStagGetGlobalSizes(dmf, &Mf, &Nf, &Pf));
  factorx = Mf / Mc;
  factory = Nf / Nc;
  factorz = Pf / Pc;
  PetscCall(DMStagGetDOF(dmc, &dof[0], &dof[1], &dof[2], &dof[3]));

  /* In 3D, each fine point can receive data from at most 8 coarse points, at most 7 of which could be off-process */
  PetscCall(MatSeqAIJSetPreallocation(A, 8, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 8, NULL, 7, NULL));

  PetscCall(DMStagGetCorners(dmf, &xf, &yf, &zf, &mf, &nf, &pf, &nExtraxf, &nExtrayf, &nExtrazf));

  /* Linear interpolation for vertices */
  for (d = 0; d < dof[0]; ++d)
    for (k = zf; k < zf + pf + nExtrazf; ++k)
      for (j = yf; j < yf + nf + nExtrayf; ++j)
        for (i = xf; i < xf + mf + nExtraxf; ++i) {
          rowf.i   = i;
          rowf.j   = j;
          rowf.k   = k;
          rowf.c   = d;
          rowf.loc = DMSTAG_BACK_DOWN_LEFT;
          for (count = 0; count < 8; ++count) {
            colc[count].i = i / factorx;
            colc[count].j = j / factory;
            colc[count].k = k / factorz;
            colc[count].c = d;
          }
          colc[0].loc = DMSTAG_BACK_DOWN_LEFT;
          colc[1].loc = DMSTAG_BACK_DOWN_RIGHT;
          colc[2].loc = DMSTAG_BACK_UP_LEFT;
          colc[3].loc = DMSTAG_BACK_UP_RIGHT;
          colc[4].loc = DMSTAG_FRONT_DOWN_LEFT;
          colc[5].loc = DMSTAG_FRONT_DOWN_RIGHT;
          colc[6].loc = DMSTAG_FRONT_UP_LEFT;
          colc[7].loc = DMSTAG_FRONT_UP_RIGHT;
          PetscCall(SetInterpolationCoefficientVertex_Private(i, factorx, &ax));
          PetscCall(SetInterpolationCoefficientVertex_Private(j, factory, &ay));
          PetscCall(SetInterpolationCoefficientVertex_Private(k, factorz, &az));
          PetscCall(SetInterpolationWeight3d_Private(ax, ay, az, weight));

          PetscCall(RemoveZeroWeights_Private(8, colc, weight, &count));
          PetscCall(DMStagStencilToIndexLocal(dmf, dim, 1, &rowf, &ir));
          PetscCall(DMStagStencilToIndexLocal(dmc, dim, count, colc, ic));
          PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
        }

  /* Linear interpolation for down left edges */
  for (d = 0; d < dof[1]; ++d)
    for (k = zf; k < zf + pf; ++k)
      for (j = yf; j < yf + nf + nExtrayf; ++j)
        for (i = xf; i < xf + mf + nExtraxf; ++i) {
          const PetscBool belowHalfz = (PetscBool)(2 * (k % factorz) + 1 < factorz);

          rowf.i   = i;
          rowf.j   = j;
          rowf.k   = k;
          rowf.c   = d;
          rowf.loc = DMSTAG_DOWN_LEFT;
          for (count = 0; count < 8; ++count) {
            colc[count].i = i / factorx;
            colc[count].j = j / factory;
            colc[count].k = k / factorz;
            if (belowHalfz) colc[count].k -= 1;
            if (count / 4 == 1) colc[count].k += 1;
            colc[count].c = d;
          }
          colc[0].loc = DMSTAG_DOWN_LEFT;
          colc[1].loc = DMSTAG_DOWN_RIGHT;
          colc[2].loc = DMSTAG_UP_LEFT;
          colc[3].loc = DMSTAG_UP_RIGHT;
          colc[4].loc = DMSTAG_DOWN_LEFT;
          colc[5].loc = DMSTAG_DOWN_RIGHT;
          colc[6].loc = DMSTAG_UP_LEFT;
          colc[7].loc = DMSTAG_UP_RIGHT;
          PetscCall(SetInterpolationCoefficientVertex_Private(i, factorx, &ax));
          PetscCall(SetInterpolationCoefficientVertex_Private(j, factory, &ay));
          PetscCall(SetInterpolationCoefficientCenter_Private(belowHalfz, k, factorz, &az));
          PetscCall(SetInterpolationWeight3d_Private(ax, ay, az, weight));
          /* Assume Neumann boundary condition */
          if (k / factorz == 0 && belowHalfz) PetscCall(MergeInterpolationWeightToFront3d_Private(weight));
          else if (k / factorz == Pc - 1 && !belowHalfz) PetscCall(MergeInterpolationWeightToBack3d_Private(weight));

          PetscCall(RemoveZeroWeights_Private(8, colc, weight, &count));
          PetscCall(DMStagStencilToIndexLocal(dmf, dim, 1, &rowf, &ir));
          PetscCall(DMStagStencilToIndexLocal(dmc, dim, count, colc, ic));
          PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
        }

  /* Linear interpolation for back left edges */
  for (d = 0; d < dof[1]; ++d)
    for (k = zf; k < zf + pf + nExtrazf; ++k)
      for (j = yf; j < yf + nf; ++j)
        for (i = xf; i < xf + mf + nExtraxf; ++i) {
          const PetscBool belowHalfy = (PetscBool)(2 * (j % factory) + 1 < factory);

          rowf.i   = i;
          rowf.j   = j;
          rowf.k   = k;
          rowf.c   = d;
          rowf.loc = DMSTAG_BACK_LEFT;
          for (count = 0; count < 8; ++count) {
            colc[count].i = i / factorx;
            colc[count].j = j / factory;
            colc[count].k = k / factorz;
            if (belowHalfy) colc[count].j -= 1;
            if ((count % 4) / 2 == 1) colc[count].j += 1;
            colc[count].c = d;
          }
          colc[0].loc = DMSTAG_BACK_LEFT;
          colc[1].loc = DMSTAG_BACK_RIGHT;
          colc[2].loc = DMSTAG_BACK_LEFT;
          colc[3].loc = DMSTAG_BACK_RIGHT;
          colc[4].loc = DMSTAG_FRONT_LEFT;
          colc[5].loc = DMSTAG_FRONT_RIGHT;
          colc[6].loc = DMSTAG_FRONT_LEFT;
          colc[7].loc = DMSTAG_FRONT_RIGHT;
          PetscCall(SetInterpolationCoefficientVertex_Private(i, factorx, &ax));
          PetscCall(SetInterpolationCoefficientCenter_Private(belowHalfy, j, factory, &ay));
          PetscCall(SetInterpolationCoefficientVertex_Private(k, factorz, &az));
          PetscCall(SetInterpolationWeight3d_Private(ax, ay, az, weight));
          /* Assume Neumann boundary condition */
          if (j / factory == 0 && belowHalfy) PetscCall(MergeInterpolationWeightToTop3d_Private(weight));
          else if (j / factory == Nc - 1 && !belowHalfy) PetscCall(MergeInterplationWeightToBottom3d_Private(weight));

          PetscCall(RemoveZeroWeights_Private(8, colc, weight, &count));
          PetscCall(DMStagStencilToIndexLocal(dmf, dim, 1, &rowf, &ir));
          PetscCall(DMStagStencilToIndexLocal(dmc, dim, count, colc, ic));
          PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
        }

  /* Linear interpolation for down back edges */
  for (d = 0; d < dof[1]; ++d)
    for (k = zf; k < zf + pf + nExtrazf; ++k)
      for (j = yf; j < yf + nf + nExtrayf; ++j)
        for (i = xf; i < xf + mf; ++i) {
          const PetscBool belowHalfx = (PetscBool)(2 * (i % factorx) + 1 < factorx);

          rowf.i   = i;
          rowf.j   = j;
          rowf.k   = k;
          rowf.c   = d;
          rowf.loc = DMSTAG_BACK_DOWN;
          for (count = 0; count < 8; ++count) {
            colc[count].i = i / factorx;
            colc[count].j = j / factory;
            colc[count].k = k / factorz;
            if (belowHalfx) colc[count].i -= 1;
            if (count % 2 == 1) colc[count].i += 1;
            colc[count].c = d;
          }
          colc[0].loc = DMSTAG_BACK_DOWN;
          colc[1].loc = DMSTAG_BACK_DOWN;
          colc[2].loc = DMSTAG_BACK_UP;
          colc[3].loc = DMSTAG_BACK_UP;
          colc[4].loc = DMSTAG_FRONT_DOWN;
          colc[5].loc = DMSTAG_FRONT_DOWN;
          colc[6].loc = DMSTAG_FRONT_UP;
          colc[7].loc = DMSTAG_FRONT_UP;
          PetscCall(SetInterpolationCoefficientCenter_Private(belowHalfx, i, factorx, &ax));
          PetscCall(SetInterpolationCoefficientVertex_Private(j, factory, &ay));
          PetscCall(SetInterpolationCoefficientVertex_Private(k, factorz, &az));
          PetscCall(SetInterpolationWeight3d_Private(ax, ay, az, weight));
          /* Assume Neumann boundary condition */
          if (i / factorx == 0 && belowHalfx) PetscCall(MergeInterpolationWeightToRight3d_Private(weight));
          else if (i / factorx == Mc - 1 && !belowHalfx) PetscCall(MergeInterpolationWeightToLeft3d_Private(weight));

          PetscCall(RemoveZeroWeights_Private(8, colc, weight, &count));
          PetscCall(DMStagStencilToIndexLocal(dmf, dim, 1, &rowf, &ir));
          PetscCall(DMStagStencilToIndexLocal(dmc, dim, count, colc, ic));
          PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
        }

  /* Linear interpolation for left faces */
  for (d = 0; d < dof[2]; ++d)
    for (k = zf; k < zf + pf; ++k)
      for (j = yf; j < yf + nf; ++j)
        for (i = xf; i < xf + mf + nExtraxf; ++i) {
          const PetscBool belowHalfy = (PetscBool)(2 * (j % factory) + 1 < factory);
          const PetscBool belowHalfz = (PetscBool)(2 * (k % factorz) + 1 < factorz);

          rowf.i   = i;
          rowf.j   = j;
          rowf.k   = k;
          rowf.c   = d;
          rowf.loc = DMSTAG_LEFT;
          for (count = 0; count < 8; ++count) {
            colc[count].i = i / factorx;
            colc[count].j = j / factory;
            colc[count].k = k / factorz;
            if (belowHalfy) colc[count].j -= 1;
            if ((count % 4) / 2 == 1) colc[count].j += 1;
            if (belowHalfz) colc[count].k -= 1;
            if (count / 4 == 1) colc[count].k += 1;
            colc[count].c = d;
          }
          colc[0].loc = DMSTAG_LEFT;
          colc[1].loc = DMSTAG_RIGHT;
          colc[2].loc = DMSTAG_LEFT;
          colc[3].loc = DMSTAG_RIGHT;
          colc[4].loc = DMSTAG_LEFT;
          colc[5].loc = DMSTAG_RIGHT;
          colc[6].loc = DMSTAG_LEFT;
          colc[7].loc = DMSTAG_RIGHT;
          PetscCall(SetInterpolationCoefficientVertex_Private(i, factorx, &ax));
          PetscCall(SetInterpolationCoefficientCenter_Private(belowHalfy, j, factory, &ay));
          PetscCall(SetInterpolationCoefficientCenter_Private(belowHalfz, k, factorz, &az));
          PetscCall(SetInterpolationWeight3d_Private(ax, ay, az, weight));
          /* Assume Neumann boundary condition */
          if (j / factory == 0 && belowHalfy) PetscCall(MergeInterpolationWeightToTop3d_Private(weight));
          else if (j / factory == Nc - 1 && !belowHalfy) PetscCall(MergeInterplationWeightToBottom3d_Private(weight));
          if (k / factorz == 0 && belowHalfz) PetscCall(MergeInterpolationWeightToFront3d_Private(weight));
          else if (k / factorz == Pc - 1 && !belowHalfz) PetscCall(MergeInterpolationWeightToBack3d_Private(weight));

          PetscCall(RemoveZeroWeights_Private(8, colc, weight, &count));
          PetscCall(DMStagStencilToIndexLocal(dmf, dim, 1, &rowf, &ir));
          PetscCall(DMStagStencilToIndexLocal(dmc, dim, count, colc, ic));
          PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
        }

  /* Linear interpolation for down faces */
  for (d = 0; d < dof[2]; ++d)
    for (k = zf; k < zf + pf; ++k)
      for (j = yf; j < yf + nf + nExtrayf; ++j)
        for (i = xf; i < xf + mf; ++i) {
          const PetscBool belowHalfx = (PetscBool)(2 * (i % factorx) + 1 < factorx);
          const PetscBool belowHalfz = (PetscBool)(2 * (k % factorz) + 1 < factorz);

          rowf.i   = i;
          rowf.j   = j;
          rowf.k   = k;
          rowf.c   = d;
          rowf.loc = DMSTAG_DOWN;
          for (count = 0; count < 8; ++count) {
            colc[count].i = i / factorx;
            colc[count].j = j / factory;
            colc[count].k = k / factorz;
            if (belowHalfx) colc[count].i -= 1;
            if (count % 2 == 1) colc[count].i += 1;
            if (belowHalfz) colc[count].k -= 1;
            if (count / 4 == 1) colc[count].k += 1;
            colc[count].c = d;
          }
          colc[0].loc = DMSTAG_DOWN;
          colc[1].loc = DMSTAG_DOWN;
          colc[2].loc = DMSTAG_UP;
          colc[3].loc = DMSTAG_UP;
          colc[4].loc = DMSTAG_DOWN;
          colc[5].loc = DMSTAG_DOWN;
          colc[6].loc = DMSTAG_UP;
          colc[7].loc = DMSTAG_UP;
          PetscCall(SetInterpolationCoefficientCenter_Private(belowHalfx, i, factorx, &ax));
          PetscCall(SetInterpolationCoefficientVertex_Private(j, factory, &ay));
          PetscCall(SetInterpolationCoefficientCenter_Private(belowHalfz, k, factorz, &az));
          PetscCall(SetInterpolationWeight3d_Private(ax, ay, az, weight));
          /* Assume Neumann boundary condition */
          if (i / factorx == 0 && belowHalfx) PetscCall(MergeInterpolationWeightToRight3d_Private(weight));
          else if (i / factorx == Mc - 1 && !belowHalfx) PetscCall(MergeInterpolationWeightToLeft3d_Private(weight));
          if (k / factorz == 0 && belowHalfz) PetscCall(MergeInterpolationWeightToFront3d_Private(weight));
          else if (k / factorz == Pc - 1 && !belowHalfz) PetscCall(MergeInterpolationWeightToBack3d_Private(weight));

          PetscCall(RemoveZeroWeights_Private(8, colc, weight, &count));
          PetscCall(DMStagStencilToIndexLocal(dmf, dim, 1, &rowf, &ir));
          PetscCall(DMStagStencilToIndexLocal(dmc, dim, count, colc, ic));
          PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
        }

  /* Linear interpolation for back faces */
  for (d = 0; d < dof[2]; ++d)
    for (k = zf; k < zf + pf + nExtrazf; ++k)
      for (j = yf; j < yf + nf; ++j)
        for (i = xf; i < xf + mf; ++i) {
          const PetscBool belowHalfx = (PetscBool)(2 * (i % factorx) + 1 < factorx);
          const PetscBool belowHalfy = (PetscBool)(2 * (j % factory) + 1 < factory);

          rowf.i   = i;
          rowf.j   = j;
          rowf.k   = k;
          rowf.c   = d;
          rowf.loc = DMSTAG_BACK;
          for (count = 0; count < 8; ++count) {
            colc[count].i = i / factorx;
            colc[count].j = j / factory;
            colc[count].k = k / factorz;
            if (belowHalfx) colc[count].i -= 1;
            if (count % 2 == 1) colc[count].i += 1;
            if (belowHalfy) colc[count].j -= 1;
            if ((count % 4) / 2 == 1) colc[count].j += 1;
            colc[count].c = d;
          }
          colc[0].loc = DMSTAG_BACK;
          colc[1].loc = DMSTAG_BACK;
          colc[2].loc = DMSTAG_BACK;
          colc[3].loc = DMSTAG_BACK;
          colc[4].loc = DMSTAG_FRONT;
          colc[5].loc = DMSTAG_FRONT;
          colc[6].loc = DMSTAG_FRONT;
          colc[7].loc = DMSTAG_FRONT;
          PetscCall(SetInterpolationCoefficientCenter_Private(belowHalfx, i, factorx, &ax));
          PetscCall(SetInterpolationCoefficientCenter_Private(belowHalfy, j, factory, &ay));
          PetscCall(SetInterpolationCoefficientVertex_Private(k, factorz, &az));
          PetscCall(SetInterpolationWeight3d_Private(ax, ay, az, weight));
          /* Assume Neumann boundary condition */
          if (i / factorx == 0 && belowHalfx) PetscCall(MergeInterpolationWeightToRight3d_Private(weight));
          else if (i / factorx == Mc - 1 && !belowHalfx) PetscCall(MergeInterpolationWeightToLeft3d_Private(weight));
          if (j / factory == 0 && belowHalfy) PetscCall(MergeInterpolationWeightToTop3d_Private(weight));
          else if (j / factory == Nc - 1 && !belowHalfy) PetscCall(MergeInterplationWeightToBottom3d_Private(weight));

          PetscCall(RemoveZeroWeights_Private(8, colc, weight, &count));
          PetscCall(DMStagStencilToIndexLocal(dmf, dim, 1, &rowf, &ir));
          PetscCall(DMStagStencilToIndexLocal(dmc, dim, count, colc, ic));
          PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
        }

  /* Nearest neighbor for elements */
  for (d = 0; d < dof[3]; ++d)
    for (k = zf; k < zf + pf; ++k)
      for (j = yf; j < yf + nf; ++j)
        for (i = xf; i < xf + mf; ++i) {
          rowf.i      = i;
          rowf.j      = j;
          rowf.k      = k;
          rowf.c      = d;
          rowf.loc    = DMSTAG_ELEMENT;
          colc[0].i   = i / factorx;
          colc[0].j   = j / factory;
          colc[0].k   = k / factorz;
          colc[0].c   = d;
          colc[0].loc = DMSTAG_ELEMENT;
          weight[0]   = 1.0;
          PetscCall(DMStagStencilToIndexLocal(dmf, dim, 1, &rowf, &ir));
          PetscCall(DMStagStencilToIndexLocal(dmc, dim, 1, colc, ic));
          PetscCall(MatSetValuesLocal(A, 1, &ir, 1, ic, weight, INSERT_VALUES));
        }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode DMStagPopulateRestriction1d_Internal(DM dmc, DM dmf, Mat A)
{
  PetscInt       Mc, Mf, factorx, dof[2];
  PetscInt       xc, mc, nExtraxc, i, d, ii, count;
  PetscInt       maxFinePoints, maxOffRankFinePoints;
  DMStagStencil  rowc;
  DMStagStencil *colf;
  PetscScalar   *weight;
  PetscInt       ir;
  PetscInt      *ic;
  const PetscInt dim = 1;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dmc, &Mc, NULL, NULL));
  PetscCall(DMStagGetGlobalSizes(dmf, &Mf, NULL, NULL));
  factorx = Mf / Mc;
  PetscCall(DMStagGetDOF(dmc, &dof[0], &dof[1], NULL, NULL));

  /* In 1D, each coarse point can receive from up to (2 * factorx - 1) fine points, (factorx - 1) of which may be off-rank */
  maxFinePoints        = 2 * factorx - 1;
  maxOffRankFinePoints = maxFinePoints - factorx;
  PetscCall(MatSeqAIJSetPreallocation(A, maxFinePoints, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, maxFinePoints, NULL, maxOffRankFinePoints, NULL));
  PetscCall(PetscMalloc3(maxFinePoints, &colf, maxFinePoints, &weight, maxFinePoints, &ic));

  PetscCall(DMStagGetCorners(dmc, &xc, NULL, NULL, &mc, NULL, NULL, &nExtraxc, NULL, NULL));

  for (d = 0; d < dof[0]; ++d)
    for (i = xc; i < xc + mc + nExtraxc; ++i) {
      rowc.i   = i;
      rowc.c   = d;
      rowc.loc = DMSTAG_LEFT;
      count    = 0;
      for (ii = -(factorx - 1); ii <= factorx - 1; ++ii) {
        colf[count].i   = i * factorx + ii;
        colf[count].c   = d;
        colf[count].loc = DMSTAG_LEFT;
        PetscCall(SetRestrictionCoefficientVertex_Private(ii, factorx, &weight[count]));
        ++count;
      }
      if (i == 0) PetscCall(MergeRestrictionWeightToRight1d_Private(weight, 2 * factorx - 1));
      else if (i == Mc) PetscCall(MergeRestrictionWeightToLeft1d_Private(weight, 2 * factorx - 1));

      PetscCall(RemoveZeroWeights_Private(2 * factorx - 1, colf, weight, &count));
      PetscCall(DMStagStencilToIndexLocal(dmc, dim, 1, &rowc, &ir));
      PetscCall(DMStagStencilToIndexLocal(dmf, dim, count, colf, ic));
      PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
    }

  for (d = 0; d < dof[1]; ++d)
    for (i = xc; i < xc + mc; ++i) {
      rowc.i   = i;
      rowc.c   = d;
      rowc.loc = DMSTAG_ELEMENT;
      count    = 0;
      for (ii = 0; ii < factorx; ++ii) {
        colf[count].i   = i * factorx + ii;
        colf[count].c   = d;
        colf[count].loc = DMSTAG_ELEMENT;
        weight[count]   = 1 / (PetscScalar)factorx;
        ++count;
      }

      PetscCall(DMStagStencilToIndexLocal(dmc, dim, 1, &rowc, &ir));
      PetscCall(DMStagStencilToIndexLocal(dmf, dim, count, colf, ic));
      PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
    }

  PetscCall(PetscFree3(colf, weight, ic));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode DMStagPopulateRestriction2d_Internal(DM dmc, DM dmf, Mat A)
{
  PetscInt       Mc, Nc, Mf, Nf, factorx, factory, dof[3];
  PetscInt       xc, yc, mc, nc, nExtraxc, nExtrayc, i, j, d, ii, jj, count;
  PetscInt       maxFinePoints, maxOffRankFinePoints;
  DMStagStencil  rowc;
  DMStagStencil *colf;
  PetscScalar    ax, ay;
  PetscScalar   *weight;
  PetscInt       ir;
  PetscInt      *ic;
  const PetscInt dim = 2;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dmc, &Mc, &Nc, NULL));
  PetscCall(DMStagGetGlobalSizes(dmf, &Mf, &Nf, NULL));
  factorx = Mf / Mc;
  factory = Nf / Nc;
  PetscCall(DMStagGetDOF(dmc, &dof[0], &dof[1], &dof[2], NULL));

  /* In 2D, each coarse point can receive from up to ((2 * factorx - 1) * (2 * factory - 1)) fine points,
     up to ((2 * factorx - 1) * (2 * factory - 1) - factorx * factory) of which may be off rank */
  maxFinePoints        = (2 * factorx - 1) * (2 * factory - 1);
  maxOffRankFinePoints = maxFinePoints - factorx * factory;
  PetscCall(MatSeqAIJSetPreallocation(A, maxFinePoints, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, maxFinePoints, NULL, maxOffRankFinePoints, NULL));
  PetscCall(PetscMalloc3(maxFinePoints, &colf, maxFinePoints, &weight, maxFinePoints, &ic));

  PetscCall(DMStagGetCorners(dmc, &xc, &yc, NULL, &mc, &nc, NULL, &nExtraxc, &nExtrayc, NULL));

  for (d = 0; d < dof[0]; ++d)
    for (j = yc; j < yc + nc + nExtrayc; ++j)
      for (i = xc; i < xc + mc + nExtraxc; ++i) {
        rowc.i   = i;
        rowc.j   = j;
        rowc.c   = d;
        rowc.loc = DMSTAG_DOWN_LEFT;
        count    = 0;
        for (jj = -(factory - 1); jj <= factory - 1; ++jj)
          for (ii = -(factorx - 1); ii <= factorx - 1; ++ii) {
            colf[count].i   = i * factorx + ii;
            colf[count].j   = j * factory + jj;
            colf[count].c   = d;
            colf[count].loc = DMSTAG_DOWN_LEFT;
            PetscCall(SetRestrictionCoefficientVertex_Private(ii, factorx, &ax));
            PetscCall(SetRestrictionCoefficientVertex_Private(jj, factory, &ay));
            weight[count] = ax * ay;
            ++count;
          }
        if (i == 0) PetscCall(MergeRestrictionWeightToRight2d_Private(weight, 2 * factorx - 1, 2 * factory - 1));
        else if (i == Mc) PetscCall(MergeRestrictionWeightToLeft2d_Private(weight, 2 * factorx - 1, 2 * factory - 1));
        if (j == 0) PetscCall(MergeRestrictionWeightToTop2d_Private(weight, 2 * factorx - 1, 2 * factory - 1));
        else if (j == Nc) PetscCall(MergeRestrictionWeightToBottom2d_Private(weight, 2 * factorx - 1, 2 * factory - 1));

        PetscCall(RemoveZeroWeights_Private((2 * factorx - 1) * (2 * factory - 1), colf, weight, &count));
        PetscCall(DMStagStencilToIndexLocal(dmc, dim, 1, &rowc, &ir));
        PetscCall(DMStagStencilToIndexLocal(dmf, dim, count, colf, ic));
        PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
      }

  for (d = 0; d < dof[1]; ++d)
    for (j = yc; j < yc + nc; ++j)
      for (i = xc; i < xc + mc + nExtraxc; ++i) {
        rowc.i   = i;
        rowc.j   = j;
        rowc.c   = d;
        rowc.loc = DMSTAG_LEFT;
        count    = 0;
        for (jj = 0; jj < factory; ++jj)
          for (ii = -(factorx - 1); ii <= factorx - 1; ++ii) {
            colf[count].i   = i * factorx + ii;
            colf[count].j   = j * factory + jj;
            colf[count].c   = d;
            colf[count].loc = DMSTAG_LEFT;
            PetscCall(SetRestrictionCoefficientVertex_Private(ii, factorx, &ax));
            PetscCall(SetRestrictionCoefficientCenter_Private(jj, factory, &ay));
            weight[count] = ax * ay;
            ++count;
          }
        if (i == 0) PetscCall(MergeRestrictionWeightToRight2d_Private(weight, 2 * factorx - 1, factory));
        else if (i == Mc) PetscCall(MergeRestrictionWeightToLeft2d_Private(weight, 2 * factorx - 1, factory));

        PetscCall(RemoveZeroWeights_Private((2 * factorx - 1) * factory, colf, weight, &count));
        PetscCall(DMStagStencilToIndexLocal(dmc, dim, 1, &rowc, &ir));
        PetscCall(DMStagStencilToIndexLocal(dmf, dim, count, colf, ic));
        PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
      }

  for (d = 0; d < dof[1]; ++d)
    for (j = yc; j < yc + nc + nExtrayc; ++j)
      for (i = xc; i < xc + mc; ++i) {
        rowc.i   = i;
        rowc.j   = j;
        rowc.c   = d;
        rowc.loc = DMSTAG_DOWN;
        count    = 0;
        for (jj = -(factory - 1); jj <= factory - 1; ++jj)
          for (ii = 0; ii < factorx; ++ii) {
            colf[count].i   = i * factorx + ii;
            colf[count].j   = j * factory + jj;
            colf[count].c   = d;
            colf[count].loc = DMSTAG_DOWN;
            PetscCall(SetRestrictionCoefficientCenter_Private(ii, factorx, &ax));
            PetscCall(SetRestrictionCoefficientVertex_Private(jj, factory, &ay));
            weight[count] = ax * ay;
            ++count;
          }
        if (j == 0) PetscCall(MergeRestrictionWeightToTop2d_Private(weight, factorx, 2 * factory - 1));
        else if (j == Nc) PetscCall(MergeRestrictionWeightToBottom2d_Private(weight, factorx, 2 * factory - 1));

        PetscCall(RemoveZeroWeights_Private(factorx * (2 * factory - 1), colf, weight, &count));
        PetscCall(DMStagStencilToIndexLocal(dmc, dim, 1, &rowc, &ir));
        PetscCall(DMStagStencilToIndexLocal(dmf, dim, count, colf, ic));
        PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
      }

  for (d = 0; d < dof[2]; ++d)
    for (j = yc; j < yc + nc; ++j)
      for (i = xc; i < xc + mc; ++i) {
        rowc.i   = i;
        rowc.j   = j;
        rowc.c   = d;
        rowc.loc = DMSTAG_ELEMENT;
        count    = 0;
        for (jj = 0; jj < factory; ++jj)
          for (ii = 0; ii < factorx; ++ii) {
            colf[count].i   = i * factorx + ii;
            colf[count].j   = j * factory + jj;
            colf[count].c   = d;
            colf[count].loc = DMSTAG_ELEMENT;
            weight[count]   = 1 / (PetscScalar)(factorx * factory);
            ++count;
          }

        PetscCall(DMStagStencilToIndexLocal(dmc, dim, 1, &rowc, &ir));
        PetscCall(DMStagStencilToIndexLocal(dmf, dim, count, colf, ic));
        PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
      }

  PetscCall(PetscFree3(colf, weight, ic));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode DMStagPopulateRestriction3d_Internal(DM dmc, DM dmf, Mat A)
{
  PetscInt       Mc, Nc, Pc, Mf, Nf, Pf, factorx, factory, factorz, dof[4];
  PetscInt       xc, yc, zc, mc, nc, pc, nExtraxc, nExtrayc, nExtrazc, i, j, k, d, ii, jj, kk, count;
  PetscInt       maxFinePoints, maxOffRankFinePoints;
  DMStagStencil  rowc;
  DMStagStencil *colf;
  PetscScalar    ax, ay, az;
  PetscScalar   *weight;
  PetscInt       ir;
  PetscInt      *ic;
  const PetscInt dim = 3;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dmc, &Mc, &Nc, &Pc));
  PetscCall(DMStagGetGlobalSizes(dmf, &Mf, &Nf, &Pf));
  factorx = Mf / Mc;
  factory = Nf / Nc;
  factorz = Pf / Pc;
  PetscCall(DMStagGetDOF(dmc, &dof[0], &dof[1], &dof[2], &dof[3]));

  /* In 2D, each coarse point can receive from up to ((2 * factorx - 1) * (2 * factory - 1) * (2 * factorz - 1)) fine points,
     up to ((2 * factorx - 1) * (2 * factory - 1) * (2 * factorz - 1) - factorx * factory * factorz) of which may be off rank */
  maxFinePoints        = (2 * factorx - 1) * (2 * factory - 1) * (2 * factorz - 1);
  maxOffRankFinePoints = maxFinePoints - factorx * factory * factorz;
  PetscCall(MatSeqAIJSetPreallocation(A, maxFinePoints, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, maxFinePoints, NULL, maxOffRankFinePoints, NULL));
  PetscCall(PetscMalloc3(maxFinePoints, &colf, maxFinePoints, &weight, maxFinePoints, &ic));

  PetscCall(DMStagGetCorners(dmc, &xc, &yc, &zc, &mc, &nc, &pc, &nExtraxc, &nExtrayc, &nExtrazc));

  for (d = 0; d < dof[0]; ++d)
    for (k = zc; k < zc + pc + nExtrazc; ++k)
      for (j = yc; j < yc + nc + nExtrayc; ++j)
        for (i = xc; i < xc + mc + nExtraxc; ++i) {
          rowc.i   = i;
          rowc.j   = j;
          rowc.k   = k;
          rowc.c   = d;
          rowc.loc = DMSTAG_BACK_DOWN_LEFT;
          count    = 0;
          for (kk = -(factorz - 1); kk <= factorz - 1; ++kk)
            for (jj = -(factory - 1); jj <= factory - 1; ++jj)
              for (ii = -(factorx - 1); ii <= factorx - 1; ++ii) {
                colf[count].i   = i * factorx + ii;
                colf[count].j   = j * factory + jj;
                colf[count].k   = k * factorz + kk;
                colf[count].c   = d;
                colf[count].loc = DMSTAG_BACK_DOWN_LEFT;
                PetscCall(SetRestrictionCoefficientVertex_Private(ii, factorx, &ax));
                PetscCall(SetRestrictionCoefficientVertex_Private(jj, factory, &ay));
                PetscCall(SetRestrictionCoefficientVertex_Private(kk, factorz, &az));
                weight[count] = ax * ay * az;
                ++count;
              }
          if (i == 0) PetscCall(MergeRestrictionWeightToRight3d_Private(weight, 2 * factorx - 1, 2 * factory - 1, 2 * factorz - 1));
          else if (i == Mc) PetscCall(MergeRestrictionWeightToLeft3d_Private(weight, 2 * factorx - 1, 2 * factory - 1, 2 * factorz - 1));
          if (j == 0) PetscCall(MergeRestrictionWeightToTop3d_Private(weight, 2 * factorx - 1, 2 * factory - 1, 2 * factorz - 1));
          else if (j == Nc) PetscCall(MergeRestrictionWeightToBottom3d_Private(weight, 2 * factorx - 1, 2 * factory - 1, 2 * factorz - 1));
          if (k == 0) PetscCall(MergeRestrictionWeightToFront3d_Private(weight, 2 * factorx - 1, 2 * factory - 1, 2 * factorz - 1));
          else if (k == Pc) PetscCall(MergeRestrictionWeightToBack3d_Private(weight, 2 * factorx - 1, 2 * factory - 1, 2 * factorz - 1));

          PetscCall(RemoveZeroWeights_Private((2 * factorx - 1) * (2 * factory - 1) * (2 * factorz - 1), colf, weight, &count));
          PetscCall(DMStagStencilToIndexLocal(dmc, dim, 1, &rowc, &ir));
          PetscCall(DMStagStencilToIndexLocal(dmf, dim, count, colf, ic));
          PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
        }

  for (d = 0; d < dof[1]; ++d)
    for (k = zc; k < zc + pc; ++k)
      for (j = yc; j < yc + nc + nExtrayc; ++j)
        for (i = xc; i < xc + mc + nExtraxc; ++i) {
          rowc.i   = i;
          rowc.j   = j;
          rowc.k   = k;
          rowc.c   = d;
          rowc.loc = DMSTAG_DOWN_LEFT;
          count    = 0;
          for (kk = 0; kk < factorz; ++kk)
            for (jj = -(factory - 1); jj <= factory - 1; ++jj)
              for (ii = -(factorx - 1); ii <= factorx - 1; ++ii) {
                colf[count].i   = i * factorx + ii;
                colf[count].j   = j * factory + jj;
                colf[count].k   = k * factorz + kk;
                colf[count].c   = d;
                colf[count].loc = DMSTAG_DOWN_LEFT;
                PetscCall(SetRestrictionCoefficientVertex_Private(ii, factorx, &ax));
                PetscCall(SetRestrictionCoefficientVertex_Private(jj, factory, &ay));
                PetscCall(SetRestrictionCoefficientCenter_Private(kk, factorz, &az));
                weight[count] = ax * ay * az;
                ++count;
              }
          if (i == 0) PetscCall(MergeRestrictionWeightToRight3d_Private(weight, 2 * factorx - 1, 2 * factory - 1, factorz));
          else if (i == Mc) PetscCall(MergeRestrictionWeightToLeft3d_Private(weight, 2 * factorx - 1, 2 * factory - 1, factorz));
          if (j == 0) PetscCall(MergeRestrictionWeightToTop3d_Private(weight, 2 * factorx - 1, 2 * factory - 1, factorz));
          else if (j == Nc) PetscCall(MergeRestrictionWeightToBottom3d_Private(weight, 2 * factorx - 1, 2 * factory - 1, factorz));

          PetscCall(RemoveZeroWeights_Private((2 * factorx - 1) * (2 * factory - 1) * factorz, colf, weight, &count));
          PetscCall(DMStagStencilToIndexLocal(dmc, dim, 1, &rowc, &ir));
          PetscCall(DMStagStencilToIndexLocal(dmf, dim, count, colf, ic));
          PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
        }

  for (d = 0; d < dof[1]; ++d)
    for (k = zc; k < zc + pc + nExtrazc; ++k)
      for (j = yc; j < yc + nc; ++j)
        for (i = xc; i < xc + mc + nExtraxc; ++i) {
          rowc.i   = i;
          rowc.j   = j;
          rowc.k   = k;
          rowc.c   = d;
          rowc.loc = DMSTAG_BACK_LEFT;
          count    = 0;
          for (kk = -(factorz - 1); kk <= factorz - 1; ++kk)
            for (jj = 0; jj < factory; ++jj)
              for (ii = -(factorx - 1); ii <= factorx - 1; ++ii) {
                colf[count].i   = i * factorx + ii;
                colf[count].j   = j * factory + jj;
                colf[count].k   = k * factorz + kk;
                colf[count].c   = d;
                colf[count].loc = DMSTAG_BACK_LEFT;
                PetscCall(SetRestrictionCoefficientVertex_Private(ii, factorx, &ax));
                PetscCall(SetRestrictionCoefficientCenter_Private(jj, factory, &ay));
                PetscCall(SetRestrictionCoefficientVertex_Private(kk, factorz, &az));
                weight[count] = ax * ay * az;
                ++count;
              }
          if (i == 0) PetscCall(MergeRestrictionWeightToRight3d_Private(weight, 2 * factorx - 1, factory, 2 * factorz - 1));
          else if (i == Mc) PetscCall(MergeRestrictionWeightToLeft3d_Private(weight, 2 * factorx - 1, factory, 2 * factorz - 1));
          if (k == 0) PetscCall(MergeRestrictionWeightToFront3d_Private(weight, 2 * factorx - 1, factory, 2 * factorz - 1));
          else if (k == Pc) PetscCall(MergeRestrictionWeightToBack3d_Private(weight, 2 * factorx - 1, factory, 2 * factorz - 1));
        }

  for (d = 0; d < dof[1]; ++d)
    for (k = zc; k < zc + pc + nExtrazc; ++k)
      for (j = yc; j < yc + nc + nExtrayc; ++j)
        for (i = xc; i < xc + mc; ++i) {
          rowc.i   = i;
          rowc.j   = j;
          rowc.k   = k;
          rowc.c   = d;
          rowc.loc = DMSTAG_BACK_DOWN;
          count    = 0;
          for (kk = -(factorz - 1); kk <= factorz - 1; ++kk)
            for (jj = -(factory - 1); jj <= factory - 1; ++jj)
              for (ii = 0; ii < factorx; ++ii) {
                colf[count].i   = i * factorx + ii;
                colf[count].j   = j * factory + jj;
                colf[count].k   = k * factorz + kk;
                colf[count].c   = d;
                colf[count].loc = DMSTAG_BACK_DOWN;
                PetscCall(SetRestrictionCoefficientCenter_Private(ii, factorx, &ax));
                PetscCall(SetRestrictionCoefficientVertex_Private(jj, factory, &ay));
                PetscCall(SetRestrictionCoefficientVertex_Private(kk, factorz, &az));
                weight[count] = ax * ay * az;
                ++count;
              }
          if (j == 0) PetscCall(MergeRestrictionWeightToTop3d_Private(weight, factorx, 2 * factory - 1, 2 * factorz - 1));
          else if (j == Nc) PetscCall(MergeRestrictionWeightToBottom3d_Private(weight, factorx, 2 * factory - 1, 2 * factorz - 1));
          if (k == 0) PetscCall(MergeRestrictionWeightToFront3d_Private(weight, factorx, 2 * factory - 1, 2 * factorz - 1));
          else if (k == Pc) PetscCall(MergeRestrictionWeightToBack3d_Private(weight, factorx, 2 * factory - 1, 2 * factorz - 1));

          PetscCall(RemoveZeroWeights_Private(factorx * (2 * factory - 1) * (2 * factorz - 1), colf, weight, &count));
          PetscCall(DMStagStencilToIndexLocal(dmc, dim, 1, &rowc, &ir));
          PetscCall(DMStagStencilToIndexLocal(dmf, dim, count, colf, ic));
          PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
        }

  for (d = 0; d < dof[2]; ++d)
    for (k = zc; k < zc + pc; ++k)
      for (j = yc; j < yc + nc; ++j)
        for (i = xc; i < xc + mc + nExtraxc; ++i) {
          rowc.i   = i;
          rowc.j   = j;
          rowc.k   = k;
          rowc.c   = d;
          rowc.loc = DMSTAG_LEFT;
          count    = 0;
          for (kk = 0; kk < factorz; ++kk)
            for (jj = 0; jj < factory; ++jj)
              for (ii = -(factorx - 1); ii <= factorx - 1; ++ii) {
                colf[count].i   = i * factorx + ii;
                colf[count].j   = j * factory + jj;
                colf[count].k   = k * factorz + kk;
                colf[count].c   = d;
                colf[count].loc = DMSTAG_LEFT;
                PetscCall(SetRestrictionCoefficientVertex_Private(ii, factorx, &ax));
                PetscCall(SetRestrictionCoefficientCenter_Private(jj, factory, &ay));
                PetscCall(SetRestrictionCoefficientCenter_Private(kk, factorz, &az));
                weight[count] = ax * ay * az;
                ++count;
              }
          if (i == 0) PetscCall(MergeRestrictionWeightToRight3d_Private(weight, 2 * factorx - 1, factory, factorz));
          else if (i == Mc) PetscCall(MergeRestrictionWeightToLeft3d_Private(weight, 2 * factorx - 1, factory, factorz));

          PetscCall(RemoveZeroWeights_Private((2 * factorx - 1) * factory * factorz, colf, weight, &count));
          PetscCall(DMStagStencilToIndexLocal(dmc, dim, 1, &rowc, &ir));
          PetscCall(DMStagStencilToIndexLocal(dmf, dim, count, colf, ic));
          PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
        }

  for (d = 0; d < dof[2]; ++d)
    for (k = zc; k < zc + pc; ++k)
      for (j = yc; j < yc + nc + nExtrayc; ++j)
        for (i = xc; i < xc + mc; ++i) {
          rowc.i   = i;
          rowc.j   = j;
          rowc.k   = k;
          rowc.c   = d;
          rowc.loc = DMSTAG_DOWN;
          count    = 0;
          for (kk = 0; kk < factorz; ++kk)
            for (jj = -(factory - 1); jj <= factory - 1; ++jj)
              for (ii = 0; ii < factorx; ++ii) {
                colf[count].i   = i * factorx + ii;
                colf[count].j   = j * factory + jj;
                colf[count].k   = k * factorz + kk;
                colf[count].c   = d;
                colf[count].loc = DMSTAG_DOWN;
                PetscCall(SetRestrictionCoefficientCenter_Private(ii, factorx, &ax));
                PetscCall(SetRestrictionCoefficientVertex_Private(jj, factory, &ay));
                PetscCall(SetRestrictionCoefficientCenter_Private(kk, factorz, &az));
                weight[count] = ax * ay * az;
                ++count;
              }
          if (j == 0) PetscCall(MergeRestrictionWeightToTop3d_Private(weight, factorx, 2 * factory - 1, factorz));
          else if (j == Nc) PetscCall(MergeRestrictionWeightToBottom3d_Private(weight, factorx, 2 * factory - 1, factorz));

          PetscCall(RemoveZeroWeights_Private(factorx * (2 * factory - 1) * factorz, colf, weight, &count));
          PetscCall(DMStagStencilToIndexLocal(dmc, dim, 1, &rowc, &ir));
          PetscCall(DMStagStencilToIndexLocal(dmf, dim, count, colf, ic));
          PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
        }

  for (d = 0; d < dof[2]; ++d)
    for (k = zc; k < zc + pc + nExtrazc; ++k)
      for (j = yc; j < yc + nc; ++j)
        for (i = xc; i < xc + mc; ++i) {
          rowc.i   = i;
          rowc.j   = j;
          rowc.k   = k;
          rowc.c   = d;
          rowc.loc = DMSTAG_BACK;
          count    = 0;
          for (kk = -(factorz - 1); kk <= factorz - 1; ++kk)
            for (jj = 0; jj < factory; ++jj)
              for (ii = 0; ii < factorx; ++ii) {
                colf[count].i   = i * factorx + ii;
                colf[count].j   = j * factory + jj;
                colf[count].k   = k * factorz + kk;
                colf[count].c   = d;
                colf[count].loc = DMSTAG_BACK;
                PetscCall(SetRestrictionCoefficientCenter_Private(ii, factorx, &ax));
                PetscCall(SetRestrictionCoefficientCenter_Private(jj, factory, &ay));
                PetscCall(SetRestrictionCoefficientVertex_Private(kk, factorz, &az));
                weight[count] = ax * ay * az;
                ++count;
              }
          if (k == 0) PetscCall(MergeRestrictionWeightToFront3d_Private(weight, factorx, factory, 2 * factorz - 1));
          else if (k == Pc) PetscCall(MergeRestrictionWeightToBack3d_Private(weight, factorx, factory, 2 * factorz - 1));

          PetscCall(RemoveZeroWeights_Private(factorx * factory * (2 * factorz - 1), colf, weight, &count));
          PetscCall(DMStagStencilToIndexLocal(dmc, dim, 1, &rowc, &ir));
          PetscCall(DMStagStencilToIndexLocal(dmf, dim, count, colf, ic));
          PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
        }

  for (d = 0; d < dof[3]; ++d)
    for (k = zc; k < zc + pc; ++k)
      for (j = yc; j < yc + nc; ++j)
        for (i = xc; i < xc + mc; ++i) {
          rowc.i   = i;
          rowc.j   = j;
          rowc.k   = k;
          rowc.c   = d;
          rowc.loc = DMSTAG_ELEMENT;
          count    = 0;
          for (kk = 0; kk < factorz; ++kk)
            for (jj = 0; jj < factory; ++jj)
              for (ii = 0; ii < factorx; ++ii) {
                colf[count].i   = i * factorx + ii;
                colf[count].j   = j * factory + jj;
                colf[count].k   = k * factorz + kk;
                colf[count].c   = d;
                colf[count].loc = DMSTAG_ELEMENT;
                weight[count]   = 1 / (PetscScalar)(factorx * factory * factorz);
                ++count;
              }

          PetscCall(DMStagStencilToIndexLocal(dmc, dim, 1, &rowc, &ir));
          PetscCall(DMStagStencilToIndexLocal(dmf, dim, count, colf, ic));
          PetscCall(MatSetValuesLocal(A, 1, &ir, count, ic, weight, INSERT_VALUES));
        }

  PetscCall(PetscFree3(colf, weight, ic));
  PetscFunctionReturn(PETSC_SUCCESS);
}
