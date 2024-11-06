#include <../src/mat/impls/aij/mpi/mpiaij.h> /*I "petscmat.h" I*/
#include <petsc/private/vecimpl.h>
#include <petsc/private/sfimpl.h>
#include <petsc/private/isimpl.h>
#include <petscblaslapack.h>
#include <petscsf.h>
#include <petsc/private/hashmapi.h>

PetscErrorCode MatDestroy_MPIAIJ(Mat mat)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *)mat->data;

  PetscFunctionBegin;
  PetscCall(PetscLogObjectState((PetscObject)mat, "Rows=%" PetscInt_FMT ", Cols=%" PetscInt_FMT, mat->rmap->N, mat->cmap->N));
  PetscCall(MatStashDestroy_Private(&mat->stash));
  PetscCall(VecDestroy(&aij->diag));
  PetscCall(MatDestroy(&aij->A));
  PetscCall(MatDestroy(&aij->B));
#if defined(PETSC_USE_CTABLE)
  PetscCall(PetscHMapIDestroy(&aij->colmap));
#else
  PetscCall(PetscFree(aij->colmap));
#endif
  PetscCall(PetscFree(aij->garray));
  PetscCall(VecDestroy(&aij->lvec));
  PetscCall(VecScatterDestroy(&aij->Mvctx));
  PetscCall(PetscFree2(aij->rowvalues, aij->rowindices));
  PetscCall(PetscFree(aij->ld));

  PetscCall(PetscFree(mat->data));

  /* may be created by MatCreateMPIAIJSumSeqAIJSymbolic */
  PetscCall(PetscObjectCompose((PetscObject)mat, "MatMergeSeqsToMPI", NULL));

  PetscCall(PetscObjectChangeTypeName((PetscObject)mat, NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatStoreValues_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatRetrieveValues_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatIsTranspose_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatMPIAIJSetPreallocation_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatResetPreallocation_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatMPIAIJSetPreallocationCSR_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatDiagonalScaleLocal_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatConvert_mpiaij_mpibaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatConvert_mpiaij_mpisbaij_C", NULL));
#if defined(PETSC_HAVE_CUDA)
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatConvert_mpiaij_mpiaijcusparse_C", NULL));
#endif
#if defined(PETSC_HAVE_HIP)
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatConvert_mpiaij_mpiaijhipsparse_C", NULL));
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatConvert_mpiaij_mpiaijkokkos_C", NULL));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatConvert_mpiaij_mpidense_C", NULL));
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatConvert_mpiaij_elemental_C", NULL));
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatConvert_mpiaij_scalapack_C", NULL));
#endif
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatConvert_mpiaij_hypre_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatProductSetFromOptions_transpose_mpiaij_mpiaij_C", NULL));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatConvert_mpiaij_is_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatProductSetFromOptions_is_mpiaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatProductSetFromOptions_mpiaij_mpiaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatMPIAIJSetUseScalableIncreaseOverlap_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatConvert_mpiaij_mpiaijperm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatConvert_mpiaij_mpiaijsell_C", NULL));
#if defined(PETSC_HAVE_MKL_SPARSE)
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatConvert_mpiaij_mpiaijmkl_C", NULL));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatConvert_mpiaij_mpiaijcrl_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatConvert_mpiaij_is_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatConvert_mpiaij_mpisell_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatSetPreallocationCOO_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatSetValuesCOO_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* defines MatSetValues_MPI_Hash(), MatAssemblyBegin_MPI_Hash(), and  MatAssemblyEnd_MPI_Hash() */
#define TYPE AIJ
#define TYPE_AIJ
#include "../src/mat/impls/aij/mpi/mpihashmat.h"
#undef TYPE
#undef TYPE_AIJ

static PetscErrorCode MatGetRowIJ_MPIAIJ(Mat A, PetscInt oshift, PetscBool symmetric, PetscBool inodecompressed, PetscInt *m, const PetscInt *ia[], const PetscInt *ja[], PetscBool *done)
{
  Mat B;

  PetscFunctionBegin;
  PetscCall(MatMPIAIJGetLocalMat(A, MAT_INITIAL_MATRIX, &B));
  PetscCall(PetscObjectCompose((PetscObject)A, "MatGetRowIJ_MPIAIJ", (PetscObject)B));
  PetscCall(MatGetRowIJ(B, oshift, symmetric, inodecompressed, m, ia, ja, done));
  PetscCall(MatDestroy(&B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatRestoreRowIJ_MPIAIJ(Mat A, PetscInt oshift, PetscBool symmetric, PetscBool inodecompressed, PetscInt *m, const PetscInt *ia[], const PetscInt *ja[], PetscBool *done)
{
  Mat B;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)A, "MatGetRowIJ_MPIAIJ", (PetscObject *)&B));
  PetscCall(MatRestoreRowIJ(B, oshift, symmetric, inodecompressed, m, ia, ja, done));
  PetscCall(PetscObjectCompose((PetscObject)A, "MatGetRowIJ_MPIAIJ", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATAIJ - MATAIJ = "aij" - A matrix type to be used for sparse matrices.

   This matrix type is identical to` MATSEQAIJ` when constructed with a single process communicator,
   and `MATMPIAIJ` otherwise.  As a result, for single process communicators,
  `MatSeqAIJSetPreallocation()` is supported, and similarly `MatMPIAIJSetPreallocation()` is supported
  for communicators controlling multiple processes.  It is recommended that you call both of
  the above preallocation routines for simplicity.

   Options Database Key:
. -mat_type aij - sets the matrix type to `MATAIJ` during a call to `MatSetFromOptions()`

  Developer Note:
  Level: beginner

    Subclasses include `MATAIJCUSPARSE`, `MATAIJPERM`, `MATAIJSELL`, `MATAIJMKL`, `MATAIJCRL`, `MATAIJKOKKOS`,and also automatically switches over to use inodes when
   enough exist.

.seealso: [](ch_matrices), `Mat`, `MATMPIAIJ`, `MATSEQAIJ`, `MatCreateAIJ()`, `MatCreateSeqAIJ()`, `MATSEQAIJ`, `MATMPIAIJ`
M*/

/*MC
   MATAIJCRL - MATAIJCRL = "aijcrl" - A matrix type to be used for sparse matrices.

   This matrix type is identical to `MATSEQAIJCRL` when constructed with a single process communicator,
   and `MATMPIAIJCRL` otherwise.  As a result, for single process communicators,
   `MatSeqAIJSetPreallocation()` is supported, and similarly `MatMPIAIJSetPreallocation()` is supported
  for communicators controlling multiple processes.  It is recommended that you call both of
  the above preallocation routines for simplicity.

   Options Database Key:
. -mat_type aijcrl - sets the matrix type to `MATMPIAIJCRL` during a call to `MatSetFromOptions()`

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatCreateMPIAIJCRL`, `MATSEQAIJCRL`, `MATMPIAIJCRL`, `MATSEQAIJCRL`, `MATMPIAIJCRL`
M*/

static PetscErrorCode MatBindToCPU_MPIAIJ(Mat A, PetscBool flg)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP) || defined(PETSC_HAVE_VIENNACL)
  A->boundtocpu = flg;
#endif
  if (a->A) PetscCall(MatBindToCPU(a->A, flg));
  if (a->B) PetscCall(MatBindToCPU(a->B, flg));

  /* In addition to binding the diagonal and off-diagonal matrices, bind the local vectors used for matrix-vector products.
   * This maybe seems a little odd for a MatBindToCPU() call to do, but it makes no sense for the binding of these vectors
   * to differ from the parent matrix. */
  if (a->lvec) PetscCall(VecBindToCPU(a->lvec, flg));
  if (a->diag) PetscCall(VecBindToCPU(a->diag, flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetBlockSizes_MPIAIJ(Mat M, PetscInt rbs, PetscInt cbs)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *)M->data;

  PetscFunctionBegin;
  if (mat->A) {
    PetscCall(MatSetBlockSizes(mat->A, rbs, cbs));
    PetscCall(MatSetBlockSizes(mat->B, rbs, 1));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFindNonzeroRows_MPIAIJ(Mat M, IS *keptrows)
{
  Mat_MPIAIJ      *mat = (Mat_MPIAIJ *)M->data;
  Mat_SeqAIJ      *a   = (Mat_SeqAIJ *)mat->A->data;
  Mat_SeqAIJ      *b   = (Mat_SeqAIJ *)mat->B->data;
  const PetscInt  *ia, *ib;
  const MatScalar *aa, *bb, *aav, *bav;
  PetscInt         na, nb, i, j, *rows, cnt = 0, n0rows;
  PetscInt         m = M->rmap->n, rstart = M->rmap->rstart;

  PetscFunctionBegin;
  *keptrows = NULL;

  ia = a->i;
  ib = b->i;
  PetscCall(MatSeqAIJGetArrayRead(mat->A, &aav));
  PetscCall(MatSeqAIJGetArrayRead(mat->B, &bav));
  for (i = 0; i < m; i++) {
    na = ia[i + 1] - ia[i];
    nb = ib[i + 1] - ib[i];
    if (!na && !nb) {
      cnt++;
      goto ok1;
    }
    aa = aav + ia[i];
    for (j = 0; j < na; j++) {
      if (aa[j] != 0.0) goto ok1;
    }
    bb = PetscSafePointerPlusOffset(bav, ib[i]);
    for (j = 0; j < nb; j++) {
      if (bb[j] != 0.0) goto ok1;
    }
    cnt++;
  ok1:;
  }
  PetscCallMPI(MPIU_Allreduce(&cnt, &n0rows, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)M)));
  if (!n0rows) {
    PetscCall(MatSeqAIJRestoreArrayRead(mat->A, &aav));
    PetscCall(MatSeqAIJRestoreArrayRead(mat->B, &bav));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscMalloc1(M->rmap->n - cnt, &rows));
  cnt = 0;
  for (i = 0; i < m; i++) {
    na = ia[i + 1] - ia[i];
    nb = ib[i + 1] - ib[i];
    if (!na && !nb) continue;
    aa = aav + ia[i];
    for (j = 0; j < na; j++) {
      if (aa[j] != 0.0) {
        rows[cnt++] = rstart + i;
        goto ok2;
      }
    }
    bb = PetscSafePointerPlusOffset(bav, ib[i]);
    for (j = 0; j < nb; j++) {
      if (bb[j] != 0.0) {
        rows[cnt++] = rstart + i;
        goto ok2;
      }
    }
  ok2:;
  }
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)M), cnt, rows, PETSC_OWN_POINTER, keptrows));
  PetscCall(MatSeqAIJRestoreArrayRead(mat->A, &aav));
  PetscCall(MatSeqAIJRestoreArrayRead(mat->B, &bav));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDiagonalSet_MPIAIJ(Mat Y, Vec D, InsertMode is)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *)Y->data;
  PetscBool   cong;

  PetscFunctionBegin;
  PetscCall(MatHasCongruentLayouts(Y, &cong));
  if (Y->assembled && cong) {
    PetscCall(MatDiagonalSet(aij->A, D, is));
  } else {
    PetscCall(MatDiagonalSet_Default(Y, D, is));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFindZeroDiagonals_MPIAIJ(Mat M, IS *zrows)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *)M->data;
  PetscInt    i, rstart, nrows, *rows;

  PetscFunctionBegin;
  *zrows = NULL;
  PetscCall(MatFindZeroDiagonals_SeqAIJ_Private(aij->A, &nrows, &rows));
  PetscCall(MatGetOwnershipRange(M, &rstart, NULL));
  for (i = 0; i < nrows; i++) rows[i] += rstart;
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)M), nrows, rows, PETSC_OWN_POINTER, zrows));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetColumnReductions_MPIAIJ(Mat A, PetscInt type, PetscReal *reductions)
{
  Mat_MPIAIJ        *aij = (Mat_MPIAIJ *)A->data;
  PetscInt           i, m, n, *garray = aij->garray;
  Mat_SeqAIJ        *a_aij = (Mat_SeqAIJ *)aij->A->data;
  Mat_SeqAIJ        *b_aij = (Mat_SeqAIJ *)aij->B->data;
  PetscReal         *work;
  const PetscScalar *dummy;
  PetscMPIInt        in;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A, &m, &n));
  PetscCall(PetscCalloc1(n, &work));
  PetscCall(MatSeqAIJGetArrayRead(aij->A, &dummy));
  PetscCall(MatSeqAIJRestoreArrayRead(aij->A, &dummy));
  PetscCall(MatSeqAIJGetArrayRead(aij->B, &dummy));
  PetscCall(MatSeqAIJRestoreArrayRead(aij->B, &dummy));
  if (type == NORM_2) {
    for (i = 0; i < a_aij->i[aij->A->rmap->n]; i++) work[A->cmap->rstart + a_aij->j[i]] += PetscAbsScalar(a_aij->a[i] * a_aij->a[i]);
    for (i = 0; i < b_aij->i[aij->B->rmap->n]; i++) work[garray[b_aij->j[i]]] += PetscAbsScalar(b_aij->a[i] * b_aij->a[i]);
  } else if (type == NORM_1) {
    for (i = 0; i < a_aij->i[aij->A->rmap->n]; i++) work[A->cmap->rstart + a_aij->j[i]] += PetscAbsScalar(a_aij->a[i]);
    for (i = 0; i < b_aij->i[aij->B->rmap->n]; i++) work[garray[b_aij->j[i]]] += PetscAbsScalar(b_aij->a[i]);
  } else if (type == NORM_INFINITY) {
    for (i = 0; i < a_aij->i[aij->A->rmap->n]; i++) work[A->cmap->rstart + a_aij->j[i]] = PetscMax(PetscAbsScalar(a_aij->a[i]), work[A->cmap->rstart + a_aij->j[i]]);
    for (i = 0; i < b_aij->i[aij->B->rmap->n]; i++) work[garray[b_aij->j[i]]] = PetscMax(PetscAbsScalar(b_aij->a[i]), work[garray[b_aij->j[i]]]);
  } else if (type == REDUCTION_SUM_REALPART || type == REDUCTION_MEAN_REALPART) {
    for (i = 0; i < a_aij->i[aij->A->rmap->n]; i++) work[A->cmap->rstart + a_aij->j[i]] += PetscRealPart(a_aij->a[i]);
    for (i = 0; i < b_aij->i[aij->B->rmap->n]; i++) work[garray[b_aij->j[i]]] += PetscRealPart(b_aij->a[i]);
  } else if (type == REDUCTION_SUM_IMAGINARYPART || type == REDUCTION_MEAN_IMAGINARYPART) {
    for (i = 0; i < a_aij->i[aij->A->rmap->n]; i++) work[A->cmap->rstart + a_aij->j[i]] += PetscImaginaryPart(a_aij->a[i]);
    for (i = 0; i < b_aij->i[aij->B->rmap->n]; i++) work[garray[b_aij->j[i]]] += PetscImaginaryPart(b_aij->a[i]);
  } else SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Unknown reduction type");
  PetscCall(PetscMPIIntCast(n, &in));
  if (type == NORM_INFINITY) {
    PetscCallMPI(MPIU_Allreduce(work, reductions, in, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)A)));
  } else {
    PetscCallMPI(MPIU_Allreduce(work, reductions, in, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)A)));
  }
  PetscCall(PetscFree(work));
  if (type == NORM_2) {
    for (i = 0; i < n; i++) reductions[i] = PetscSqrtReal(reductions[i]);
  } else if (type == REDUCTION_MEAN_REALPART || type == REDUCTION_MEAN_IMAGINARYPART) {
    for (i = 0; i < n; i++) reductions[i] /= m;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFindOffBlockDiagonalEntries_MPIAIJ(Mat A, IS *is)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ *)A->data;
  IS              sis, gis;
  const PetscInt *isis, *igis;
  PetscInt        n, *iis, nsis, ngis, rstart, i;

  PetscFunctionBegin;
  PetscCall(MatFindOffBlockDiagonalEntries(a->A, &sis));
  PetscCall(MatFindNonzeroRows(a->B, &gis));
  PetscCall(ISGetSize(gis, &ngis));
  PetscCall(ISGetSize(sis, &nsis));
  PetscCall(ISGetIndices(sis, &isis));
  PetscCall(ISGetIndices(gis, &igis));

  PetscCall(PetscMalloc1(ngis + nsis, &iis));
  PetscCall(PetscArraycpy(iis, igis, ngis));
  PetscCall(PetscArraycpy(iis + ngis, isis, nsis));
  n = ngis + nsis;
  PetscCall(PetscSortRemoveDupsInt(&n, iis));
  PetscCall(MatGetOwnershipRange(A, &rstart, NULL));
  for (i = 0; i < n; i++) iis[i] += rstart;
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)A), n, iis, PETSC_OWN_POINTER, is));

  PetscCall(ISRestoreIndices(sis, &isis));
  PetscCall(ISRestoreIndices(gis, &igis));
  PetscCall(ISDestroy(&sis));
  PetscCall(ISDestroy(&gis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Local utility routine that creates a mapping from the global column
number to the local number in the off-diagonal part of the local
storage of the matrix.  When PETSC_USE_CTABLE is used this is scalable at
a slightly higher hash table cost; without it it is not scalable (each processor
has an order N integer array but is fast to access.
*/
PetscErrorCode MatCreateColmap_MPIAIJ_Private(Mat mat)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *)mat->data;
  PetscInt    n   = aij->B->cmap->n, i;

  PetscFunctionBegin;
  PetscCheck(!n || aij->garray, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MPIAIJ Matrix was assembled but is missing garray");
#if defined(PETSC_USE_CTABLE)
  PetscCall(PetscHMapICreateWithSize(n, &aij->colmap));
  for (i = 0; i < n; i++) PetscCall(PetscHMapISet(aij->colmap, aij->garray[i] + 1, i + 1));
#else
  PetscCall(PetscCalloc1(mat->cmap->N + 1, &aij->colmap));
  for (i = 0; i < n; i++) aij->colmap[aij->garray[i]] = i + 1;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define MatSetValues_SeqAIJ_A_Private(row, col, value, addv, orow, ocol) \
  do { \
    if (col <= lastcol1) low1 = 0; \
    else high1 = nrow1; \
    lastcol1 = col; \
    while (high1 - low1 > 5) { \
      t = (low1 + high1) / 2; \
      if (rp1[t] > col) high1 = t; \
      else low1 = t; \
    } \
    for (_i = low1; _i < high1; _i++) { \
      if (rp1[_i] > col) break; \
      if (rp1[_i] == col) { \
        if (addv == ADD_VALUES) { \
          ap1[_i] += value; \
          /* Not sure LogFlops will slow dow the code or not */ \
          (void)PetscLogFlops(1.0); \
        } else ap1[_i] = value; \
        goto a_noinsert; \
      } \
    } \
    if (value == 0.0 && ignorezeroentries && row != col) { \
      low1  = 0; \
      high1 = nrow1; \
      goto a_noinsert; \
    } \
    if (nonew == 1) { \
      low1  = 0; \
      high1 = nrow1; \
      goto a_noinsert; \
    } \
    PetscCheck(nonew != -1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Inserting a new nonzero at global row/column (%" PetscInt_FMT ", %" PetscInt_FMT ") into matrix", orow, ocol); \
    MatSeqXAIJReallocateAIJ(A, am, 1, nrow1, row, col, rmax1, aa, ai, aj, rp1, ap1, aimax, nonew, MatScalar); \
    N = nrow1++ - 1; \
    a->nz++; \
    high1++; \
    /* shift up all the later entries in this row */ \
    PetscCall(PetscArraymove(rp1 + _i + 1, rp1 + _i, N - _i + 1)); \
    PetscCall(PetscArraymove(ap1 + _i + 1, ap1 + _i, N - _i + 1)); \
    rp1[_i] = col; \
    ap1[_i] = value; \
  a_noinsert:; \
    ailen[row] = nrow1; \
  } while (0)

#define MatSetValues_SeqAIJ_B_Private(row, col, value, addv, orow, ocol) \
  do { \
    if (col <= lastcol2) low2 = 0; \
    else high2 = nrow2; \
    lastcol2 = col; \
    while (high2 - low2 > 5) { \
      t = (low2 + high2) / 2; \
      if (rp2[t] > col) high2 = t; \
      else low2 = t; \
    } \
    for (_i = low2; _i < high2; _i++) { \
      if (rp2[_i] > col) break; \
      if (rp2[_i] == col) { \
        if (addv == ADD_VALUES) { \
          ap2[_i] += value; \
          (void)PetscLogFlops(1.0); \
        } else ap2[_i] = value; \
        goto b_noinsert; \
      } \
    } \
    if (value == 0.0 && ignorezeroentries) { \
      low2  = 0; \
      high2 = nrow2; \
      goto b_noinsert; \
    } \
    if (nonew == 1) { \
      low2  = 0; \
      high2 = nrow2; \
      goto b_noinsert; \
    } \
    PetscCheck(nonew != -1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Inserting a new nonzero at global row/column (%" PetscInt_FMT ", %" PetscInt_FMT ") into matrix", orow, ocol); \
    MatSeqXAIJReallocateAIJ(B, bm, 1, nrow2, row, col, rmax2, ba, bi, bj, rp2, ap2, bimax, nonew, MatScalar); \
    N = nrow2++ - 1; \
    b->nz++; \
    high2++; \
    /* shift up all the later entries in this row */ \
    PetscCall(PetscArraymove(rp2 + _i + 1, rp2 + _i, N - _i + 1)); \
    PetscCall(PetscArraymove(ap2 + _i + 1, ap2 + _i, N - _i + 1)); \
    rp2[_i] = col; \
    ap2[_i] = value; \
  b_noinsert:; \
    bilen[row] = nrow2; \
  } while (0)

static PetscErrorCode MatSetValuesRow_MPIAIJ(Mat A, PetscInt row, const PetscScalar v[])
{
  Mat_MPIAIJ  *mat = (Mat_MPIAIJ *)A->data;
  Mat_SeqAIJ  *a = (Mat_SeqAIJ *)mat->A->data, *b = (Mat_SeqAIJ *)mat->B->data;
  PetscInt     l, *garray                         = mat->garray, diag;
  PetscScalar *aa, *ba;

  PetscFunctionBegin;
  /* code only works for square matrices A */

  /* find size of row to the left of the diagonal part */
  PetscCall(MatGetOwnershipRange(A, &diag, NULL));
  row = row - diag;
  for (l = 0; l < b->i[row + 1] - b->i[row]; l++) {
    if (garray[b->j[b->i[row] + l]] > diag) break;
  }
  if (l) {
    PetscCall(MatSeqAIJGetArray(mat->B, &ba));
    PetscCall(PetscArraycpy(ba + b->i[row], v, l));
    PetscCall(MatSeqAIJRestoreArray(mat->B, &ba));
  }

  /* diagonal part */
  if (a->i[row + 1] - a->i[row]) {
    PetscCall(MatSeqAIJGetArray(mat->A, &aa));
    PetscCall(PetscArraycpy(aa + a->i[row], v + l, a->i[row + 1] - a->i[row]));
    PetscCall(MatSeqAIJRestoreArray(mat->A, &aa));
  }

  /* right of diagonal part */
  if (b->i[row + 1] - b->i[row] - l) {
    PetscCall(MatSeqAIJGetArray(mat->B, &ba));
    PetscCall(PetscArraycpy(ba + b->i[row] + l, v + l + a->i[row + 1] - a->i[row], b->i[row + 1] - b->i[row] - l));
    PetscCall(MatSeqAIJRestoreArray(mat->B, &ba));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSetValues_MPIAIJ(Mat mat, PetscInt m, const PetscInt im[], PetscInt n, const PetscInt in[], const PetscScalar v[], InsertMode addv)
{
  Mat_MPIAIJ *aij   = (Mat_MPIAIJ *)mat->data;
  PetscScalar value = 0.0;
  PetscInt    i, j, rstart = mat->rmap->rstart, rend = mat->rmap->rend;
  PetscInt    cstart = mat->cmap->rstart, cend = mat->cmap->rend, row, col;
  PetscBool   roworiented = aij->roworiented;

  /* Some Variables required in the macro */
  Mat         A     = aij->A;
  Mat_SeqAIJ *a     = (Mat_SeqAIJ *)A->data;
  PetscInt   *aimax = a->imax, *ai = a->i, *ailen = a->ilen, *aj = a->j;
  PetscBool   ignorezeroentries = a->ignorezeroentries;
  Mat         B                 = aij->B;
  Mat_SeqAIJ *b                 = (Mat_SeqAIJ *)B->data;
  PetscInt   *bimax = b->imax, *bi = b->i, *bilen = b->ilen, *bj = b->j, bm = aij->B->rmap->n, am = aij->A->rmap->n;
  MatScalar  *aa, *ba;
  PetscInt   *rp1, *rp2, ii, nrow1, nrow2, _i, rmax1, rmax2, N, low1, high1, low2, high2, t, lastcol1, lastcol2;
  PetscInt    nonew;
  MatScalar  *ap1, *ap2;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetArray(A, &aa));
  PetscCall(MatSeqAIJGetArray(B, &ba));
  for (i = 0; i < m; i++) {
    if (im[i] < 0) continue;
    PetscCheck(im[i] < mat->rmap->N, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT, im[i], mat->rmap->N - 1);
    if (im[i] >= rstart && im[i] < rend) {
      row      = im[i] - rstart;
      lastcol1 = -1;
      rp1      = PetscSafePointerPlusOffset(aj, ai[row]);
      ap1      = PetscSafePointerPlusOffset(aa, ai[row]);
      rmax1    = aimax[row];
      nrow1    = ailen[row];
      low1     = 0;
      high1    = nrow1;
      lastcol2 = -1;
      rp2      = PetscSafePointerPlusOffset(bj, bi[row]);
      ap2      = PetscSafePointerPlusOffset(ba, bi[row]);
      rmax2    = bimax[row];
      nrow2    = bilen[row];
      low2     = 0;
      high2    = nrow2;

      for (j = 0; j < n; j++) {
        if (v) value = roworiented ? v[i * n + j] : v[i + j * m];
        if (ignorezeroentries && value == 0.0 && (addv == ADD_VALUES) && im[i] != in[j]) continue;
        if (in[j] >= cstart && in[j] < cend) {
          col   = in[j] - cstart;
          nonew = a->nonew;
          MatSetValues_SeqAIJ_A_Private(row, col, value, addv, im[i], in[j]);
        } else if (in[j] < 0) {
          continue;
        } else {
          PetscCheck(in[j] < mat->cmap->N, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Column too large: col %" PetscInt_FMT " max %" PetscInt_FMT, in[j], mat->cmap->N - 1);
          if (mat->was_assembled) {
            if (!aij->colmap) PetscCall(MatCreateColmap_MPIAIJ_Private(mat));
#if defined(PETSC_USE_CTABLE)
            PetscCall(PetscHMapIGetWithDefault(aij->colmap, in[j] + 1, 0, &col)); /* map global col ids to local ones */
            col--;
#else
            col = aij->colmap[in[j]] - 1;
#endif
            if (col < 0 && !((Mat_SeqAIJ *)aij->B->data)->nonew) { /* col < 0 means in[j] is a new col for B */
              PetscCall(MatDisAssemble_MPIAIJ(mat));               /* Change aij->B from reduced/local format to expanded/global format */
              col = in[j];
              /* Reinitialize the variables required by MatSetValues_SeqAIJ_B_Private() */
              B     = aij->B;
              b     = (Mat_SeqAIJ *)B->data;
              bimax = b->imax;
              bi    = b->i;
              bilen = b->ilen;
              bj    = b->j;
              ba    = b->a;
              rp2   = PetscSafePointerPlusOffset(bj, bi[row]);
              ap2   = PetscSafePointerPlusOffset(ba, bi[row]);
              rmax2 = bimax[row];
              nrow2 = bilen[row];
              low2  = 0;
              high2 = nrow2;
              bm    = aij->B->rmap->n;
              ba    = b->a;
            } else if (col < 0 && !(ignorezeroentries && value == 0.0)) {
              if (1 == ((Mat_SeqAIJ *)aij->B->data)->nonew) {
                PetscCall(PetscInfo(mat, "Skipping of insertion of new nonzero location in off-diagonal portion of matrix %g(%" PetscInt_FMT ",%" PetscInt_FMT ")\n", (double)PetscRealPart(value), im[i], in[j]));
              } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Inserting a new nonzero at global row/column (%" PetscInt_FMT ", %" PetscInt_FMT ") into matrix", im[i], in[j]);
            }
          } else col = in[j];
          nonew = b->nonew;
          MatSetValues_SeqAIJ_B_Private(row, col, value, addv, im[i], in[j]);
        }
      }
    } else {
      PetscCheck(!mat->nooffprocentries, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Setting off process row %" PetscInt_FMT " even though MatSetOption(,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) was set", im[i]);
      if (!aij->donotstash) {
        mat->assembled = PETSC_FALSE;
        if (roworiented) {
          PetscCall(MatStashValuesRow_Private(&mat->stash, im[i], n, in, PetscSafePointerPlusOffset(v, i * n), (PetscBool)(ignorezeroentries && (addv == ADD_VALUES))));
        } else {
          PetscCall(MatStashValuesCol_Private(&mat->stash, im[i], n, in, PetscSafePointerPlusOffset(v, i), m, (PetscBool)(ignorezeroentries && (addv == ADD_VALUES))));
        }
      }
    }
  }
  PetscCall(MatSeqAIJRestoreArray(A, &aa)); /* aa, bb might have been free'd due to reallocation above. But we don't access them here */
  PetscCall(MatSeqAIJRestoreArray(B, &ba));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    This function sets the j and ilen arrays (of the diagonal and off-diagonal part) of an MPIAIJ-matrix.
    The values in mat_i have to be sorted and the values in mat_j have to be sorted for each row (CSR-like).
    No off-processor parts off the matrix are allowed here and mat->was_assembled has to be PETSC_FALSE.
*/
PetscErrorCode MatSetValues_MPIAIJ_CopyFromCSRFormat_Symbolic(Mat mat, const PetscInt mat_j[], const PetscInt mat_i[])
{
  Mat_MPIAIJ *aij    = (Mat_MPIAIJ *)mat->data;
  Mat         A      = aij->A; /* diagonal part of the matrix */
  Mat         B      = aij->B; /* off-diagonal part of the matrix */
  Mat_SeqAIJ *a      = (Mat_SeqAIJ *)A->data;
  Mat_SeqAIJ *b      = (Mat_SeqAIJ *)B->data;
  PetscInt    cstart = mat->cmap->rstart, cend = mat->cmap->rend, col;
  PetscInt   *ailen = a->ilen, *aj = a->j;
  PetscInt   *bilen = b->ilen, *bj = b->j;
  PetscInt    am          = aij->A->rmap->n, j;
  PetscInt    diag_so_far = 0, dnz;
  PetscInt    offd_so_far = 0, onz;

  PetscFunctionBegin;
  /* Iterate over all rows of the matrix */
  for (j = 0; j < am; j++) {
    dnz = onz = 0;
    /*  Iterate over all non-zero columns of the current row */
    for (col = mat_i[j]; col < mat_i[j + 1]; col++) {
      /* If column is in the diagonal */
      if (mat_j[col] >= cstart && mat_j[col] < cend) {
        aj[diag_so_far++] = mat_j[col] - cstart;
        dnz++;
      } else { /* off-diagonal entries */
        bj[offd_so_far++] = mat_j[col];
        onz++;
      }
    }
    ailen[j] = dnz;
    bilen[j] = onz;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    This function sets the local j, a and ilen arrays (of the diagonal and off-diagonal part) of an MPIAIJ-matrix.
    The values in mat_i have to be sorted and the values in mat_j have to be sorted for each row (CSR-like).
    No off-processor parts off the matrix are allowed here, they are set at a later point by MatSetValues_MPIAIJ.
    Also, mat->was_assembled has to be false, otherwise the statement aj[rowstart_diag+dnz_row] = mat_j[col] - cstart;
    would not be true and the more complex MatSetValues_MPIAIJ has to be used.
*/
PetscErrorCode MatSetValues_MPIAIJ_CopyFromCSRFormat(Mat mat, const PetscInt mat_j[], const PetscInt mat_i[], const PetscScalar mat_a[])
{
  Mat_MPIAIJ  *aij  = (Mat_MPIAIJ *)mat->data;
  Mat          A    = aij->A; /* diagonal part of the matrix */
  Mat          B    = aij->B; /* off-diagonal part of the matrix */
  Mat_SeqAIJ  *aijd = (Mat_SeqAIJ *)aij->A->data, *aijo = (Mat_SeqAIJ *)aij->B->data;
  Mat_SeqAIJ  *a      = (Mat_SeqAIJ *)A->data;
  Mat_SeqAIJ  *b      = (Mat_SeqAIJ *)B->data;
  PetscInt     cstart = mat->cmap->rstart, cend = mat->cmap->rend;
  PetscInt    *ailen = a->ilen, *aj = a->j;
  PetscInt    *bilen = b->ilen, *bj = b->j;
  PetscInt     am          = aij->A->rmap->n, j;
  PetscInt    *full_diag_i = aijd->i, *full_offd_i = aijo->i; /* These variables can also include non-local elements, which are set at a later point. */
  PetscInt     col, dnz_row, onz_row, rowstart_diag, rowstart_offd;
  PetscScalar *aa = a->a, *ba = b->a;

  PetscFunctionBegin;
  /* Iterate over all rows of the matrix */
  for (j = 0; j < am; j++) {
    dnz_row = onz_row = 0;
    rowstart_offd     = full_offd_i[j];
    rowstart_diag     = full_diag_i[j];
    /*  Iterate over all non-zero columns of the current row */
    for (col = mat_i[j]; col < mat_i[j + 1]; col++) {
      /* If column is in the diagonal */
      if (mat_j[col] >= cstart && mat_j[col] < cend) {
        aj[rowstart_diag + dnz_row] = mat_j[col] - cstart;
        aa[rowstart_diag + dnz_row] = mat_a[col];
        dnz_row++;
      } else { /* off-diagonal entries */
        bj[rowstart_offd + onz_row] = mat_j[col];
        ba[rowstart_offd + onz_row] = mat_a[col];
        onz_row++;
      }
    }
    ailen[j] = dnz_row;
    bilen[j] = onz_row;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetValues_MPIAIJ(Mat mat, PetscInt m, const PetscInt idxm[], PetscInt n, const PetscInt idxn[], PetscScalar v[])
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *)mat->data;
  PetscInt    i, j, rstart = mat->rmap->rstart, rend = mat->rmap->rend;
  PetscInt    cstart = mat->cmap->rstart, cend = mat->cmap->rend, row, col;

  PetscFunctionBegin;
  for (i = 0; i < m; i++) {
    if (idxm[i] < 0) continue; /* negative row */
    PetscCheck(idxm[i] < mat->rmap->N, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT, idxm[i], mat->rmap->N - 1);
    PetscCheck(idxm[i] >= rstart && idxm[i] < rend, PETSC_COMM_SELF, PETSC_ERR_SUP, "Only local values currently supported, row requested %" PetscInt_FMT " range [%" PetscInt_FMT " %" PetscInt_FMT ")", idxm[i], rstart, rend);
    row = idxm[i] - rstart;
    for (j = 0; j < n; j++) {
      if (idxn[j] < 0) continue; /* negative column */
      PetscCheck(idxn[j] < mat->cmap->N, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Column too large: col %" PetscInt_FMT " max %" PetscInt_FMT, idxn[j], mat->cmap->N - 1);
      if (idxn[j] >= cstart && idxn[j] < cend) {
        col = idxn[j] - cstart;
        PetscCall(MatGetValues(aij->A, 1, &row, 1, &col, v + i * n + j));
      } else {
        if (!aij->colmap) PetscCall(MatCreateColmap_MPIAIJ_Private(mat));
#if defined(PETSC_USE_CTABLE)
        PetscCall(PetscHMapIGetWithDefault(aij->colmap, idxn[j] + 1, 0, &col));
        col--;
#else
        col = aij->colmap[idxn[j]] - 1;
#endif
        if ((col < 0) || (aij->garray[col] != idxn[j])) *(v + i * n + j) = 0.0;
        else PetscCall(MatGetValues(aij->B, 1, &row, 1, &col, v + i * n + j));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAssemblyBegin_MPIAIJ(Mat mat, MatAssemblyType mode)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *)mat->data;
  PetscInt    nstash, reallocs;

  PetscFunctionBegin;
  if (aij->donotstash || mat->nooffprocentries) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(MatStashScatterBegin_Private(mat, &mat->stash, mat->rmap->range));
  PetscCall(MatStashGetInfo_Private(&mat->stash, &nstash, &reallocs));
  PetscCall(PetscInfo(aij->A, "Stash has %" PetscInt_FMT " entries, uses %" PetscInt_FMT " mallocs.\n", nstash, reallocs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatAssemblyEnd_MPIAIJ(Mat mat, MatAssemblyType mode)
{
  Mat_MPIAIJ  *aij = (Mat_MPIAIJ *)mat->data;
  PetscMPIInt  n;
  PetscInt     i, j, rstart, ncols, flg;
  PetscInt    *row, *col;
  PetscBool    other_disassembled;
  PetscScalar *val;

  /* do not use 'b = (Mat_SeqAIJ*)aij->B->data' as B can be reset in disassembly */

  PetscFunctionBegin;
  if (!aij->donotstash && !mat->nooffprocentries) {
    while (1) {
      PetscCall(MatStashScatterGetMesg_Private(&mat->stash, &n, &row, &col, &val, &flg));
      if (!flg) break;

      for (i = 0; i < n;) {
        /* Now identify the consecutive vals belonging to the same row */
        for (j = i, rstart = row[j]; j < n; j++) {
          if (row[j] != rstart) break;
        }
        if (j < n) ncols = j - i;
        else ncols = n - i;
        /* Now assemble all these values with a single function call */
        PetscCall(MatSetValues_MPIAIJ(mat, 1, row + i, ncols, col + i, val + i, mat->insertmode));
        i = j;
      }
    }
    PetscCall(MatStashScatterEnd_Private(&mat->stash));
  }
#if defined(PETSC_HAVE_DEVICE)
  if (mat->offloadmask == PETSC_OFFLOAD_CPU) aij->A->offloadmask = PETSC_OFFLOAD_CPU;
  /* We call MatBindToCPU() on aij->A and aij->B here, because if MatBindToCPU_MPIAIJ() is called before assembly, it cannot bind these. */
  if (mat->boundtocpu) {
    PetscCall(MatBindToCPU(aij->A, PETSC_TRUE));
    PetscCall(MatBindToCPU(aij->B, PETSC_TRUE));
  }
#endif
  PetscCall(MatAssemblyBegin(aij->A, mode));
  PetscCall(MatAssemblyEnd(aij->A, mode));

  /* determine if any processor has disassembled, if so we must
     also disassemble ourself, in order that we may reassemble. */
  /*
     if nonzero structure of submatrix B cannot change then we know that
     no processor disassembled thus we can skip this stuff
  */
  if (!((Mat_SeqAIJ *)aij->B->data)->nonew) {
    PetscCallMPI(MPIU_Allreduce(&mat->was_assembled, &other_disassembled, 1, MPIU_BOOL, MPI_LAND, PetscObjectComm((PetscObject)mat)));
    if (mat->was_assembled && !other_disassembled) { /* mat on this rank has reduced off-diag B with local col ids, but globally it does not */
      PetscCall(MatDisAssemble_MPIAIJ(mat));
    }
  }
  if (!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) PetscCall(MatSetUpMultiply_MPIAIJ(mat));
  PetscCall(MatSetOption(aij->B, MAT_USE_INODES, PETSC_FALSE));
#if defined(PETSC_HAVE_DEVICE)
  if (mat->offloadmask == PETSC_OFFLOAD_CPU && aij->B->offloadmask != PETSC_OFFLOAD_UNALLOCATED) aij->B->offloadmask = PETSC_OFFLOAD_CPU;
#endif
  PetscCall(MatAssemblyBegin(aij->B, mode));
  PetscCall(MatAssemblyEnd(aij->B, mode));

  PetscCall(PetscFree2(aij->rowvalues, aij->rowindices));

  aij->rowvalues = NULL;

  PetscCall(VecDestroy(&aij->diag));

  /* if no new nonzero locations are allowed in matrix then only set the matrix state the first time through */
  if ((!mat->was_assembled && mode == MAT_FINAL_ASSEMBLY) || !((Mat_SeqAIJ *)aij->A->data)->nonew) {
    PetscObjectState state = aij->A->nonzerostate + aij->B->nonzerostate;
    PetscCallMPI(MPIU_Allreduce(&state, &mat->nonzerostate, 1, MPIU_INT64, MPI_SUM, PetscObjectComm((PetscObject)mat)));
  }
#if defined(PETSC_HAVE_DEVICE)
  mat->offloadmask = PETSC_OFFLOAD_BOTH;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroEntries_MPIAIJ(Mat A)
{
  Mat_MPIAIJ *l = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatZeroEntries(l->A));
  PetscCall(MatZeroEntries(l->B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroRows_MPIAIJ(Mat A, PetscInt N, const PetscInt rows[], PetscScalar diag, Vec x, Vec b)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *)A->data;
  PetscInt   *lrows;
  PetscInt    r, len;
  PetscBool   cong;

  PetscFunctionBegin;
  /* get locally owned rows */
  PetscCall(MatZeroRowsMapLocal_Private(A, N, rows, &len, &lrows));
  PetscCall(MatHasCongruentLayouts(A, &cong));
  /* fix right-hand side if needed */
  if (x && b) {
    const PetscScalar *xx;
    PetscScalar       *bb;

    PetscCheck(cong, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Need matching row/col layout");
    PetscCall(VecGetArrayRead(x, &xx));
    PetscCall(VecGetArray(b, &bb));
    for (r = 0; r < len; ++r) bb[lrows[r]] = diag * xx[lrows[r]];
    PetscCall(VecRestoreArrayRead(x, &xx));
    PetscCall(VecRestoreArray(b, &bb));
  }

  if (diag != 0.0 && cong) {
    PetscCall(MatZeroRows(mat->A, len, lrows, diag, NULL, NULL));
    PetscCall(MatZeroRows(mat->B, len, lrows, 0.0, NULL, NULL));
  } else if (diag != 0.0) { /* non-square or non congruent layouts -> if keepnonzeropattern is false, we allow for new insertion */
    Mat_SeqAIJ *aijA = (Mat_SeqAIJ *)mat->A->data;
    Mat_SeqAIJ *aijB = (Mat_SeqAIJ *)mat->B->data;
    PetscInt    nnwA, nnwB;
    PetscBool   nnzA, nnzB;

    nnwA = aijA->nonew;
    nnwB = aijB->nonew;
    nnzA = aijA->keepnonzeropattern;
    nnzB = aijB->keepnonzeropattern;
    if (!nnzA) {
      PetscCall(PetscInfo(mat->A, "Requested to not keep the pattern and add a nonzero diagonal; may encounter reallocations on diagonal block.\n"));
      aijA->nonew = 0;
    }
    if (!nnzB) {
      PetscCall(PetscInfo(mat->B, "Requested to not keep the pattern and add a nonzero diagonal; may encounter reallocations on off-diagonal block.\n"));
      aijB->nonew = 0;
    }
    /* Must zero here before the next loop */
    PetscCall(MatZeroRows(mat->A, len, lrows, 0.0, NULL, NULL));
    PetscCall(MatZeroRows(mat->B, len, lrows, 0.0, NULL, NULL));
    for (r = 0; r < len; ++r) {
      const PetscInt row = lrows[r] + A->rmap->rstart;
      if (row >= A->cmap->N) continue;
      PetscCall(MatSetValues(A, 1, &row, 1, &row, &diag, INSERT_VALUES));
    }
    aijA->nonew = nnwA;
    aijB->nonew = nnwB;
  } else {
    PetscCall(MatZeroRows(mat->A, len, lrows, 0.0, NULL, NULL));
    PetscCall(MatZeroRows(mat->B, len, lrows, 0.0, NULL, NULL));
  }
  PetscCall(PetscFree(lrows));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /* only change matrix nonzero state if pattern was allowed to be changed */
  if (!((Mat_SeqAIJ *)mat->A->data)->keepnonzeropattern || !((Mat_SeqAIJ *)mat->A->data)->nonew) {
    PetscObjectState state = mat->A->nonzerostate + mat->B->nonzerostate;
    PetscCallMPI(MPIU_Allreduce(&state, &A->nonzerostate, 1, MPIU_INT64, MPI_SUM, PetscObjectComm((PetscObject)A)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroRowsColumns_MPIAIJ(Mat A, PetscInt N, const PetscInt rows[], PetscScalar diag, Vec x, Vec b)
{
  Mat_MPIAIJ        *l = (Mat_MPIAIJ *)A->data;
  PetscInt           n = A->rmap->n;
  PetscInt           i, j, r, m, len = 0;
  PetscInt          *lrows, *owners = A->rmap->range;
  PetscMPIInt        p = 0;
  PetscSFNode       *rrows;
  PetscSF            sf;
  const PetscScalar *xx;
  PetscScalar       *bb, *mask, *aij_a;
  Vec                xmask, lmask;
  Mat_SeqAIJ        *aij = (Mat_SeqAIJ *)l->B->data;
  const PetscInt    *aj, *ii, *ridx;
  PetscScalar       *aa;

  PetscFunctionBegin;
  /* Create SF where leaves are input rows and roots are owned rows */
  PetscCall(PetscMalloc1(n, &lrows));
  for (r = 0; r < n; ++r) lrows[r] = -1;
  PetscCall(PetscMalloc1(N, &rrows));
  for (r = 0; r < N; ++r) {
    const PetscInt idx = rows[r];
    PetscCheck(idx >= 0 && A->rmap->N > idx, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Row %" PetscInt_FMT " out of range [0,%" PetscInt_FMT ")", idx, A->rmap->N);
    if (idx < owners[p] || owners[p + 1] <= idx) { /* short-circuit the search if the last p owns this row too */
      PetscCall(PetscLayoutFindOwner(A->rmap, idx, &p));
    }
    rrows[r].rank  = p;
    rrows[r].index = rows[r] - owners[p];
  }
  PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)A), &sf));
  PetscCall(PetscSFSetGraph(sf, n, N, NULL, PETSC_OWN_POINTER, rrows, PETSC_OWN_POINTER));
  /* Collect flags for rows to be zeroed */
  PetscCall(PetscSFReduceBegin(sf, MPIU_INT, (PetscInt *)rows, lrows, MPI_LOR));
  PetscCall(PetscSFReduceEnd(sf, MPIU_INT, (PetscInt *)rows, lrows, MPI_LOR));
  PetscCall(PetscSFDestroy(&sf));
  /* Compress and put in row numbers */
  for (r = 0; r < n; ++r)
    if (lrows[r] >= 0) lrows[len++] = r;
  /* zero diagonal part of matrix */
  PetscCall(MatZeroRowsColumns(l->A, len, lrows, diag, x, b));
  /* handle off-diagonal part of matrix */
  PetscCall(MatCreateVecs(A, &xmask, NULL));
  PetscCall(VecDuplicate(l->lvec, &lmask));
  PetscCall(VecGetArray(xmask, &bb));
  for (i = 0; i < len; i++) bb[lrows[i]] = 1;
  PetscCall(VecRestoreArray(xmask, &bb));
  PetscCall(VecScatterBegin(l->Mvctx, xmask, lmask, ADD_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(l->Mvctx, xmask, lmask, ADD_VALUES, SCATTER_FORWARD));
  PetscCall(VecDestroy(&xmask));
  if (x && b) { /* this code is buggy when the row and column layout don't match */
    PetscBool cong;

    PetscCall(MatHasCongruentLayouts(A, &cong));
    PetscCheck(cong, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Need matching row/col layout");
    PetscCall(VecScatterBegin(l->Mvctx, x, l->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(l->Mvctx, x, l->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecGetArrayRead(l->lvec, &xx));
    PetscCall(VecGetArray(b, &bb));
  }
  PetscCall(VecGetArray(lmask, &mask));
  /* remove zeroed rows of off-diagonal matrix */
  PetscCall(MatSeqAIJGetArray(l->B, &aij_a));
  ii = aij->i;
  for (i = 0; i < len; i++) PetscCall(PetscArrayzero(PetscSafePointerPlusOffset(aij_a, ii[lrows[i]]), ii[lrows[i] + 1] - ii[lrows[i]]));
  /* loop over all elements of off process part of matrix zeroing removed columns*/
  if (aij->compressedrow.use) {
    m    = aij->compressedrow.nrows;
    ii   = aij->compressedrow.i;
    ridx = aij->compressedrow.rindex;
    for (i = 0; i < m; i++) {
      n  = ii[i + 1] - ii[i];
      aj = aij->j + ii[i];
      aa = aij_a + ii[i];

      for (j = 0; j < n; j++) {
        if (PetscAbsScalar(mask[*aj])) {
          if (b) bb[*ridx] -= *aa * xx[*aj];
          *aa = 0.0;
        }
        aa++;
        aj++;
      }
      ridx++;
    }
  } else { /* do not use compressed row format */
    m = l->B->rmap->n;
    for (i = 0; i < m; i++) {
      n  = ii[i + 1] - ii[i];
      aj = aij->j + ii[i];
      aa = aij_a + ii[i];
      for (j = 0; j < n; j++) {
        if (PetscAbsScalar(mask[*aj])) {
          if (b) bb[i] -= *aa * xx[*aj];
          *aa = 0.0;
        }
        aa++;
        aj++;
      }
    }
  }
  if (x && b) {
    PetscCall(VecRestoreArray(b, &bb));
    PetscCall(VecRestoreArrayRead(l->lvec, &xx));
  }
  PetscCall(MatSeqAIJRestoreArray(l->B, &aij_a));
  PetscCall(VecRestoreArray(lmask, &mask));
  PetscCall(VecDestroy(&lmask));
  PetscCall(PetscFree(lrows));

  /* only change matrix nonzero state if pattern was allowed to be changed */
  if (!((Mat_SeqAIJ *)l->A->data)->nonew) {
    PetscObjectState state = l->A->nonzerostate + l->B->nonzerostate;
    PetscCallMPI(MPIU_Allreduce(&state, &A->nonzerostate, 1, MPIU_INT64, MPI_SUM, PetscObjectComm((PetscObject)A)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_MPIAIJ(Mat A, Vec xx, Vec yy)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;
  PetscInt    nt;
  VecScatter  Mvctx = a->Mvctx;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(xx, &nt));
  PetscCheck(nt == A->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible partition of A (%" PetscInt_FMT ") and xx (%" PetscInt_FMT ")", A->cmap->n, nt);
  PetscCall(VecScatterBegin(Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscUseTypeMethod(a->A, mult, xx, yy);
  PetscCall(VecScatterEnd(Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscUseTypeMethod(a->B, multadd, a->lvec, yy, yy);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultDiagonalBlock_MPIAIJ(Mat A, Vec bb, Vec xx)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatMultDiagonalBlock(a->A, bb, xx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultAdd_MPIAIJ(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_MPIAIJ *a     = (Mat_MPIAIJ *)A->data;
  VecScatter  Mvctx = a->Mvctx;

  PetscFunctionBegin;
  PetscCall(VecScatterBegin(Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*a->A->ops->multadd)(a->A, xx, yy, zz));
  PetscCall(VecScatterEnd(Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*a->B->ops->multadd)(a->B, a->lvec, zz, zz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTranspose_MPIAIJ(Mat A, Vec xx, Vec yy)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  /* do nondiagonal part */
  PetscCall((*a->B->ops->multtranspose)(a->B, xx, a->lvec));
  /* do local part */
  PetscCall((*a->A->ops->multtranspose)(a->A, xx, yy));
  /* add partial results together */
  PetscCall(VecScatterBegin(a->Mvctx, a->lvec, yy, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(a->Mvctx, a->lvec, yy, ADD_VALUES, SCATTER_REVERSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatIsTranspose_MPIAIJ(Mat Amat, Mat Bmat, PetscReal tol, PetscBool *f)
{
  MPI_Comm    comm;
  Mat_MPIAIJ *Aij = (Mat_MPIAIJ *)Amat->data, *Bij = (Mat_MPIAIJ *)Bmat->data;
  Mat         Adia = Aij->A, Bdia = Bij->A, Aoff, Boff, *Aoffs, *Boffs;
  IS          Me, Notme;
  PetscInt    M, N, first, last, *notme, i;
  PetscBool   lf;
  PetscMPIInt size;

  PetscFunctionBegin;
  /* Easy test: symmetric diagonal block */
  PetscCall(MatIsTranspose(Adia, Bdia, tol, &lf));
  PetscCallMPI(MPIU_Allreduce(&lf, f, 1, MPIU_BOOL, MPI_LAND, PetscObjectComm((PetscObject)Amat)));
  if (!*f) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectGetComm((PetscObject)Amat, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size == 1) PetscFunctionReturn(PETSC_SUCCESS);

  /* Hard test: off-diagonal block. This takes a MatCreateSubMatrix. */
  PetscCall(MatGetSize(Amat, &M, &N));
  PetscCall(MatGetOwnershipRange(Amat, &first, &last));
  PetscCall(PetscMalloc1(N - last + first, &notme));
  for (i = 0; i < first; i++) notme[i] = i;
  for (i = last; i < M; i++) notme[i - last + first] = i;
  PetscCall(ISCreateGeneral(MPI_COMM_SELF, N - last + first, notme, PETSC_COPY_VALUES, &Notme));
  PetscCall(ISCreateStride(MPI_COMM_SELF, last - first, first, 1, &Me));
  PetscCall(MatCreateSubMatrices(Amat, 1, &Me, &Notme, MAT_INITIAL_MATRIX, &Aoffs));
  Aoff = Aoffs[0];
  PetscCall(MatCreateSubMatrices(Bmat, 1, &Notme, &Me, MAT_INITIAL_MATRIX, &Boffs));
  Boff = Boffs[0];
  PetscCall(MatIsTranspose(Aoff, Boff, tol, f));
  PetscCall(MatDestroyMatrices(1, &Aoffs));
  PetscCall(MatDestroyMatrices(1, &Boffs));
  PetscCall(ISDestroy(&Me));
  PetscCall(ISDestroy(&Notme));
  PetscCall(PetscFree(notme));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTransposeAdd_MPIAIJ(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  /* do nondiagonal part */
  PetscCall((*a->B->ops->multtranspose)(a->B, xx, a->lvec));
  /* do local part */
  PetscCall((*a->A->ops->multtransposeadd)(a->A, xx, yy, zz));
  /* add partial results together */
  PetscCall(VecScatterBegin(a->Mvctx, a->lvec, zz, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(a->Mvctx, a->lvec, zz, ADD_VALUES, SCATTER_REVERSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  This only works correctly for square matrices where the subblock A->A is the
   diagonal block
*/
static PetscErrorCode MatGetDiagonal_MPIAIJ(Mat A, Vec v)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCheck(A->rmap->N == A->cmap->N, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Supports only square matrix where A->A is diag block");
  PetscCheck(A->rmap->rstart == A->cmap->rstart && A->rmap->rend == A->cmap->rend, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "row partition must equal col partition");
  PetscCall(MatGetDiagonal(a->A, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatScale_MPIAIJ(Mat A, PetscScalar aa)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatScale(a->A, aa));
  PetscCall(MatScale(a->B, aa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_MPIAIJ_Binary(Mat mat, PetscViewer viewer)
{
  Mat_MPIAIJ        *aij    = (Mat_MPIAIJ *)mat->data;
  Mat_SeqAIJ        *A      = (Mat_SeqAIJ *)aij->A->data;
  Mat_SeqAIJ        *B      = (Mat_SeqAIJ *)aij->B->data;
  const PetscInt    *garray = aij->garray;
  const PetscScalar *aa, *ba;
  PetscInt           header[4], M, N, m, rs, cs, cnt, i, ja, jb;
  PetscInt64         nz, hnz;
  PetscInt          *rowlens;
  PetscInt          *colidxs;
  PetscScalar       *matvals;
  PetscMPIInt        rank;

  PetscFunctionBegin;
  PetscCall(PetscViewerSetUp(viewer));

  M  = mat->rmap->N;
  N  = mat->cmap->N;
  m  = mat->rmap->n;
  rs = mat->rmap->rstart;
  cs = mat->cmap->rstart;
  nz = A->nz + B->nz;

  /* write matrix header */
  header[0] = MAT_FILE_CLASSID;
  header[1] = M;
  header[2] = N;
  PetscCallMPI(MPI_Reduce(&nz, &hnz, 1, MPIU_INT64, MPI_SUM, 0, PetscObjectComm((PetscObject)mat)));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mat), &rank));
  if (rank == 0) {
    if (hnz > PETSC_INT_MAX) header[3] = PETSC_INT_MAX;
    else header[3] = (PetscInt)hnz;
  }
  PetscCall(PetscViewerBinaryWrite(viewer, header, 4, PETSC_INT));

  /* fill in and store row lengths  */
  PetscCall(PetscMalloc1(m, &rowlens));
  for (i = 0; i < m; i++) rowlens[i] = A->i[i + 1] - A->i[i] + B->i[i + 1] - B->i[i];
  PetscCall(PetscViewerBinaryWriteAll(viewer, rowlens, m, rs, M, PETSC_INT));
  PetscCall(PetscFree(rowlens));

  /* fill in and store column indices */
  PetscCall(PetscMalloc1(nz, &colidxs));
  for (cnt = 0, i = 0; i < m; i++) {
    for (jb = B->i[i]; jb < B->i[i + 1]; jb++) {
      if (garray[B->j[jb]] > cs) break;
      colidxs[cnt++] = garray[B->j[jb]];
    }
    for (ja = A->i[i]; ja < A->i[i + 1]; ja++) colidxs[cnt++] = A->j[ja] + cs;
    for (; jb < B->i[i + 1]; jb++) colidxs[cnt++] = garray[B->j[jb]];
  }
  PetscCheck(cnt == nz, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Internal PETSc error: cnt = %" PetscInt_FMT " nz = %" PetscInt64_FMT, cnt, nz);
  PetscCall(PetscViewerBinaryWriteAll(viewer, colidxs, nz, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_INT));
  PetscCall(PetscFree(colidxs));

  /* fill in and store nonzero values */
  PetscCall(MatSeqAIJGetArrayRead(aij->A, &aa));
  PetscCall(MatSeqAIJGetArrayRead(aij->B, &ba));
  PetscCall(PetscMalloc1(nz, &matvals));
  for (cnt = 0, i = 0; i < m; i++) {
    for (jb = B->i[i]; jb < B->i[i + 1]; jb++) {
      if (garray[B->j[jb]] > cs) break;
      matvals[cnt++] = ba[jb];
    }
    for (ja = A->i[i]; ja < A->i[i + 1]; ja++) matvals[cnt++] = aa[ja];
    for (; jb < B->i[i + 1]; jb++) matvals[cnt++] = ba[jb];
  }
  PetscCall(MatSeqAIJRestoreArrayRead(aij->A, &aa));
  PetscCall(MatSeqAIJRestoreArrayRead(aij->B, &ba));
  PetscCheck(cnt == nz, PETSC_COMM_SELF, PETSC_ERR_LIB, "Internal PETSc error: cnt = %" PetscInt_FMT " nz = %" PetscInt64_FMT, cnt, nz);
  PetscCall(PetscViewerBinaryWriteAll(viewer, matvals, nz, PETSC_DETERMINE, PETSC_DETERMINE, PETSC_SCALAR));
  PetscCall(PetscFree(matvals));

  /* write block size option to the viewer's .info file */
  PetscCall(MatView_Binary_BlockSizes(mat, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscdraw.h>
static PetscErrorCode MatView_MPIAIJ_ASCIIorDraworSocket(Mat mat, PetscViewer viewer)
{
  Mat_MPIAIJ       *aij  = (Mat_MPIAIJ *)mat->data;
  PetscMPIInt       rank = aij->rank, size = aij->size;
  PetscBool         isdraw, iascii, isbinary;
  PetscViewer       sviewer;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERDRAW, &isdraw));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  if (iascii) {
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_LOAD_BALANCE) {
      PetscInt i, nmax = 0, nmin = PETSC_INT_MAX, navg = 0, *nz, nzlocal = ((Mat_SeqAIJ *)aij->A->data)->nz + ((Mat_SeqAIJ *)aij->B->data)->nz;
      PetscCall(PetscMalloc1(size, &nz));
      PetscCallMPI(MPI_Allgather(&nzlocal, 1, MPIU_INT, nz, 1, MPIU_INT, PetscObjectComm((PetscObject)mat)));
      for (i = 0; i < (PetscInt)size; i++) {
        nmax = PetscMax(nmax, nz[i]);
        nmin = PetscMin(nmin, nz[i]);
        navg += nz[i];
      }
      PetscCall(PetscFree(nz));
      navg = navg / size;
      PetscCall(PetscViewerASCIIPrintf(viewer, "Load Balance - Nonzeros: Min %" PetscInt_FMT "  avg %" PetscInt_FMT "  max %" PetscInt_FMT "\n", nmin, navg, nmax));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      MatInfo   info;
      PetscInt *inodes = NULL;

      PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mat), &rank));
      PetscCall(MatGetInfo(mat, MAT_LOCAL, &info));
      PetscCall(MatInodeGetInodeSizes(aij->A, NULL, &inodes, NULL));
      PetscCall(PetscViewerASCIIPushSynchronized(viewer));
      if (!inodes) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Local rows %" PetscInt_FMT " nz %" PetscInt_FMT " nz alloced %" PetscInt_FMT " mem %g, not using I-node routines\n", rank, mat->rmap->n, (PetscInt)info.nz_used, (PetscInt)info.nz_allocated,
                                                     (double)info.memory));
      } else {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] Local rows %" PetscInt_FMT " nz %" PetscInt_FMT " nz alloced %" PetscInt_FMT " mem %g, using I-node routines\n", rank, mat->rmap->n, (PetscInt)info.nz_used, (PetscInt)info.nz_allocated,
                                                     (double)info.memory));
      }
      PetscCall(MatGetInfo(aij->A, MAT_LOCAL, &info));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] on-diagonal part: nz %" PetscInt_FMT " \n", rank, (PetscInt)info.nz_used));
      PetscCall(MatGetInfo(aij->B, MAT_LOCAL, &info));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "[%d] off-diagonal part: nz %" PetscInt_FMT " \n", rank, (PetscInt)info.nz_used));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPopSynchronized(viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Information on VecScatter used in matrix-vector product: \n"));
      PetscCall(VecScatterView(aij->Mvctx, viewer));
      PetscFunctionReturn(PETSC_SUCCESS);
    } else if (format == PETSC_VIEWER_ASCII_INFO) {
      PetscInt inodecount, inodelimit, *inodes;
      PetscCall(MatInodeGetInodeSizes(aij->A, &inodecount, &inodes, &inodelimit));
      if (inodes) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "using I-node (on process 0) routines: found %" PetscInt_FMT " nodes, limit used is %" PetscInt_FMT "\n", inodecount, inodelimit));
      } else {
        PetscCall(PetscViewerASCIIPrintf(viewer, "not using I-node (on process 0) routines\n"));
      }
      PetscFunctionReturn(PETSC_SUCCESS);
    } else if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  } else if (isbinary) {
    if (size == 1) {
      PetscCall(PetscObjectSetName((PetscObject)aij->A, ((PetscObject)mat)->name));
      PetscCall(MatView(aij->A, viewer));
    } else {
      PetscCall(MatView_MPIAIJ_Binary(mat, viewer));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  } else if (iascii && size == 1) {
    PetscCall(PetscObjectSetName((PetscObject)aij->A, ((PetscObject)mat)->name));
    PetscCall(MatView(aij->A, viewer));
    PetscFunctionReturn(PETSC_SUCCESS);
  } else if (isdraw) {
    PetscDraw draw;
    PetscBool isnull;
    PetscCall(PetscViewerDrawGetDraw(viewer, 0, &draw));
    PetscCall(PetscDrawIsNull(draw, &isnull));
    if (isnull) PetscFunctionReturn(PETSC_SUCCESS);
  }

  { /* assemble the entire matrix onto first processor */
    Mat A = NULL, Av;
    IS  isrow, iscol;

    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)mat), rank == 0 ? mat->rmap->N : 0, 0, 1, &isrow));
    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)mat), rank == 0 ? mat->cmap->N : 0, 0, 1, &iscol));
    PetscCall(MatCreateSubMatrix(mat, isrow, iscol, MAT_INITIAL_MATRIX, &A));
    PetscCall(MatMPIAIJGetSeqAIJ(A, &Av, NULL, NULL));
    /*  The commented code uses MatCreateSubMatrices instead */
    /*
    Mat *AA, A = NULL, Av;
    IS  isrow,iscol;

    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)mat),rank == 0 ? mat->rmap->N : 0,0,1,&isrow));
    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)mat),rank == 0 ? mat->cmap->N : 0,0,1,&iscol));
    PetscCall(MatCreateSubMatrices(mat,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&AA));
    if (rank == 0) {
       PetscCall(PetscObjectReference((PetscObject)AA[0]));
       A    = AA[0];
       Av   = AA[0];
    }
    PetscCall(MatDestroySubMatrices(1,&AA));
*/
    PetscCall(ISDestroy(&iscol));
    PetscCall(ISDestroy(&isrow));
    /*
       Everyone has to call to draw the matrix since the graphics waits are
       synchronized across all processors that share the PetscDraw object
    */
    PetscCall(PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &sviewer));
    if (rank == 0) {
      if (((PetscObject)mat)->name) PetscCall(PetscObjectSetName((PetscObject)Av, ((PetscObject)mat)->name));
      PetscCall(MatView_SeqAIJ(Av, sviewer));
    }
    PetscCall(PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &sviewer));
    PetscCall(MatDestroy(&A));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatView_MPIAIJ(Mat mat, PetscViewer viewer)
{
  PetscBool iascii, isdraw, issocket, isbinary;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERDRAW, &isdraw));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSOCKET, &issocket));
  if (iascii || isdraw || isbinary || issocket) PetscCall(MatView_MPIAIJ_ASCIIorDraworSocket(mat, viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSOR_MPIAIJ(Mat matin, Vec bb, PetscReal omega, MatSORType flag, PetscReal fshift, PetscInt its, PetscInt lits, Vec xx)
{
  Mat_MPIAIJ *mat = (Mat_MPIAIJ *)matin->data;
  Vec         bb1 = NULL;
  PetscBool   hasop;

  PetscFunctionBegin;
  if (flag == SOR_APPLY_UPPER) {
    PetscCall((*mat->A->ops->sor)(mat->A, bb, omega, flag, fshift, lits, 1, xx));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (its > 1 || ~flag & SOR_ZERO_INITIAL_GUESS || flag & SOR_EISENSTAT) PetscCall(VecDuplicate(bb, &bb1));

  if ((flag & SOR_LOCAL_SYMMETRIC_SWEEP) == SOR_LOCAL_SYMMETRIC_SWEEP) {
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      PetscCall((*mat->A->ops->sor)(mat->A, bb, omega, flag, fshift, lits, 1, xx));
      its--;
    }

    while (its--) {
      PetscCall(VecScatterBegin(mat->Mvctx, xx, mat->lvec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(mat->Mvctx, xx, mat->lvec, INSERT_VALUES, SCATTER_FORWARD));

      /* update rhs: bb1 = bb - B*x */
      PetscCall(VecScale(mat->lvec, -1.0));
      PetscCall((*mat->B->ops->multadd)(mat->B, mat->lvec, bb, bb1));

      /* local sweep */
      PetscCall((*mat->A->ops->sor)(mat->A, bb1, omega, SOR_SYMMETRIC_SWEEP, fshift, lits, 1, xx));
    }
  } else if (flag & SOR_LOCAL_FORWARD_SWEEP) {
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      PetscCall((*mat->A->ops->sor)(mat->A, bb, omega, flag, fshift, lits, 1, xx));
      its--;
    }
    while (its--) {
      PetscCall(VecScatterBegin(mat->Mvctx, xx, mat->lvec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(mat->Mvctx, xx, mat->lvec, INSERT_VALUES, SCATTER_FORWARD));

      /* update rhs: bb1 = bb - B*x */
      PetscCall(VecScale(mat->lvec, -1.0));
      PetscCall((*mat->B->ops->multadd)(mat->B, mat->lvec, bb, bb1));

      /* local sweep */
      PetscCall((*mat->A->ops->sor)(mat->A, bb1, omega, SOR_FORWARD_SWEEP, fshift, lits, 1, xx));
    }
  } else if (flag & SOR_LOCAL_BACKWARD_SWEEP) {
    if (flag & SOR_ZERO_INITIAL_GUESS) {
      PetscCall((*mat->A->ops->sor)(mat->A, bb, omega, flag, fshift, lits, 1, xx));
      its--;
    }
    while (its--) {
      PetscCall(VecScatterBegin(mat->Mvctx, xx, mat->lvec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(mat->Mvctx, xx, mat->lvec, INSERT_VALUES, SCATTER_FORWARD));

      /* update rhs: bb1 = bb - B*x */
      PetscCall(VecScale(mat->lvec, -1.0));
      PetscCall((*mat->B->ops->multadd)(mat->B, mat->lvec, bb, bb1));

      /* local sweep */
      PetscCall((*mat->A->ops->sor)(mat->A, bb1, omega, SOR_BACKWARD_SWEEP, fshift, lits, 1, xx));
    }
  } else if (flag & SOR_EISENSTAT) {
    Vec xx1;

    PetscCall(VecDuplicate(bb, &xx1));
    PetscCall((*mat->A->ops->sor)(mat->A, bb, omega, (MatSORType)(SOR_ZERO_INITIAL_GUESS | SOR_LOCAL_BACKWARD_SWEEP), fshift, lits, 1, xx));

    PetscCall(VecScatterBegin(mat->Mvctx, xx, mat->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(mat->Mvctx, xx, mat->lvec, INSERT_VALUES, SCATTER_FORWARD));
    if (!mat->diag) {
      PetscCall(MatCreateVecs(matin, &mat->diag, NULL));
      PetscCall(MatGetDiagonal(matin, mat->diag));
    }
    PetscCall(MatHasOperation(matin, MATOP_MULT_DIAGONAL_BLOCK, &hasop));
    if (hasop) {
      PetscCall(MatMultDiagonalBlock(matin, xx, bb1));
    } else {
      PetscCall(VecPointwiseMult(bb1, mat->diag, xx));
    }
    PetscCall(VecAYPX(bb1, (omega - 2.0) / omega, bb));

    PetscCall(MatMultAdd(mat->B, mat->lvec, bb1, bb1));

    /* local sweep */
    PetscCall((*mat->A->ops->sor)(mat->A, bb1, omega, (MatSORType)(SOR_ZERO_INITIAL_GUESS | SOR_LOCAL_FORWARD_SWEEP), fshift, lits, 1, xx1));
    PetscCall(VecAXPY(xx, 1.0, xx1));
    PetscCall(VecDestroy(&xx1));
  } else SETERRQ(PetscObjectComm((PetscObject)matin), PETSC_ERR_SUP, "Parallel SOR not supported");

  PetscCall(VecDestroy(&bb1));

  matin->factorerrortype = mat->A->factorerrortype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatPermute_MPIAIJ(Mat A, IS rowp, IS colp, Mat *B)
{
  Mat             aA, aB, Aperm;
  const PetscInt *rwant, *cwant, *gcols, *ai, *bi, *aj, *bj;
  PetscScalar    *aa, *ba;
  PetscInt        i, j, m, n, ng, anz, bnz, *dnnz, *onnz, *tdnnz, *tonnz, *rdest, *cdest, *work, *gcdest;
  PetscSF         rowsf, sf;
  IS              parcolp = NULL;
  PetscBool       done;

  PetscFunctionBegin;
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(ISGetIndices(rowp, &rwant));
  PetscCall(ISGetIndices(colp, &cwant));
  PetscCall(PetscMalloc3(PetscMax(m, n), &work, m, &rdest, n, &cdest));

  /* Invert row permutation to find out where my rows should go */
  PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)A), &rowsf));
  PetscCall(PetscSFSetGraphLayout(rowsf, A->rmap, A->rmap->n, NULL, PETSC_OWN_POINTER, rwant));
  PetscCall(PetscSFSetFromOptions(rowsf));
  for (i = 0; i < m; i++) work[i] = A->rmap->rstart + i;
  PetscCall(PetscSFReduceBegin(rowsf, MPIU_INT, work, rdest, MPI_REPLACE));
  PetscCall(PetscSFReduceEnd(rowsf, MPIU_INT, work, rdest, MPI_REPLACE));

  /* Invert column permutation to find out where my columns should go */
  PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)A), &sf));
  PetscCall(PetscSFSetGraphLayout(sf, A->cmap, A->cmap->n, NULL, PETSC_OWN_POINTER, cwant));
  PetscCall(PetscSFSetFromOptions(sf));
  for (i = 0; i < n; i++) work[i] = A->cmap->rstart + i;
  PetscCall(PetscSFReduceBegin(sf, MPIU_INT, work, cdest, MPI_REPLACE));
  PetscCall(PetscSFReduceEnd(sf, MPIU_INT, work, cdest, MPI_REPLACE));
  PetscCall(PetscSFDestroy(&sf));

  PetscCall(ISRestoreIndices(rowp, &rwant));
  PetscCall(ISRestoreIndices(colp, &cwant));
  PetscCall(MatMPIAIJGetSeqAIJ(A, &aA, &aB, &gcols));

  /* Find out where my gcols should go */
  PetscCall(MatGetSize(aB, NULL, &ng));
  PetscCall(PetscMalloc1(ng, &gcdest));
  PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)A), &sf));
  PetscCall(PetscSFSetGraphLayout(sf, A->cmap, ng, NULL, PETSC_OWN_POINTER, gcols));
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, cdest, gcdest, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, cdest, gcdest, MPI_REPLACE));
  PetscCall(PetscSFDestroy(&sf));

  PetscCall(PetscCalloc4(m, &dnnz, m, &onnz, m, &tdnnz, m, &tonnz));
  PetscCall(MatGetRowIJ(aA, 0, PETSC_FALSE, PETSC_FALSE, &anz, &ai, &aj, &done));
  PetscCall(MatGetRowIJ(aB, 0, PETSC_FALSE, PETSC_FALSE, &bnz, &bi, &bj, &done));
  for (i = 0; i < m; i++) {
    PetscInt    row = rdest[i];
    PetscMPIInt rowner;
    PetscCall(PetscLayoutFindOwner(A->rmap, row, &rowner));
    for (j = ai[i]; j < ai[i + 1]; j++) {
      PetscInt    col = cdest[aj[j]];
      PetscMPIInt cowner;
      PetscCall(PetscLayoutFindOwner(A->cmap, col, &cowner)); /* Could build an index for the columns to eliminate this search */
      if (rowner == cowner) dnnz[i]++;
      else onnz[i]++;
    }
    for (j = bi[i]; j < bi[i + 1]; j++) {
      PetscInt    col = gcdest[bj[j]];
      PetscMPIInt cowner;
      PetscCall(PetscLayoutFindOwner(A->cmap, col, &cowner));
      if (rowner == cowner) dnnz[i]++;
      else onnz[i]++;
    }
  }
  PetscCall(PetscSFBcastBegin(rowsf, MPIU_INT, dnnz, tdnnz, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(rowsf, MPIU_INT, dnnz, tdnnz, MPI_REPLACE));
  PetscCall(PetscSFBcastBegin(rowsf, MPIU_INT, onnz, tonnz, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(rowsf, MPIU_INT, onnz, tonnz, MPI_REPLACE));
  PetscCall(PetscSFDestroy(&rowsf));

  PetscCall(MatCreateAIJ(PetscObjectComm((PetscObject)A), A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N, 0, tdnnz, 0, tonnz, &Aperm));
  PetscCall(MatSeqAIJGetArray(aA, &aa));
  PetscCall(MatSeqAIJGetArray(aB, &ba));
  for (i = 0; i < m; i++) {
    PetscInt *acols = dnnz, *bcols = onnz; /* Repurpose now-unneeded arrays */
    PetscInt  j0, rowlen;
    rowlen = ai[i + 1] - ai[i];
    for (j0 = j = 0; j < rowlen; j0 = j) { /* rowlen could be larger than number of rows m, so sum in batches */
      for (; j < PetscMin(rowlen, j0 + m); j++) acols[j - j0] = cdest[aj[ai[i] + j]];
      PetscCall(MatSetValues(Aperm, 1, &rdest[i], j - j0, acols, aa + ai[i] + j0, INSERT_VALUES));
    }
    rowlen = bi[i + 1] - bi[i];
    for (j0 = j = 0; j < rowlen; j0 = j) {
      for (; j < PetscMin(rowlen, j0 + m); j++) bcols[j - j0] = gcdest[bj[bi[i] + j]];
      PetscCall(MatSetValues(Aperm, 1, &rdest[i], j - j0, bcols, ba + bi[i] + j0, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(Aperm, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Aperm, MAT_FINAL_ASSEMBLY));
  PetscCall(MatRestoreRowIJ(aA, 0, PETSC_FALSE, PETSC_FALSE, &anz, &ai, &aj, &done));
  PetscCall(MatRestoreRowIJ(aB, 0, PETSC_FALSE, PETSC_FALSE, &bnz, &bi, &bj, &done));
  PetscCall(MatSeqAIJRestoreArray(aA, &aa));
  PetscCall(MatSeqAIJRestoreArray(aB, &ba));
  PetscCall(PetscFree4(dnnz, onnz, tdnnz, tonnz));
  PetscCall(PetscFree3(work, rdest, cdest));
  PetscCall(PetscFree(gcdest));
  if (parcolp) PetscCall(ISDestroy(&colp));
  *B = Aperm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetGhosts_MPIAIJ(Mat mat, PetscInt *nghosts, const PetscInt *ghosts[])
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *)mat->data;

  PetscFunctionBegin;
  PetscCall(MatGetSize(aij->B, NULL, nghosts));
  if (ghosts) *ghosts = aij->garray;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetInfo_MPIAIJ(Mat matin, MatInfoType flag, MatInfo *info)
{
  Mat_MPIAIJ    *mat = (Mat_MPIAIJ *)matin->data;
  Mat            A = mat->A, B = mat->B;
  PetscLogDouble isend[5], irecv[5];

  PetscFunctionBegin;
  info->block_size = 1.0;
  PetscCall(MatGetInfo(A, MAT_LOCAL, info));

  isend[0] = info->nz_used;
  isend[1] = info->nz_allocated;
  isend[2] = info->nz_unneeded;
  isend[3] = info->memory;
  isend[4] = info->mallocs;

  PetscCall(MatGetInfo(B, MAT_LOCAL, info));

  isend[0] += info->nz_used;
  isend[1] += info->nz_allocated;
  isend[2] += info->nz_unneeded;
  isend[3] += info->memory;
  isend[4] += info->mallocs;
  if (flag == MAT_LOCAL) {
    info->nz_used      = isend[0];
    info->nz_allocated = isend[1];
    info->nz_unneeded  = isend[2];
    info->memory       = isend[3];
    info->mallocs      = isend[4];
  } else if (flag == MAT_GLOBAL_MAX) {
    PetscCallMPI(MPIU_Allreduce(isend, irecv, 5, MPIU_PETSCLOGDOUBLE, MPI_MAX, PetscObjectComm((PetscObject)matin)));

    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    PetscCallMPI(MPIU_Allreduce(isend, irecv, 5, MPIU_PETSCLOGDOUBLE, MPI_SUM, PetscObjectComm((PetscObject)matin)));

    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  }
  info->fill_ratio_given  = 0; /* no parallel LU/ILU/Cholesky */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSetOption_MPIAIJ(Mat A, MatOption op, PetscBool flg)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  switch (op) {
  case MAT_NEW_NONZERO_LOCATIONS:
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
  case MAT_UNUSED_NONZERO_LOCATION_ERR:
  case MAT_KEEP_NONZERO_PATTERN:
  case MAT_NEW_NONZERO_LOCATION_ERR:
  case MAT_USE_INODES:
  case MAT_IGNORE_ZERO_ENTRIES:
  case MAT_FORM_EXPLICIT_TRANSPOSE:
    MatCheckPreallocated(A, 1);
    PetscCall(MatSetOption(a->A, op, flg));
    PetscCall(MatSetOption(a->B, op, flg));
    break;
  case MAT_ROW_ORIENTED:
    MatCheckPreallocated(A, 1);
    a->roworiented = flg;

    PetscCall(MatSetOption(a->A, op, flg));
    PetscCall(MatSetOption(a->B, op, flg));
    break;
  case MAT_FORCE_DIAGONAL_ENTRIES:
  case MAT_SORTED_FULL:
    PetscCall(PetscInfo(A, "Option %s ignored\n", MatOptions[op]));
    break;
  case MAT_IGNORE_OFF_PROC_ENTRIES:
    a->donotstash = flg;
    break;
  /* Symmetry flags are handled directly by MatSetOption() and they don't affect preallocation */
  case MAT_SPD:
  case MAT_SYMMETRIC:
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_HERMITIAN:
  case MAT_SYMMETRY_ETERNAL:
  case MAT_STRUCTURAL_SYMMETRY_ETERNAL:
  case MAT_SPD_ETERNAL:
    /* if the diagonal matrix is square it inherits some of the properties above */
    break;
  case MAT_SUBMAT_SINGLEIS:
    A->submat_singleis = flg;
    break;
  case MAT_STRUCTURE_ONLY:
    /* The option is handled directly by MatSetOption() */
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "unknown option %d", op);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatGetRow_MPIAIJ(Mat matin, PetscInt row, PetscInt *nz, PetscInt **idx, PetscScalar **v)
{
  Mat_MPIAIJ  *mat = (Mat_MPIAIJ *)matin->data;
  PetscScalar *vworkA, *vworkB, **pvA, **pvB, *v_p;
  PetscInt     i, *cworkA, *cworkB, **pcA, **pcB, cstart = matin->cmap->rstart;
  PetscInt     nztot, nzA, nzB, lrow, rstart = matin->rmap->rstart, rend = matin->rmap->rend;
  PetscInt    *cmap, *idx_p;

  PetscFunctionBegin;
  PetscCheck(!mat->getrowactive, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Already active");
  mat->getrowactive = PETSC_TRUE;

  if (!mat->rowvalues && (idx || v)) {
    /*
        allocate enough space to hold information from the longest row.
    */
    Mat_SeqAIJ *Aa = (Mat_SeqAIJ *)mat->A->data, *Ba = (Mat_SeqAIJ *)mat->B->data;
    PetscInt    max = 1, tmp;
    for (i = 0; i < matin->rmap->n; i++) {
      tmp = Aa->i[i + 1] - Aa->i[i] + Ba->i[i + 1] - Ba->i[i];
      if (max < tmp) max = tmp;
    }
    PetscCall(PetscMalloc2(max, &mat->rowvalues, max, &mat->rowindices));
  }

  PetscCheck(row >= rstart && row < rend, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Only local rows");
  lrow = row - rstart;

  pvA = &vworkA;
  pcA = &cworkA;
  pvB = &vworkB;
  pcB = &cworkB;
  if (!v) {
    pvA = NULL;
    pvB = NULL;
  }
  if (!idx) {
    pcA = NULL;
    if (!v) pcB = NULL;
  }
  PetscCall((*mat->A->ops->getrow)(mat->A, lrow, &nzA, pcA, pvA));
  PetscCall((*mat->B->ops->getrow)(mat->B, lrow, &nzB, pcB, pvB));
  nztot = nzA + nzB;

  cmap = mat->garray;
  if (v || idx) {
    if (nztot) {
      /* Sort by increasing column numbers, assuming A and B already sorted */
      PetscInt imark = -1;
      if (v) {
        *v = v_p = mat->rowvalues;
        for (i = 0; i < nzB; i++) {
          if (cmap[cworkB[i]] < cstart) v_p[i] = vworkB[i];
          else break;
        }
        imark = i;
        for (i = 0; i < nzA; i++) v_p[imark + i] = vworkA[i];
        for (i = imark; i < nzB; i++) v_p[nzA + i] = vworkB[i];
      }
      if (idx) {
        *idx = idx_p = mat->rowindices;
        if (imark > -1) {
          for (i = 0; i < imark; i++) idx_p[i] = cmap[cworkB[i]];
        } else {
          for (i = 0; i < nzB; i++) {
            if (cmap[cworkB[i]] < cstart) idx_p[i] = cmap[cworkB[i]];
            else break;
          }
          imark = i;
        }
        for (i = 0; i < nzA; i++) idx_p[imark + i] = cstart + cworkA[i];
        for (i = imark; i < nzB; i++) idx_p[nzA + i] = cmap[cworkB[i]];
      }
    } else {
      if (idx) *idx = NULL;
      if (v) *v = NULL;
    }
  }
  *nz = nztot;
  PetscCall((*mat->A->ops->restorerow)(mat->A, lrow, &nzA, pcA, pvA));
  PetscCall((*mat->B->ops->restorerow)(mat->B, lrow, &nzB, pcB, pvB));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatRestoreRow_MPIAIJ(Mat mat, PetscInt row, PetscInt *nz, PetscInt **idx, PetscScalar **v)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *)mat->data;

  PetscFunctionBegin;
  PetscCheck(aij->getrowactive, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "MatGetRow() must be called first");
  aij->getrowactive = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatNorm_MPIAIJ(Mat mat, NormType type, PetscReal *norm)
{
  Mat_MPIAIJ      *aij  = (Mat_MPIAIJ *)mat->data;
  Mat_SeqAIJ      *amat = (Mat_SeqAIJ *)aij->A->data, *bmat = (Mat_SeqAIJ *)aij->B->data;
  PetscInt         i, j, cstart = mat->cmap->rstart;
  PetscReal        sum = 0.0;
  const MatScalar *v, *amata, *bmata;
  PetscMPIInt      iN;

  PetscFunctionBegin;
  if (aij->size == 1) {
    PetscCall(MatNorm(aij->A, type, norm));
  } else {
    PetscCall(MatSeqAIJGetArrayRead(aij->A, &amata));
    PetscCall(MatSeqAIJGetArrayRead(aij->B, &bmata));
    if (type == NORM_FROBENIUS) {
      v = amata;
      for (i = 0; i < amat->nz; i++) {
        sum += PetscRealPart(PetscConj(*v) * (*v));
        v++;
      }
      v = bmata;
      for (i = 0; i < bmat->nz; i++) {
        sum += PetscRealPart(PetscConj(*v) * (*v));
        v++;
      }
      PetscCallMPI(MPIU_Allreduce(&sum, norm, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)mat)));
      *norm = PetscSqrtReal(*norm);
      PetscCall(PetscLogFlops(2.0 * amat->nz + 2.0 * bmat->nz));
    } else if (type == NORM_1) { /* max column norm */
      PetscReal *tmp, *tmp2;
      PetscInt  *jj, *garray = aij->garray;
      PetscCall(PetscCalloc1(mat->cmap->N + 1, &tmp));
      PetscCall(PetscMalloc1(mat->cmap->N + 1, &tmp2));
      *norm = 0.0;
      v     = amata;
      jj    = amat->j;
      for (j = 0; j < amat->nz; j++) {
        tmp[cstart + *jj++] += PetscAbsScalar(*v);
        v++;
      }
      v  = bmata;
      jj = bmat->j;
      for (j = 0; j < bmat->nz; j++) {
        tmp[garray[*jj++]] += PetscAbsScalar(*v);
        v++;
      }
      PetscCall(PetscMPIIntCast(mat->cmap->N, &iN));
      PetscCallMPI(MPIU_Allreduce(tmp, tmp2, iN, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)mat)));
      for (j = 0; j < mat->cmap->N; j++) {
        if (tmp2[j] > *norm) *norm = tmp2[j];
      }
      PetscCall(PetscFree(tmp));
      PetscCall(PetscFree(tmp2));
      PetscCall(PetscLogFlops(PetscMax(amat->nz + bmat->nz - 1, 0)));
    } else if (type == NORM_INFINITY) { /* max row norm */
      PetscReal ntemp = 0.0;
      for (j = 0; j < aij->A->rmap->n; j++) {
        v   = PetscSafePointerPlusOffset(amata, amat->i[j]);
        sum = 0.0;
        for (i = 0; i < amat->i[j + 1] - amat->i[j]; i++) {
          sum += PetscAbsScalar(*v);
          v++;
        }
        v = PetscSafePointerPlusOffset(bmata, bmat->i[j]);
        for (i = 0; i < bmat->i[j + 1] - bmat->i[j]; i++) {
          sum += PetscAbsScalar(*v);
          v++;
        }
        if (sum > ntemp) ntemp = sum;
      }
      PetscCallMPI(MPIU_Allreduce(&ntemp, norm, 1, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)mat)));
      PetscCall(PetscLogFlops(PetscMax(amat->nz + bmat->nz - 1, 0)));
    } else SETERRQ(PetscObjectComm((PetscObject)mat), PETSC_ERR_SUP, "No support for two norm");
    PetscCall(MatSeqAIJRestoreArrayRead(aij->A, &amata));
    PetscCall(MatSeqAIJRestoreArrayRead(aij->B, &bmata));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatTranspose_MPIAIJ(Mat A, MatReuse reuse, Mat *matout)
{
  Mat_MPIAIJ      *a    = (Mat_MPIAIJ *)A->data, *b;
  Mat_SeqAIJ      *Aloc = (Mat_SeqAIJ *)a->A->data, *Bloc = (Mat_SeqAIJ *)a->B->data, *sub_B_diag;
  PetscInt         M = A->rmap->N, N = A->cmap->N, ma, na, mb, nb, row, *cols, *cols_tmp, *B_diag_ilen, i, ncol, A_diag_ncol;
  const PetscInt  *ai, *aj, *bi, *bj, *B_diag_i;
  Mat              B, A_diag, *B_diag;
  const MatScalar *pbv, *bv;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) PetscCall(MatTransposeCheckNonzeroState_Private(A, *matout));
  ma = A->rmap->n;
  na = A->cmap->n;
  mb = a->B->rmap->n;
  nb = a->B->cmap->n;
  ai = Aloc->i;
  aj = Aloc->j;
  bi = Bloc->i;
  bj = Bloc->j;
  if (reuse == MAT_INITIAL_MATRIX || *matout == A) {
    PetscInt            *d_nnz, *g_nnz, *o_nnz;
    PetscSFNode         *oloc;
    PETSC_UNUSED PetscSF sf;

    PetscCall(PetscMalloc4(na, &d_nnz, na, &o_nnz, nb, &g_nnz, nb, &oloc));
    /* compute d_nnz for preallocation */
    PetscCall(PetscArrayzero(d_nnz, na));
    for (i = 0; i < ai[ma]; i++) d_nnz[aj[i]]++;
    /* compute local off-diagonal contributions */
    PetscCall(PetscArrayzero(g_nnz, nb));
    for (i = 0; i < bi[ma]; i++) g_nnz[bj[i]]++;
    /* map those to global */
    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)A), &sf));
    PetscCall(PetscSFSetGraphLayout(sf, A->cmap, nb, NULL, PETSC_USE_POINTER, a->garray));
    PetscCall(PetscSFSetFromOptions(sf));
    PetscCall(PetscArrayzero(o_nnz, na));
    PetscCall(PetscSFReduceBegin(sf, MPIU_INT, g_nnz, o_nnz, MPI_SUM));
    PetscCall(PetscSFReduceEnd(sf, MPIU_INT, g_nnz, o_nnz, MPI_SUM));
    PetscCall(PetscSFDestroy(&sf));

    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
    PetscCall(MatSetSizes(B, A->cmap->n, A->rmap->n, N, M));
    PetscCall(MatSetBlockSizes(B, PetscAbs(A->cmap->bs), PetscAbs(A->rmap->bs)));
    PetscCall(MatSetType(B, ((PetscObject)A)->type_name));
    PetscCall(MatMPIAIJSetPreallocation(B, 0, d_nnz, 0, o_nnz));
    PetscCall(PetscFree4(d_nnz, o_nnz, g_nnz, oloc));
  } else {
    B = *matout;
    PetscCall(MatSetOption(B, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  }

  b           = (Mat_MPIAIJ *)B->data;
  A_diag      = a->A;
  B_diag      = &b->A;
  sub_B_diag  = (Mat_SeqAIJ *)(*B_diag)->data;
  A_diag_ncol = A_diag->cmap->N;
  B_diag_ilen = sub_B_diag->ilen;
  B_diag_i    = sub_B_diag->i;

  /* Set ilen for diagonal of B */
  for (i = 0; i < A_diag_ncol; i++) B_diag_ilen[i] = B_diag_i[i + 1] - B_diag_i[i];

  /* Transpose the diagonal part of the matrix. In contrast to the off-diagonal part, this can be done
  very quickly (=without using MatSetValues), because all writes are local. */
  PetscCall(MatTransposeSetPrecursor(A_diag, *B_diag));
  PetscCall(MatTranspose(A_diag, MAT_REUSE_MATRIX, B_diag));

  /* copy over the B part */
  PetscCall(PetscMalloc1(bi[mb], &cols));
  PetscCall(MatSeqAIJGetArrayRead(a->B, &bv));
  pbv = bv;
  row = A->rmap->rstart;
  for (i = 0; i < bi[mb]; i++) cols[i] = a->garray[bj[i]];
  cols_tmp = cols;
  for (i = 0; i < mb; i++) {
    ncol = bi[i + 1] - bi[i];
    PetscCall(MatSetValues(B, ncol, cols_tmp, 1, &row, pbv, INSERT_VALUES));
    row++;
    if (pbv) pbv += ncol;
    if (cols_tmp) cols_tmp += ncol;
  }
  PetscCall(PetscFree(cols));
  PetscCall(MatSeqAIJRestoreArrayRead(a->B, &bv));

  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_REUSE_MATRIX) {
    *matout = B;
  } else {
    PetscCall(MatHeaderMerge(A, &B));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDiagonalScale_MPIAIJ(Mat mat, Vec ll, Vec rr)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *)mat->data;
  Mat         a = aij->A, b = aij->B;
  PetscInt    s1, s2, s3;

  PetscFunctionBegin;
  PetscCall(MatGetLocalSize(mat, &s2, &s3));
  if (rr) {
    PetscCall(VecGetLocalSize(rr, &s1));
    PetscCheck(s1 == s3, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "right vector non-conforming local size");
    /* Overlap communication with computation. */
    PetscCall(VecScatterBegin(aij->Mvctx, rr, aij->lvec, INSERT_VALUES, SCATTER_FORWARD));
  }
  if (ll) {
    PetscCall(VecGetLocalSize(ll, &s1));
    PetscCheck(s1 == s2, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "left vector non-conforming local size");
    PetscUseTypeMethod(b, diagonalscale, ll, NULL);
  }
  /* scale  the diagonal block */
  PetscUseTypeMethod(a, diagonalscale, ll, rr);

  if (rr) {
    /* Do a scatter end and then right scale the off-diagonal block */
    PetscCall(VecScatterEnd(aij->Mvctx, rr, aij->lvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscUseTypeMethod(b, diagonalscale, NULL, aij->lvec);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetUnfactored_MPIAIJ(Mat A)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatSetUnfactored(a->A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatEqual_MPIAIJ(Mat A, Mat B, PetscBool *flag)
{
  Mat_MPIAIJ *matB = (Mat_MPIAIJ *)B->data, *matA = (Mat_MPIAIJ *)A->data;
  Mat         a, b, c, d;
  PetscBool   flg;

  PetscFunctionBegin;
  a = matA->A;
  b = matA->B;
  c = matB->A;
  d = matB->B;

  PetscCall(MatEqual(a, c, &flg));
  if (flg) PetscCall(MatEqual(b, d, &flg));
  PetscCallMPI(MPIU_Allreduce(&flg, flag, 1, MPIU_BOOL, MPI_LAND, PetscObjectComm((PetscObject)A)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCopy_MPIAIJ(Mat A, Mat B, MatStructure str)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;
  Mat_MPIAIJ *b = (Mat_MPIAIJ *)B->data;

  PetscFunctionBegin;
  /* If the two matrices don't have the same copy implementation, they aren't compatible for fast copy. */
  if ((str != SAME_NONZERO_PATTERN) || (A->ops->copy != B->ops->copy)) {
    /* because of the column compression in the off-processor part of the matrix a->B,
       the number of columns in a->B and b->B may be different, hence we cannot call
       the MatCopy() directly on the two parts. If need be, we can provide a more
       efficient copy than the MatCopy_Basic() by first uncompressing the a->B matrices
       then copying the submatrices */
    PetscCall(MatCopy_Basic(A, B, str));
  } else {
    PetscCall(MatCopy(a->A, b->A, str));
    PetscCall(MatCopy(a->B, b->B, str));
  }
  PetscCall(PetscObjectStateIncrease((PetscObject)B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Computes the number of nonzeros per row needed for preallocation when X and Y
   have different nonzero structure.
*/
PetscErrorCode MatAXPYGetPreallocation_MPIX_private(PetscInt m, const PetscInt *xi, const PetscInt *xj, const PetscInt *xltog, const PetscInt *yi, const PetscInt *yj, const PetscInt *yltog, PetscInt *nnz)
{
  PetscInt i, j, k, nzx, nzy;

  PetscFunctionBegin;
  /* Set the number of nonzeros in the new matrix */
  for (i = 0; i < m; i++) {
    const PetscInt *xjj = PetscSafePointerPlusOffset(xj, xi[i]), *yjj = PetscSafePointerPlusOffset(yj, yi[i]);
    nzx    = xi[i + 1] - xi[i];
    nzy    = yi[i + 1] - yi[i];
    nnz[i] = 0;
    for (j = 0, k = 0; j < nzx; j++) {                                /* Point in X */
      for (; k < nzy && yltog[yjj[k]] < xltog[xjj[j]]; k++) nnz[i]++; /* Catch up to X */
      if (k < nzy && yltog[yjj[k]] == xltog[xjj[j]]) k++;             /* Skip duplicate */
      nnz[i]++;
    }
    for (; k < nzy; k++) nnz[i]++;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* This is the same as MatAXPYGetPreallocation_SeqAIJ, except that the local-to-global map is provided */
static PetscErrorCode MatAXPYGetPreallocation_MPIAIJ(Mat Y, const PetscInt *yltog, Mat X, const PetscInt *xltog, PetscInt *nnz)
{
  PetscInt    m = Y->rmap->N;
  Mat_SeqAIJ *x = (Mat_SeqAIJ *)X->data;
  Mat_SeqAIJ *y = (Mat_SeqAIJ *)Y->data;

  PetscFunctionBegin;
  PetscCall(MatAXPYGetPreallocation_MPIX_private(m, x->i, x->j, xltog, y->i, y->j, yltog, nnz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAXPY_MPIAIJ(Mat Y, PetscScalar a, Mat X, MatStructure str)
{
  Mat_MPIAIJ *xx = (Mat_MPIAIJ *)X->data, *yy = (Mat_MPIAIJ *)Y->data;

  PetscFunctionBegin;
  if (str == SAME_NONZERO_PATTERN) {
    PetscCall(MatAXPY(yy->A, a, xx->A, str));
    PetscCall(MatAXPY(yy->B, a, xx->B, str));
  } else if (str == SUBSET_NONZERO_PATTERN) { /* nonzeros of X is a subset of Y's */
    PetscCall(MatAXPY_Basic(Y, a, X, str));
  } else {
    Mat       B;
    PetscInt *nnz_d, *nnz_o;

    PetscCall(PetscMalloc1(yy->A->rmap->N, &nnz_d));
    PetscCall(PetscMalloc1(yy->B->rmap->N, &nnz_o));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)Y), &B));
    PetscCall(PetscObjectSetName((PetscObject)B, ((PetscObject)Y)->name));
    PetscCall(MatSetLayouts(B, Y->rmap, Y->cmap));
    PetscCall(MatSetType(B, ((PetscObject)Y)->type_name));
    PetscCall(MatAXPYGetPreallocation_SeqAIJ(yy->A, xx->A, nnz_d));
    PetscCall(MatAXPYGetPreallocation_MPIAIJ(yy->B, yy->garray, xx->B, xx->garray, nnz_o));
    PetscCall(MatMPIAIJSetPreallocation(B, 0, nnz_d, 0, nnz_o));
    PetscCall(MatAXPY_BasicWithPreallocation(B, Y, a, X, str));
    PetscCall(MatHeaderMerge(Y, &B));
    PetscCall(PetscFree(nnz_d));
    PetscCall(PetscFree(nnz_o));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConjugate_SeqAIJ(Mat);

static PetscErrorCode MatConjugate_MPIAIJ(Mat mat)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_COMPLEX)) {
    Mat_MPIAIJ *aij = (Mat_MPIAIJ *)mat->data;

    PetscCall(MatConjugate_SeqAIJ(aij->A));
    PetscCall(MatConjugate_SeqAIJ(aij->B));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatRealPart_MPIAIJ(Mat A)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatRealPart(a->A));
  PetscCall(MatRealPart(a->B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatImaginaryPart_MPIAIJ(Mat A)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatImaginaryPart(a->A));
  PetscCall(MatImaginaryPart(a->B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetRowMaxAbs_MPIAIJ(Mat A, Vec v, PetscInt idx[])
{
  Mat_MPIAIJ        *a = (Mat_MPIAIJ *)A->data;
  PetscInt           i, *idxb = NULL, m = A->rmap->n;
  PetscScalar       *va, *vv;
  Vec                vB, vA;
  const PetscScalar *vb;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(a->A, NULL, &vA));
  PetscCall(MatGetRowMaxAbs(a->A, vA, idx));

  PetscCall(VecGetArrayWrite(vA, &va));
  if (idx) {
    for (i = 0; i < m; i++) {
      if (PetscAbsScalar(va[i])) idx[i] += A->cmap->rstart;
    }
  }

  PetscCall(MatCreateVecs(a->B, NULL, &vB));
  PetscCall(PetscMalloc1(m, &idxb));
  PetscCall(MatGetRowMaxAbs(a->B, vB, idxb));

  PetscCall(VecGetArrayWrite(v, &vv));
  PetscCall(VecGetArrayRead(vB, &vb));
  for (i = 0; i < m; i++) {
    if (PetscAbsScalar(va[i]) < PetscAbsScalar(vb[i])) {
      vv[i] = vb[i];
      if (idx) idx[i] = a->garray[idxb[i]];
    } else {
      vv[i] = va[i];
      if (idx && PetscAbsScalar(va[i]) == PetscAbsScalar(vb[i]) && idxb[i] != -1 && idx[i] > a->garray[idxb[i]]) idx[i] = a->garray[idxb[i]];
    }
  }
  PetscCall(VecRestoreArrayWrite(vA, &vv));
  PetscCall(VecRestoreArrayWrite(vA, &va));
  PetscCall(VecRestoreArrayRead(vB, &vb));
  PetscCall(PetscFree(idxb));
  PetscCall(VecDestroy(&vA));
  PetscCall(VecDestroy(&vB));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetRowSumAbs_MPIAIJ(Mat A, Vec v)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;
  Vec         vB, vA;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(a->A, NULL, &vA));
  PetscCall(MatGetRowSumAbs(a->A, vA));
  PetscCall(MatCreateVecs(a->B, NULL, &vB));
  PetscCall(MatGetRowSumAbs(a->B, vB));
  PetscCall(VecAXPY(vA, 1.0, vB));
  PetscCall(VecDestroy(&vB));
  PetscCall(VecCopy(vA, v));
  PetscCall(VecDestroy(&vA));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetRowMinAbs_MPIAIJ(Mat A, Vec v, PetscInt idx[])
{
  Mat_MPIAIJ        *mat = (Mat_MPIAIJ *)A->data;
  PetscInt           m = A->rmap->n, n = A->cmap->n;
  PetscInt           cstart = A->cmap->rstart, cend = A->cmap->rend;
  PetscInt          *cmap = mat->garray;
  PetscInt          *diagIdx, *offdiagIdx;
  Vec                diagV, offdiagV;
  PetscScalar       *a, *diagA, *offdiagA;
  const PetscScalar *ba, *bav;
  PetscInt           r, j, col, ncols, *bi, *bj;
  Mat                B = mat->B;
  Mat_SeqAIJ        *b = (Mat_SeqAIJ *)B->data;

  PetscFunctionBegin;
  /* When a process holds entire A and other processes have no entry */
  if (A->cmap->N == n) {
    PetscCall(VecGetArrayWrite(v, &diagA));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, m, diagA, &diagV));
    PetscCall(MatGetRowMinAbs(mat->A, diagV, idx));
    PetscCall(VecDestroy(&diagV));
    PetscCall(VecRestoreArrayWrite(v, &diagA));
    PetscFunctionReturn(PETSC_SUCCESS);
  } else if (n == 0) {
    if (m) {
      PetscCall(VecGetArrayWrite(v, &a));
      for (r = 0; r < m; r++) {
        a[r] = 0.0;
        if (idx) idx[r] = -1;
      }
      PetscCall(VecRestoreArrayWrite(v, &a));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscMalloc2(m, &diagIdx, m, &offdiagIdx));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, m, &diagV));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, m, &offdiagV));
  PetscCall(MatGetRowMinAbs(mat->A, diagV, diagIdx));

  /* Get offdiagIdx[] for implicit 0.0 */
  PetscCall(MatSeqAIJGetArrayRead(B, &bav));
  ba = bav;
  bi = b->i;
  bj = b->j;
  PetscCall(VecGetArrayWrite(offdiagV, &offdiagA));
  for (r = 0; r < m; r++) {
    ncols = bi[r + 1] - bi[r];
    if (ncols == A->cmap->N - n) { /* Brow is dense */
      offdiagA[r]   = *ba;
      offdiagIdx[r] = cmap[0];
    } else { /* Brow is sparse so already KNOW maximum is 0.0 or higher */
      offdiagA[r] = 0.0;

      /* Find first hole in the cmap */
      for (j = 0; j < ncols; j++) {
        col = cmap[bj[j]]; /* global column number = cmap[B column number] */
        if (col > j && j < cstart) {
          offdiagIdx[r] = j; /* global column number of first implicit 0.0 */
          break;
        } else if (col > j + n && j >= cstart) {
          offdiagIdx[r] = j + n; /* global column number of first implicit 0.0 */
          break;
        }
      }
      if (j == ncols && ncols < A->cmap->N - n) {
        /* a hole is outside compressed Bcols */
        if (ncols == 0) {
          if (cstart) {
            offdiagIdx[r] = 0;
          } else offdiagIdx[r] = cend;
        } else { /* ncols > 0 */
          offdiagIdx[r] = cmap[ncols - 1] + 1;
          if (offdiagIdx[r] == cstart) offdiagIdx[r] += n;
        }
      }
    }

    for (j = 0; j < ncols; j++) {
      if (PetscAbsScalar(offdiagA[r]) > PetscAbsScalar(*ba)) {
        offdiagA[r]   = *ba;
        offdiagIdx[r] = cmap[*bj];
      }
      ba++;
      bj++;
    }
  }

  PetscCall(VecGetArrayWrite(v, &a));
  PetscCall(VecGetArrayRead(diagV, (const PetscScalar **)&diagA));
  for (r = 0; r < m; ++r) {
    if (PetscAbsScalar(diagA[r]) < PetscAbsScalar(offdiagA[r])) {
      a[r] = diagA[r];
      if (idx) idx[r] = cstart + diagIdx[r];
    } else if (PetscAbsScalar(diagA[r]) == PetscAbsScalar(offdiagA[r])) {
      a[r] = diagA[r];
      if (idx) {
        if (cstart + diagIdx[r] <= offdiagIdx[r]) {
          idx[r] = cstart + diagIdx[r];
        } else idx[r] = offdiagIdx[r];
      }
    } else {
      a[r] = offdiagA[r];
      if (idx) idx[r] = offdiagIdx[r];
    }
  }
  PetscCall(MatSeqAIJRestoreArrayRead(B, &bav));
  PetscCall(VecRestoreArrayWrite(v, &a));
  PetscCall(VecRestoreArrayRead(diagV, (const PetscScalar **)&diagA));
  PetscCall(VecRestoreArrayWrite(offdiagV, &offdiagA));
  PetscCall(VecDestroy(&diagV));
  PetscCall(VecDestroy(&offdiagV));
  PetscCall(PetscFree2(diagIdx, offdiagIdx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetRowMin_MPIAIJ(Mat A, Vec v, PetscInt idx[])
{
  Mat_MPIAIJ        *mat = (Mat_MPIAIJ *)A->data;
  PetscInt           m = A->rmap->n, n = A->cmap->n;
  PetscInt           cstart = A->cmap->rstart, cend = A->cmap->rend;
  PetscInt          *cmap = mat->garray;
  PetscInt          *diagIdx, *offdiagIdx;
  Vec                diagV, offdiagV;
  PetscScalar       *a, *diagA, *offdiagA;
  const PetscScalar *ba, *bav;
  PetscInt           r, j, col, ncols, *bi, *bj;
  Mat                B = mat->B;
  Mat_SeqAIJ        *b = (Mat_SeqAIJ *)B->data;

  PetscFunctionBegin;
  /* When a process holds entire A and other processes have no entry */
  if (A->cmap->N == n) {
    PetscCall(VecGetArrayWrite(v, &diagA));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, m, diagA, &diagV));
    PetscCall(MatGetRowMin(mat->A, diagV, idx));
    PetscCall(VecDestroy(&diagV));
    PetscCall(VecRestoreArrayWrite(v, &diagA));
    PetscFunctionReturn(PETSC_SUCCESS);
  } else if (n == 0) {
    if (m) {
      PetscCall(VecGetArrayWrite(v, &a));
      for (r = 0; r < m; r++) {
        a[r] = PETSC_MAX_REAL;
        if (idx) idx[r] = -1;
      }
      PetscCall(VecRestoreArrayWrite(v, &a));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscCalloc2(m, &diagIdx, m, &offdiagIdx));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, m, &diagV));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, m, &offdiagV));
  PetscCall(MatGetRowMin(mat->A, diagV, diagIdx));

  /* Get offdiagIdx[] for implicit 0.0 */
  PetscCall(MatSeqAIJGetArrayRead(B, &bav));
  ba = bav;
  bi = b->i;
  bj = b->j;
  PetscCall(VecGetArrayWrite(offdiagV, &offdiagA));
  for (r = 0; r < m; r++) {
    ncols = bi[r + 1] - bi[r];
    if (ncols == A->cmap->N - n) { /* Brow is dense */
      offdiagA[r]   = *ba;
      offdiagIdx[r] = cmap[0];
    } else { /* Brow is sparse so already KNOW maximum is 0.0 or higher */
      offdiagA[r] = 0.0;

      /* Find first hole in the cmap */
      for (j = 0; j < ncols; j++) {
        col = cmap[bj[j]]; /* global column number = cmap[B column number] */
        if (col > j && j < cstart) {
          offdiagIdx[r] = j; /* global column number of first implicit 0.0 */
          break;
        } else if (col > j + n && j >= cstart) {
          offdiagIdx[r] = j + n; /* global column number of first implicit 0.0 */
          break;
        }
      }
      if (j == ncols && ncols < A->cmap->N - n) {
        /* a hole is outside compressed Bcols */
        if (ncols == 0) {
          if (cstart) {
            offdiagIdx[r] = 0;
          } else offdiagIdx[r] = cend;
        } else { /* ncols > 0 */
          offdiagIdx[r] = cmap[ncols - 1] + 1;
          if (offdiagIdx[r] == cstart) offdiagIdx[r] += n;
        }
      }
    }

    for (j = 0; j < ncols; j++) {
      if (PetscRealPart(offdiagA[r]) > PetscRealPart(*ba)) {
        offdiagA[r]   = *ba;
        offdiagIdx[r] = cmap[*bj];
      }
      ba++;
      bj++;
    }
  }

  PetscCall(VecGetArrayWrite(v, &a));
  PetscCall(VecGetArrayRead(diagV, (const PetscScalar **)&diagA));
  for (r = 0; r < m; ++r) {
    if (PetscRealPart(diagA[r]) < PetscRealPart(offdiagA[r])) {
      a[r] = diagA[r];
      if (idx) idx[r] = cstart + diagIdx[r];
    } else if (PetscRealPart(diagA[r]) == PetscRealPart(offdiagA[r])) {
      a[r] = diagA[r];
      if (idx) {
        if (cstart + diagIdx[r] <= offdiagIdx[r]) {
          idx[r] = cstart + diagIdx[r];
        } else idx[r] = offdiagIdx[r];
      }
    } else {
      a[r] = offdiagA[r];
      if (idx) idx[r] = offdiagIdx[r];
    }
  }
  PetscCall(MatSeqAIJRestoreArrayRead(B, &bav));
  PetscCall(VecRestoreArrayWrite(v, &a));
  PetscCall(VecRestoreArrayRead(diagV, (const PetscScalar **)&diagA));
  PetscCall(VecRestoreArrayWrite(offdiagV, &offdiagA));
  PetscCall(VecDestroy(&diagV));
  PetscCall(VecDestroy(&offdiagV));
  PetscCall(PetscFree2(diagIdx, offdiagIdx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetRowMax_MPIAIJ(Mat A, Vec v, PetscInt idx[])
{
  Mat_MPIAIJ        *mat = (Mat_MPIAIJ *)A->data;
  PetscInt           m = A->rmap->n, n = A->cmap->n;
  PetscInt           cstart = A->cmap->rstart, cend = A->cmap->rend;
  PetscInt          *cmap = mat->garray;
  PetscInt          *diagIdx, *offdiagIdx;
  Vec                diagV, offdiagV;
  PetscScalar       *a, *diagA, *offdiagA;
  const PetscScalar *ba, *bav;
  PetscInt           r, j, col, ncols, *bi, *bj;
  Mat                B = mat->B;
  Mat_SeqAIJ        *b = (Mat_SeqAIJ *)B->data;

  PetscFunctionBegin;
  /* When a process holds entire A and other processes have no entry */
  if (A->cmap->N == n) {
    PetscCall(VecGetArrayWrite(v, &diagA));
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, m, diagA, &diagV));
    PetscCall(MatGetRowMax(mat->A, diagV, idx));
    PetscCall(VecDestroy(&diagV));
    PetscCall(VecRestoreArrayWrite(v, &diagA));
    PetscFunctionReturn(PETSC_SUCCESS);
  } else if (n == 0) {
    if (m) {
      PetscCall(VecGetArrayWrite(v, &a));
      for (r = 0; r < m; r++) {
        a[r] = PETSC_MIN_REAL;
        if (idx) idx[r] = -1;
      }
      PetscCall(VecRestoreArrayWrite(v, &a));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscMalloc2(m, &diagIdx, m, &offdiagIdx));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, m, &diagV));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, m, &offdiagV));
  PetscCall(MatGetRowMax(mat->A, diagV, diagIdx));

  /* Get offdiagIdx[] for implicit 0.0 */
  PetscCall(MatSeqAIJGetArrayRead(B, &bav));
  ba = bav;
  bi = b->i;
  bj = b->j;
  PetscCall(VecGetArrayWrite(offdiagV, &offdiagA));
  for (r = 0; r < m; r++) {
    ncols = bi[r + 1] - bi[r];
    if (ncols == A->cmap->N - n) { /* Brow is dense */
      offdiagA[r]   = *ba;
      offdiagIdx[r] = cmap[0];
    } else { /* Brow is sparse so already KNOW maximum is 0.0 or higher */
      offdiagA[r] = 0.0;

      /* Find first hole in the cmap */
      for (j = 0; j < ncols; j++) {
        col = cmap[bj[j]]; /* global column number = cmap[B column number] */
        if (col > j && j < cstart) {
          offdiagIdx[r] = j; /* global column number of first implicit 0.0 */
          break;
        } else if (col > j + n && j >= cstart) {
          offdiagIdx[r] = j + n; /* global column number of first implicit 0.0 */
          break;
        }
      }
      if (j == ncols && ncols < A->cmap->N - n) {
        /* a hole is outside compressed Bcols */
        if (ncols == 0) {
          if (cstart) {
            offdiagIdx[r] = 0;
          } else offdiagIdx[r] = cend;
        } else { /* ncols > 0 */
          offdiagIdx[r] = cmap[ncols - 1] + 1;
          if (offdiagIdx[r] == cstart) offdiagIdx[r] += n;
        }
      }
    }

    for (j = 0; j < ncols; j++) {
      if (PetscRealPart(offdiagA[r]) < PetscRealPart(*ba)) {
        offdiagA[r]   = *ba;
        offdiagIdx[r] = cmap[*bj];
      }
      ba++;
      bj++;
    }
  }

  PetscCall(VecGetArrayWrite(v, &a));
  PetscCall(VecGetArrayRead(diagV, (const PetscScalar **)&diagA));
  for (r = 0; r < m; ++r) {
    if (PetscRealPart(diagA[r]) > PetscRealPart(offdiagA[r])) {
      a[r] = diagA[r];
      if (idx) idx[r] = cstart + diagIdx[r];
    } else if (PetscRealPart(diagA[r]) == PetscRealPart(offdiagA[r])) {
      a[r] = diagA[r];
      if (idx) {
        if (cstart + diagIdx[r] <= offdiagIdx[r]) {
          idx[r] = cstart + diagIdx[r];
        } else idx[r] = offdiagIdx[r];
      }
    } else {
      a[r] = offdiagA[r];
      if (idx) idx[r] = offdiagIdx[r];
    }
  }
  PetscCall(MatSeqAIJRestoreArrayRead(B, &bav));
  PetscCall(VecRestoreArrayWrite(v, &a));
  PetscCall(VecRestoreArrayRead(diagV, (const PetscScalar **)&diagA));
  PetscCall(VecRestoreArrayWrite(offdiagV, &offdiagA));
  PetscCall(VecDestroy(&diagV));
  PetscCall(VecDestroy(&offdiagV));
  PetscCall(PetscFree2(diagIdx, offdiagIdx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatGetSeqNonzeroStructure_MPIAIJ(Mat mat, Mat *newmat)
{
  Mat *dummy;

  PetscFunctionBegin;
  PetscCall(MatCreateSubMatrix_MPIAIJ_All(mat, MAT_DO_NOT_GET_VALUES, MAT_INITIAL_MATRIX, &dummy));
  *newmat = *dummy;
  PetscCall(PetscFree(dummy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatInvertBlockDiagonal_MPIAIJ(Mat A, const PetscScalar **values)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatInvertBlockDiagonal(a->A, values));
  A->factorerrortype = a->A->factorerrortype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetRandom_MPIAIJ(Mat x, PetscRandom rctx)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *)x->data;

  PetscFunctionBegin;
  PetscCheck(x->assembled || x->preallocated, PetscObjectComm((PetscObject)x), PETSC_ERR_ARG_WRONGSTATE, "MatSetRandom on an unassembled and unpreallocated MATMPIAIJ is not allowed");
  PetscCall(MatSetRandom(aij->A, rctx));
  if (x->assembled) {
    PetscCall(MatSetRandom(aij->B, rctx));
  } else {
    PetscCall(MatSetRandomSkipColumnRange_SeqAIJ_Private(aij->B, x->cmap->rstart, x->cmap->rend, rctx));
  }
  PetscCall(MatAssemblyBegin(x, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(x, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMPIAIJSetUseScalableIncreaseOverlap_MPIAIJ(Mat A, PetscBool sc)
{
  PetscFunctionBegin;
  if (sc) A->ops->increaseoverlap = MatIncreaseOverlap_MPIAIJ_Scalable;
  else A->ops->increaseoverlap = MatIncreaseOverlap_MPIAIJ;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMPIAIJGetNumberNonzeros - gets the number of nonzeros in the matrix on this MPI rank

  Not Collective

  Input Parameter:
. A - the matrix

  Output Parameter:
. nz - the number of nonzeros

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATMPIAIJ`
@*/
PetscErrorCode MatMPIAIJGetNumberNonzeros(Mat A, PetscCount *nz)
{
  Mat_MPIAIJ *maij = (Mat_MPIAIJ *)A->data;
  Mat_SeqAIJ *aaij = (Mat_SeqAIJ *)maij->A->data, *baij = (Mat_SeqAIJ *)maij->B->data;
  PetscBool   isaij;

  PetscFunctionBegin;
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATMPIAIJ, &isaij));
  PetscCheck(isaij, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Not for type %s", ((PetscObject)A)->type_name);
  *nz = aaij->i[A->rmap->n] + baij->i[A->rmap->n];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMPIAIJSetUseScalableIncreaseOverlap - Determine if the matrix uses a scalable algorithm to compute the overlap

  Collective

  Input Parameters:
+ A  - the matrix
- sc - `PETSC_TRUE` indicates use the scalable algorithm (default is not to use the scalable algorithm)

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATMPIAIJ`
@*/
PetscErrorCode MatMPIAIJSetUseScalableIncreaseOverlap(Mat A, PetscBool sc)
{
  PetscFunctionBegin;
  PetscTryMethod(A, "MatMPIAIJSetUseScalableIncreaseOverlap_C", (Mat, PetscBool), (A, sc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSetFromOptions_MPIAIJ(Mat A, PetscOptionItems *PetscOptionsObject)
{
  PetscBool sc = PETSC_FALSE, flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "MPIAIJ options");
  if (A->ops->increaseoverlap == MatIncreaseOverlap_MPIAIJ_Scalable) sc = PETSC_TRUE;
  PetscCall(PetscOptionsBool("-mat_increase_overlap_scalable", "Use a scalable algorithm to compute the overlap", "MatIncreaseOverlap", sc, &sc, &flg));
  if (flg) PetscCall(MatMPIAIJSetUseScalableIncreaseOverlap(A, sc));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatShift_MPIAIJ(Mat Y, PetscScalar a)
{
  Mat_MPIAIJ *maij = (Mat_MPIAIJ *)Y->data;
  Mat_SeqAIJ *aij  = (Mat_SeqAIJ *)maij->A->data;

  PetscFunctionBegin;
  if (!Y->preallocated) {
    PetscCall(MatMPIAIJSetPreallocation(Y, 1, NULL, 0, NULL));
  } else if (!aij->nz) { /* It does not matter if diagonals of Y only partially lie in maij->A. We just need an estimated preallocation. */
    PetscInt nonew = aij->nonew;
    PetscCall(MatSeqAIJSetPreallocation(maij->A, 1, NULL));
    aij->nonew = nonew;
  }
  PetscCall(MatShift_Basic(Y, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMissingDiagonal_MPIAIJ(Mat A, PetscBool *missing, PetscInt *d)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCheck(A->rmap->n == A->cmap->n, PETSC_COMM_SELF, PETSC_ERR_SUP, "Only works for square matrices");
  PetscCall(MatMissingDiagonal(a->A, missing, d));
  if (d) {
    PetscInt rstart;
    PetscCall(MatGetOwnershipRange(A, &rstart, NULL));
    *d += rstart;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatInvertVariableBlockDiagonal_MPIAIJ(Mat A, PetscInt nblocks, const PetscInt *bsizes, PetscScalar *diag)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatInvertVariableBlockDiagonal(a->A, nblocks, bsizes, diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatEliminateZeros_MPIAIJ(Mat A, PetscBool keep)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatEliminateZeros_SeqAIJ(a->A, keep));        // possibly keep zero diagonal coefficients
  PetscCall(MatEliminateZeros_SeqAIJ(a->B, PETSC_FALSE)); // never keep zero diagonal coefficients
  PetscFunctionReturn(PETSC_SUCCESS);
}

static struct _MatOps MatOps_Values = {MatSetValues_MPIAIJ,
                                       MatGetRow_MPIAIJ,
                                       MatRestoreRow_MPIAIJ,
                                       MatMult_MPIAIJ,
                                       /* 4*/ MatMultAdd_MPIAIJ,
                                       MatMultTranspose_MPIAIJ,
                                       MatMultTransposeAdd_MPIAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       /*10*/ NULL,
                                       NULL,
                                       NULL,
                                       MatSOR_MPIAIJ,
                                       MatTranspose_MPIAIJ,
                                       /*15*/ MatGetInfo_MPIAIJ,
                                       MatEqual_MPIAIJ,
                                       MatGetDiagonal_MPIAIJ,
                                       MatDiagonalScale_MPIAIJ,
                                       MatNorm_MPIAIJ,
                                       /*20*/ MatAssemblyBegin_MPIAIJ,
                                       MatAssemblyEnd_MPIAIJ,
                                       MatSetOption_MPIAIJ,
                                       MatZeroEntries_MPIAIJ,
                                       /*24*/ MatZeroRows_MPIAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       /*29*/ MatSetUp_MPI_Hash,
                                       NULL,
                                       NULL,
                                       MatGetDiagonalBlock_MPIAIJ,
                                       NULL,
                                       /*34*/ MatDuplicate_MPIAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       /*39*/ MatAXPY_MPIAIJ,
                                       MatCreateSubMatrices_MPIAIJ,
                                       MatIncreaseOverlap_MPIAIJ,
                                       MatGetValues_MPIAIJ,
                                       MatCopy_MPIAIJ,
                                       /*44*/ MatGetRowMax_MPIAIJ,
                                       MatScale_MPIAIJ,
                                       MatShift_MPIAIJ,
                                       MatDiagonalSet_MPIAIJ,
                                       MatZeroRowsColumns_MPIAIJ,
                                       /*49*/ MatSetRandom_MPIAIJ,
                                       MatGetRowIJ_MPIAIJ,
                                       MatRestoreRowIJ_MPIAIJ,
                                       NULL,
                                       NULL,
                                       /*54*/ MatFDColoringCreate_MPIXAIJ,
                                       NULL,
                                       MatSetUnfactored_MPIAIJ,
                                       MatPermute_MPIAIJ,
                                       NULL,
                                       /*59*/ MatCreateSubMatrix_MPIAIJ,
                                       MatDestroy_MPIAIJ,
                                       MatView_MPIAIJ,
                                       NULL,
                                       NULL,
                                       /*64*/ NULL,
                                       MatMatMatMultNumeric_MPIAIJ_MPIAIJ_MPIAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       /*69*/ MatGetRowMaxAbs_MPIAIJ,
                                       MatGetRowMinAbs_MPIAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       /*75*/ MatFDColoringApply_AIJ,
                                       MatSetFromOptions_MPIAIJ,
                                       NULL,
                                       NULL,
                                       MatFindZeroDiagonals_MPIAIJ,
                                       /*80*/ NULL,
                                       NULL,
                                       NULL,
                                       /*83*/ MatLoad_MPIAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       /*89*/ NULL,
                                       NULL,
                                       MatMatMultNumeric_MPIAIJ_MPIAIJ,
                                       NULL,
                                       NULL,
                                       /*94*/ MatPtAPNumeric_MPIAIJ_MPIAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       MatBindToCPU_MPIAIJ,
                                       /*99*/ MatProductSetFromOptions_MPIAIJ,
                                       NULL,
                                       NULL,
                                       MatConjugate_MPIAIJ,
                                       NULL,
                                       /*104*/ MatSetValuesRow_MPIAIJ,
                                       MatRealPart_MPIAIJ,
                                       MatImaginaryPart_MPIAIJ,
                                       NULL,
                                       NULL,
                                       /*109*/ NULL,
                                       NULL,
                                       MatGetRowMin_MPIAIJ,
                                       NULL,
                                       MatMissingDiagonal_MPIAIJ,
                                       /*114*/ MatGetSeqNonzeroStructure_MPIAIJ,
                                       NULL,
                                       MatGetGhosts_MPIAIJ,
                                       NULL,
                                       NULL,
                                       /*119*/ MatMultDiagonalBlock_MPIAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       MatGetMultiProcBlock_MPIAIJ,
                                       /*124*/ MatFindNonzeroRows_MPIAIJ,
                                       MatGetColumnReductions_MPIAIJ,
                                       MatInvertBlockDiagonal_MPIAIJ,
                                       MatInvertVariableBlockDiagonal_MPIAIJ,
                                       MatCreateSubMatricesMPI_MPIAIJ,
                                       /*129*/ NULL,
                                       NULL,
                                       NULL,
                                       MatTransposeMatMultNumeric_MPIAIJ_MPIAIJ,
                                       NULL,
                                       /*134*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       /*139*/ MatSetBlockSizes_MPIAIJ,
                                       NULL,
                                       NULL,
                                       MatFDColoringSetUp_MPIXAIJ,
                                       MatFindOffBlockDiagonalEntries_MPIAIJ,
                                       MatCreateMPIMatConcatenateSeqMat_MPIAIJ,
                                       /*145*/ NULL,
                                       NULL,
                                       NULL,
                                       MatCreateGraph_Simple_AIJ,
                                       NULL,
                                       /*150*/ NULL,
                                       MatEliminateZeros_MPIAIJ,
                                       MatGetRowSumAbs_MPIAIJ,
                                       NULL,
                                       NULL,
                                       NULL};

static PetscErrorCode MatStoreValues_MPIAIJ(Mat mat)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *)mat->data;

  PetscFunctionBegin;
  PetscCall(MatStoreValues(aij->A));
  PetscCall(MatStoreValues(aij->B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatRetrieveValues_MPIAIJ(Mat mat)
{
  Mat_MPIAIJ *aij = (Mat_MPIAIJ *)mat->data;

  PetscFunctionBegin;
  PetscCall(MatRetrieveValues(aij->A));
  PetscCall(MatRetrieveValues(aij->B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMPIAIJSetPreallocation_MPIAIJ(Mat B, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[])
{
  Mat_MPIAIJ *b = (Mat_MPIAIJ *)B->data;
  PetscMPIInt size;

  PetscFunctionBegin;
  if (B->hash_active) {
    B->ops[0]      = b->cops;
    B->hash_active = PETSC_FALSE;
  }
  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));

#if defined(PETSC_USE_CTABLE)
  PetscCall(PetscHMapIDestroy(&b->colmap));
#else
  PetscCall(PetscFree(b->colmap));
#endif
  PetscCall(PetscFree(b->garray));
  PetscCall(VecDestroy(&b->lvec));
  PetscCall(VecScatterDestroy(&b->Mvctx));

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)B), &size));

  MatSeqXAIJGetOptions_Private(b->B);
  PetscCall(MatDestroy(&b->B));
  PetscCall(MatCreate(PETSC_COMM_SELF, &b->B));
  PetscCall(MatSetSizes(b->B, B->rmap->n, size > 1 ? B->cmap->N : 0, B->rmap->n, size > 1 ? B->cmap->N : 0));
  PetscCall(MatSetBlockSizesFromMats(b->B, B, B));
  PetscCall(MatSetType(b->B, MATSEQAIJ));
  MatSeqXAIJRestoreOptions_Private(b->B);

  MatSeqXAIJGetOptions_Private(b->A);
  PetscCall(MatDestroy(&b->A));
  PetscCall(MatCreate(PETSC_COMM_SELF, &b->A));
  PetscCall(MatSetSizes(b->A, B->rmap->n, B->cmap->n, B->rmap->n, B->cmap->n));
  PetscCall(MatSetBlockSizesFromMats(b->A, B, B));
  PetscCall(MatSetType(b->A, MATSEQAIJ));
  MatSeqXAIJRestoreOptions_Private(b->A);

  PetscCall(MatSeqAIJSetPreallocation(b->A, d_nz, d_nnz));
  PetscCall(MatSeqAIJSetPreallocation(b->B, o_nz, o_nnz));
  B->preallocated  = PETSC_TRUE;
  B->was_assembled = PETSC_FALSE;
  B->assembled     = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatResetPreallocation_MPIAIJ(Mat B)
{
  Mat_MPIAIJ *b = (Mat_MPIAIJ *)B->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));

#if defined(PETSC_USE_CTABLE)
  PetscCall(PetscHMapIDestroy(&b->colmap));
#else
  PetscCall(PetscFree(b->colmap));
#endif
  PetscCall(PetscFree(b->garray));
  PetscCall(VecDestroy(&b->lvec));
  PetscCall(VecScatterDestroy(&b->Mvctx));

  PetscCall(MatResetPreallocation(b->A));
  PetscCall(MatResetPreallocation(b->B));
  B->preallocated  = PETSC_TRUE;
  B->was_assembled = PETSC_FALSE;
  B->assembled     = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatDuplicate_MPIAIJ(Mat matin, MatDuplicateOption cpvalues, Mat *newmat)
{
  Mat         mat;
  Mat_MPIAIJ *a, *oldmat = (Mat_MPIAIJ *)matin->data;

  PetscFunctionBegin;
  *newmat = NULL;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)matin), &mat));
  PetscCall(MatSetSizes(mat, matin->rmap->n, matin->cmap->n, matin->rmap->N, matin->cmap->N));
  PetscCall(MatSetBlockSizesFromMats(mat, matin, matin));
  PetscCall(MatSetType(mat, ((PetscObject)matin)->type_name));
  a = (Mat_MPIAIJ *)mat->data;

  mat->factortype = matin->factortype;
  mat->assembled  = matin->assembled;
  mat->insertmode = NOT_SET_VALUES;

  a->size         = oldmat->size;
  a->rank         = oldmat->rank;
  a->donotstash   = oldmat->donotstash;
  a->roworiented  = oldmat->roworiented;
  a->rowindices   = NULL;
  a->rowvalues    = NULL;
  a->getrowactive = PETSC_FALSE;

  PetscCall(PetscLayoutReference(matin->rmap, &mat->rmap));
  PetscCall(PetscLayoutReference(matin->cmap, &mat->cmap));
  if (matin->hash_active) {
    PetscCall(MatSetUp(mat));
  } else {
    mat->preallocated = matin->preallocated;
    if (oldmat->colmap) {
#if defined(PETSC_USE_CTABLE)
      PetscCall(PetscHMapIDuplicate(oldmat->colmap, &a->colmap));
#else
      PetscCall(PetscMalloc1(mat->cmap->N, &a->colmap));
      PetscCall(PetscArraycpy(a->colmap, oldmat->colmap, mat->cmap->N));
#endif
    } else a->colmap = NULL;
    if (oldmat->garray) {
      PetscInt len;
      len = oldmat->B->cmap->n;
      PetscCall(PetscMalloc1(len + 1, &a->garray));
      if (len) PetscCall(PetscArraycpy(a->garray, oldmat->garray, len));
    } else a->garray = NULL;

    /* It may happen MatDuplicate is called with a non-assembled matrix
      In fact, MatDuplicate only requires the matrix to be preallocated
      This may happen inside a DMCreateMatrix_Shell */
    if (oldmat->lvec) PetscCall(VecDuplicate(oldmat->lvec, &a->lvec));
    if (oldmat->Mvctx) {
      a->Mvctx = oldmat->Mvctx;
      PetscCall(PetscObjectReference((PetscObject)oldmat->Mvctx));
    }
    PetscCall(MatDuplicate(oldmat->A, cpvalues, &a->A));
    PetscCall(MatDuplicate(oldmat->B, cpvalues, &a->B));
  }
  PetscCall(PetscFunctionListDuplicate(((PetscObject)matin)->qlist, &((PetscObject)mat)->qlist));
  *newmat = mat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatLoad_MPIAIJ(Mat newMat, PetscViewer viewer)
{
  PetscBool isbinary, ishdf5;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(newMat, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  /* force binary viewer to load .info file if it has not yet done so */
  PetscCall(PetscViewerSetUp(viewer));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERHDF5, &ishdf5));
  if (isbinary) {
    PetscCall(MatLoad_MPIAIJ_Binary(newMat, viewer));
  } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    PetscCall(MatLoad_AIJ_HDF5(newMat, viewer));
#else
    SETERRQ(PetscObjectComm((PetscObject)newMat), PETSC_ERR_SUP, "HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  } else {
    SETERRQ(PetscObjectComm((PetscObject)newMat), PETSC_ERR_SUP, "Viewer type %s not yet supported for reading %s matrices", ((PetscObject)viewer)->type_name, ((PetscObject)newMat)->type_name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatLoad_MPIAIJ_Binary(Mat mat, PetscViewer viewer)
{
  PetscInt     header[4], M, N, m, nz, rows, cols, sum, i;
  PetscInt    *rowidxs, *colidxs;
  PetscScalar *matvals;

  PetscFunctionBegin;
  PetscCall(PetscViewerSetUp(viewer));

  /* read in matrix header */
  PetscCall(PetscViewerBinaryRead(viewer, header, 4, NULL, PETSC_INT));
  PetscCheck(header[0] == MAT_FILE_CLASSID, PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "Not a matrix object in file");
  M  = header[1];
  N  = header[2];
  nz = header[3];
  PetscCheck(M >= 0, PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "Matrix row size (%" PetscInt_FMT ") in file is negative", M);
  PetscCheck(N >= 0, PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "Matrix column size (%" PetscInt_FMT ") in file is negative", N);
  PetscCheck(nz >= 0, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Matrix stored in special format on disk, cannot load as MPIAIJ");

  /* set block sizes from the viewer's .info file */
  PetscCall(MatLoad_Binary_BlockSizes(mat, viewer));
  /* set global sizes if not set already */
  if (mat->rmap->N < 0) mat->rmap->N = M;
  if (mat->cmap->N < 0) mat->cmap->N = N;
  PetscCall(PetscLayoutSetUp(mat->rmap));
  PetscCall(PetscLayoutSetUp(mat->cmap));

  /* check if the matrix sizes are correct */
  PetscCall(MatGetSize(mat, &rows, &cols));
  PetscCheck(M == rows && N == cols, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Matrix in file of different sizes (%" PetscInt_FMT ", %" PetscInt_FMT ") than the input matrix (%" PetscInt_FMT ", %" PetscInt_FMT ")", M, N, rows, cols);

  /* read in row lengths and build row indices */
  PetscCall(MatGetLocalSize(mat, &m, NULL));
  PetscCall(PetscMalloc1(m + 1, &rowidxs));
  PetscCall(PetscViewerBinaryReadAll(viewer, rowidxs + 1, m, PETSC_DECIDE, M, PETSC_INT));
  rowidxs[0] = 0;
  for (i = 0; i < m; i++) rowidxs[i + 1] += rowidxs[i];
  if (nz != PETSC_INT_MAX) {
    PetscCallMPI(MPIU_Allreduce(&rowidxs[m], &sum, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)viewer)));
    PetscCheck(sum == nz, PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_UNEXPECTED, "Inconsistent matrix data in file: nonzeros = %" PetscInt_FMT ", sum-row-lengths = %" PetscInt_FMT, nz, sum);
  }

  /* read in column indices and matrix values */
  PetscCall(PetscMalloc2(rowidxs[m], &colidxs, rowidxs[m], &matvals));
  PetscCall(PetscViewerBinaryReadAll(viewer, colidxs, rowidxs[m], PETSC_DETERMINE, PETSC_DETERMINE, PETSC_INT));
  PetscCall(PetscViewerBinaryReadAll(viewer, matvals, rowidxs[m], PETSC_DETERMINE, PETSC_DETERMINE, PETSC_SCALAR));
  /* store matrix indices and values */
  PetscCall(MatMPIAIJSetPreallocationCSR(mat, rowidxs, colidxs, matvals));
  PetscCall(PetscFree(rowidxs));
  PetscCall(PetscFree2(colidxs, matvals));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Not scalable because of ISAllGather() unless getting all columns. */
static PetscErrorCode ISGetSeqIS_Private(Mat mat, IS iscol, IS *isseq)
{
  IS          iscol_local;
  PetscBool   isstride;
  PetscMPIInt lisstride = 0, gisstride;

  PetscFunctionBegin;
  /* check if we are grabbing all columns*/
  PetscCall(PetscObjectTypeCompare((PetscObject)iscol, ISSTRIDE, &isstride));

  if (isstride) {
    PetscInt start, len, mstart, mlen;
    PetscCall(ISStrideGetInfo(iscol, &start, NULL));
    PetscCall(ISGetLocalSize(iscol, &len));
    PetscCall(MatGetOwnershipRangeColumn(mat, &mstart, &mlen));
    if (mstart == start && mlen - mstart == len) lisstride = 1;
  }

  PetscCallMPI(MPIU_Allreduce(&lisstride, &gisstride, 1, MPI_INT, MPI_MIN, PetscObjectComm((PetscObject)mat)));
  if (gisstride) {
    PetscInt N;
    PetscCall(MatGetSize(mat, NULL, &N));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, N, 0, 1, &iscol_local));
    PetscCall(ISSetIdentity(iscol_local));
    PetscCall(PetscInfo(mat, "Optimizing for obtaining all columns of the matrix; skipping ISAllGather()\n"));
  } else {
    PetscInt cbs;
    PetscCall(ISGetBlockSize(iscol, &cbs));
    PetscCall(ISAllGather(iscol, &iscol_local));
    PetscCall(ISSetBlockSize(iscol_local, cbs));
  }

  *isseq = iscol_local;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 Used by MatCreateSubMatrix_MPIAIJ_SameRowColDist() to avoid ISAllGather() and global size of iscol_local
 (see MatCreateSubMatrix_MPIAIJ_nonscalable)

 Input Parameters:
+   mat - matrix
.   isrow - parallel row index set; its local indices are a subset of local columns of `mat`,
           i.e., mat->rstart <= isrow[i] < mat->rend
-   iscol - parallel column index set; its local indices are a subset of local columns of `mat`,
           i.e., mat->cstart <= iscol[i] < mat->cend

 Output Parameters:
+   isrow_d - sequential row index set for retrieving mat->A
.   iscol_d - sequential  column index set for retrieving mat->A
.   iscol_o - sequential column index set for retrieving mat->B
-   garray - column map; garray[i] indicates global location of iscol_o[i] in `iscol`
 */
static PetscErrorCode ISGetSeqIS_SameColDist_Private(Mat mat, IS isrow, IS iscol, IS *isrow_d, IS *iscol_d, IS *iscol_o, const PetscInt *garray[])
{
  Vec             x, cmap;
  const PetscInt *is_idx;
  PetscScalar    *xarray, *cmaparray;
  PetscInt        ncols, isstart, *idx, m, rstart, *cmap1, count;
  Mat_MPIAIJ     *a    = (Mat_MPIAIJ *)mat->data;
  Mat             B    = a->B;
  Vec             lvec = a->lvec, lcmap;
  PetscInt        i, cstart, cend, Bn = B->cmap->N;
  MPI_Comm        comm;
  VecScatter      Mvctx = a->Mvctx;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)mat, &comm));
  PetscCall(ISGetLocalSize(iscol, &ncols));

  /* (1) iscol is a sub-column vector of mat, pad it with '-1.' to form a full vector x */
  PetscCall(MatCreateVecs(mat, &x, NULL));
  PetscCall(VecSet(x, -1.0));
  PetscCall(VecDuplicate(x, &cmap));
  PetscCall(VecSet(cmap, -1.0));

  /* Get start indices */
  PetscCallMPI(MPI_Scan(&ncols, &isstart, 1, MPIU_INT, MPI_SUM, comm));
  isstart -= ncols;
  PetscCall(MatGetOwnershipRangeColumn(mat, &cstart, &cend));

  PetscCall(ISGetIndices(iscol, &is_idx));
  PetscCall(VecGetArray(x, &xarray));
  PetscCall(VecGetArray(cmap, &cmaparray));
  PetscCall(PetscMalloc1(ncols, &idx));
  for (i = 0; i < ncols; i++) {
    xarray[is_idx[i] - cstart]    = (PetscScalar)is_idx[i];
    cmaparray[is_idx[i] - cstart] = i + isstart;        /* global index of iscol[i] */
    idx[i]                        = is_idx[i] - cstart; /* local index of iscol[i]  */
  }
  PetscCall(VecRestoreArray(x, &xarray));
  PetscCall(VecRestoreArray(cmap, &cmaparray));
  PetscCall(ISRestoreIndices(iscol, &is_idx));

  /* Get iscol_d */
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, ncols, idx, PETSC_OWN_POINTER, iscol_d));
  PetscCall(ISGetBlockSize(iscol, &i));
  PetscCall(ISSetBlockSize(*iscol_d, i));

  /* Get isrow_d */
  PetscCall(ISGetLocalSize(isrow, &m));
  rstart = mat->rmap->rstart;
  PetscCall(PetscMalloc1(m, &idx));
  PetscCall(ISGetIndices(isrow, &is_idx));
  for (i = 0; i < m; i++) idx[i] = is_idx[i] - rstart;
  PetscCall(ISRestoreIndices(isrow, &is_idx));

  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, m, idx, PETSC_OWN_POINTER, isrow_d));
  PetscCall(ISGetBlockSize(isrow, &i));
  PetscCall(ISSetBlockSize(*isrow_d, i));

  /* (2) Scatter x and cmap using aij->Mvctx to get their off-process portions (see MatMult_MPIAIJ) */
  PetscCall(VecScatterBegin(Mvctx, x, lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(Mvctx, x, lvec, INSERT_VALUES, SCATTER_FORWARD));

  PetscCall(VecDuplicate(lvec, &lcmap));

  PetscCall(VecScatterBegin(Mvctx, cmap, lcmap, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(Mvctx, cmap, lcmap, INSERT_VALUES, SCATTER_FORWARD));

  /* (3) create sequential iscol_o (a subset of iscol) and isgarray */
  /* off-process column indices */
  count = 0;
  PetscCall(PetscMalloc1(Bn, &idx));
  PetscCall(PetscMalloc1(Bn, &cmap1));

  PetscCall(VecGetArray(lvec, &xarray));
  PetscCall(VecGetArray(lcmap, &cmaparray));
  for (i = 0; i < Bn; i++) {
    if (PetscRealPart(xarray[i]) > -1.0) {
      idx[count]   = i;                                     /* local column index in off-diagonal part B */
      cmap1[count] = (PetscInt)PetscRealPart(cmaparray[i]); /* column index in submat */
      count++;
    }
  }
  PetscCall(VecRestoreArray(lvec, &xarray));
  PetscCall(VecRestoreArray(lcmap, &cmaparray));

  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, count, idx, PETSC_COPY_VALUES, iscol_o));
  /* cannot ensure iscol_o has same blocksize as iscol! */

  PetscCall(PetscFree(idx));
  *garray = cmap1;

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&cmap));
  PetscCall(VecDestroy(&lcmap));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* isrow and iscol have same processor distribution as mat, output *submat is a submatrix of local mat */
PetscErrorCode MatCreateSubMatrix_MPIAIJ_SameRowColDist(Mat mat, IS isrow, IS iscol, MatReuse call, Mat *submat)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)mat->data, *asub;
  Mat         M = NULL;
  MPI_Comm    comm;
  IS          iscol_d, isrow_d, iscol_o;
  Mat         Asub = NULL, Bsub = NULL;
  PetscInt    n;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)mat, &comm));

  if (call == MAT_REUSE_MATRIX) {
    /* Retrieve isrow_d, iscol_d and iscol_o from submat */
    PetscCall(PetscObjectQuery((PetscObject)*submat, "isrow_d", (PetscObject *)&isrow_d));
    PetscCheck(isrow_d, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "isrow_d passed in was not used before, cannot reuse");

    PetscCall(PetscObjectQuery((PetscObject)*submat, "iscol_d", (PetscObject *)&iscol_d));
    PetscCheck(iscol_d, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "iscol_d passed in was not used before, cannot reuse");

    PetscCall(PetscObjectQuery((PetscObject)*submat, "iscol_o", (PetscObject *)&iscol_o));
    PetscCheck(iscol_o, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "iscol_o passed in was not used before, cannot reuse");

    /* Update diagonal and off-diagonal portions of submat */
    asub = (Mat_MPIAIJ *)(*submat)->data;
    PetscCall(MatCreateSubMatrix_SeqAIJ(a->A, isrow_d, iscol_d, PETSC_DECIDE, MAT_REUSE_MATRIX, &asub->A));
    PetscCall(ISGetLocalSize(iscol_o, &n));
    if (n) PetscCall(MatCreateSubMatrix_SeqAIJ(a->B, isrow_d, iscol_o, PETSC_DECIDE, MAT_REUSE_MATRIX, &asub->B));
    PetscCall(MatAssemblyBegin(*submat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*submat, MAT_FINAL_ASSEMBLY));

  } else { /* call == MAT_INITIAL_MATRIX) */
    const PetscInt *garray;
    PetscInt        BsubN;

    /* Create isrow_d, iscol_d, iscol_o and isgarray (replace isgarray with array?) */
    PetscCall(ISGetSeqIS_SameColDist_Private(mat, isrow, iscol, &isrow_d, &iscol_d, &iscol_o, &garray));

    /* Create local submatrices Asub and Bsub */
    PetscCall(MatCreateSubMatrix_SeqAIJ(a->A, isrow_d, iscol_d, PETSC_DECIDE, MAT_INITIAL_MATRIX, &Asub));
    PetscCall(MatCreateSubMatrix_SeqAIJ(a->B, isrow_d, iscol_o, PETSC_DECIDE, MAT_INITIAL_MATRIX, &Bsub));

    /* Create submatrix M */
    PetscCall(MatCreateMPIAIJWithSeqAIJ(comm, Asub, Bsub, garray, &M));

    /* If Bsub has empty columns, compress iscol_o such that it will retrieve condensed Bsub from a->B during reuse */
    asub = (Mat_MPIAIJ *)M->data;

    PetscCall(ISGetLocalSize(iscol_o, &BsubN));
    n = asub->B->cmap->N;
    if (BsubN > n) {
      /* This case can be tested using ~petsc/src/tao/bound/tutorials/runplate2_3 */
      const PetscInt *idx;
      PetscInt        i, j, *idx_new, *subgarray = asub->garray;
      PetscCall(PetscInfo(M, "submatrix Bn %" PetscInt_FMT " != BsubN %" PetscInt_FMT ", update iscol_o\n", n, BsubN));

      PetscCall(PetscMalloc1(n, &idx_new));
      j = 0;
      PetscCall(ISGetIndices(iscol_o, &idx));
      for (i = 0; i < n; i++) {
        if (j >= BsubN) break;
        while (subgarray[i] > garray[j]) j++;

        if (subgarray[i] == garray[j]) {
          idx_new[i] = idx[j++];
        } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "subgarray[%" PetscInt_FMT "]=%" PetscInt_FMT " cannot < garray[%" PetscInt_FMT "]=%" PetscInt_FMT, i, subgarray[i], j, garray[j]);
      }
      PetscCall(ISRestoreIndices(iscol_o, &idx));

      PetscCall(ISDestroy(&iscol_o));
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF, n, idx_new, PETSC_OWN_POINTER, &iscol_o));

    } else if (BsubN < n) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Columns of Bsub (%" PetscInt_FMT ") cannot be smaller than B's (%" PetscInt_FMT ")", BsubN, asub->B->cmap->N);
    }

    PetscCall(PetscFree(garray));
    *submat = M;

    /* Save isrow_d, iscol_d and iscol_o used in processor for next request */
    PetscCall(PetscObjectCompose((PetscObject)M, "isrow_d", (PetscObject)isrow_d));
    PetscCall(ISDestroy(&isrow_d));

    PetscCall(PetscObjectCompose((PetscObject)M, "iscol_d", (PetscObject)iscol_d));
    PetscCall(ISDestroy(&iscol_d));

    PetscCall(PetscObjectCompose((PetscObject)M, "iscol_o", (PetscObject)iscol_o));
    PetscCall(ISDestroy(&iscol_o));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCreateSubMatrix_MPIAIJ(Mat mat, IS isrow, IS iscol, MatReuse call, Mat *newmat)
{
  IS        iscol_local = NULL, isrow_d;
  PetscInt  csize;
  PetscInt  n, i, j, start, end;
  PetscBool sameRowDist = PETSC_FALSE, sameDist[2], tsameDist[2];
  MPI_Comm  comm;

  PetscFunctionBegin;
  /* If isrow has same processor distribution as mat,
     call MatCreateSubMatrix_MPIAIJ_SameRowDist() to avoid using a hash table with global size of iscol */
  if (call == MAT_REUSE_MATRIX) {
    PetscCall(PetscObjectQuery((PetscObject)*newmat, "isrow_d", (PetscObject *)&isrow_d));
    if (isrow_d) {
      sameRowDist  = PETSC_TRUE;
      tsameDist[1] = PETSC_TRUE; /* sameColDist */
    } else {
      PetscCall(PetscObjectQuery((PetscObject)*newmat, "SubIScol", (PetscObject *)&iscol_local));
      if (iscol_local) {
        sameRowDist  = PETSC_TRUE;
        tsameDist[1] = PETSC_FALSE; /* !sameColDist */
      }
    }
  } else {
    /* Check if isrow has same processor distribution as mat */
    sameDist[0] = PETSC_FALSE;
    PetscCall(ISGetLocalSize(isrow, &n));
    if (!n) {
      sameDist[0] = PETSC_TRUE;
    } else {
      PetscCall(ISGetMinMax(isrow, &i, &j));
      PetscCall(MatGetOwnershipRange(mat, &start, &end));
      if (i >= start && j < end) sameDist[0] = PETSC_TRUE;
    }

    /* Check if iscol has same processor distribution as mat */
    sameDist[1] = PETSC_FALSE;
    PetscCall(ISGetLocalSize(iscol, &n));
    if (!n) {
      sameDist[1] = PETSC_TRUE;
    } else {
      PetscCall(ISGetMinMax(iscol, &i, &j));
      PetscCall(MatGetOwnershipRangeColumn(mat, &start, &end));
      if (i >= start && j < end) sameDist[1] = PETSC_TRUE;
    }

    PetscCall(PetscObjectGetComm((PetscObject)mat, &comm));
    PetscCallMPI(MPIU_Allreduce(&sameDist, &tsameDist, 2, MPIU_BOOL, MPI_LAND, comm));
    sameRowDist = tsameDist[0];
  }

  if (sameRowDist) {
    if (tsameDist[1]) { /* sameRowDist & sameColDist */
      /* isrow and iscol have same processor distribution as mat */
      PetscCall(MatCreateSubMatrix_MPIAIJ_SameRowColDist(mat, isrow, iscol, call, newmat));
      PetscFunctionReturn(PETSC_SUCCESS);
    } else { /* sameRowDist */
      /* isrow has same processor distribution as mat */
      if (call == MAT_INITIAL_MATRIX) {
        PetscBool sorted;
        PetscCall(ISGetSeqIS_Private(mat, iscol, &iscol_local));
        PetscCall(ISGetLocalSize(iscol_local, &n)); /* local size of iscol_local = global columns of newmat */
        PetscCall(ISGetSize(iscol, &i));
        PetscCheck(n == i, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "n %" PetscInt_FMT " != size of iscol %" PetscInt_FMT, n, i);

        PetscCall(ISSorted(iscol_local, &sorted));
        if (sorted) {
          /* MatCreateSubMatrix_MPIAIJ_SameRowDist() requires iscol_local be sorted; it can have duplicate indices */
          PetscCall(MatCreateSubMatrix_MPIAIJ_SameRowDist(mat, isrow, iscol, iscol_local, MAT_INITIAL_MATRIX, newmat));
          PetscFunctionReturn(PETSC_SUCCESS);
        }
      } else { /* call == MAT_REUSE_MATRIX */
        IS iscol_sub;
        PetscCall(PetscObjectQuery((PetscObject)*newmat, "SubIScol", (PetscObject *)&iscol_sub));
        if (iscol_sub) {
          PetscCall(MatCreateSubMatrix_MPIAIJ_SameRowDist(mat, isrow, iscol, NULL, call, newmat));
          PetscFunctionReturn(PETSC_SUCCESS);
        }
      }
    }
  }

  /* General case: iscol -> iscol_local which has global size of iscol */
  if (call == MAT_REUSE_MATRIX) {
    PetscCall(PetscObjectQuery((PetscObject)*newmat, "ISAllGather", (PetscObject *)&iscol_local));
    PetscCheck(iscol_local, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Submatrix passed in was not used before, cannot reuse");
  } else {
    if (!iscol_local) PetscCall(ISGetSeqIS_Private(mat, iscol, &iscol_local));
  }

  PetscCall(ISGetLocalSize(iscol, &csize));
  PetscCall(MatCreateSubMatrix_MPIAIJ_nonscalable(mat, isrow, iscol_local, csize, call, newmat));

  if (call == MAT_INITIAL_MATRIX) {
    PetscCall(PetscObjectCompose((PetscObject)*newmat, "ISAllGather", (PetscObject)iscol_local));
    PetscCall(ISDestroy(&iscol_local));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatCreateMPIAIJWithSeqAIJ - creates a `MATMPIAIJ` matrix using `MATSEQAIJ` matrices that contain the "diagonal"
  and "off-diagonal" part of the matrix in CSR format.

  Collective

  Input Parameters:
+ comm   - MPI communicator
. A      - "diagonal" portion of matrix
. B      - "off-diagonal" portion of matrix, may have empty columns, will be destroyed by this routine
- garray - global index of `B` columns

  Output Parameter:
. mat - the matrix, with input `A` as its local diagonal matrix

  Level: advanced

  Notes:
  See `MatCreateAIJ()` for the definition of "diagonal" and "off-diagonal" portion of the matrix.

  `A` becomes part of output mat, `B` is destroyed by this routine. The user cannot use `A` and `B` anymore.

.seealso: [](ch_matrices), `Mat`, `MATMPIAIJ`, `MATSEQAIJ`, `MatCreateMPIAIJWithSplitArrays()`
@*/
PetscErrorCode MatCreateMPIAIJWithSeqAIJ(MPI_Comm comm, Mat A, Mat B, const PetscInt garray[], Mat *mat)
{
  Mat_MPIAIJ        *maij;
  Mat_SeqAIJ        *b  = (Mat_SeqAIJ *)B->data, *bnew;
  PetscInt          *oi = b->i, *oj = b->j, i, nz, col;
  const PetscScalar *oa;
  Mat                Bnew;
  PetscInt           m, n, N;
  MatType            mpi_mat_type;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm, mat));
  PetscCall(MatGetSize(A, &m, &n));
  PetscCheck(m == B->rmap->N, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Am %" PetscInt_FMT " != Bm %" PetscInt_FMT, m, B->rmap->N);
  PetscCheck(PetscAbs(A->rmap->bs) == PetscAbs(B->rmap->bs), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "A row bs %" PetscInt_FMT " != B row bs %" PetscInt_FMT, A->rmap->bs, B->rmap->bs);
  /* remove check below; When B is created using iscol_o from ISGetSeqIS_SameColDist_Private(), its bs may not be same as A */
  /* PetscCheck(A->cmap->bs == B->cmap->bs,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"A column bs %" PetscInt_FMT " != B column bs %" PetscInt_FMT,A->cmap->bs,B->cmap->bs); */

  /* Get global columns of mat */
  PetscCallMPI(MPIU_Allreduce(&n, &N, 1, MPIU_INT, MPI_SUM, comm));

  PetscCall(MatSetSizes(*mat, m, n, PETSC_DECIDE, N));
  /* Determine the type of MPI matrix that should be created from the type of matrix A, which holds the "diagonal" portion. */
  PetscCall(MatGetMPIMatType_Private(A, &mpi_mat_type));
  PetscCall(MatSetType(*mat, mpi_mat_type));

  if (A->rmap->bs > 1 || A->cmap->bs > 1) PetscCall(MatSetBlockSizes(*mat, A->rmap->bs, A->cmap->bs));
  maij = (Mat_MPIAIJ *)(*mat)->data;

  (*mat)->preallocated = PETSC_TRUE;

  PetscCall(PetscLayoutSetUp((*mat)->rmap));
  PetscCall(PetscLayoutSetUp((*mat)->cmap));

  /* Set A as diagonal portion of *mat */
  maij->A = A;

  nz = oi[m];
  for (i = 0; i < nz; i++) {
    col   = oj[i];
    oj[i] = garray[col];
  }

  /* Set Bnew as off-diagonal portion of *mat */
  PetscCall(MatSeqAIJGetArrayRead(B, &oa));
  PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, m, N, oi, oj, (PetscScalar *)oa, &Bnew));
  PetscCall(MatSeqAIJRestoreArrayRead(B, &oa));
  bnew        = (Mat_SeqAIJ *)Bnew->data;
  bnew->maxnz = b->maxnz; /* allocated nonzeros of B */
  maij->B     = Bnew;

  PetscCheck(B->rmap->N == Bnew->rmap->N, PETSC_COMM_SELF, PETSC_ERR_PLIB, "BN %" PetscInt_FMT " != BnewN %" PetscInt_FMT, B->rmap->N, Bnew->rmap->N);

  b->free_a  = PETSC_FALSE;
  b->free_ij = PETSC_FALSE;
  PetscCall(MatDestroy(&B));

  bnew->free_a  = PETSC_TRUE;
  bnew->free_ij = PETSC_TRUE;

  /* condense columns of maij->B */
  PetscCall(MatSetOption(*mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(*mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_FALSE));
  PetscCall(MatSetOption(*mat, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

extern PetscErrorCode MatCreateSubMatrices_MPIAIJ_SingleIS_Local(Mat, PetscInt, const IS[], const IS[], MatReuse, PetscBool, Mat *);

PetscErrorCode MatCreateSubMatrix_MPIAIJ_SameRowDist(Mat mat, IS isrow, IS iscol, IS iscol_local, MatReuse call, Mat *newmat)
{
  PetscInt        i, m, n, rstart, row, rend, nz, j, bs, cbs;
  PetscInt       *ii, *jj, nlocal, *dlens, *olens, dlen, olen, jend, mglobal;
  Mat_MPIAIJ     *a = (Mat_MPIAIJ *)mat->data;
  Mat             M, Msub, B = a->B;
  MatScalar      *aa;
  Mat_SeqAIJ     *aij;
  PetscInt       *garray = a->garray, *colsub, Ncols;
  PetscInt        count, Bn = B->cmap->N, cstart = mat->cmap->rstart, cend = mat->cmap->rend;
  IS              iscol_sub, iscmap;
  const PetscInt *is_idx, *cmap;
  PetscBool       allcolumns = PETSC_FALSE;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)mat, &comm));
  if (call == MAT_REUSE_MATRIX) {
    PetscCall(PetscObjectQuery((PetscObject)*newmat, "SubIScol", (PetscObject *)&iscol_sub));
    PetscCheck(iscol_sub, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "SubIScol passed in was not used before, cannot reuse");
    PetscCall(ISGetLocalSize(iscol_sub, &count));

    PetscCall(PetscObjectQuery((PetscObject)*newmat, "Subcmap", (PetscObject *)&iscmap));
    PetscCheck(iscmap, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Subcmap passed in was not used before, cannot reuse");

    PetscCall(PetscObjectQuery((PetscObject)*newmat, "SubMatrix", (PetscObject *)&Msub));
    PetscCheck(Msub, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Submatrix passed in was not used before, cannot reuse");

    PetscCall(MatCreateSubMatrices_MPIAIJ_SingleIS_Local(mat, 1, &isrow, &iscol_sub, MAT_REUSE_MATRIX, PETSC_FALSE, &Msub));

  } else { /* call == MAT_INITIAL_MATRIX) */
    PetscBool flg;

    PetscCall(ISGetLocalSize(iscol, &n));
    PetscCall(ISGetSize(iscol, &Ncols));

    /* (1) iscol -> nonscalable iscol_local */
    /* Check for special case: each processor gets entire matrix columns */
    PetscCall(ISIdentity(iscol_local, &flg));
    if (flg && n == mat->cmap->N) allcolumns = PETSC_TRUE;
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &allcolumns, 1, MPIU_BOOL, MPI_LAND, PetscObjectComm((PetscObject)mat)));
    if (allcolumns) {
      iscol_sub = iscol_local;
      PetscCall(PetscObjectReference((PetscObject)iscol_local));
      PetscCall(ISCreateStride(PETSC_COMM_SELF, n, 0, 1, &iscmap));

    } else {
      /* (2) iscol_local -> iscol_sub and iscmap. Implementation below requires iscol_local be sorted, it can have duplicate indices */
      PetscInt *idx, *cmap1, k;
      PetscCall(PetscMalloc1(Ncols, &idx));
      PetscCall(PetscMalloc1(Ncols, &cmap1));
      PetscCall(ISGetIndices(iscol_local, &is_idx));
      count = 0;
      k     = 0;
      for (i = 0; i < Ncols; i++) {
        j = is_idx[i];
        if (j >= cstart && j < cend) {
          /* diagonal part of mat */
          idx[count]     = j;
          cmap1[count++] = i; /* column index in submat */
        } else if (Bn) {
          /* off-diagonal part of mat */
          if (j == garray[k]) {
            idx[count]     = j;
            cmap1[count++] = i; /* column index in submat */
          } else if (j > garray[k]) {
            while (j > garray[k] && k < Bn - 1) k++;
            if (j == garray[k]) {
              idx[count]     = j;
              cmap1[count++] = i; /* column index in submat */
            }
          }
        }
      }
      PetscCall(ISRestoreIndices(iscol_local, &is_idx));

      PetscCall(ISCreateGeneral(PETSC_COMM_SELF, count, idx, PETSC_OWN_POINTER, &iscol_sub));
      PetscCall(ISGetBlockSize(iscol, &cbs));
      PetscCall(ISSetBlockSize(iscol_sub, cbs));

      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)iscol_local), count, cmap1, PETSC_OWN_POINTER, &iscmap));
    }

    /* (3) Create sequential Msub */
    PetscCall(MatCreateSubMatrices_MPIAIJ_SingleIS_Local(mat, 1, &isrow, &iscol_sub, MAT_INITIAL_MATRIX, allcolumns, &Msub));
  }

  PetscCall(ISGetLocalSize(iscol_sub, &count));
  aij = (Mat_SeqAIJ *)Msub->data;
  ii  = aij->i;
  PetscCall(ISGetIndices(iscmap, &cmap));

  /*
      m - number of local rows
      Ncols - number of columns (same on all processors)
      rstart - first row in new global matrix generated
  */
  PetscCall(MatGetSize(Msub, &m, NULL));

  if (call == MAT_INITIAL_MATRIX) {
    /* (4) Create parallel newmat */
    PetscMPIInt rank, size;
    PetscInt    csize;

    PetscCallMPI(MPI_Comm_size(comm, &size));
    PetscCallMPI(MPI_Comm_rank(comm, &rank));

    /*
        Determine the number of non-zeros in the diagonal and off-diagonal
        portions of the matrix in order to do correct preallocation
    */

    /* first get start and end of "diagonal" columns */
    PetscCall(ISGetLocalSize(iscol, &csize));
    if (csize == PETSC_DECIDE) {
      PetscCall(ISGetSize(isrow, &mglobal));
      if (mglobal == Ncols) { /* square matrix */
        nlocal = m;
      } else {
        nlocal = Ncols / size + ((Ncols % size) > rank);
      }
    } else {
      nlocal = csize;
    }
    PetscCallMPI(MPI_Scan(&nlocal, &rend, 1, MPIU_INT, MPI_SUM, comm));
    rstart = rend - nlocal;
    PetscCheck(rank != size - 1 || rend == Ncols, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local column sizes %" PetscInt_FMT " do not add up to total number of columns %" PetscInt_FMT, rend, Ncols);

    /* next, compute all the lengths */
    jj = aij->j;
    PetscCall(PetscMalloc1(2 * m + 1, &dlens));
    olens = dlens + m;
    for (i = 0; i < m; i++) {
      jend = ii[i + 1] - ii[i];
      olen = 0;
      dlen = 0;
      for (j = 0; j < jend; j++) {
        if (cmap[*jj] < rstart || cmap[*jj] >= rend) olen++;
        else dlen++;
        jj++;
      }
      olens[i] = olen;
      dlens[i] = dlen;
    }

    PetscCall(ISGetBlockSize(isrow, &bs));
    PetscCall(ISGetBlockSize(iscol, &cbs));

    PetscCall(MatCreate(comm, &M));
    PetscCall(MatSetSizes(M, m, nlocal, PETSC_DECIDE, Ncols));
    PetscCall(MatSetBlockSizes(M, bs, cbs));
    PetscCall(MatSetType(M, ((PetscObject)mat)->type_name));
    PetscCall(MatMPIAIJSetPreallocation(M, 0, dlens, 0, olens));
    PetscCall(PetscFree(dlens));

  } else { /* call == MAT_REUSE_MATRIX */
    M = *newmat;
    PetscCall(MatGetLocalSize(M, &i, NULL));
    PetscCheck(i == m, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Previous matrix must be same size/layout as request");
    PetscCall(MatZeroEntries(M));
    /*
         The next two lines are needed so we may call MatSetValues_MPIAIJ() below directly,
       rather than the slower MatSetValues().
    */
    M->was_assembled = PETSC_TRUE;
    M->assembled     = PETSC_FALSE;
  }

  /* (5) Set values of Msub to *newmat */
  PetscCall(PetscMalloc1(count, &colsub));
  PetscCall(MatGetOwnershipRange(M, &rstart, NULL));

  jj = aij->j;
  PetscCall(MatSeqAIJGetArrayRead(Msub, (const PetscScalar **)&aa));
  for (i = 0; i < m; i++) {
    row = rstart + i;
    nz  = ii[i + 1] - ii[i];
    for (j = 0; j < nz; j++) colsub[j] = cmap[jj[j]];
    PetscCall(MatSetValues_MPIAIJ(M, 1, &row, nz, colsub, aa, INSERT_VALUES));
    jj += nz;
    aa += nz;
  }
  PetscCall(MatSeqAIJRestoreArrayRead(Msub, (const PetscScalar **)&aa));
  PetscCall(ISRestoreIndices(iscmap, &cmap));

  PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));

  PetscCall(PetscFree(colsub));

  /* save Msub, iscol_sub and iscmap used in processor for next request */
  if (call == MAT_INITIAL_MATRIX) {
    *newmat = M;
    PetscCall(PetscObjectCompose((PetscObject)*newmat, "SubMatrix", (PetscObject)Msub));
    PetscCall(MatDestroy(&Msub));

    PetscCall(PetscObjectCompose((PetscObject)*newmat, "SubIScol", (PetscObject)iscol_sub));
    PetscCall(ISDestroy(&iscol_sub));

    PetscCall(PetscObjectCompose((PetscObject)*newmat, "Subcmap", (PetscObject)iscmap));
    PetscCall(ISDestroy(&iscmap));

    if (iscol_local) {
      PetscCall(PetscObjectCompose((PetscObject)*newmat, "ISAllGather", (PetscObject)iscol_local));
      PetscCall(ISDestroy(&iscol_local));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    Not great since it makes two copies of the submatrix, first an SeqAIJ
  in local and then by concatenating the local matrices the end result.
  Writing it directly would be much like MatCreateSubMatrices_MPIAIJ()

  This requires a sequential iscol with all indices.
*/
PetscErrorCode MatCreateSubMatrix_MPIAIJ_nonscalable(Mat mat, IS isrow, IS iscol, PetscInt csize, MatReuse call, Mat *newmat)
{
  PetscMPIInt rank, size;
  PetscInt    i, m, n, rstart, row, rend, nz, *cwork, j, bs, cbs;
  PetscInt   *ii, *jj, nlocal, *dlens, *olens, dlen, olen, jend, mglobal;
  Mat         M, Mreuse;
  MatScalar  *aa, *vwork;
  MPI_Comm    comm;
  Mat_SeqAIJ *aij;
  PetscBool   colflag, allcolumns = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)mat, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  /* Check for special case: each processor gets entire matrix columns */
  PetscCall(ISIdentity(iscol, &colflag));
  PetscCall(ISGetLocalSize(iscol, &n));
  if (colflag && n == mat->cmap->N) allcolumns = PETSC_TRUE;
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &allcolumns, 1, MPIU_BOOL, MPI_LAND, PetscObjectComm((PetscObject)mat)));

  if (call == MAT_REUSE_MATRIX) {
    PetscCall(PetscObjectQuery((PetscObject)*newmat, "SubMatrix", (PetscObject *)&Mreuse));
    PetscCheck(Mreuse, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Submatrix passed in was not used before, cannot reuse");
    PetscCall(MatCreateSubMatrices_MPIAIJ_SingleIS_Local(mat, 1, &isrow, &iscol, MAT_REUSE_MATRIX, allcolumns, &Mreuse));
  } else {
    PetscCall(MatCreateSubMatrices_MPIAIJ_SingleIS_Local(mat, 1, &isrow, &iscol, MAT_INITIAL_MATRIX, allcolumns, &Mreuse));
  }

  /*
      m - number of local rows
      n - number of columns (same on all processors)
      rstart - first row in new global matrix generated
  */
  PetscCall(MatGetSize(Mreuse, &m, &n));
  PetscCall(MatGetBlockSizes(Mreuse, &bs, &cbs));
  if (call == MAT_INITIAL_MATRIX) {
    aij = (Mat_SeqAIJ *)Mreuse->data;
    ii  = aij->i;
    jj  = aij->j;

    /*
        Determine the number of non-zeros in the diagonal and off-diagonal
        portions of the matrix in order to do correct preallocation
    */

    /* first get start and end of "diagonal" columns */
    if (csize == PETSC_DECIDE) {
      PetscCall(ISGetSize(isrow, &mglobal));
      if (mglobal == n) { /* square matrix */
        nlocal = m;
      } else {
        nlocal = n / size + ((n % size) > rank);
      }
    } else {
      nlocal = csize;
    }
    PetscCallMPI(MPI_Scan(&nlocal, &rend, 1, MPIU_INT, MPI_SUM, comm));
    rstart = rend - nlocal;
    PetscCheck(rank != size - 1 || rend == n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Local column sizes %" PetscInt_FMT " do not add up to total number of columns %" PetscInt_FMT, rend, n);

    /* next, compute all the lengths */
    PetscCall(PetscMalloc1(2 * m + 1, &dlens));
    olens = dlens + m;
    for (i = 0; i < m; i++) {
      jend = ii[i + 1] - ii[i];
      olen = 0;
      dlen = 0;
      for (j = 0; j < jend; j++) {
        if (*jj < rstart || *jj >= rend) olen++;
        else dlen++;
        jj++;
      }
      olens[i] = olen;
      dlens[i] = dlen;
    }
    PetscCall(MatCreate(comm, &M));
    PetscCall(MatSetSizes(M, m, nlocal, PETSC_DECIDE, n));
    PetscCall(MatSetBlockSizes(M, bs, cbs));
    PetscCall(MatSetType(M, ((PetscObject)mat)->type_name));
    PetscCall(MatMPIAIJSetPreallocation(M, 0, dlens, 0, olens));
    PetscCall(PetscFree(dlens));
  } else {
    PetscInt ml, nl;

    M = *newmat;
    PetscCall(MatGetLocalSize(M, &ml, &nl));
    PetscCheck(ml == m, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Previous matrix must be same size/layout as request");
    PetscCall(MatZeroEntries(M));
    /*
         The next two lines are needed so we may call MatSetValues_MPIAIJ() below directly,
       rather than the slower MatSetValues().
    */
    M->was_assembled = PETSC_TRUE;
    M->assembled     = PETSC_FALSE;
  }
  PetscCall(MatGetOwnershipRange(M, &rstart, &rend));
  aij = (Mat_SeqAIJ *)Mreuse->data;
  ii  = aij->i;
  jj  = aij->j;

  /* trigger copy to CPU if needed */
  PetscCall(MatSeqAIJGetArrayRead(Mreuse, (const PetscScalar **)&aa));
  for (i = 0; i < m; i++) {
    row   = rstart + i;
    nz    = ii[i + 1] - ii[i];
    cwork = jj;
    jj    = PetscSafePointerPlusOffset(jj, nz);
    vwork = aa;
    aa    = PetscSafePointerPlusOffset(aa, nz);
    PetscCall(MatSetValues_MPIAIJ(M, 1, &row, nz, cwork, vwork, INSERT_VALUES));
  }
  PetscCall(MatSeqAIJRestoreArrayRead(Mreuse, (const PetscScalar **)&aa));

  PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));
  *newmat = M;

  /* save submatrix used in processor for next request */
  if (call == MAT_INITIAL_MATRIX) {
    PetscCall(PetscObjectCompose((PetscObject)M, "SubMatrix", (PetscObject)Mreuse));
    PetscCall(MatDestroy(&Mreuse));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMPIAIJSetPreallocationCSR_MPIAIJ(Mat B, const PetscInt Ii[], const PetscInt J[], const PetscScalar v[])
{
  PetscInt        m, cstart, cend, j, nnz, i, d, *ld;
  PetscInt       *d_nnz, *o_nnz, nnz_max = 0, rstart, ii, irstart;
  const PetscInt *JJ;
  PetscBool       nooffprocentries;
  Mat_MPIAIJ     *Aij = (Mat_MPIAIJ *)B->data;

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));
  m       = B->rmap->n;
  cstart  = B->cmap->rstart;
  cend    = B->cmap->rend;
  rstart  = B->rmap->rstart;
  irstart = Ii[0];

  PetscCall(PetscCalloc2(m, &d_nnz, m, &o_nnz));

  if (PetscDefined(USE_DEBUG)) {
    for (i = 0; i < m; i++) {
      nnz = Ii[i + 1] - Ii[i];
      JJ  = PetscSafePointerPlusOffset(J, Ii[i] - irstart);
      PetscCheck(nnz >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Local row %" PetscInt_FMT " has a negative %" PetscInt_FMT " number of columns", i, nnz);
      PetscCheck(!nnz || !(JJ[0] < 0), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Row %" PetscInt_FMT " starts with negative column index %" PetscInt_FMT, i, JJ[0]);
      PetscCheck(!nnz || !(JJ[nnz - 1] >= B->cmap->N), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Row %" PetscInt_FMT " ends with too large a column index %" PetscInt_FMT " (max allowed %" PetscInt_FMT ")", i, JJ[nnz - 1], B->cmap->N);
    }
  }

  for (i = 0; i < m; i++) {
    nnz     = Ii[i + 1] - Ii[i];
    JJ      = PetscSafePointerPlusOffset(J, Ii[i] - irstart);
    nnz_max = PetscMax(nnz_max, nnz);
    d       = 0;
    for (j = 0; j < nnz; j++) {
      if (cstart <= JJ[j] && JJ[j] < cend) d++;
    }
    d_nnz[i] = d;
    o_nnz[i] = nnz - d;
  }
  PetscCall(MatMPIAIJSetPreallocation(B, 0, d_nnz, 0, o_nnz));
  PetscCall(PetscFree2(d_nnz, o_nnz));

  for (i = 0; i < m; i++) {
    ii = i + rstart;
    PetscCall(MatSetValues_MPIAIJ(B, 1, &ii, Ii[i + 1] - Ii[i], PetscSafePointerPlusOffset(J, Ii[i] - irstart), PetscSafePointerPlusOffset(v, Ii[i] - irstart), INSERT_VALUES));
  }
  nooffprocentries    = B->nooffprocentries;
  B->nooffprocentries = PETSC_TRUE;
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  B->nooffprocentries = nooffprocentries;

  /* count number of entries below block diagonal */
  PetscCall(PetscFree(Aij->ld));
  PetscCall(PetscCalloc1(m, &ld));
  Aij->ld = ld;
  for (i = 0; i < m; i++) {
    nnz = Ii[i + 1] - Ii[i];
    j   = 0;
    while (j < nnz && J[j] < cstart) j++;
    ld[i] = j;
    if (J) J += nnz;
  }

  PetscCall(MatSetOption(B, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMPIAIJSetPreallocationCSR - Allocates memory for a sparse parallel matrix in `MATAIJ` format
  (the default parallel PETSc format).

  Collective

  Input Parameters:
+ B - the matrix
. i - the indices into `j` for the start of each local row (indices start with zero)
. j - the column indices for each local row (indices start with zero)
- v - optional values in the matrix

  Level: developer

  Notes:
  The `i`, `j`, and `v` arrays ARE copied by this routine into the internal format used by PETSc;
  thus you CANNOT change the matrix entries by changing the values of `v` after you have
  called this routine. Use `MatCreateMPIAIJWithSplitArrays()` to avoid needing to copy the arrays.

  The `i` and `j` indices are 0 based, and `i` indices are indices corresponding to the local `j` array.

  A convenience routine for this functionality is `MatCreateMPIAIJWithArrays()`.

  You can update the matrix with new numerical values using `MatUpdateMPIAIJWithArrays()` after this call if the column indices in `j` are sorted.

  If you do **not** use `MatUpdateMPIAIJWithArrays()`, the column indices in `j` do not need to be sorted. If you will use
  `MatUpdateMPIAIJWithArrays()`, the column indices **must** be sorted.

  The format which is used for the sparse matrix input, is equivalent to a
  row-major ordering.. i.e for the following matrix, the input data expected is
  as shown
.vb
        1 0 0
        2 0 3     P0
       -------
        4 5 6     P1

     Process0 [P0] rows_owned=[0,1]
        i =  {0,1,3}  [size = nrow+1  = 2+1]
        j =  {0,0,2}  [size = 3]
        v =  {1,2,3}  [size = 3]

     Process1 [P1] rows_owned=[2]
        i =  {0,3}    [size = nrow+1  = 1+1]
        j =  {0,1,2}  [size = 3]
        v =  {4,5,6}  [size = 3]
.ve

.seealso: [](ch_matrices), `Mat`, `MATMPIAIJ`, `MatCreate()`, `MatCreateSeqAIJ()`, `MatSetValues()`, `MatMPIAIJSetPreallocation()`, `MatCreateAIJ()`,
          `MatCreateSeqAIJWithArrays()`, `MatCreateMPIAIJWithSplitArrays()`, `MatCreateMPIAIJWithArrays()`, `MatSetPreallocationCOO()`, `MatSetValuesCOO()`
@*/
PetscErrorCode MatMPIAIJSetPreallocationCSR(Mat B, const PetscInt i[], const PetscInt j[], const PetscScalar v[])
{
  PetscFunctionBegin;
  PetscTryMethod(B, "MatMPIAIJSetPreallocationCSR_C", (Mat, const PetscInt[], const PetscInt[], const PetscScalar[]), (B, i, j, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMPIAIJSetPreallocation - Preallocates memory for a sparse parallel matrix in `MATMPIAIJ` format
  (the default parallel PETSc format).  For good matrix assembly performance
  the user should preallocate the matrix storage by setting the parameters
  `d_nz` (or `d_nnz`) and `o_nz` (or `o_nnz`).

  Collective

  Input Parameters:
+ B     - the matrix
. d_nz  - number of nonzeros per row in DIAGONAL portion of local submatrix
           (same value is used for all local rows)
. d_nnz - array containing the number of nonzeros in the various rows of the
           DIAGONAL portion of the local submatrix (possibly different for each row)
           or `NULL` (`PETSC_NULL_INTEGER` in Fortran), if `d_nz` is used to specify the nonzero structure.
           The size of this array is equal to the number of local rows, i.e 'm'.
           For matrices that will be factored, you must leave room for (and set)
           the diagonal entry even if it is zero.
. o_nz  - number of nonzeros per row in the OFF-DIAGONAL portion of local
           submatrix (same value is used for all local rows).
- o_nnz - array containing the number of nonzeros in the various rows of the
           OFF-DIAGONAL portion of the local submatrix (possibly different for
           each row) or `NULL` (`PETSC_NULL_INTEGER` in Fortran), if `o_nz` is used to specify the nonzero
           structure. The size of this array is equal to the number
           of local rows, i.e 'm'.

  Example Usage:
  Consider the following 8x8 matrix with 34 non-zero values, that is
  assembled across 3 processors. Lets assume that proc0 owns 3 rows,
  proc1 owns 3 rows, proc2 owns 2 rows. This division can be shown
  as follows

.vb
            1  2  0  |  0  3  0  |  0  4
    Proc0   0  5  6  |  7  0  0  |  8  0
            9  0 10  | 11  0  0  | 12  0
    -------------------------------------
           13  0 14  | 15 16 17  |  0  0
    Proc1   0 18  0  | 19 20 21  |  0  0
            0  0  0  | 22 23  0  | 24  0
    -------------------------------------
    Proc2  25 26 27  |  0  0 28  | 29  0
           30  0  0  | 31 32 33  |  0 34
.ve

  This can be represented as a collection of submatrices as
.vb
      A B C
      D E F
      G H I
.ve

  Where the submatrices A,B,C are owned by proc0, D,E,F are
  owned by proc1, G,H,I are owned by proc2.

  The 'm' parameters for proc0,proc1,proc2 are 3,3,2 respectively.
  The 'n' parameters for proc0,proc1,proc2 are 3,3,2 respectively.
  The 'M','N' parameters are 8,8, and have the same values on all procs.

  The DIAGONAL submatrices corresponding to proc0,proc1,proc2 are
  submatrices [A], [E], [I] respectively. The OFF-DIAGONAL submatrices
  corresponding to proc0,proc1,proc2 are [BC], [DF], [GH] respectively.
  Internally, each processor stores the DIAGONAL part, and the OFF-DIAGONAL
  part as `MATSEQAIJ` matrices. For example, proc1 will store [E] as a `MATSEQAIJ`
  matrix, ans [DF] as another `MATSEQAIJ` matrix.

  When `d_nz`, `o_nz` parameters are specified, `d_nz` storage elements are
  allocated for every row of the local diagonal submatrix, and `o_nz`
  storage locations are allocated for every row of the OFF-DIAGONAL submat.
  One way to choose `d_nz` and `o_nz` is to use the max nonzerors per local
  rows for each of the local DIAGONAL, and the OFF-DIAGONAL submatrices.
  In this case, the values of `d_nz`, `o_nz` are
.vb
     proc0  dnz = 2, o_nz = 2
     proc1  dnz = 3, o_nz = 2
     proc2  dnz = 1, o_nz = 4
.ve
  We are allocating `m`*(`d_nz`+`o_nz`) storage locations for every proc. This
  translates to 3*(2+2)=12 for proc0, 3*(3+2)=15 for proc1, 2*(1+4)=10
  for proc3. i.e we are using 12+15+10=37 storage locations to store
  34 values.

  When `d_nnz`, `o_nnz` parameters are specified, the storage is specified
  for every row, corresponding to both DIAGONAL and OFF-DIAGONAL submatrices.
  In the above case the values for `d_nnz`, `o_nnz` are
.vb
     proc0 d_nnz = [2,2,2] and o_nnz = [2,2,2]
     proc1 d_nnz = [3,3,2] and o_nnz = [2,1,1]
     proc2 d_nnz = [1,1]   and o_nnz = [4,4]
.ve
  Here the space allocated is sum of all the above values i.e 34, and
  hence pre-allocation is perfect.

  Level: intermediate

  Notes:
  If the *_nnz parameter is given then the *_nz parameter is ignored

  The `MATAIJ` format, also called compressed row storage (CSR), is compatible with standard Fortran
  storage.  The stored row and column indices begin with zero.
  See [Sparse Matrices](sec_matsparse) for details.

  The parallel matrix is partitioned such that the first m0 rows belong to
  process 0, the next m1 rows belong to process 1, the next m2 rows belong
  to process 2 etc.. where m0,m1,m2... are the input parameter 'm'.

  The DIAGONAL portion of the local submatrix of a processor can be defined
  as the submatrix which is obtained by extraction the part corresponding to
  the rows r1-r2 and columns c1-c2 of the global matrix, where r1 is the
  first row that belongs to the processor, r2 is the last row belonging to
  the this processor, and c1-c2 is range of indices of the local part of a
  vector suitable for applying the matrix to.  This is an mxn matrix.  In the
  common case of a square matrix, the row and column ranges are the same and
  the DIAGONAL part is also square. The remaining portion of the local
  submatrix (mxN) constitute the OFF-DIAGONAL portion.

  If `o_nnz` and `d_nnz` are specified, then `o_nz` and `d_nz` are ignored.

  You can call `MatGetInfo()` to get information on how effective the preallocation was;
  for example the fields mallocs,nz_allocated,nz_used,nz_unneeded;
  You can also run with the option `-info` and look for messages with the string
  malloc in them to see if additional memory allocation was needed.

.seealso: [](ch_matrices), `Mat`, [Sparse Matrices](sec_matsparse), `MATMPIAIJ`, `MATAIJ`, `MatCreate()`, `MatCreateSeqAIJ()`, `MatSetValues()`, `MatCreateAIJ()`, `MatMPIAIJSetPreallocationCSR()`,
          `MatGetInfo()`, `PetscSplitOwnership()`, `MatSetPreallocationCOO()`, `MatSetValuesCOO()`
@*/
PetscErrorCode MatMPIAIJSetPreallocation(Mat B, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B, MAT_CLASSID, 1);
  PetscValidType(B, 1);
  PetscTryMethod(B, "MatMPIAIJSetPreallocation_C", (Mat, PetscInt, const PetscInt[], PetscInt, const PetscInt[]), (B, d_nz, d_nnz, o_nz, o_nnz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateMPIAIJWithArrays - creates a `MATMPIAIJ` matrix using arrays that contain in standard
  CSR format for the local rows.

  Collective

  Input Parameters:
+ comm - MPI communicator
. m    - number of local rows (Cannot be `PETSC_DECIDE`)
. n    - This value should be the same as the local size used in creating the
         x vector for the matrix-vector product $ y = Ax$. (or `PETSC_DECIDE` to have
         calculated if `N` is given) For square matrices n is almost always `m`.
. M    - number of global rows (or `PETSC_DETERMINE` to have calculated if `m` is given)
. N    - number of global columns (or `PETSC_DETERMINE` to have calculated if `n` is given)
. i    - row indices (of length m+1); that is i[0] = 0, i[row] = i[row-1] + number of elements in that row of the matrix
. j    - global column indices
- a    - optional matrix values

  Output Parameter:
. mat - the matrix

  Level: intermediate

  Notes:
  The `i`, `j`, and `a` arrays ARE copied by this routine into the internal format used by PETSc;
  thus you CANNOT change the matrix entries by changing the values of `a[]` after you have
  called this routine. Use `MatCreateMPIAIJWithSplitArrays()` to avoid needing to copy the arrays.

  The `i` and `j` indices are 0 based, and `i` indices are indices corresponding to the local `j` array.

  Once you have created the matrix you can update it with new numerical values using `MatUpdateMPIAIJWithArray()`

  If you do **not** use `MatUpdateMPIAIJWithArray()`, the column indices in `j` do not need to be sorted. If you will use
  `MatUpdateMPIAIJWithArrays()`, the column indices **must** be sorted.

  The format which is used for the sparse matrix input, is equivalent to a
  row-major ordering, i.e., for the following matrix, the input data expected is
  as shown
.vb
        1 0 0
        2 0 3     P0
       -------
        4 5 6     P1

     Process0 [P0] rows_owned=[0,1]
        i =  {0,1,3}  [size = nrow+1  = 2+1]
        j =  {0,0,2}  [size = 3]
        v =  {1,2,3}  [size = 3]

     Process1 [P1] rows_owned=[2]
        i =  {0,3}    [size = nrow+1  = 1+1]
        j =  {0,1,2}  [size = 3]
        v =  {4,5,6}  [size = 3]
.ve

.seealso: [](ch_matrices), `Mat`, `MatCreate()`, `MatCreateSeqAIJ()`, `MatSetValues()`, `MatMPIAIJSetPreallocation()`, `MatMPIAIJSetPreallocationCSR()`,
          `MATMPIAIJ`, `MatCreateAIJ()`, `MatCreateMPIAIJWithSplitArrays()`, `MatUpdateMPIAIJWithArray()`, `MatSetPreallocationCOO()`, `MatSetValuesCOO()`
@*/
PetscErrorCode MatCreateMPIAIJWithArrays(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, const PetscInt i[], const PetscInt j[], const PetscScalar a[], Mat *mat)
{
  PetscFunctionBegin;
  PetscCheck(!i || !i[0], PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "i (row indices) must start with 0");
  PetscCheck(m >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "local number of rows (m) cannot be PETSC_DECIDE, or negative");
  PetscCall(MatCreate(comm, mat));
  PetscCall(MatSetSizes(*mat, m, n, M, N));
  /* PetscCall(MatSetBlockSizes(M,bs,cbs)); */
  PetscCall(MatSetType(*mat, MATMPIAIJ));
  PetscCall(MatMPIAIJSetPreallocationCSR(*mat, i, j, a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatUpdateMPIAIJWithArrays - updates a `MATMPIAIJ` matrix using arrays that contain in standard
  CSR format for the local rows. Only the numerical values are updated the other arrays must be identical to what was passed
  from `MatCreateMPIAIJWithArrays()`

  Deprecated: Use `MatUpdateMPIAIJWithArray()`

  Collective

  Input Parameters:
+ mat - the matrix
. m   - number of local rows (Cannot be `PETSC_DECIDE`)
. n   - This value should be the same as the local size used in creating the
       x vector for the matrix-vector product y = Ax. (or `PETSC_DECIDE` to have
       calculated if N is given) For square matrices n is almost always m.
. M   - number of global rows (or `PETSC_DETERMINE` to have calculated if m is given)
. N   - number of global columns (or `PETSC_DETERMINE` to have calculated if n is given)
. Ii  - row indices; that is Ii[0] = 0, Ii[row] = Ii[row-1] + number of elements in that row of the matrix
. J   - column indices
- v   - matrix values

  Level: deprecated

.seealso: [](ch_matrices), `Mat`, `MATMPIAIJ`, `MatCreate()`, `MatCreateSeqAIJ()`, `MatSetValues()`, `MatMPIAIJSetPreallocation()`, `MatMPIAIJSetPreallocationCSR()`,
          `MatCreateAIJ()`, `MatCreateMPIAIJWithSplitArrays()`, `MatUpdateMPIAIJWithArray()`, `MatSetPreallocationCOO()`, `MatSetValuesCOO()`
@*/
PetscErrorCode MatUpdateMPIAIJWithArrays(Mat mat, PetscInt m, PetscInt n, PetscInt M, PetscInt N, const PetscInt Ii[], const PetscInt J[], const PetscScalar v[])
{
  PetscInt        nnz, i;
  PetscBool       nooffprocentries;
  Mat_MPIAIJ     *Aij = (Mat_MPIAIJ *)mat->data;
  Mat_SeqAIJ     *Ad  = (Mat_SeqAIJ *)Aij->A->data;
  PetscScalar    *ad, *ao;
  PetscInt        ldi, Iii, md;
  const PetscInt *Adi = Ad->i;
  PetscInt       *ld  = Aij->ld;

  PetscFunctionBegin;
  PetscCheck(Ii[0] == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "i (row indices) must start with 0");
  PetscCheck(m >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "local number of rows (m) cannot be PETSC_DECIDE, or negative");
  PetscCheck(m == mat->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Local number of rows cannot change from call to MatUpdateMPIAIJWithArrays()");
  PetscCheck(n == mat->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Local number of columns cannot change from call to MatUpdateMPIAIJWithArrays()");

  PetscCall(MatSeqAIJGetArrayWrite(Aij->A, &ad));
  PetscCall(MatSeqAIJGetArrayWrite(Aij->B, &ao));

  for (i = 0; i < m; i++) {
    if (PetscDefined(USE_DEBUG)) {
      for (PetscInt j = Ii[i] + 1; j < Ii[i + 1]; ++j) {
        PetscCheck(J[j] >= J[j - 1], PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Column entry number %" PetscInt_FMT " (actual column %" PetscInt_FMT ") in row %" PetscInt_FMT " is not sorted", j - Ii[i], J[j], i);
        PetscCheck(J[j] != J[j - 1], PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Column entry number %" PetscInt_FMT " (actual column %" PetscInt_FMT ") in row %" PetscInt_FMT " is identical to previous entry", j - Ii[i], J[j], i);
      }
    }
    nnz = Ii[i + 1] - Ii[i];
    Iii = Ii[i];
    ldi = ld[i];
    md  = Adi[i + 1] - Adi[i];
    PetscCall(PetscArraycpy(ao, v + Iii, ldi));
    PetscCall(PetscArraycpy(ad, v + Iii + ldi, md));
    PetscCall(PetscArraycpy(ao + ldi, v + Iii + ldi + md, nnz - ldi - md));
    ad += md;
    ao += nnz - md;
  }
  nooffprocentries      = mat->nooffprocentries;
  mat->nooffprocentries = PETSC_TRUE;
  PetscCall(MatSeqAIJRestoreArrayWrite(Aij->A, &ad));
  PetscCall(MatSeqAIJRestoreArrayWrite(Aij->B, &ao));
  PetscCall(PetscObjectStateIncrease((PetscObject)Aij->A));
  PetscCall(PetscObjectStateIncrease((PetscObject)Aij->B));
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  mat->nooffprocentries = nooffprocentries;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatUpdateMPIAIJWithArray - updates an `MATMPIAIJ` matrix using an array that contains the nonzero values

  Collective

  Input Parameters:
+ mat - the matrix
- v   - matrix values, stored by row

  Level: intermediate

  Notes:
  The matrix must have been obtained with `MatCreateMPIAIJWithArrays()` or `MatMPIAIJSetPreallocationCSR()`

  The column indices in the call to `MatCreateMPIAIJWithArrays()` or `MatMPIAIJSetPreallocationCSR()` must have been sorted for this call to work correctly

.seealso: [](ch_matrices), `Mat`, `MatCreate()`, `MatCreateSeqAIJ()`, `MatSetValues()`, `MatMPIAIJSetPreallocation()`, `MatMPIAIJSetPreallocationCSR()`,
          `MATMPIAIJ`, `MatCreateAIJ()`, `MatCreateMPIAIJWithSplitArrays()`, `MatUpdateMPIAIJWithArrays()`, `MatSetPreallocationCOO()`, `MatSetValuesCOO()`
@*/
PetscErrorCode MatUpdateMPIAIJWithArray(Mat mat, const PetscScalar v[])
{
  PetscInt        nnz, i, m;
  PetscBool       nooffprocentries;
  Mat_MPIAIJ     *Aij = (Mat_MPIAIJ *)mat->data;
  Mat_SeqAIJ     *Ad  = (Mat_SeqAIJ *)Aij->A->data;
  Mat_SeqAIJ     *Ao  = (Mat_SeqAIJ *)Aij->B->data;
  PetscScalar    *ad, *ao;
  const PetscInt *Adi = Ad->i, *Adj = Ao->i;
  PetscInt        ldi, Iii, md;
  PetscInt       *ld = Aij->ld;

  PetscFunctionBegin;
  m = mat->rmap->n;

  PetscCall(MatSeqAIJGetArrayWrite(Aij->A, &ad));
  PetscCall(MatSeqAIJGetArrayWrite(Aij->B, &ao));
  Iii = 0;
  for (i = 0; i < m; i++) {
    nnz = Adi[i + 1] - Adi[i] + Adj[i + 1] - Adj[i];
    ldi = ld[i];
    md  = Adi[i + 1] - Adi[i];
    PetscCall(PetscArraycpy(ad, v + Iii + ldi, md));
    ad += md;
    if (ao) {
      PetscCall(PetscArraycpy(ao, v + Iii, ldi));
      PetscCall(PetscArraycpy(ao + ldi, v + Iii + ldi + md, nnz - ldi - md));
      ao += nnz - md;
    }
    Iii += nnz;
  }
  nooffprocentries      = mat->nooffprocentries;
  mat->nooffprocentries = PETSC_TRUE;
  PetscCall(MatSeqAIJRestoreArrayWrite(Aij->A, &ad));
  PetscCall(MatSeqAIJRestoreArrayWrite(Aij->B, &ao));
  PetscCall(PetscObjectStateIncrease((PetscObject)Aij->A));
  PetscCall(PetscObjectStateIncrease((PetscObject)Aij->B));
  PetscCall(PetscObjectStateIncrease((PetscObject)mat));
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  mat->nooffprocentries = nooffprocentries;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateAIJ - Creates a sparse parallel matrix in `MATAIJ` format
  (the default parallel PETSc format).  For good matrix assembly performance
  the user should preallocate the matrix storage by setting the parameters
  `d_nz` (or `d_nnz`) and `o_nz` (or `o_nnz`).

  Collective

  Input Parameters:
+ comm  - MPI communicator
. m     - number of local rows (or `PETSC_DECIDE` to have calculated if M is given)
          This value should be the same as the local size used in creating the
          y vector for the matrix-vector product y = Ax.
. n     - This value should be the same as the local size used in creating the
          x vector for the matrix-vector product y = Ax. (or `PETSC_DECIDE` to have
          calculated if N is given) For square matrices n is almost always m.
. M     - number of global rows (or `PETSC_DETERMINE` to have calculated if m is given)
. N     - number of global columns (or `PETSC_DETERMINE` to have calculated if n is given)
. d_nz  - number of nonzeros per row in DIAGONAL portion of local submatrix
          (same value is used for all local rows)
. d_nnz - array containing the number of nonzeros in the various rows of the
          DIAGONAL portion of the local submatrix (possibly different for each row)
          or `NULL`, if `d_nz` is used to specify the nonzero structure.
          The size of this array is equal to the number of local rows, i.e 'm'.
. o_nz  - number of nonzeros per row in the OFF-DIAGONAL portion of local
          submatrix (same value is used for all local rows).
- o_nnz - array containing the number of nonzeros in the various rows of the
          OFF-DIAGONAL portion of the local submatrix (possibly different for
          each row) or `NULL`, if `o_nz` is used to specify the nonzero
          structure. The size of this array is equal to the number
          of local rows, i.e 'm'.

  Output Parameter:
. A - the matrix

  Options Database Keys:
+ -mat_no_inode                     - Do not use inodes
. -mat_inode_limit <limit>          - Sets inode limit (max limit=5)
- -matmult_vecscatter_view <viewer> - View the vecscatter (i.e., communication pattern) used in `MatMult()` of sparse parallel matrices.
                                      See viewer types in manual of `MatView()`. Of them, ascii_matlab, draw or binary cause the `VecScatter`
                                      to be viewed as a matrix. Entry (i,j) is the size of message (in bytes) rank i sends to rank j in one `MatMult()` call.

  Level: intermediate

  Notes:
  It is recommended that one use `MatCreateFromOptions()` or the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()`,
  MatXXXXSetPreallocation() paradigm instead of this routine directly.
  [MatXXXXSetPreallocation() is, for example, `MatSeqAIJSetPreallocation()`]

  If the *_nnz parameter is given then the *_nz parameter is ignored

  The `m`,`n`,`M`,`N` parameters specify the size of the matrix, and its partitioning across
  processors, while `d_nz`,`d_nnz`,`o_nz`,`o_nnz` parameters specify the approximate
  storage requirements for this matrix.

  If `PETSC_DECIDE` or  `PETSC_DETERMINE` is used for a particular argument on one
  processor than it must be used on all processors that share the object for
  that argument.

  If `m` and `n` are not `PETSC_DECIDE`, then the values determine the `PetscLayout` of the matrix and the ranges returned by
  `MatGetOwnershipRange()`, `MatGetOwnershipRanges()`, `MatGetOwnershipRangeColumn()`, and `MatGetOwnershipRangesColumn()`.

  The user MUST specify either the local or global matrix dimensions
  (possibly both).

  The parallel matrix is partitioned across processors such that the
  first `m0` rows belong to process 0, the next `m1` rows belong to
  process 1, the next `m2` rows belong to process 2, etc., where
  `m0`, `m1`, `m2`... are the input parameter `m` on each MPI process. I.e., each MPI process stores
  values corresponding to [m x N] submatrix.

  The columns are logically partitioned with the n0 columns belonging
  to 0th partition, the next n1 columns belonging to the next
  partition etc.. where n0,n1,n2... are the input parameter 'n'.

  The DIAGONAL portion of the local submatrix on any given processor
  is the submatrix corresponding to the rows and columns m,n
  corresponding to the given processor. i.e diagonal matrix on
  process 0 is [m0 x n0], diagonal matrix on process 1 is [m1 x n1]
  etc. The remaining portion of the local submatrix [m x (N-n)]
  constitute the OFF-DIAGONAL portion. The example below better
  illustrates this concept.

  For a square global matrix we define each processor's diagonal portion
  to be its local rows and the corresponding columns (a square submatrix);
  each processor's off-diagonal portion encompasses the remainder of the
  local matrix (a rectangular submatrix).

  If `o_nnz`, `d_nnz` are specified, then `o_nz`, and `d_nz` are ignored.

  When calling this routine with a single process communicator, a matrix of
  type `MATSEQAIJ` is returned.  If a matrix of type `MATMPIAIJ` is desired for this
  type of communicator, use the construction mechanism
.vb
  MatCreate(..., &A);
  MatSetType(A, MATMPIAIJ);
  MatSetSizes(A, m, n, M, N);
  MatMPIAIJSetPreallocation(A, ...);
.ve

  By default, this format uses inodes (identical nodes) when possible.
  We search for consecutive rows with the same nonzero structure, thereby
  reusing matrix information to achieve increased efficiency.

  Example Usage:
  Consider the following 8x8 matrix with 34 non-zero values, that is
  assembled across 3 processors. Lets assume that proc0 owns 3 rows,
  proc1 owns 3 rows, proc2 owns 2 rows. This division can be shown
  as follows

.vb
            1  2  0  |  0  3  0  |  0  4
    Proc0   0  5  6  |  7  0  0  |  8  0
            9  0 10  | 11  0  0  | 12  0
    -------------------------------------
           13  0 14  | 15 16 17  |  0  0
    Proc1   0 18  0  | 19 20 21  |  0  0
            0  0  0  | 22 23  0  | 24  0
    -------------------------------------
    Proc2  25 26 27  |  0  0 28  | 29  0
           30  0  0  | 31 32 33  |  0 34
.ve

  This can be represented as a collection of submatrices as

.vb
      A B C
      D E F
      G H I
.ve

  Where the submatrices A,B,C are owned by proc0, D,E,F are
  owned by proc1, G,H,I are owned by proc2.

  The 'm' parameters for proc0,proc1,proc2 are 3,3,2 respectively.
  The 'n' parameters for proc0,proc1,proc2 are 3,3,2 respectively.
  The 'M','N' parameters are 8,8, and have the same values on all procs.

  The DIAGONAL submatrices corresponding to proc0,proc1,proc2 are
  submatrices [A], [E], [I] respectively. The OFF-DIAGONAL submatrices
  corresponding to proc0,proc1,proc2 are [BC], [DF], [GH] respectively.
  Internally, each processor stores the DIAGONAL part, and the OFF-DIAGONAL
  part as `MATSEQAIJ` matrices. For example, proc1 will store [E] as a `MATSEQAIJ`
  matrix, ans [DF] as another SeqAIJ matrix.

  When `d_nz`, `o_nz` parameters are specified, `d_nz` storage elements are
  allocated for every row of the local diagonal submatrix, and `o_nz`
  storage locations are allocated for every row of the OFF-DIAGONAL submat.
  One way to choose `d_nz` and `o_nz` is to use the max nonzerors per local
  rows for each of the local DIAGONAL, and the OFF-DIAGONAL submatrices.
  In this case, the values of `d_nz`,`o_nz` are
.vb
     proc0  dnz = 2, o_nz = 2
     proc1  dnz = 3, o_nz = 2
     proc2  dnz = 1, o_nz = 4
.ve
  We are allocating m*(`d_nz`+`o_nz`) storage locations for every proc. This
  translates to 3*(2+2)=12 for proc0, 3*(3+2)=15 for proc1, 2*(1+4)=10
  for proc3. i.e we are using 12+15+10=37 storage locations to store
  34 values.

  When `d_nnz`, `o_nnz` parameters are specified, the storage is specified
  for every row, corresponding to both DIAGONAL and OFF-DIAGONAL submatrices.
  In the above case the values for d_nnz,o_nnz are
.vb
     proc0 d_nnz = [2,2,2] and o_nnz = [2,2,2]
     proc1 d_nnz = [3,3,2] and o_nnz = [2,1,1]
     proc2 d_nnz = [1,1]   and o_nnz = [4,4]
.ve
  Here the space allocated is sum of all the above values i.e 34, and
  hence pre-allocation is perfect.

.seealso: [](ch_matrices), `Mat`, [Sparse Matrix Creation](sec_matsparse), `MatCreate()`, `MatCreateSeqAIJ()`, `MatSetValues()`, `MatMPIAIJSetPreallocation()`, `MatMPIAIJSetPreallocationCSR()`,
          `MATMPIAIJ`, `MatCreateMPIAIJWithArrays()`, `MatGetOwnershipRange()`, `MatGetOwnershipRanges()`, `MatGetOwnershipRangeColumn()`,
          `MatGetOwnershipRangesColumn()`, `PetscLayout`
@*/
PetscErrorCode MatCreateAIJ(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[], Mat *A)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, m, n, M, N));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size > 1) {
    PetscCall(MatSetType(*A, MATMPIAIJ));
    PetscCall(MatMPIAIJSetPreallocation(*A, d_nz, d_nnz, o_nz, o_nnz));
  } else {
    PetscCall(MatSetType(*A, MATSEQAIJ));
    PetscCall(MatSeqAIJSetPreallocation(*A, d_nz, d_nnz));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
    MatMPIAIJGetSeqAIJF90 - Returns the local pieces of this distributed matrix

    Synopsis:
    MatMPIAIJGetSeqAIJF90(Mat A, Mat Ad, Mat Ao, {PetscInt, pointer :: colmap(:)},integer ierr)

    Not Collective

    Input Parameter:
.   A - the `MATMPIAIJ` matrix

    Output Parameters:
+   Ad - the diagonal portion of the matrix
.   Ao - the off-diagonal portion of the matrix
.   colmap - An array mapping local column numbers of `Ao` to global column numbers of the parallel matrix
-   ierr - error code

     Level: advanced

    Note:
    Use  `MatMPIAIJRestoreSeqAIJF90()` when you no longer need access to the matrices and `colmap`

.seealso: [](ch_matrices), `Mat`, [](sec_fortranarrays), `Mat`, `MATMPIAIJ`, `MatMPIAIJGetSeqAIJ()`, `MatMPIAIJRestoreSeqAIJF90()`
M*/

/*MC
    MatMPIAIJRestoreSeqAIJF90 - call after `MatMPIAIJGetSeqAIJF90()` when you no longer need access to the matrices and `colmap`

    Synopsis:
    MatMPIAIJRestoreSeqAIJF90(Mat A, Mat Ad, Mat Ao, {PetscInt, pointer :: colmap(:)},integer ierr)

    Not Collective

    Input Parameters:
+   A - the `MATMPIAIJ` matrix
.   Ad - the diagonal portion of the matrix
.   Ao - the off-diagonal portion of the matrix
.   colmap - An array mapping local column numbers of `Ao` to global column numbers of the parallel matrix
-   ierr - error code

     Level: advanced

.seealso: [](ch_matrices), `Mat`, [](sec_fortranarrays), `Mat`, `MATMPIAIJ`, `MatMPIAIJGetSeqAIJ()`, `MatMPIAIJGetSeqAIJF90()`
M*/

/*@C
  MatMPIAIJGetSeqAIJ - Returns the local pieces of this distributed matrix

  Not Collective

  Input Parameter:
. A - The `MATMPIAIJ` matrix

  Output Parameters:
+ Ad     - The local diagonal block as a `MATSEQAIJ` matrix
. Ao     - The local off-diagonal block as a `MATSEQAIJ` matrix
- colmap - An array mapping local column numbers of `Ao` to global column numbers of the parallel matrix

  Level: intermediate

  Note:
  The rows in `Ad` and `Ao` are in [0, Nr), where Nr is the number of local rows on this process. The columns
  in `Ad` are in [0, Nc) where Nc is the number of local columns. The columns are `Ao` are in [0, Nco), where Nco is
  the number of nonzero columns in the local off-diagonal piece of the matrix `A`. The array colmap maps these
  local column numbers to global column numbers in the original matrix.

  Fortran Notes:
  `MatMPIAIJGetSeqAIJ()` Fortran binding is deprecated (since PETSc 3.19), use `MatMPIAIJGetSeqAIJF90()`

.seealso: [](ch_matrices), `Mat`, `MATMPIAIJ`, `MatMPIAIJGetSeqAIJF90()`, `MatMPIAIJRestoreSeqAIJF90()`, `MatMPIAIJGetLocalMat()`, `MatMPIAIJGetLocalMatCondensed()`, `MatCreateAIJ()`, `MATSEQAIJ`
@*/
PetscErrorCode MatMPIAIJGetSeqAIJ(Mat A, Mat *Ad, Mat *Ao, const PetscInt *colmap[])
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;
  PetscBool   flg;

  PetscFunctionBegin;
  PetscCall(PetscStrbeginswith(((PetscObject)A)->type_name, MATMPIAIJ, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "This function requires a MATMPIAIJ matrix as input");
  if (Ad) *Ad = a->A;
  if (Ao) *Ao = a->B;
  if (colmap) *colmap = a->garray;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCreateMPIMatConcatenateSeqMat_MPIAIJ(MPI_Comm comm, Mat inmat, PetscInt n, MatReuse scall, Mat *outmat)
{
  PetscInt     m, N, i, rstart, nnz, Ii;
  PetscInt    *indx;
  PetscScalar *values;
  MatType      rootType;

  PetscFunctionBegin;
  PetscCall(MatGetSize(inmat, &m, &N));
  if (scall == MAT_INITIAL_MATRIX) { /* symbolic phase */
    PetscInt *dnz, *onz, sum, bs, cbs;

    if (n == PETSC_DECIDE) PetscCall(PetscSplitOwnership(comm, &n, &N));
    /* Check sum(n) = N */
    PetscCallMPI(MPIU_Allreduce(&n, &sum, 1, MPIU_INT, MPI_SUM, comm));
    PetscCheck(sum == N, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Sum of local columns %" PetscInt_FMT " != global columns %" PetscInt_FMT, sum, N);

    PetscCallMPI(MPI_Scan(&m, &rstart, 1, MPIU_INT, MPI_SUM, comm));
    rstart -= m;

    MatPreallocateBegin(comm, m, n, dnz, onz);
    for (i = 0; i < m; i++) {
      PetscCall(MatGetRow_SeqAIJ(inmat, i, &nnz, &indx, NULL));
      PetscCall(MatPreallocateSet(i + rstart, nnz, indx, dnz, onz));
      PetscCall(MatRestoreRow_SeqAIJ(inmat, i, &nnz, &indx, NULL));
    }

    PetscCall(MatCreate(comm, outmat));
    PetscCall(MatSetSizes(*outmat, m, n, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatGetBlockSizes(inmat, &bs, &cbs));
    PetscCall(MatSetBlockSizes(*outmat, bs, cbs));
    PetscCall(MatGetRootType_Private(inmat, &rootType));
    PetscCall(MatSetType(*outmat, rootType));
    PetscCall(MatSeqAIJSetPreallocation(*outmat, 0, dnz));
    PetscCall(MatMPIAIJSetPreallocation(*outmat, 0, dnz, 0, onz));
    MatPreallocateEnd(dnz, onz);
    PetscCall(MatSetOption(*outmat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
  }

  /* numeric phase */
  PetscCall(MatGetOwnershipRange(*outmat, &rstart, NULL));
  for (i = 0; i < m; i++) {
    PetscCall(MatGetRow_SeqAIJ(inmat, i, &nnz, &indx, &values));
    Ii = i + rstart;
    PetscCall(MatSetValues(*outmat, 1, &Ii, nnz, indx, values, INSERT_VALUES));
    PetscCall(MatRestoreRow_SeqAIJ(inmat, i, &nnz, &indx, &values));
  }
  PetscCall(MatAssemblyBegin(*outmat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*outmat, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_MPIAIJ_SeqsToMPI(void *data)
{
  Mat_Merge_SeqsToMPI *merge = (Mat_Merge_SeqsToMPI *)data;

  PetscFunctionBegin;
  if (!merge) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFree(merge->id_r));
  PetscCall(PetscFree(merge->len_s));
  PetscCall(PetscFree(merge->len_r));
  PetscCall(PetscFree(merge->bi));
  PetscCall(PetscFree(merge->bj));
  PetscCall(PetscFree(merge->buf_ri[0]));
  PetscCall(PetscFree(merge->buf_ri));
  PetscCall(PetscFree(merge->buf_rj[0]));
  PetscCall(PetscFree(merge->buf_rj));
  PetscCall(PetscFree(merge->coi));
  PetscCall(PetscFree(merge->coj));
  PetscCall(PetscFree(merge->owners_co));
  PetscCall(PetscLayoutDestroy(&merge->rowmap));
  PetscCall(PetscFree(merge));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <../src/mat/utils/freespace.h>
#include <petscbt.h>

PetscErrorCode MatCreateMPIAIJSumSeqAIJNumeric(Mat seqmat, Mat mpimat)
{
  MPI_Comm             comm;
  Mat_SeqAIJ          *a = (Mat_SeqAIJ *)seqmat->data;
  PetscMPIInt          size, rank, taga, *len_s;
  PetscInt             N = mpimat->cmap->N, i, j, *owners, *ai = a->i, *aj, m;
  PetscMPIInt          proc, k;
  PetscInt           **buf_ri, **buf_rj;
  PetscInt             anzi, *bj_i, *bi, *bj, arow, bnzi, nextaj;
  PetscInt             nrows, **buf_ri_k, **nextrow, **nextai;
  MPI_Request         *s_waits, *r_waits;
  MPI_Status          *status;
  const MatScalar     *aa, *a_a;
  MatScalar          **abuf_r, *ba_i;
  Mat_Merge_SeqsToMPI *merge;
  PetscContainer       container;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)mpimat, &comm));
  PetscCall(PetscLogEventBegin(MAT_Seqstompinum, seqmat, 0, 0, 0));

  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(PetscObjectQuery((PetscObject)mpimat, "MatMergeSeqsToMPI", (PetscObject *)&container));
  PetscCheck(container, PetscObjectComm((PetscObject)mpimat), PETSC_ERR_PLIB, "Mat not created from MatCreateMPIAIJSumSeqAIJSymbolic");
  PetscCall(PetscContainerGetPointer(container, (void **)&merge));
  PetscCall(MatSeqAIJGetArrayRead(seqmat, &a_a));
  aa = a_a;

  bi     = merge->bi;
  bj     = merge->bj;
  buf_ri = merge->buf_ri;
  buf_rj = merge->buf_rj;

  PetscCall(PetscMalloc1(size, &status));
  owners = merge->rowmap->range;
  len_s  = merge->len_s;

  /* send and recv matrix values */
  PetscCall(PetscObjectGetNewTag((PetscObject)mpimat, &taga));
  PetscCall(PetscPostIrecvScalar(comm, taga, merge->nrecv, merge->id_r, merge->len_r, &abuf_r, &r_waits));

  PetscCall(PetscMalloc1(merge->nsend + 1, &s_waits));
  for (proc = 0, k = 0; proc < size; proc++) {
    if (!len_s[proc]) continue;
    i = owners[proc];
    PetscCallMPI(MPIU_Isend(aa + ai[i], len_s[proc], MPIU_MATSCALAR, proc, taga, comm, s_waits + k));
    k++;
  }

  if (merge->nrecv) PetscCallMPI(MPI_Waitall(merge->nrecv, r_waits, status));
  if (merge->nsend) PetscCallMPI(MPI_Waitall(merge->nsend, s_waits, status));
  PetscCall(PetscFree(status));

  PetscCall(PetscFree(s_waits));
  PetscCall(PetscFree(r_waits));

  /* insert mat values of mpimat */
  PetscCall(PetscMalloc1(N, &ba_i));
  PetscCall(PetscMalloc3(merge->nrecv, &buf_ri_k, merge->nrecv, &nextrow, merge->nrecv, &nextai));

  for (k = 0; k < merge->nrecv; k++) {
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *buf_ri_k[k];
    nextrow[k]  = buf_ri_k[k] + 1;           /* next row number of k-th recved i-structure */
    nextai[k]   = buf_ri_k[k] + (nrows + 1); /* points to the next i-structure of k-th recved i-structure  */
  }

  /* set values of ba */
  m = merge->rowmap->n;
  for (i = 0; i < m; i++) {
    arow = owners[rank] + i;
    bj_i = bj + bi[i]; /* col indices of the i-th row of mpimat */
    bnzi = bi[i + 1] - bi[i];
    PetscCall(PetscArrayzero(ba_i, bnzi));

    /* add local non-zero vals of this proc's seqmat into ba */
    anzi   = ai[arow + 1] - ai[arow];
    aj     = a->j + ai[arow];
    aa     = a_a + ai[arow];
    nextaj = 0;
    for (j = 0; nextaj < anzi; j++) {
      if (*(bj_i + j) == aj[nextaj]) { /* bcol == acol */
        ba_i[j] += aa[nextaj++];
      }
    }

    /* add received vals into ba */
    for (k = 0; k < merge->nrecv; k++) { /* k-th received message */
      /* i-th row */
      if (i == *nextrow[k]) {
        anzi   = *(nextai[k] + 1) - *nextai[k];
        aj     = buf_rj[k] + *nextai[k];
        aa     = abuf_r[k] + *nextai[k];
        nextaj = 0;
        for (j = 0; nextaj < anzi; j++) {
          if (*(bj_i + j) == aj[nextaj]) { /* bcol == acol */
            ba_i[j] += aa[nextaj++];
          }
        }
        nextrow[k]++;
        nextai[k]++;
      }
    }
    PetscCall(MatSetValues(mpimat, 1, &arow, bnzi, bj_i, ba_i, INSERT_VALUES));
  }
  PetscCall(MatSeqAIJRestoreArrayRead(seqmat, &a_a));
  PetscCall(MatAssemblyBegin(mpimat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mpimat, MAT_FINAL_ASSEMBLY));

  PetscCall(PetscFree(abuf_r[0]));
  PetscCall(PetscFree(abuf_r));
  PetscCall(PetscFree(ba_i));
  PetscCall(PetscFree3(buf_ri_k, nextrow, nextai));
  PetscCall(PetscLogEventEnd(MAT_Seqstompinum, seqmat, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCreateMPIAIJSumSeqAIJSymbolic(MPI_Comm comm, Mat seqmat, PetscInt m, PetscInt n, Mat *mpimat)
{
  Mat                  B_mpi;
  Mat_SeqAIJ          *a = (Mat_SeqAIJ *)seqmat->data;
  PetscMPIInt          size, rank, tagi, tagj, *len_s, *len_si, *len_ri;
  PetscInt           **buf_rj, **buf_ri, **buf_ri_k;
  PetscInt             M = seqmat->rmap->n, N = seqmat->cmap->n, i, *owners, *ai = a->i, *aj = a->j;
  PetscInt             len, *dnz, *onz, bs, cbs;
  PetscInt             k, anzi, *bi, *bj, *lnk, nlnk, arow, bnzi;
  PetscInt             nrows, *buf_s, *buf_si, *buf_si_i, **nextrow, **nextai;
  MPI_Request         *si_waits, *sj_waits, *ri_waits, *rj_waits;
  MPI_Status          *status;
  PetscFreeSpaceList   free_space = NULL, current_space = NULL;
  PetscBT              lnkbt;
  Mat_Merge_SeqsToMPI *merge;
  PetscContainer       container;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(MAT_Seqstompisym, seqmat, 0, 0, 0));

  /* make sure it is a PETSc comm */
  PetscCall(PetscCommDuplicate(comm, &comm, NULL));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  PetscCall(PetscNew(&merge));
  PetscCall(PetscMalloc1(size, &status));

  /* determine row ownership */
  PetscCall(PetscLayoutCreate(comm, &merge->rowmap));
  PetscCall(PetscLayoutSetLocalSize(merge->rowmap, m));
  PetscCall(PetscLayoutSetSize(merge->rowmap, M));
  PetscCall(PetscLayoutSetBlockSize(merge->rowmap, 1));
  PetscCall(PetscLayoutSetUp(merge->rowmap));
  PetscCall(PetscMalloc1(size, &len_si));
  PetscCall(PetscMalloc1(size, &merge->len_s));

  m      = merge->rowmap->n;
  owners = merge->rowmap->range;

  /* determine the number of messages to send, their lengths */
  len_s = merge->len_s;

  len          = 0; /* length of buf_si[] */
  merge->nsend = 0;
  for (PetscMPIInt proc = 0; proc < size; proc++) {
    len_si[proc] = 0;
    if (proc == rank) {
      len_s[proc] = 0;
    } else {
      PetscCall(PetscMPIIntCast(owners[proc + 1] - owners[proc] + 1, &len_si[proc]));
      PetscCall(PetscMPIIntCast(ai[owners[proc + 1]] - ai[owners[proc]], &len_s[proc])); /* num of rows to be sent to [proc] */
    }
    if (len_s[proc]) {
      merge->nsend++;
      nrows = 0;
      for (i = owners[proc]; i < owners[proc + 1]; i++) {
        if (ai[i + 1] > ai[i]) nrows++;
      }
      PetscCall(PetscMPIIntCast(2 * (nrows + 1), &len_si[proc]));
      len += len_si[proc];
    }
  }

  /* determine the number and length of messages to receive for ij-structure */
  PetscCall(PetscGatherNumberOfMessages(comm, NULL, len_s, &merge->nrecv));
  PetscCall(PetscGatherMessageLengths2(comm, merge->nsend, merge->nrecv, len_s, len_si, &merge->id_r, &merge->len_r, &len_ri));

  /* post the Irecv of j-structure */
  PetscCall(PetscCommGetNewTag(comm, &tagj));
  PetscCall(PetscPostIrecvInt(comm, tagj, merge->nrecv, merge->id_r, merge->len_r, &buf_rj, &rj_waits));

  /* post the Isend of j-structure */
  PetscCall(PetscMalloc2(merge->nsend, &si_waits, merge->nsend, &sj_waits));

  for (PetscMPIInt proc = 0, k = 0; proc < size; proc++) {
    if (!len_s[proc]) continue;
    i = owners[proc];
    PetscCallMPI(MPIU_Isend(aj + ai[i], len_s[proc], MPIU_INT, proc, tagj, comm, sj_waits + k));
    k++;
  }

  /* receives and sends of j-structure are complete */
  if (merge->nrecv) PetscCallMPI(MPI_Waitall(merge->nrecv, rj_waits, status));
  if (merge->nsend) PetscCallMPI(MPI_Waitall(merge->nsend, sj_waits, status));

  /* send and recv i-structure */
  PetscCall(PetscCommGetNewTag(comm, &tagi));
  PetscCall(PetscPostIrecvInt(comm, tagi, merge->nrecv, merge->id_r, len_ri, &buf_ri, &ri_waits));

  PetscCall(PetscMalloc1(len + 1, &buf_s));
  buf_si = buf_s; /* points to the beginning of k-th msg to be sent */
  for (PetscMPIInt proc = 0, k = 0; proc < size; proc++) {
    if (!len_s[proc]) continue;
    /* form outgoing message for i-structure:
         buf_si[0]:                 nrows to be sent
               [1:nrows]:           row index (global)
               [nrows+1:2*nrows+1]: i-structure index
    */
    nrows       = len_si[proc] / 2 - 1;
    buf_si_i    = buf_si + nrows + 1;
    buf_si[0]   = nrows;
    buf_si_i[0] = 0;
    nrows       = 0;
    for (i = owners[proc]; i < owners[proc + 1]; i++) {
      anzi = ai[i + 1] - ai[i];
      if (anzi) {
        buf_si_i[nrows + 1] = buf_si_i[nrows] + anzi; /* i-structure */
        buf_si[nrows + 1]   = i - owners[proc];       /* local row index */
        nrows++;
      }
    }
    PetscCallMPI(MPIU_Isend(buf_si, len_si[proc], MPIU_INT, proc, tagi, comm, si_waits + k));
    k++;
    buf_si += len_si[proc];
  }

  if (merge->nrecv) PetscCallMPI(MPI_Waitall(merge->nrecv, ri_waits, status));
  if (merge->nsend) PetscCallMPI(MPI_Waitall(merge->nsend, si_waits, status));

  PetscCall(PetscInfo(seqmat, "nsend: %d, nrecv: %d\n", merge->nsend, merge->nrecv));
  for (i = 0; i < merge->nrecv; i++) PetscCall(PetscInfo(seqmat, "recv len_ri=%d, len_rj=%d from [%d]\n", len_ri[i], merge->len_r[i], merge->id_r[i]));

  PetscCall(PetscFree(len_si));
  PetscCall(PetscFree(len_ri));
  PetscCall(PetscFree(rj_waits));
  PetscCall(PetscFree2(si_waits, sj_waits));
  PetscCall(PetscFree(ri_waits));
  PetscCall(PetscFree(buf_s));
  PetscCall(PetscFree(status));

  /* compute a local seq matrix in each processor */
  /* allocate bi array and free space for accumulating nonzero column info */
  PetscCall(PetscMalloc1(m + 1, &bi));
  bi[0] = 0;

  /* create and initialize a linked list */
  nlnk = N + 1;
  PetscCall(PetscLLCreate(N, N, nlnk, lnk, lnkbt));

  /* initial FreeSpace size is 2*(num of local nnz(seqmat)) */
  len = ai[owners[rank + 1]] - ai[owners[rank]];
  PetscCall(PetscFreeSpaceGet(PetscIntMultTruncate(2, len) + 1, &free_space));

  current_space = free_space;

  /* determine symbolic info for each local row */
  PetscCall(PetscMalloc3(merge->nrecv, &buf_ri_k, merge->nrecv, &nextrow, merge->nrecv, &nextai));

  for (k = 0; k < merge->nrecv; k++) {
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *buf_ri_k[k];
    nextrow[k]  = buf_ri_k[k] + 1;           /* next row number of k-th recved i-structure */
    nextai[k]   = buf_ri_k[k] + (nrows + 1); /* points to the next i-structure of k-th recved i-structure  */
  }

  MatPreallocateBegin(comm, m, n, dnz, onz);
  len = 0;
  for (i = 0; i < m; i++) {
    bnzi = 0;
    /* add local non-zero cols of this proc's seqmat into lnk */
    arow = owners[rank] + i;
    anzi = ai[arow + 1] - ai[arow];
    aj   = a->j + ai[arow];
    PetscCall(PetscLLAddSorted(anzi, aj, N, &nlnk, lnk, lnkbt));
    bnzi += nlnk;
    /* add received col data into lnk */
    for (k = 0; k < merge->nrecv; k++) { /* k-th received message */
      if (i == *nextrow[k]) {            /* i-th row */
        anzi = *(nextai[k] + 1) - *nextai[k];
        aj   = buf_rj[k] + *nextai[k];
        PetscCall(PetscLLAddSorted(anzi, aj, N, &nlnk, lnk, lnkbt));
        bnzi += nlnk;
        nextrow[k]++;
        nextai[k]++;
      }
    }
    if (len < bnzi) len = bnzi; /* =max(bnzi) */

    /* if free space is not available, make more free space */
    if (current_space->local_remaining < bnzi) PetscCall(PetscFreeSpaceGet(PetscIntSumTruncate(bnzi, current_space->total_array_size), &current_space));
    /* copy data into free space, then initialize lnk */
    PetscCall(PetscLLClean(N, N, bnzi, lnk, current_space->array, lnkbt));
    PetscCall(MatPreallocateSet(i + owners[rank], bnzi, current_space->array, dnz, onz));

    current_space->array += bnzi;
    current_space->local_used += bnzi;
    current_space->local_remaining -= bnzi;

    bi[i + 1] = bi[i] + bnzi;
  }

  PetscCall(PetscFree3(buf_ri_k, nextrow, nextai));

  PetscCall(PetscMalloc1(bi[m] + 1, &bj));
  PetscCall(PetscFreeSpaceContiguous(&free_space, bj));
  PetscCall(PetscLLDestroy(lnk, lnkbt));

  /* create symbolic parallel matrix B_mpi */
  PetscCall(MatGetBlockSizes(seqmat, &bs, &cbs));
  PetscCall(MatCreate(comm, &B_mpi));
  if (n == PETSC_DECIDE) {
    PetscCall(MatSetSizes(B_mpi, m, n, PETSC_DETERMINE, N));
  } else {
    PetscCall(MatSetSizes(B_mpi, m, n, PETSC_DETERMINE, PETSC_DETERMINE));
  }
  PetscCall(MatSetBlockSizes(B_mpi, bs, cbs));
  PetscCall(MatSetType(B_mpi, MATMPIAIJ));
  PetscCall(MatMPIAIJSetPreallocation(B_mpi, 0, dnz, 0, onz));
  MatPreallocateEnd(dnz, onz);
  PetscCall(MatSetOption(B_mpi, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  /* B_mpi is not ready for use - assembly will be done by MatCreateMPIAIJSumSeqAIJNumeric() */
  B_mpi->assembled = PETSC_FALSE;
  merge->bi        = bi;
  merge->bj        = bj;
  merge->buf_ri    = buf_ri;
  merge->buf_rj    = buf_rj;
  merge->coi       = NULL;
  merge->coj       = NULL;
  merge->owners_co = NULL;

  PetscCall(PetscCommDestroy(&comm));

  /* attach the supporting struct to B_mpi for reuse */
  PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &container));
  PetscCall(PetscContainerSetPointer(container, merge));
  PetscCall(PetscContainerSetUserDestroy(container, MatDestroy_MPIAIJ_SeqsToMPI));
  PetscCall(PetscObjectCompose((PetscObject)B_mpi, "MatMergeSeqsToMPI", (PetscObject)container));
  PetscCall(PetscContainerDestroy(&container));
  *mpimat = B_mpi;

  PetscCall(PetscLogEventEnd(MAT_Seqstompisym, seqmat, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateMPIAIJSumSeqAIJ - Creates a `MATMPIAIJ` matrix by adding sequential
  matrices from each processor

  Collective

  Input Parameters:
+ comm   - the communicators the parallel matrix will live on
. seqmat - the input sequential matrices
. m      - number of local rows (or `PETSC_DECIDE`)
. n      - number of local columns (or `PETSC_DECIDE`)
- scall  - either `MAT_INITIAL_MATRIX` or `MAT_REUSE_MATRIX`

  Output Parameter:
. mpimat - the parallel matrix generated

  Level: advanced

  Note:
  The dimensions of the sequential matrix in each processor MUST be the same.
  The input seqmat is included into the container "Mat_Merge_SeqsToMPI", and will be
  destroyed when `mpimat` is destroyed. Call `PetscObjectQuery()` to access `seqmat`.

.seealso: [](ch_matrices), `Mat`, `MatCreateAIJ()`
@*/
PetscErrorCode MatCreateMPIAIJSumSeqAIJ(MPI_Comm comm, Mat seqmat, PetscInt m, PetscInt n, MatReuse scall, Mat *mpimat)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size == 1) {
    PetscCall(PetscLogEventBegin(MAT_Seqstompi, seqmat, 0, 0, 0));
    if (scall == MAT_INITIAL_MATRIX) {
      PetscCall(MatDuplicate(seqmat, MAT_COPY_VALUES, mpimat));
    } else {
      PetscCall(MatCopy(seqmat, *mpimat, SAME_NONZERO_PATTERN));
    }
    PetscCall(PetscLogEventEnd(MAT_Seqstompi, seqmat, 0, 0, 0));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscLogEventBegin(MAT_Seqstompi, seqmat, 0, 0, 0));
  if (scall == MAT_INITIAL_MATRIX) PetscCall(MatCreateMPIAIJSumSeqAIJSymbolic(comm, seqmat, m, n, mpimat));
  PetscCall(MatCreateMPIAIJSumSeqAIJNumeric(seqmat, *mpimat));
  PetscCall(PetscLogEventEnd(MAT_Seqstompi, seqmat, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatAIJGetLocalMat - Creates a `MATSEQAIJ` from a `MATAIJ` matrix.

  Not Collective

  Input Parameter:
. A - the matrix

  Output Parameter:
. A_loc - the local sequential matrix generated

  Level: developer

  Notes:
  The matrix is created by taking `A`'s local rows and putting them into a sequential matrix
  with `mlocal` rows and `n` columns. Where `mlocal` is obtained with `MatGetLocalSize()` and
  `n` is the global column count obtained with `MatGetSize()`

  In other words combines the two parts of a parallel `MATMPIAIJ` matrix on each process to a single matrix.

  For parallel matrices this creates an entirely new matrix. If the matrix is sequential it merely increases the reference count.

  Destroy the matrix with `MatDestroy()`

.seealso: [](ch_matrices), `Mat`, `MatMPIAIJGetLocalMat()`
@*/
PetscErrorCode MatAIJGetLocalMat(Mat A, Mat *A_loc)
{
  PetscBool mpi;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATMPIAIJ, &mpi));
  if (mpi) {
    PetscCall(MatMPIAIJGetLocalMat(A, MAT_INITIAL_MATRIX, A_loc));
  } else {
    *A_loc = A;
    PetscCall(PetscObjectReference((PetscObject)*A_loc));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMPIAIJGetLocalMat - Creates a `MATSEQAIJ` from a `MATMPIAIJ` matrix.

  Not Collective

  Input Parameters:
+ A     - the matrix
- scall - either `MAT_INITIAL_MATRIX` or `MAT_REUSE_MATRIX`

  Output Parameter:
. A_loc - the local sequential matrix generated

  Level: developer

  Notes:
  The matrix is created by taking all `A`'s local rows and putting them into a sequential
  matrix with `mlocal` rows and `n` columns.`mlocal` is the row count obtained with
  `MatGetLocalSize()` and `n` is the global column count obtained with `MatGetSize()`.

  In other words combines the two parts of a parallel `MATMPIAIJ` matrix on each process to a single matrix.

  When `A` is sequential and `MAT_INITIAL_MATRIX` is requested, the matrix returned is the diagonal part of `A` (which contains the entire matrix),
  with its reference count increased by one. Hence changing values of `A_loc` changes `A`. If `MAT_REUSE_MATRIX` is requested on a sequential matrix
  then `MatCopy`(Adiag,*`A_loc`,`SAME_NONZERO_PATTERN`) is called to fill `A_loc`. Thus one can preallocate the appropriate sequential matrix `A_loc`
  and then call this routine with `MAT_REUSE_MATRIX`. In this case, one can modify the values of `A_loc` without affecting the original sequential matrix.

.seealso: [](ch_matrices), `Mat`, `MATMPIAIJ`, `MatGetOwnershipRange()`, `MatMPIAIJGetLocalMatCondensed()`, `MatMPIAIJGetLocalMatMerge()`
@*/
PetscErrorCode MatMPIAIJGetLocalMat(Mat A, MatReuse scall, Mat *A_loc)
{
  Mat_MPIAIJ        *mpimat = (Mat_MPIAIJ *)A->data;
  Mat_SeqAIJ        *mat, *a, *b;
  PetscInt          *ai, *aj, *bi, *bj, *cmap = mpimat->garray;
  const PetscScalar *aa, *ba, *aav, *bav;
  PetscScalar       *ca, *cam;
  PetscMPIInt        size;
  PetscInt           am = A->rmap->n, i, j, k, cstart = A->cmap->rstart;
  PetscInt          *ci, *cj, col, ncols_d, ncols_o, jo;
  PetscBool          match;

  PetscFunctionBegin;
  PetscCall(PetscStrbeginswith(((PetscObject)A)->type_name, MATMPIAIJ, &match));
  PetscCheck(match, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Requires MATMPIAIJ matrix as input");
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
  if (size == 1) {
    if (scall == MAT_INITIAL_MATRIX) {
      PetscCall(PetscObjectReference((PetscObject)mpimat->A));
      *A_loc = mpimat->A;
    } else if (scall == MAT_REUSE_MATRIX) {
      PetscCall(MatCopy(mpimat->A, *A_loc, SAME_NONZERO_PATTERN));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscLogEventBegin(MAT_Getlocalmat, A, 0, 0, 0));
  a  = (Mat_SeqAIJ *)mpimat->A->data;
  b  = (Mat_SeqAIJ *)mpimat->B->data;
  ai = a->i;
  aj = a->j;
  bi = b->i;
  bj = b->j;
  PetscCall(MatSeqAIJGetArrayRead(mpimat->A, &aav));
  PetscCall(MatSeqAIJGetArrayRead(mpimat->B, &bav));
  aa = aav;
  ba = bav;
  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(PetscMalloc1(1 + am, &ci));
    ci[0] = 0;
    for (i = 0; i < am; i++) ci[i + 1] = ci[i] + (ai[i + 1] - ai[i]) + (bi[i + 1] - bi[i]);
    PetscCall(PetscMalloc1(1 + ci[am], &cj));
    PetscCall(PetscMalloc1(1 + ci[am], &ca));
    k = 0;
    for (i = 0; i < am; i++) {
      ncols_o = bi[i + 1] - bi[i];
      ncols_d = ai[i + 1] - ai[i];
      /* off-diagonal portion of A */
      for (jo = 0; jo < ncols_o; jo++) {
        col = cmap[*bj];
        if (col >= cstart) break;
        cj[k] = col;
        bj++;
        ca[k++] = *ba++;
      }
      /* diagonal portion of A */
      for (j = 0; j < ncols_d; j++) {
        cj[k]   = cstart + *aj++;
        ca[k++] = *aa++;
      }
      /* off-diagonal portion of A */
      for (j = jo; j < ncols_o; j++) {
        cj[k]   = cmap[*bj++];
        ca[k++] = *ba++;
      }
    }
    /* put together the new matrix */
    PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, am, A->cmap->N, ci, cj, ca, A_loc));
    /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
    /* Since these are PETSc arrays, change flags to free them as necessary. */
    mat          = (Mat_SeqAIJ *)(*A_loc)->data;
    mat->free_a  = PETSC_TRUE;
    mat->free_ij = PETSC_TRUE;
    mat->nonew   = 0;
  } else if (scall == MAT_REUSE_MATRIX) {
    mat = (Mat_SeqAIJ *)(*A_loc)->data;
    ci  = mat->i;
    cj  = mat->j;
    PetscCall(MatSeqAIJGetArrayWrite(*A_loc, &cam));
    for (i = 0; i < am; i++) {
      /* off-diagonal portion of A */
      ncols_o = bi[i + 1] - bi[i];
      for (jo = 0; jo < ncols_o; jo++) {
        col = cmap[*bj];
        if (col >= cstart) break;
        *cam++ = *ba++;
        bj++;
      }
      /* diagonal portion of A */
      ncols_d = ai[i + 1] - ai[i];
      for (j = 0; j < ncols_d; j++) *cam++ = *aa++;
      /* off-diagonal portion of A */
      for (j = jo; j < ncols_o; j++) {
        *cam++ = *ba++;
        bj++;
      }
    }
    PetscCall(MatSeqAIJRestoreArrayWrite(*A_loc, &cam));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid MatReuse %d", (int)scall);
  PetscCall(MatSeqAIJRestoreArrayRead(mpimat->A, &aav));
  PetscCall(MatSeqAIJRestoreArrayRead(mpimat->B, &bav));
  PetscCall(PetscLogEventEnd(MAT_Getlocalmat, A, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMPIAIJGetLocalMatMerge - Creates a `MATSEQAIJ` from a `MATMPIAIJ` matrix by taking all its local rows and putting them into a sequential matrix with
  mlocal rows and n columns. Where n is the sum of the number of columns of the diagonal and off-diagonal part

  Not Collective

  Input Parameters:
+ A     - the matrix
- scall - either `MAT_INITIAL_MATRIX` or `MAT_REUSE_MATRIX`

  Output Parameters:
+ glob  - sequential `IS` with global indices associated with the columns of the local sequential matrix generated (can be `NULL`)
- A_loc - the local sequential matrix generated

  Level: developer

  Note:
  This is different from `MatMPIAIJGetLocalMat()` since the first columns in the returning matrix are those associated with the diagonal
  part, then those associated with the off-diagonal part (in its local ordering)

.seealso: [](ch_matrices), `Mat`, `MATMPIAIJ`, `MatGetOwnershipRange()`, `MatMPIAIJGetLocalMat()`, `MatMPIAIJGetLocalMatCondensed()`
@*/
PetscErrorCode MatMPIAIJGetLocalMatMerge(Mat A, MatReuse scall, IS *glob, Mat *A_loc)
{
  Mat             Ao, Ad;
  const PetscInt *cmap;
  PetscMPIInt     size;
  PetscErrorCode (*f)(Mat, MatReuse, IS *, Mat *);

  PetscFunctionBegin;
  PetscCall(MatMPIAIJGetSeqAIJ(A, &Ad, &Ao, &cmap));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
  if (size == 1) {
    if (scall == MAT_INITIAL_MATRIX) {
      PetscCall(PetscObjectReference((PetscObject)Ad));
      *A_loc = Ad;
    } else if (scall == MAT_REUSE_MATRIX) {
      PetscCall(MatCopy(Ad, *A_loc, SAME_NONZERO_PATTERN));
    }
    if (glob) PetscCall(ISCreateStride(PetscObjectComm((PetscObject)Ad), Ad->cmap->n, Ad->cmap->rstart, 1, glob));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscObjectQueryFunction((PetscObject)A, "MatMPIAIJGetLocalMatMerge_C", &f));
  PetscCall(PetscLogEventBegin(MAT_Getlocalmat, A, 0, 0, 0));
  if (f) {
    PetscCall((*f)(A, scall, glob, A_loc));
  } else {
    Mat_SeqAIJ        *a = (Mat_SeqAIJ *)Ad->data;
    Mat_SeqAIJ        *b = (Mat_SeqAIJ *)Ao->data;
    Mat_SeqAIJ        *c;
    PetscInt          *ai = a->i, *aj = a->j;
    PetscInt          *bi = b->i, *bj = b->j;
    PetscInt          *ci, *cj;
    const PetscScalar *aa, *ba;
    PetscScalar       *ca;
    PetscInt           i, j, am, dn, on;

    PetscCall(MatGetLocalSize(Ad, &am, &dn));
    PetscCall(MatGetLocalSize(Ao, NULL, &on));
    PetscCall(MatSeqAIJGetArrayRead(Ad, &aa));
    PetscCall(MatSeqAIJGetArrayRead(Ao, &ba));
    if (scall == MAT_INITIAL_MATRIX) {
      PetscInt k;
      PetscCall(PetscMalloc1(1 + am, &ci));
      PetscCall(PetscMalloc1(ai[am] + bi[am], &cj));
      PetscCall(PetscMalloc1(ai[am] + bi[am], &ca));
      ci[0] = 0;
      for (i = 0, k = 0; i < am; i++) {
        const PetscInt ncols_o = bi[i + 1] - bi[i];
        const PetscInt ncols_d = ai[i + 1] - ai[i];
        ci[i + 1]              = ci[i] + ncols_o + ncols_d;
        /* diagonal portion of A */
        for (j = 0; j < ncols_d; j++, k++) {
          cj[k] = *aj++;
          ca[k] = *aa++;
        }
        /* off-diagonal portion of A */
        for (j = 0; j < ncols_o; j++, k++) {
          cj[k] = dn + *bj++;
          ca[k] = *ba++;
        }
      }
      /* put together the new matrix */
      PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, am, dn + on, ci, cj, ca, A_loc));
      /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
      /* Since these are PETSc arrays, change flags to free them as necessary. */
      c          = (Mat_SeqAIJ *)(*A_loc)->data;
      c->free_a  = PETSC_TRUE;
      c->free_ij = PETSC_TRUE;
      c->nonew   = 0;
      PetscCall(MatSetType(*A_loc, ((PetscObject)Ad)->type_name));
    } else if (scall == MAT_REUSE_MATRIX) {
      PetscCall(MatSeqAIJGetArrayWrite(*A_loc, &ca));
      for (i = 0; i < am; i++) {
        const PetscInt ncols_d = ai[i + 1] - ai[i];
        const PetscInt ncols_o = bi[i + 1] - bi[i];
        /* diagonal portion of A */
        for (j = 0; j < ncols_d; j++) *ca++ = *aa++;
        /* off-diagonal portion of A */
        for (j = 0; j < ncols_o; j++) *ca++ = *ba++;
      }
      PetscCall(MatSeqAIJRestoreArrayWrite(*A_loc, &ca));
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid MatReuse %d", (int)scall);
    PetscCall(MatSeqAIJRestoreArrayRead(Ad, &aa));
    PetscCall(MatSeqAIJRestoreArrayRead(Ao, &aa));
    if (glob) {
      PetscInt cst, *gidx;

      PetscCall(MatGetOwnershipRangeColumn(A, &cst, NULL));
      PetscCall(PetscMalloc1(dn + on, &gidx));
      for (i = 0; i < dn; i++) gidx[i] = cst + i;
      for (i = 0; i < on; i++) gidx[i + dn] = cmap[i];
      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)Ad), dn + on, gidx, PETSC_OWN_POINTER, glob));
    }
  }
  PetscCall(PetscLogEventEnd(MAT_Getlocalmat, A, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatMPIAIJGetLocalMatCondensed - Creates a `MATSEQAIJ` matrix from an `MATMPIAIJ` matrix by taking all its local rows and NON-ZERO columns

  Not Collective

  Input Parameters:
+ A     - the matrix
. scall - either `MAT_INITIAL_MATRIX` or `MAT_REUSE_MATRIX`
. row   - index set of rows to extract (or `NULL`)
- col   - index set of columns to extract (or `NULL`)

  Output Parameter:
. A_loc - the local sequential matrix generated

  Level: developer

.seealso: [](ch_matrices), `Mat`, `MATMPIAIJ`, `MatGetOwnershipRange()`, `MatMPIAIJGetLocalMat()`
@*/
PetscErrorCode MatMPIAIJGetLocalMatCondensed(Mat A, MatReuse scall, IS *row, IS *col, Mat *A_loc)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;
  PetscInt    i, start, end, ncols, nzA, nzB, *cmap, imark, *idx;
  IS          isrowa, iscola;
  Mat        *aloc;
  PetscBool   match;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATMPIAIJ, &match));
  PetscCheck(match, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Requires MATMPIAIJ matrix as input");
  PetscCall(PetscLogEventBegin(MAT_Getlocalmatcondensed, A, 0, 0, 0));
  if (!row) {
    start = A->rmap->rstart;
    end   = A->rmap->rend;
    PetscCall(ISCreateStride(PETSC_COMM_SELF, end - start, start, 1, &isrowa));
  } else {
    isrowa = *row;
  }
  if (!col) {
    start = A->cmap->rstart;
    cmap  = a->garray;
    nzA   = a->A->cmap->n;
    nzB   = a->B->cmap->n;
    PetscCall(PetscMalloc1(nzA + nzB, &idx));
    ncols = 0;
    for (i = 0; i < nzB; i++) {
      if (cmap[i] < start) idx[ncols++] = cmap[i];
      else break;
    }
    imark = i;
    for (i = 0; i < nzA; i++) idx[ncols++] = start + i;
    for (i = imark; i < nzB; i++) idx[ncols++] = cmap[i];
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, ncols, idx, PETSC_OWN_POINTER, &iscola));
  } else {
    iscola = *col;
  }
  if (scall != MAT_INITIAL_MATRIX) {
    PetscCall(PetscMalloc1(1, &aloc));
    aloc[0] = *A_loc;
  }
  PetscCall(MatCreateSubMatrices(A, 1, &isrowa, &iscola, scall, &aloc));
  if (!col) { /* attach global id of condensed columns */
    PetscCall(PetscObjectCompose((PetscObject)aloc[0], "_petsc_GetLocalMatCondensed_iscol", (PetscObject)iscola));
  }
  *A_loc = aloc[0];
  PetscCall(PetscFree(aloc));
  if (!row) PetscCall(ISDestroy(&isrowa));
  if (!col) PetscCall(ISDestroy(&iscola));
  PetscCall(PetscLogEventEnd(MAT_Getlocalmatcondensed, A, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * Create a sequential AIJ matrix based on row indices. a whole column is extracted once a row is matched.
 * Row could be local or remote.The routine is designed to be scalable in memory so that nothing is based
 * on a global size.
 * */
static PetscErrorCode MatCreateSeqSubMatrixWithRows_Private(Mat P, IS rows, Mat *P_oth)
{
  Mat_MPIAIJ            *p  = (Mat_MPIAIJ *)P->data;
  Mat_SeqAIJ            *pd = (Mat_SeqAIJ *)p->A->data, *po = (Mat_SeqAIJ *)p->B->data, *p_oth;
  PetscInt               plocalsize, nrows, *ilocal, *oilocal, i, lidx, *nrcols, *nlcols, ncol;
  PetscMPIInt            owner;
  PetscSFNode           *iremote, *oiremote;
  const PetscInt        *lrowindices;
  PetscSF                sf, osf;
  PetscInt               pcstart, *roffsets, *loffsets, *pnnz, j;
  PetscInt               ontotalcols, dntotalcols, ntotalcols, nout;
  MPI_Comm               comm;
  ISLocalToGlobalMapping mapping;
  const PetscScalar     *pd_a, *po_a;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)P, &comm));
  /* plocalsize is the number of roots
   * nrows is the number of leaves
   * */
  PetscCall(MatGetLocalSize(P, &plocalsize, NULL));
  PetscCall(ISGetLocalSize(rows, &nrows));
  PetscCall(PetscCalloc1(nrows, &iremote));
  PetscCall(ISGetIndices(rows, &lrowindices));
  for (i = 0; i < nrows; i++) {
    /* Find a remote index and an owner for a row
     * The row could be local or remote
     * */
    owner = 0;
    lidx  = 0;
    PetscCall(PetscLayoutFindOwnerIndex(P->rmap, lrowindices[i], &owner, &lidx));
    iremote[i].index = lidx;
    iremote[i].rank  = owner;
  }
  /* Create SF to communicate how many nonzero columns for each row */
  PetscCall(PetscSFCreate(comm, &sf));
  /* SF will figure out the number of nonzero columns for each row, and their
   * offsets
   * */
  PetscCall(PetscSFSetGraph(sf, plocalsize, nrows, NULL, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(PetscSFSetUp(sf));

  PetscCall(PetscCalloc1(2 * (plocalsize + 1), &roffsets));
  PetscCall(PetscCalloc1(2 * plocalsize, &nrcols));
  PetscCall(PetscCalloc1(nrows, &pnnz));
  roffsets[0] = 0;
  roffsets[1] = 0;
  for (i = 0; i < plocalsize; i++) {
    /* diagonal */
    nrcols[i * 2 + 0] = pd->i[i + 1] - pd->i[i];
    /* off-diagonal */
    nrcols[i * 2 + 1] = po->i[i + 1] - po->i[i];
    /* compute offsets so that we relative location for each row */
    roffsets[(i + 1) * 2 + 0] = roffsets[i * 2 + 0] + nrcols[i * 2 + 0];
    roffsets[(i + 1) * 2 + 1] = roffsets[i * 2 + 1] + nrcols[i * 2 + 1];
  }
  PetscCall(PetscCalloc1(2 * nrows, &nlcols));
  PetscCall(PetscCalloc1(2 * nrows, &loffsets));
  /* 'r' means root, and 'l' means leaf */
  PetscCall(PetscSFBcastBegin(sf, MPIU_2INT, nrcols, nlcols, MPI_REPLACE));
  PetscCall(PetscSFBcastBegin(sf, MPIU_2INT, roffsets, loffsets, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_2INT, nrcols, nlcols, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_2INT, roffsets, loffsets, MPI_REPLACE));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscFree(roffsets));
  PetscCall(PetscFree(nrcols));
  dntotalcols = 0;
  ontotalcols = 0;
  ncol        = 0;
  for (i = 0; i < nrows; i++) {
    pnnz[i] = nlcols[i * 2 + 0] + nlcols[i * 2 + 1];
    ncol    = PetscMax(pnnz[i], ncol);
    /* diagonal */
    dntotalcols += nlcols[i * 2 + 0];
    /* off-diagonal */
    ontotalcols += nlcols[i * 2 + 1];
  }
  /* We do not need to figure the right number of columns
   * since all the calculations will be done by going through the raw data
   * */
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, nrows, ncol, 0, pnnz, P_oth));
  PetscCall(MatSetUp(*P_oth));
  PetscCall(PetscFree(pnnz));
  p_oth = (Mat_SeqAIJ *)(*P_oth)->data;
  /* diagonal */
  PetscCall(PetscCalloc1(dntotalcols, &iremote));
  /* off-diagonal */
  PetscCall(PetscCalloc1(ontotalcols, &oiremote));
  /* diagonal */
  PetscCall(PetscCalloc1(dntotalcols, &ilocal));
  /* off-diagonal */
  PetscCall(PetscCalloc1(ontotalcols, &oilocal));
  dntotalcols = 0;
  ontotalcols = 0;
  ntotalcols  = 0;
  for (i = 0; i < nrows; i++) {
    owner = 0;
    PetscCall(PetscLayoutFindOwnerIndex(P->rmap, lrowindices[i], &owner, NULL));
    /* Set iremote for diag matrix */
    for (j = 0; j < nlcols[i * 2 + 0]; j++) {
      iremote[dntotalcols].index = loffsets[i * 2 + 0] + j;
      iremote[dntotalcols].rank  = owner;
      /* P_oth is seqAIJ so that ilocal need to point to the first part of memory */
      ilocal[dntotalcols++] = ntotalcols++;
    }
    /* off-diagonal */
    for (j = 0; j < nlcols[i * 2 + 1]; j++) {
      oiremote[ontotalcols].index = loffsets[i * 2 + 1] + j;
      oiremote[ontotalcols].rank  = owner;
      oilocal[ontotalcols++]      = ntotalcols++;
    }
  }
  PetscCall(ISRestoreIndices(rows, &lrowindices));
  PetscCall(PetscFree(loffsets));
  PetscCall(PetscFree(nlcols));
  PetscCall(PetscSFCreate(comm, &sf));
  /* P serves as roots and P_oth is leaves
   * Diag matrix
   * */
  PetscCall(PetscSFSetGraph(sf, pd->i[plocalsize], dntotalcols, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(PetscSFSetUp(sf));

  PetscCall(PetscSFCreate(comm, &osf));
  /* off-diagonal */
  PetscCall(PetscSFSetGraph(osf, po->i[plocalsize], ontotalcols, oilocal, PETSC_OWN_POINTER, oiremote, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetFromOptions(osf));
  PetscCall(PetscSFSetUp(osf));
  PetscCall(MatSeqAIJGetArrayRead(p->A, &pd_a));
  PetscCall(MatSeqAIJGetArrayRead(p->B, &po_a));
  /* operate on the matrix internal data to save memory */
  PetscCall(PetscSFBcastBegin(sf, MPIU_SCALAR, pd_a, p_oth->a, MPI_REPLACE));
  PetscCall(PetscSFBcastBegin(osf, MPIU_SCALAR, po_a, p_oth->a, MPI_REPLACE));
  PetscCall(MatGetOwnershipRangeColumn(P, &pcstart, NULL));
  /* Convert to global indices for diag matrix */
  for (i = 0; i < pd->i[plocalsize]; i++) pd->j[i] += pcstart;
  PetscCall(PetscSFBcastBegin(sf, MPIU_INT, pd->j, p_oth->j, MPI_REPLACE));
  /* We want P_oth store global indices */
  PetscCall(ISLocalToGlobalMappingCreate(comm, 1, p->B->cmap->n, p->garray, PETSC_COPY_VALUES, &mapping));
  /* Use memory scalable approach */
  PetscCall(ISLocalToGlobalMappingSetType(mapping, ISLOCALTOGLOBALMAPPINGHASH));
  PetscCall(ISLocalToGlobalMappingApply(mapping, po->i[plocalsize], po->j, po->j));
  PetscCall(PetscSFBcastBegin(osf, MPIU_INT, po->j, p_oth->j, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sf, MPIU_INT, pd->j, p_oth->j, MPI_REPLACE));
  /* Convert back to local indices */
  for (i = 0; i < pd->i[plocalsize]; i++) pd->j[i] -= pcstart;
  PetscCall(PetscSFBcastEnd(osf, MPIU_INT, po->j, p_oth->j, MPI_REPLACE));
  nout = 0;
  PetscCall(ISGlobalToLocalMappingApply(mapping, IS_GTOLM_DROP, po->i[plocalsize], po->j, &nout, po->j));
  PetscCheck(nout == po->i[plocalsize], comm, PETSC_ERR_ARG_INCOMP, "n %" PetscInt_FMT " does not equal to nout %" PetscInt_FMT " ", po->i[plocalsize], nout);
  PetscCall(ISLocalToGlobalMappingDestroy(&mapping));
  /* Exchange values */
  PetscCall(PetscSFBcastEnd(sf, MPIU_SCALAR, pd_a, p_oth->a, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(osf, MPIU_SCALAR, po_a, p_oth->a, MPI_REPLACE));
  PetscCall(MatSeqAIJRestoreArrayRead(p->A, &pd_a));
  PetscCall(MatSeqAIJRestoreArrayRead(p->B, &po_a));
  /* Stop PETSc from shrinking memory */
  for (i = 0; i < nrows; i++) p_oth->ilen[i] = p_oth->imax[i];
  PetscCall(MatAssemblyBegin(*P_oth, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*P_oth, MAT_FINAL_ASSEMBLY));
  /* Attach PetscSF objects to P_oth so that we can reuse it later */
  PetscCall(PetscObjectCompose((PetscObject)*P_oth, "diagsf", (PetscObject)sf));
  PetscCall(PetscObjectCompose((PetscObject)*P_oth, "offdiagsf", (PetscObject)osf));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscSFDestroy(&osf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 * Creates a SeqAIJ matrix by taking rows of B that equal to nonzero columns of local A
 * This supports MPIAIJ and MAIJ
 * */
PetscErrorCode MatGetBrowsOfAcols_MPIXAIJ(Mat A, Mat P, PetscInt dof, MatReuse reuse, Mat *P_oth)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data, *p = (Mat_MPIAIJ *)P->data;
  Mat_SeqAIJ *p_oth;
  IS          rows, map;
  PetscHMapI  hamp;
  PetscInt    i, htsize, *rowindices, off, *mapping, key, count;
  MPI_Comm    comm;
  PetscSF     sf, osf;
  PetscBool   has;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCall(PetscLogEventBegin(MAT_GetBrowsOfAocols, A, P, 0, 0));
  /* If it is the first time, create an index set of off-diag nonzero columns of A,
   *  and then create a submatrix (that often is an overlapping matrix)
   * */
  if (reuse == MAT_INITIAL_MATRIX) {
    /* Use a hash table to figure out unique keys */
    PetscCall(PetscHMapICreateWithSize(a->B->cmap->n, &hamp));
    PetscCall(PetscCalloc1(a->B->cmap->n, &mapping));
    count = 0;
    /* Assume that  a->g is sorted, otherwise the following does not make sense */
    for (i = 0; i < a->B->cmap->n; i++) {
      key = a->garray[i] / dof;
      PetscCall(PetscHMapIHas(hamp, key, &has));
      if (!has) {
        mapping[i] = count;
        PetscCall(PetscHMapISet(hamp, key, count++));
      } else {
        /* Current 'i' has the same value the previous step */
        mapping[i] = count - 1;
      }
    }
    PetscCall(ISCreateGeneral(comm, a->B->cmap->n, mapping, PETSC_OWN_POINTER, &map));
    PetscCall(PetscHMapIGetSize(hamp, &htsize));
    PetscCheck(htsize == count, comm, PETSC_ERR_ARG_INCOMP, " Size of hash map %" PetscInt_FMT " is inconsistent with count %" PetscInt_FMT, htsize, count);
    PetscCall(PetscCalloc1(htsize, &rowindices));
    off = 0;
    PetscCall(PetscHMapIGetKeys(hamp, &off, rowindices));
    PetscCall(PetscHMapIDestroy(&hamp));
    PetscCall(PetscSortInt(htsize, rowindices));
    PetscCall(ISCreateGeneral(comm, htsize, rowindices, PETSC_OWN_POINTER, &rows));
    /* In case, the matrix was already created but users want to recreate the matrix */
    PetscCall(MatDestroy(P_oth));
    PetscCall(MatCreateSeqSubMatrixWithRows_Private(P, rows, P_oth));
    PetscCall(PetscObjectCompose((PetscObject)*P_oth, "aoffdiagtopothmapping", (PetscObject)map));
    PetscCall(ISDestroy(&map));
    PetscCall(ISDestroy(&rows));
  } else if (reuse == MAT_REUSE_MATRIX) {
    /* If matrix was already created, we simply update values using SF objects
     * that as attached to the matrix earlier.
     */
    const PetscScalar *pd_a, *po_a;

    PetscCall(PetscObjectQuery((PetscObject)*P_oth, "diagsf", (PetscObject *)&sf));
    PetscCall(PetscObjectQuery((PetscObject)*P_oth, "offdiagsf", (PetscObject *)&osf));
    PetscCheck(sf && osf, comm, PETSC_ERR_ARG_NULL, "Matrix is not initialized yet");
    p_oth = (Mat_SeqAIJ *)(*P_oth)->data;
    /* Update values in place */
    PetscCall(MatSeqAIJGetArrayRead(p->A, &pd_a));
    PetscCall(MatSeqAIJGetArrayRead(p->B, &po_a));
    PetscCall(PetscSFBcastBegin(sf, MPIU_SCALAR, pd_a, p_oth->a, MPI_REPLACE));
    PetscCall(PetscSFBcastBegin(osf, MPIU_SCALAR, po_a, p_oth->a, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf, MPIU_SCALAR, pd_a, p_oth->a, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(osf, MPIU_SCALAR, po_a, p_oth->a, MPI_REPLACE));
    PetscCall(MatSeqAIJRestoreArrayRead(p->A, &pd_a));
    PetscCall(MatSeqAIJRestoreArrayRead(p->B, &po_a));
  } else SETERRQ(comm, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown reuse type");
  PetscCall(PetscLogEventEnd(MAT_GetBrowsOfAocols, A, P, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatGetBrowsOfAcols - Returns `IS` that contain rows of `B` that equal to nonzero columns of local `A`

  Collective

  Input Parameters:
+ A     - the first matrix in `MATMPIAIJ` format
. B     - the second matrix in `MATMPIAIJ` format
- scall - either `MAT_INITIAL_MATRIX` or `MAT_REUSE_MATRIX`

  Output Parameters:
+ rowb  - On input index sets of rows of B to extract (or `NULL`), modified on output
. colb  - On input index sets of columns of B to extract (or `NULL`), modified on output
- B_seq - the sequential matrix generated

  Level: developer

.seealso: `Mat`, `MATMPIAIJ`, `IS`, `MatReuse`
@*/
PetscErrorCode MatGetBrowsOfAcols(Mat A, Mat B, MatReuse scall, IS *rowb, IS *colb, Mat *B_seq)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;
  PetscInt   *idx, i, start, ncols, nzA, nzB, *cmap, imark;
  IS          isrowb, iscolb;
  Mat        *bseq = NULL;

  PetscFunctionBegin;
  PetscCheck(A->cmap->rstart == B->rmap->rstart && A->cmap->rend == B->rmap->rend, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Matrix local dimensions are incompatible, (%" PetscInt_FMT ", %" PetscInt_FMT ") != (%" PetscInt_FMT ",%" PetscInt_FMT ")",
             A->cmap->rstart, A->cmap->rend, B->rmap->rstart, B->rmap->rend);
  PetscCall(PetscLogEventBegin(MAT_GetBrowsOfAcols, A, B, 0, 0));

  if (scall == MAT_INITIAL_MATRIX) {
    start = A->cmap->rstart;
    cmap  = a->garray;
    nzA   = a->A->cmap->n;
    nzB   = a->B->cmap->n;
    PetscCall(PetscMalloc1(nzA + nzB, &idx));
    ncols = 0;
    for (i = 0; i < nzB; i++) { /* row < local row index */
      if (cmap[i] < start) idx[ncols++] = cmap[i];
      else break;
    }
    imark = i;
    for (i = 0; i < nzA; i++) idx[ncols++] = start + i;   /* local rows */
    for (i = imark; i < nzB; i++) idx[ncols++] = cmap[i]; /* row > local row index */
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, ncols, idx, PETSC_OWN_POINTER, &isrowb));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, B->cmap->N, 0, 1, &iscolb));
  } else {
    PetscCheck(rowb && colb, PETSC_COMM_SELF, PETSC_ERR_SUP, "IS rowb and colb must be provided for MAT_REUSE_MATRIX");
    isrowb = *rowb;
    iscolb = *colb;
    PetscCall(PetscMalloc1(1, &bseq));
    bseq[0] = *B_seq;
  }
  PetscCall(MatCreateSubMatrices(B, 1, &isrowb, &iscolb, scall, &bseq));
  *B_seq = bseq[0];
  PetscCall(PetscFree(bseq));
  if (!rowb) {
    PetscCall(ISDestroy(&isrowb));
  } else {
    *rowb = isrowb;
  }
  if (!colb) {
    PetscCall(ISDestroy(&iscolb));
  } else {
    *colb = iscolb;
  }
  PetscCall(PetscLogEventEnd(MAT_GetBrowsOfAcols, A, B, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    MatGetBrowsOfAoCols_MPIAIJ - Creates a `MATSEQAIJ` matrix by taking rows of B that equal to nonzero columns
    of the OFF-DIAGONAL portion of local A

    Collective

   Input Parameters:
+    A,B - the matrices in `MATMPIAIJ` format
-    scall - either `MAT_INITIAL_MATRIX` or `MAT_REUSE_MATRIX`

   Output Parameter:
+    startsj_s - starting point in B's sending j-arrays, saved for MAT_REUSE (or NULL)
.    startsj_r - starting point in B's receiving j-arrays, saved for MAT_REUSE (or NULL)
.    bufa_ptr - array for sending matrix values, saved for MAT_REUSE (or NULL)
-    B_oth - the sequential matrix generated with size aBn=a->B->cmap->n by B->cmap->N

    Developer Note:
    This directly accesses information inside the VecScatter associated with the matrix-vector product
     for this matrix. This is not desirable..

    Level: developer

*/

PetscErrorCode MatGetBrowsOfAoCols_MPIAIJ(Mat A, Mat B, MatReuse scall, PetscInt **startsj_s, PetscInt **startsj_r, MatScalar **bufa_ptr, Mat *B_oth)
{
  Mat_MPIAIJ        *a = (Mat_MPIAIJ *)A->data;
  VecScatter         ctx;
  MPI_Comm           comm;
  const PetscMPIInt *rprocs, *sprocs;
  PetscMPIInt        nrecvs, nsends;
  const PetscInt    *srow, *rstarts, *sstarts;
  PetscInt          *rowlen, *bufj, *bufJ, ncols = 0, aBn = a->B->cmap->n, row, *b_othi, *b_othj, *rvalues = NULL, *svalues = NULL, *cols, sbs, rbs;
  PetscInt           i, j, k = 0, l, ll, nrows, *rstartsj = NULL, *sstartsj, len;
  PetscScalar       *b_otha, *bufa, *bufA, *vals = NULL;
  MPI_Request       *reqs = NULL, *rwaits = NULL, *swaits = NULL;
  PetscMPIInt        size, tag, rank, nreqs;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  PetscCheck(A->cmap->rstart == B->rmap->rstart && A->cmap->rend == B->rmap->rend, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Matrix local dimensions are incompatible, (%" PetscInt_FMT ", %" PetscInt_FMT ") != (%" PetscInt_FMT ",%" PetscInt_FMT ")",
             A->cmap->rstart, A->cmap->rend, B->rmap->rstart, B->rmap->rend);
  PetscCall(PetscLogEventBegin(MAT_GetBrowsOfAocols, A, B, 0, 0));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  if (size == 1) {
    startsj_s = NULL;
    bufa_ptr  = NULL;
    *B_oth    = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  ctx = a->Mvctx;
  tag = ((PetscObject)ctx)->tag;

  PetscCall(VecScatterGetRemote_Private(ctx, PETSC_TRUE /*send*/, &nsends, &sstarts, &srow, &sprocs, &sbs));
  /* rprocs[] must be ordered so that indices received from them are ordered in rvalues[], which is key to algorithms used in this subroutine */
  PetscCall(VecScatterGetRemoteOrdered_Private(ctx, PETSC_FALSE /*recv*/, &nrecvs, &rstarts, NULL /*indices not needed*/, &rprocs, &rbs));
  PetscCall(PetscMPIIntCast(nsends + nrecvs, &nreqs));
  PetscCall(PetscMalloc1(nreqs, &reqs));
  rwaits = reqs;
  swaits = PetscSafePointerPlusOffset(reqs, nrecvs);

  if (!startsj_s || !bufa_ptr) scall = MAT_INITIAL_MATRIX;
  if (scall == MAT_INITIAL_MATRIX) {
    /* i-array */
    /*  post receives */
    if (nrecvs) PetscCall(PetscMalloc1(rbs * (rstarts[nrecvs] - rstarts[0]), &rvalues)); /* rstarts can be NULL when nrecvs=0 */
    for (i = 0; i < nrecvs; i++) {
      rowlen = rvalues + rstarts[i] * rbs;
      nrows  = (rstarts[i + 1] - rstarts[i]) * rbs; /* num of indices to be received */
      PetscCallMPI(MPIU_Irecv(rowlen, nrows, MPIU_INT, rprocs[i], tag, comm, rwaits + i));
    }

    /* pack the outgoing message */
    PetscCall(PetscMalloc2(nsends + 1, &sstartsj, nrecvs + 1, &rstartsj));

    sstartsj[0] = 0;
    rstartsj[0] = 0;
    len         = 0; /* total length of j or a array to be sent */
    if (nsends) {
      k = sstarts[0]; /* ATTENTION: sstarts[0] and rstarts[0] are not necessarily zero */
      PetscCall(PetscMalloc1(sbs * (sstarts[nsends] - sstarts[0]), &svalues));
    }
    for (i = 0; i < nsends; i++) {
      rowlen = svalues + (sstarts[i] - sstarts[0]) * sbs;
      nrows  = sstarts[i + 1] - sstarts[i]; /* num of block rows */
      for (j = 0; j < nrows; j++) {
        row = srow[k] + B->rmap->range[rank]; /* global row idx */
        for (l = 0; l < sbs; l++) {
          PetscCall(MatGetRow_MPIAIJ(B, row + l, &ncols, NULL, NULL)); /* rowlength */

          rowlen[j * sbs + l] = ncols;

          len += ncols;
          PetscCall(MatRestoreRow_MPIAIJ(B, row + l, &ncols, NULL, NULL));
        }
        k++;
      }
      PetscCallMPI(MPIU_Isend(rowlen, nrows * sbs, MPIU_INT, sprocs[i], tag, comm, swaits + i));

      sstartsj[i + 1] = len; /* starting point of (i+1)-th outgoing msg in bufj and bufa */
    }
    /* recvs and sends of i-array are completed */
    if (nreqs) PetscCallMPI(MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE));
    PetscCall(PetscFree(svalues));

    /* allocate buffers for sending j and a arrays */
    PetscCall(PetscMalloc1(len + 1, &bufj));
    PetscCall(PetscMalloc1(len + 1, &bufa));

    /* create i-array of B_oth */
    PetscCall(PetscMalloc1(aBn + 2, &b_othi));

    b_othi[0] = 0;
    len       = 0; /* total length of j or a array to be received */
    k         = 0;
    for (i = 0; i < nrecvs; i++) {
      rowlen = rvalues + (rstarts[i] - rstarts[0]) * rbs;
      nrows  = (rstarts[i + 1] - rstarts[i]) * rbs; /* num of rows to be received */
      for (j = 0; j < nrows; j++) {
        b_othi[k + 1] = b_othi[k] + rowlen[j];
        PetscCall(PetscIntSumError(rowlen[j], len, &len));
        k++;
      }
      rstartsj[i + 1] = len; /* starting point of (i+1)-th incoming msg in bufj and bufa */
    }
    PetscCall(PetscFree(rvalues));

    /* allocate space for j and a arrays of B_oth */
    PetscCall(PetscMalloc1(b_othi[aBn] + 1, &b_othj));
    PetscCall(PetscMalloc1(b_othi[aBn] + 1, &b_otha));

    /* j-array */
    /*  post receives of j-array */
    for (i = 0; i < nrecvs; i++) {
      nrows = rstartsj[i + 1] - rstartsj[i]; /* length of the msg received */
      PetscCallMPI(MPIU_Irecv(b_othj + rstartsj[i], nrows, MPIU_INT, rprocs[i], tag, comm, rwaits + i));
    }

    /* pack the outgoing message j-array */
    if (nsends) k = sstarts[0];
    for (i = 0; i < nsends; i++) {
      nrows = sstarts[i + 1] - sstarts[i]; /* num of block rows */
      bufJ  = bufj + sstartsj[i];
      for (j = 0; j < nrows; j++) {
        row = srow[k++] + B->rmap->range[rank]; /* global row idx */
        for (ll = 0; ll < sbs; ll++) {
          PetscCall(MatGetRow_MPIAIJ(B, row + ll, &ncols, &cols, NULL));
          for (l = 0; l < ncols; l++) *bufJ++ = cols[l];
          PetscCall(MatRestoreRow_MPIAIJ(B, row + ll, &ncols, &cols, NULL));
        }
      }
      PetscCallMPI(MPIU_Isend(bufj + sstartsj[i], sstartsj[i + 1] - sstartsj[i], MPIU_INT, sprocs[i], tag, comm, swaits + i));
    }

    /* recvs and sends of j-array are completed */
    if (nreqs) PetscCallMPI(MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE));
  } else if (scall == MAT_REUSE_MATRIX) {
    sstartsj = *startsj_s;
    rstartsj = *startsj_r;
    bufa     = *bufa_ptr;
    PetscCall(MatSeqAIJGetArrayWrite(*B_oth, &b_otha));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix P does not possess an object container");

  /* a-array */
  /*  post receives of a-array */
  for (i = 0; i < nrecvs; i++) {
    nrows = rstartsj[i + 1] - rstartsj[i]; /* length of the msg received */
    PetscCallMPI(MPIU_Irecv(b_otha + rstartsj[i], nrows, MPIU_SCALAR, rprocs[i], tag, comm, rwaits + i));
  }

  /* pack the outgoing message a-array */
  if (nsends) k = sstarts[0];
  for (i = 0; i < nsends; i++) {
    nrows = sstarts[i + 1] - sstarts[i]; /* num of block rows */
    bufA  = bufa + sstartsj[i];
    for (j = 0; j < nrows; j++) {
      row = srow[k++] + B->rmap->range[rank]; /* global row idx */
      for (ll = 0; ll < sbs; ll++) {
        PetscCall(MatGetRow_MPIAIJ(B, row + ll, &ncols, NULL, &vals));
        for (l = 0; l < ncols; l++) *bufA++ = vals[l];
        PetscCall(MatRestoreRow_MPIAIJ(B, row + ll, &ncols, NULL, &vals));
      }
    }
    PetscCallMPI(MPIU_Isend(bufa + sstartsj[i], sstartsj[i + 1] - sstartsj[i], MPIU_SCALAR, sprocs[i], tag, comm, swaits + i));
  }
  /* recvs and sends of a-array are completed */
  if (nreqs) PetscCallMPI(MPI_Waitall(nreqs, reqs, MPI_STATUSES_IGNORE));
  PetscCall(PetscFree(reqs));

  if (scall == MAT_INITIAL_MATRIX) {
    Mat_SeqAIJ *b_oth;

    /* put together the new matrix */
    PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, aBn, B->cmap->N, b_othi, b_othj, b_otha, B_oth));

    /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
    /* Since these are PETSc arrays, change flags to free them as necessary. */
    b_oth          = (Mat_SeqAIJ *)(*B_oth)->data;
    b_oth->free_a  = PETSC_TRUE;
    b_oth->free_ij = PETSC_TRUE;
    b_oth->nonew   = 0;

    PetscCall(PetscFree(bufj));
    if (!startsj_s || !bufa_ptr) {
      PetscCall(PetscFree2(sstartsj, rstartsj));
      PetscCall(PetscFree(bufa_ptr));
    } else {
      *startsj_s = sstartsj;
      *startsj_r = rstartsj;
      *bufa_ptr  = bufa;
    }
  } else if (scall == MAT_REUSE_MATRIX) {
    PetscCall(MatSeqAIJRestoreArrayWrite(*B_oth, &b_otha));
  }

  PetscCall(VecScatterRestoreRemote_Private(ctx, PETSC_TRUE, &nsends, &sstarts, &srow, &sprocs, &sbs));
  PetscCall(VecScatterRestoreRemoteOrdered_Private(ctx, PETSC_FALSE, &nrecvs, &rstarts, NULL, &rprocs, &rbs));
  PetscCall(PetscLogEventEnd(MAT_GetBrowsOfAocols, A, B, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJCRL(Mat, MatType, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJPERM(Mat, MatType, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJSELL(Mat, MatType, MatReuse, Mat *);
#if defined(PETSC_HAVE_MKL_SPARSE)
PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJMKL(Mat, MatType, MatReuse, Mat *);
#endif
PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIBAIJ(Mat, MatType, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPISBAIJ(Mat, MatType, MatReuse, Mat *);
#if defined(PETSC_HAVE_ELEMENTAL)
PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_Elemental(Mat, MatType, MatReuse, Mat *);
#endif
#if defined(PETSC_HAVE_SCALAPACK)
PETSC_INTERN PetscErrorCode MatConvert_AIJ_ScaLAPACK(Mat, MatType, MatReuse, Mat *);
#endif
#if defined(PETSC_HAVE_HYPRE)
PETSC_INTERN PetscErrorCode MatConvert_AIJ_HYPRE(Mat, MatType, MatReuse, Mat *);
#endif
#if defined(PETSC_HAVE_CUDA)
PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJCUSPARSE(Mat, MatType, MatReuse, Mat *);
#endif
#if defined(PETSC_HAVE_HIP)
PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJHIPSPARSE(Mat, MatType, MatReuse, Mat *);
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJKokkos(Mat, MatType, MatReuse, Mat *);
#endif
PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPISELL(Mat, MatType, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatConvert_XAIJ_IS(Mat, MatType, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatProductSetFromOptions_IS_XAIJ(Mat);

/*
    Computes (B'*A')' since computing B*A directly is untenable

               n                       p                          p
        [             ]       [             ]         [                 ]
      m [      A      ]  *  n [       B     ]   =   m [         C       ]
        [             ]       [             ]         [                 ]

*/
static PetscErrorCode MatMatMultNumeric_MPIDense_MPIAIJ(Mat A, Mat B, Mat C)
{
  Mat At, Bt, Ct;

  PetscFunctionBegin;
  PetscCall(MatTranspose(A, MAT_INITIAL_MATRIX, &At));
  PetscCall(MatTranspose(B, MAT_INITIAL_MATRIX, &Bt));
  PetscCall(MatMatMult(Bt, At, MAT_INITIAL_MATRIX, PETSC_CURRENT, &Ct));
  PetscCall(MatDestroy(&At));
  PetscCall(MatDestroy(&Bt));
  PetscCall(MatTransposeSetPrecursor(Ct, C));
  PetscCall(MatTranspose(Ct, MAT_REUSE_MATRIX, &C));
  PetscCall(MatDestroy(&Ct));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatMultSymbolic_MPIDense_MPIAIJ(Mat A, Mat B, PetscReal fill, Mat C)
{
  PetscBool cisdense;

  PetscFunctionBegin;
  PetscCheck(A->cmap->n == B->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "A->cmap->n %" PetscInt_FMT " != B->rmap->n %" PetscInt_FMT, A->cmap->n, B->rmap->n);
  PetscCall(MatSetSizes(C, A->rmap->n, B->cmap->n, A->rmap->N, B->cmap->N));
  PetscCall(MatSetBlockSizesFromMats(C, A, B));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)C, &cisdense, MATMPIDENSE, MATMPIDENSECUDA, MATMPIDENSEHIP, ""));
  if (!cisdense) PetscCall(MatSetType(C, ((PetscObject)A)->type_name));
  PetscCall(MatSetUp(C));

  C->ops->matmultnumeric = MatMatMultNumeric_MPIDense_MPIAIJ;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSetFromOptions_MPIDense_MPIAIJ_AB(Mat C)
{
  Mat_Product *product = C->product;
  Mat          A = product->A, B = product->B;

  PetscFunctionBegin;
  PetscCheck(A->cmap->rstart == B->rmap->rstart && A->cmap->rend == B->rmap->rend, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Matrix local dimensions are incompatible, (%" PetscInt_FMT ", %" PetscInt_FMT ") != (%" PetscInt_FMT ",%" PetscInt_FMT ")",
             A->cmap->rstart, A->cmap->rend, B->rmap->rstart, B->rmap->rend);
  C->ops->matmultsymbolic = MatMatMultSymbolic_MPIDense_MPIAIJ;
  C->ops->productsymbolic = MatProductSymbolic_AB;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_MPIDense_MPIAIJ(Mat C)
{
  Mat_Product *product = C->product;

  PetscFunctionBegin;
  if (product->type == MATPRODUCT_AB) PetscCall(MatProductSetFromOptions_MPIDense_MPIAIJ_AB(C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Merge two sets of sorted nonzeros and return a CSR for the merged (sequential) matrix

  Input Parameters:

    j1,rowBegin1,rowEnd1,jmap1: describe the first set of nonzeros (Set1)
    j2,rowBegin2,rowEnd2,jmap2: describe the second set of nonzeros (Set2)

    mat: both sets' nonzeros are on m rows, where m is the number of local rows of the matrix mat

    For Set1, j1[] contains column indices of the nonzeros.
    For the k-th row (0<=k<m), [rowBegin1[k],rowEnd1[k]) index into j1[] and point to the begin/end nonzero in row k
    respectively (note rowEnd1[k] is not necessarily equal to rwoBegin1[k+1]). Indices in this range of j1[] are sorted,
    but might have repeats. jmap1[t+1] - jmap1[t] is the number of repeats for the t-th unique nonzero in Set1.

    Similar for Set2.

    This routine merges the two sets of nonzeros row by row and removes repeats.

  Output Parameters: (memory is allocated by the caller)

    i[],j[]: the CSR of the merged matrix, which has m rows.
    imap1[]: the k-th unique nonzero in Set1 (k=0,1,...) corresponds to imap1[k]-th unique nonzero in the merged matrix.
    imap2[]: similar to imap1[], but for Set2.
    Note we order nonzeros row-by-row and from left to right.
*/
static PetscErrorCode MatMergeEntries_Internal(Mat mat, const PetscInt j1[], const PetscInt j2[], const PetscCount rowBegin1[], const PetscCount rowEnd1[], const PetscCount rowBegin2[], const PetscCount rowEnd2[], const PetscCount jmap1[], const PetscCount jmap2[], PetscCount imap1[], PetscCount imap2[], PetscInt i[], PetscInt j[])
{
  PetscInt   r, m; /* Row index of mat */
  PetscCount t, t1, t2, b1, e1, b2, e2;

  PetscFunctionBegin;
  PetscCall(MatGetLocalSize(mat, &m, NULL));
  t1 = t2 = t = 0; /* Count unique nonzeros of in Set1, Set1 and the merged respectively */
  i[0]        = 0;
  for (r = 0; r < m; r++) { /* Do row by row merging */
    b1 = rowBegin1[r];
    e1 = rowEnd1[r];
    b2 = rowBegin2[r];
    e2 = rowEnd2[r];
    while (b1 < e1 && b2 < e2) {
      if (j1[b1] == j2[b2]) { /* Same column index and hence same nonzero */
        j[t]      = j1[b1];
        imap1[t1] = t;
        imap2[t2] = t;
        b1 += jmap1[t1 + 1] - jmap1[t1]; /* Jump to next unique local nonzero */
        b2 += jmap2[t2 + 1] - jmap2[t2]; /* Jump to next unique remote nonzero */
        t1++;
        t2++;
        t++;
      } else if (j1[b1] < j2[b2]) {
        j[t]      = j1[b1];
        imap1[t1] = t;
        b1 += jmap1[t1 + 1] - jmap1[t1];
        t1++;
        t++;
      } else {
        j[t]      = j2[b2];
        imap2[t2] = t;
        b2 += jmap2[t2 + 1] - jmap2[t2];
        t2++;
        t++;
      }
    }
    /* Merge the remaining in either j1[] or j2[] */
    while (b1 < e1) {
      j[t]      = j1[b1];
      imap1[t1] = t;
      b1 += jmap1[t1 + 1] - jmap1[t1];
      t1++;
      t++;
    }
    while (b2 < e2) {
      j[t]      = j2[b2];
      imap2[t2] = t;
      b2 += jmap2[t2 + 1] - jmap2[t2];
      t2++;
      t++;
    }
    PetscCall(PetscIntCast(t, i + r + 1));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Split nonzeros in a block of local rows into two subsets: those in the diagonal block and those in the off-diagonal block

  Input Parameters:
    mat: an MPI matrix that provides row and column layout information for splitting. Let's say its number of local rows is m.
    n,i[],j[],perm[]: there are n input entries, belonging to m rows. Row/col indices of the entries are stored in i[] and j[]
      respectively, along with a permutation array perm[]. Length of the i[],j[],perm[] arrays is n.

      i[] is already sorted, but within a row, j[] is not sorted and might have repeats.
      i[] might contain negative indices at the beginning, which means the corresponding entries should be ignored in the splitting.

  Output Parameters:
    j[],perm[]: the routine needs to sort j[] within each row along with perm[].
    rowBegin[],rowMid[],rowEnd[]: of length m, and the memory is preallocated and zeroed by the caller.
      They contain indices pointing to j[]. For 0<=r<m, [rowBegin[r],rowMid[r]) point to begin/end entries of row r of the diagonal block,
      and [rowMid[r],rowEnd[r]) point to begin/end entries of row r of the off-diagonal block.

    Aperm[],Ajmap[],Atot,Annz: Arrays are allocated by this routine.
      Atot: number of entries belonging to the diagonal block.
      Annz: number of unique nonzeros belonging to the diagonal block.
      Aperm[Atot] stores values from perm[] for entries belonging to the diagonal block. Length of Aperm[] is Atot, though it may also count
        repeats (i.e., same 'i,j' pair).
      Ajmap[Annz+1] stores the number of repeats of each unique entry belonging to the diagonal block. More precisely, Ajmap[t+1] - Ajmap[t]
        is the number of repeats for the t-th unique entry in the diagonal block. Ajmap[0] is always 0.

      Atot: number of entries belonging to the diagonal block
      Annz: number of unique nonzeros belonging to the diagonal block.

    Bperm[], Bjmap[], Btot, Bnnz are similar but for the off-diagonal block.

    Aperm[],Bperm[],Ajmap[] and Bjmap[] are allocated separately by this routine with PetscMalloc1().
*/
static PetscErrorCode MatSplitEntries_Internal(Mat mat, PetscCount n, const PetscInt i[], PetscInt j[], PetscCount perm[], PetscCount rowBegin[], PetscCount rowMid[], PetscCount rowEnd[], PetscCount *Atot_, PetscCount **Aperm_, PetscCount *Annz_, PetscCount **Ajmap_, PetscCount *Btot_, PetscCount **Bperm_, PetscCount *Bnnz_, PetscCount **Bjmap_)
{
  PetscInt    cstart, cend, rstart, rend, row, col;
  PetscCount  Atot = 0, Btot = 0; /* Total number of nonzeros in the diagonal and off-diagonal blocks */
  PetscCount  Annz = 0, Bnnz = 0; /* Number of unique nonzeros in the diagonal and off-diagonal blocks */
  PetscCount  k, m, p, q, r, s, mid;
  PetscCount *Aperm, *Bperm, *Ajmap, *Bjmap;

  PetscFunctionBegin;
  PetscCall(PetscLayoutGetRange(mat->rmap, &rstart, &rend));
  PetscCall(PetscLayoutGetRange(mat->cmap, &cstart, &cend));
  m = rend - rstart;

  /* Skip negative rows */
  for (k = 0; k < n; k++)
    if (i[k] >= 0) break;

  /* Process [k,n): sort and partition each local row into diag and offdiag portions,
     fill rowBegin[], rowMid[], rowEnd[], and count Atot, Btot, Annz, Bnnz.
  */
  while (k < n) {
    row = i[k];
    /* Entries in [k,s) are in one row. Shift diagonal block col indices so that diag is ahead of offdiag after sorting the row */
    for (s = k; s < n; s++)
      if (i[s] != row) break;

    /* Shift diag columns to range of [-PETSC_INT_MAX, -1] */
    for (p = k; p < s; p++) {
      if (j[p] >= cstart && j[p] < cend) j[p] -= PETSC_INT_MAX;
      else PetscAssert((j[p] >= 0) && (j[p] <= mat->cmap->N), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Column index %" PetscInt_FMT " is out of range", j[p]);
    }
    PetscCall(PetscSortIntWithCountArray(s - k, j + k, perm + k));
    PetscCall(PetscSortedIntUpperBound(j, k, s, -1, &mid)); /* Separate [k,s) into [k,mid) for diag and [mid,s) for offdiag */
    rowBegin[row - rstart] = k;
    rowMid[row - rstart]   = mid;
    rowEnd[row - rstart]   = s;

    /* Count nonzeros of this diag/offdiag row, which might have repeats */
    Atot += mid - k;
    Btot += s - mid;

    /* Count unique nonzeros of this diag row */
    for (p = k; p < mid;) {
      col = j[p];
      do {
        j[p] += PETSC_INT_MAX; /* Revert the modified diagonal indices */
        p++;
      } while (p < mid && j[p] == col);
      Annz++;
    }

    /* Count unique nonzeros of this offdiag row */
    for (p = mid; p < s;) {
      col = j[p];
      do {
        p++;
      } while (p < s && j[p] == col);
      Bnnz++;
    }
    k = s;
  }

  /* Allocation according to Atot, Btot, Annz, Bnnz */
  PetscCall(PetscMalloc1(Atot, &Aperm));
  PetscCall(PetscMalloc1(Btot, &Bperm));
  PetscCall(PetscMalloc1(Annz + 1, &Ajmap));
  PetscCall(PetscMalloc1(Bnnz + 1, &Bjmap));

  /* Re-scan indices and copy diag/offdiag permutation indices to Aperm, Bperm and also fill Ajmap and Bjmap */
  Ajmap[0] = Bjmap[0] = Atot = Btot = Annz = Bnnz = 0;
  for (r = 0; r < m; r++) {
    k   = rowBegin[r];
    mid = rowMid[r];
    s   = rowEnd[r];
    PetscCall(PetscArraycpy(PetscSafePointerPlusOffset(Aperm, Atot), PetscSafePointerPlusOffset(perm, k), mid - k));
    PetscCall(PetscArraycpy(PetscSafePointerPlusOffset(Bperm, Btot), PetscSafePointerPlusOffset(perm, mid), s - mid));
    Atot += mid - k;
    Btot += s - mid;

    /* Scan column indices in this row and find out how many repeats each unique nonzero has */
    for (p = k; p < mid;) {
      col = j[p];
      q   = p;
      do {
        p++;
      } while (p < mid && j[p] == col);
      Ajmap[Annz + 1] = Ajmap[Annz] + (p - q);
      Annz++;
    }

    for (p = mid; p < s;) {
      col = j[p];
      q   = p;
      do {
        p++;
      } while (p < s && j[p] == col);
      Bjmap[Bnnz + 1] = Bjmap[Bnnz] + (p - q);
      Bnnz++;
    }
  }
  /* Output */
  *Aperm_ = Aperm;
  *Annz_  = Annz;
  *Atot_  = Atot;
  *Ajmap_ = Ajmap;
  *Bperm_ = Bperm;
  *Bnnz_  = Bnnz;
  *Btot_  = Btot;
  *Bjmap_ = Bjmap;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Expand the jmap[] array to make a new one in view of nonzeros in the merged matrix

  Input Parameters:
    nnz1: number of unique nonzeros in a set that was used to produce imap[], jmap[]
    nnz:  number of unique nonzeros in the merged matrix
    imap[nnz1]: i-th nonzero in the set is the imap[i]-th nonzero in the merged matrix
    jmap[nnz1+1]: i-th nonzero in the set has jmap[i+1] - jmap[i] repeats in the set

  Output Parameter: (memory is allocated by the caller)
    jmap_new[nnz+1]: i-th nonzero in the merged matrix has jmap_new[i+1] - jmap_new[i] repeats in the set

  Example:
    nnz1 = 4
    nnz  = 6
    imap = [1,3,4,5]
    jmap = [0,3,5,6,7]
   then,
    jmap_new = [0,0,3,3,5,6,7]
*/
static PetscErrorCode ExpandJmap_Internal(PetscCount nnz1, PetscCount nnz, const PetscCount imap[], const PetscCount jmap[], PetscCount jmap_new[])
{
  PetscCount k, p;

  PetscFunctionBegin;
  jmap_new[0] = 0;
  p           = nnz;                /* p loops over jmap_new[] backwards */
  for (k = nnz1 - 1; k >= 0; k--) { /* k loops over imap[] */
    for (; p > imap[k]; p--) jmap_new[p] = jmap[k + 1];
  }
  for (; p >= 0; p--) jmap_new[p] = jmap[0];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCOOStructDestroy_MPIAIJ(void *data)
{
  MatCOOStruct_MPIAIJ *coo = (MatCOOStruct_MPIAIJ *)data;

  PetscFunctionBegin;
  PetscCall(PetscSFDestroy(&coo->sf));
  PetscCall(PetscFree(coo->Aperm1));
  PetscCall(PetscFree(coo->Bperm1));
  PetscCall(PetscFree(coo->Ajmap1));
  PetscCall(PetscFree(coo->Bjmap1));
  PetscCall(PetscFree(coo->Aimap2));
  PetscCall(PetscFree(coo->Bimap2));
  PetscCall(PetscFree(coo->Aperm2));
  PetscCall(PetscFree(coo->Bperm2));
  PetscCall(PetscFree(coo->Ajmap2));
  PetscCall(PetscFree(coo->Bjmap2));
  PetscCall(PetscFree(coo->Cperm1));
  PetscCall(PetscFree2(coo->sendbuf, coo->recvbuf));
  PetscCall(PetscFree(coo));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSetPreallocationCOO_MPIAIJ(Mat mat, PetscCount coo_n, PetscInt coo_i[], PetscInt coo_j[])
{
  MPI_Comm             comm;
  PetscMPIInt          rank, size;
  PetscInt             m, n, M, N, rstart, rend, cstart, cend; /* Sizes, indices of row/col, therefore with type PetscInt */
  PetscCount           k, p, q, rem;                           /* Loop variables over coo arrays */
  Mat_MPIAIJ          *mpiaij = (Mat_MPIAIJ *)mat->data;
  PetscContainer       container;
  MatCOOStruct_MPIAIJ *coo;

  PetscFunctionBegin;
  PetscCall(PetscFree(mpiaij->garray));
  PetscCall(VecDestroy(&mpiaij->lvec));
#if defined(PETSC_USE_CTABLE)
  PetscCall(PetscHMapIDestroy(&mpiaij->colmap));
#else
  PetscCall(PetscFree(mpiaij->colmap));
#endif
  PetscCall(VecScatterDestroy(&mpiaij->Mvctx));
  mat->assembled     = PETSC_FALSE;
  mat->was_assembled = PETSC_FALSE;

  PetscCall(PetscObjectGetComm((PetscObject)mat, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscLayoutSetUp(mat->rmap));
  PetscCall(PetscLayoutSetUp(mat->cmap));
  PetscCall(PetscLayoutGetRange(mat->rmap, &rstart, &rend));
  PetscCall(PetscLayoutGetRange(mat->cmap, &cstart, &cend));
  PetscCall(MatGetLocalSize(mat, &m, &n));
  PetscCall(MatGetSize(mat, &M, &N));

  /* Sort (i,j) by row along with a permutation array, so that the to-be-ignored */
  /* entries come first, then local rows, then remote rows.                     */
  PetscCount n1 = coo_n, *perm1;
  PetscInt  *i1 = coo_i, *j1 = coo_j;

  PetscCall(PetscMalloc1(n1, &perm1));
  for (k = 0; k < n1; k++) perm1[k] = k;

  /* Manipulate indices so that entries with negative row or col indices will have smallest
     row indices, local entries will have greater but negative row indices, and remote entries
     will have positive row indices.
  */
  for (k = 0; k < n1; k++) {
    if (i1[k] < 0 || j1[k] < 0) i1[k] = PETSC_INT_MIN;                /* e.g., -2^31, minimal to move them ahead */
    else if (i1[k] >= rstart && i1[k] < rend) i1[k] -= PETSC_INT_MAX; /* e.g., minus 2^31-1 to shift local rows to range of [-PETSC_INT_MAX, -1] */
    else {
      PetscCheck(!mat->nooffprocentries, PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "MAT_NO_OFF_PROC_ENTRIES is set but insert to remote rows");
      if (mpiaij->donotstash) i1[k] = PETSC_INT_MIN; /* Ignore offproc entries as if they had negative indices */
    }
  }

  /* Sort by row; after that, [0,k) have ignored entries, [k,rem) have local rows and [rem,n1) have remote rows */
  PetscCall(PetscSortIntWithIntCountArrayPair(n1, i1, j1, perm1));

  /* Advance k to the first entry we need to take care of */
  for (k = 0; k < n1; k++)
    if (i1[k] > PETSC_INT_MIN) break;
  PetscCount i1start = k;

  PetscCall(PetscSortedIntUpperBound(i1, k, n1, rend - 1 - PETSC_INT_MAX, &rem)); /* rem is upper bound of the last local row */
  for (; k < rem; k++) i1[k] += PETSC_INT_MAX;                                    /* Revert row indices of local rows*/

  /*           Send remote rows to their owner                                  */
  /* Find which rows should be sent to which remote ranks*/
  PetscInt        nsend = 0; /* Number of MPI ranks to send data to */
  PetscMPIInt    *sendto;    /* [nsend], storing remote ranks */
  PetscInt       *nentries;  /* [nsend], storing number of entries sent to remote ranks; Assume PetscInt is big enough for this count, and error if not */
  const PetscInt *ranges;
  PetscInt        maxNsend = size >= 128 ? 128 : size; /* Assume max 128 neighbors; realloc when needed */

  PetscCall(PetscLayoutGetRanges(mat->rmap, &ranges));
  PetscCall(PetscMalloc2(maxNsend, &sendto, maxNsend, &nentries));
  for (k = rem; k < n1;) {
    PetscMPIInt owner;
    PetscInt    firstRow, lastRow;

    /* Locate a row range */
    firstRow = i1[k]; /* first row of this owner */
    PetscCall(PetscLayoutFindOwner(mat->rmap, firstRow, &owner));
    lastRow = ranges[owner + 1] - 1; /* last row of this owner */

    /* Find the first index 'p' in [k,n) with i[p] belonging to next owner */
    PetscCall(PetscSortedIntUpperBound(i1, k, n1, lastRow, &p));

    /* All entries in [k,p) belong to this remote owner */
    if (nsend >= maxNsend) { /* Double the remote ranks arrays if not long enough */
      PetscMPIInt *sendto2;
      PetscInt    *nentries2;
      PetscInt     maxNsend2 = (maxNsend <= size / 2) ? maxNsend * 2 : size;

      PetscCall(PetscMalloc2(maxNsend2, &sendto2, maxNsend2, &nentries2));
      PetscCall(PetscArraycpy(sendto2, sendto, maxNsend));
      PetscCall(PetscArraycpy(nentries2, nentries2, maxNsend + 1));
      PetscCall(PetscFree2(sendto, nentries2));
      sendto   = sendto2;
      nentries = nentries2;
      maxNsend = maxNsend2;
    }
    sendto[nsend] = owner;
    PetscCall(PetscIntCast(p - k, &nentries[nsend]));
    nsend++;
    k = p;
  }

  /* Build 1st SF to know offsets on remote to send data */
  PetscSF      sf1;
  PetscInt     nroots = 1, nroots2 = 0;
  PetscInt     nleaves = nsend, nleaves2 = 0;
  PetscInt    *offsets;
  PetscSFNode *iremote;

  PetscCall(PetscSFCreate(comm, &sf1));
  PetscCall(PetscMalloc1(nsend, &iremote));
  PetscCall(PetscMalloc1(nsend, &offsets));
  for (k = 0; k < nsend; k++) {
    iremote[k].rank  = sendto[k];
    iremote[k].index = 0;
    nleaves2 += nentries[k];
    PetscCheck(nleaves2 >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of SF leaves is too large for PetscInt");
  }
  PetscCall(PetscSFSetGraph(sf1, nroots, nleaves, NULL, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  PetscCall(PetscSFFetchAndOpWithMemTypeBegin(sf1, MPIU_INT, PETSC_MEMTYPE_HOST, &nroots2 /*rootdata*/, PETSC_MEMTYPE_HOST, nentries /*leafdata*/, PETSC_MEMTYPE_HOST, offsets /*leafupdate*/, MPI_SUM));
  PetscCall(PetscSFFetchAndOpEnd(sf1, MPIU_INT, &nroots2, nentries, offsets, MPI_SUM)); /* Would nroots2 overflow, we check offsets[] below */
  PetscCall(PetscSFDestroy(&sf1));
  PetscAssert(nleaves2 == n1 - rem, PETSC_COMM_SELF, PETSC_ERR_PLIB, "nleaves2 %" PetscInt_FMT " != number of remote entries %" PetscCount_FMT, nleaves2, n1 - rem);

  /* Build 2nd SF to send remote COOs to their owner */
  PetscSF sf2;
  nroots  = nroots2;
  nleaves = nleaves2;
  PetscCall(PetscSFCreate(comm, &sf2));
  PetscCall(PetscSFSetFromOptions(sf2));
  PetscCall(PetscMalloc1(nleaves, &iremote));
  p = 0;
  for (k = 0; k < nsend; k++) {
    PetscCheck(offsets[k] >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of SF roots is too large for PetscInt");
    for (q = 0; q < nentries[k]; q++, p++) {
      iremote[p].rank = sendto[k];
      PetscCall(PetscIntCast(offsets[k] + q, &iremote[p].index));
    }
  }
  PetscCall(PetscSFSetGraph(sf2, nroots, nleaves, NULL, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));

  /* Send the remote COOs to their owner */
  PetscInt    n2 = nroots, *i2, *j2; /* Buffers for received COOs from other ranks, along with a permutation array */
  PetscCount *perm2;                 /* Though PetscInt is enough for remote entries, we use PetscCount here as we want to reuse MatSplitEntries_Internal() */
  PetscCall(PetscMalloc3(n2, &i2, n2, &j2, n2, &perm2));
  PetscAssert(rem == 0 || i1 != NULL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cannot add nonzero offset to null");
  PetscAssert(rem == 0 || j1 != NULL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cannot add nonzero offset to null");
  PetscInt *i1prem = PetscSafePointerPlusOffset(i1, rem);
  PetscInt *j1prem = PetscSafePointerPlusOffset(j1, rem);
  PetscCall(PetscSFReduceWithMemTypeBegin(sf2, MPIU_INT, PETSC_MEMTYPE_HOST, i1prem, PETSC_MEMTYPE_HOST, i2, MPI_REPLACE));
  PetscCall(PetscSFReduceEnd(sf2, MPIU_INT, i1prem, i2, MPI_REPLACE));
  PetscCall(PetscSFReduceWithMemTypeBegin(sf2, MPIU_INT, PETSC_MEMTYPE_HOST, j1prem, PETSC_MEMTYPE_HOST, j2, MPI_REPLACE));
  PetscCall(PetscSFReduceEnd(sf2, MPIU_INT, j1prem, j2, MPI_REPLACE));

  PetscCall(PetscFree(offsets));
  PetscCall(PetscFree2(sendto, nentries));

  /* Sort received COOs by row along with the permutation array     */
  for (k = 0; k < n2; k++) perm2[k] = k;
  PetscCall(PetscSortIntWithIntCountArrayPair(n2, i2, j2, perm2));

  /* sf2 only sends contiguous leafdata to contiguous rootdata. We record the permutation which will be used to fill leafdata */
  PetscCount *Cperm1;
  PetscAssert(rem == 0 || perm1 != NULL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cannot add nonzero offset to null");
  PetscCount *perm1prem = PetscSafePointerPlusOffset(perm1, rem);
  PetscCall(PetscMalloc1(nleaves, &Cperm1));
  PetscCall(PetscArraycpy(Cperm1, perm1prem, nleaves));

  /* Support for HYPRE matrices, kind of a hack.
     Swap min column with diagonal so that diagonal values will go first */
  PetscBool hypre;
  PetscCall(PetscStrcmp("_internal_COO_mat_for_hypre", ((PetscObject)mat)->name, &hypre));
  if (hypre) {
    PetscInt *minj;
    PetscBT   hasdiag;

    PetscCall(PetscBTCreate(m, &hasdiag));
    PetscCall(PetscMalloc1(m, &minj));
    for (k = 0; k < m; k++) minj[k] = PETSC_INT_MAX;
    for (k = i1start; k < rem; k++) {
      if (j1[k] < cstart || j1[k] >= cend) continue;
      const PetscInt rindex = i1[k] - rstart;
      if ((j1[k] - cstart) == rindex) PetscCall(PetscBTSet(hasdiag, rindex));
      minj[rindex] = PetscMin(minj[rindex], j1[k]);
    }
    for (k = 0; k < n2; k++) {
      if (j2[k] < cstart || j2[k] >= cend) continue;
      const PetscInt rindex = i2[k] - rstart;
      if ((j2[k] - cstart) == rindex) PetscCall(PetscBTSet(hasdiag, rindex));
      minj[rindex] = PetscMin(minj[rindex], j2[k]);
    }
    for (k = i1start; k < rem; k++) {
      const PetscInt rindex = i1[k] - rstart;
      if (j1[k] < cstart || j1[k] >= cend || !PetscBTLookup(hasdiag, rindex)) continue;
      if (j1[k] == minj[rindex]) j1[k] = i1[k] + (cstart - rstart);
      else if ((j1[k] - cstart) == rindex) j1[k] = minj[rindex];
    }
    for (k = 0; k < n2; k++) {
      const PetscInt rindex = i2[k] - rstart;
      if (j2[k] < cstart || j2[k] >= cend || !PetscBTLookup(hasdiag, rindex)) continue;
      if (j2[k] == minj[rindex]) j2[k] = i2[k] + (cstart - rstart);
      else if ((j2[k] - cstart) == rindex) j2[k] = minj[rindex];
    }
    PetscCall(PetscBTDestroy(&hasdiag));
    PetscCall(PetscFree(minj));
  }

  /* Split local COOs and received COOs into diag/offdiag portions */
  PetscCount *rowBegin1, *rowMid1, *rowEnd1;
  PetscCount *Ajmap1, *Aperm1, *Bjmap1, *Bperm1;
  PetscCount  Annz1, Bnnz1, Atot1, Btot1;
  PetscCount *rowBegin2, *rowMid2, *rowEnd2;
  PetscCount *Ajmap2, *Aperm2, *Bjmap2, *Bperm2;
  PetscCount  Annz2, Bnnz2, Atot2, Btot2;

  PetscCall(PetscCalloc3(m, &rowBegin1, m, &rowMid1, m, &rowEnd1));
  PetscCall(PetscCalloc3(m, &rowBegin2, m, &rowMid2, m, &rowEnd2));
  PetscCall(MatSplitEntries_Internal(mat, rem, i1, j1, perm1, rowBegin1, rowMid1, rowEnd1, &Atot1, &Aperm1, &Annz1, &Ajmap1, &Btot1, &Bperm1, &Bnnz1, &Bjmap1));
  PetscCall(MatSplitEntries_Internal(mat, n2, i2, j2, perm2, rowBegin2, rowMid2, rowEnd2, &Atot2, &Aperm2, &Annz2, &Ajmap2, &Btot2, &Bperm2, &Bnnz2, &Bjmap2));

  /* Merge local COOs with received COOs: diag with diag, offdiag with offdiag */
  PetscInt *Ai, *Bi;
  PetscInt *Aj, *Bj;

  PetscCall(PetscMalloc1(m + 1, &Ai));
  PetscCall(PetscMalloc1(m + 1, &Bi));
  PetscCall(PetscMalloc1(Annz1 + Annz2, &Aj)); /* Since local and remote entries might have dups, we might allocate excess memory */
  PetscCall(PetscMalloc1(Bnnz1 + Bnnz2, &Bj));

  PetscCount *Aimap1, *Bimap1, *Aimap2, *Bimap2;
  PetscCall(PetscMalloc1(Annz1, &Aimap1));
  PetscCall(PetscMalloc1(Bnnz1, &Bimap1));
  PetscCall(PetscMalloc1(Annz2, &Aimap2));
  PetscCall(PetscMalloc1(Bnnz2, &Bimap2));

  PetscCall(MatMergeEntries_Internal(mat, j1, j2, rowBegin1, rowMid1, rowBegin2, rowMid2, Ajmap1, Ajmap2, Aimap1, Aimap2, Ai, Aj));
  PetscCall(MatMergeEntries_Internal(mat, j1, j2, rowMid1, rowEnd1, rowMid2, rowEnd2, Bjmap1, Bjmap2, Bimap1, Bimap2, Bi, Bj));

  /* Expand Ajmap1/Bjmap1 to make them based off nonzeros in A/B, since we     */
  /* expect nonzeros in A/B most likely have local contributing entries        */
  PetscInt    Annz = Ai[m];
  PetscInt    Bnnz = Bi[m];
  PetscCount *Ajmap1_new, *Bjmap1_new;

  PetscCall(PetscMalloc1(Annz + 1, &Ajmap1_new));
  PetscCall(PetscMalloc1(Bnnz + 1, &Bjmap1_new));

  PetscCall(ExpandJmap_Internal(Annz1, Annz, Aimap1, Ajmap1, Ajmap1_new));
  PetscCall(ExpandJmap_Internal(Bnnz1, Bnnz, Bimap1, Bjmap1, Bjmap1_new));

  PetscCall(PetscFree(Aimap1));
  PetscCall(PetscFree(Ajmap1));
  PetscCall(PetscFree(Bimap1));
  PetscCall(PetscFree(Bjmap1));
  PetscCall(PetscFree3(rowBegin1, rowMid1, rowEnd1));
  PetscCall(PetscFree3(rowBegin2, rowMid2, rowEnd2));
  PetscCall(PetscFree(perm1));
  PetscCall(PetscFree3(i2, j2, perm2));

  Ajmap1 = Ajmap1_new;
  Bjmap1 = Bjmap1_new;

  /* Reallocate Aj, Bj once we know actual numbers of unique nonzeros in A and B */
  if (Annz < Annz1 + Annz2) {
    PetscInt *Aj_new;
    PetscCall(PetscMalloc1(Annz, &Aj_new));
    PetscCall(PetscArraycpy(Aj_new, Aj, Annz));
    PetscCall(PetscFree(Aj));
    Aj = Aj_new;
  }

  if (Bnnz < Bnnz1 + Bnnz2) {
    PetscInt *Bj_new;
    PetscCall(PetscMalloc1(Bnnz, &Bj_new));
    PetscCall(PetscArraycpy(Bj_new, Bj, Bnnz));
    PetscCall(PetscFree(Bj));
    Bj = Bj_new;
  }

  /* Create new submatrices for on-process and off-process coupling                  */
  PetscScalar     *Aa, *Ba;
  MatType          rtype;
  Mat_SeqAIJ      *a, *b;
  PetscObjectState state;
  PetscCall(PetscCalloc1(Annz, &Aa)); /* Zero matrix on device */
  PetscCall(PetscCalloc1(Bnnz, &Ba));
  /* make Aj[] local, i.e, based off the start column of the diagonal portion */
  if (cstart) {
    for (k = 0; k < Annz; k++) Aj[k] -= cstart;
  }

  PetscCall(MatGetRootType_Private(mat, &rtype));

  MatSeqXAIJGetOptions_Private(mpiaij->A);
  PetscCall(MatDestroy(&mpiaij->A));
  PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, m, n, Ai, Aj, Aa, &mpiaij->A));
  PetscCall(MatSetBlockSizesFromMats(mpiaij->A, mat, mat));
  MatSeqXAIJRestoreOptions_Private(mpiaij->A);

  MatSeqXAIJGetOptions_Private(mpiaij->B);
  PetscCall(MatDestroy(&mpiaij->B));
  PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, m, mat->cmap->N, Bi, Bj, Ba, &mpiaij->B));
  PetscCall(MatSetBlockSizesFromMats(mpiaij->B, mat, mat));
  MatSeqXAIJRestoreOptions_Private(mpiaij->B);

  PetscCall(MatSetUpMultiply_MPIAIJ(mat));
  mat->was_assembled = PETSC_TRUE; // was_assembled in effect means the Mvctx is built; doing so avoids redundant MatSetUpMultiply_MPIAIJ
  state              = mpiaij->A->nonzerostate + mpiaij->B->nonzerostate;
  PetscCallMPI(MPIU_Allreduce(&state, &mat->nonzerostate, 1, MPIU_INT64, MPI_SUM, PetscObjectComm((PetscObject)mat)));

  a          = (Mat_SeqAIJ *)mpiaij->A->data;
  b          = (Mat_SeqAIJ *)mpiaij->B->data;
  a->free_a  = PETSC_TRUE;
  a->free_ij = PETSC_TRUE;
  b->free_a  = PETSC_TRUE;
  b->free_ij = PETSC_TRUE;
  a->maxnz   = a->nz;
  b->maxnz   = b->nz;

  /* conversion must happen AFTER multiply setup */
  PetscCall(MatConvert(mpiaij->A, rtype, MAT_INPLACE_MATRIX, &mpiaij->A));
  PetscCall(MatConvert(mpiaij->B, rtype, MAT_INPLACE_MATRIX, &mpiaij->B));
  PetscCall(VecDestroy(&mpiaij->lvec));
  PetscCall(MatCreateVecs(mpiaij->B, &mpiaij->lvec, NULL));

  // Put the COO struct in a container and then attach that to the matrix
  PetscCall(PetscMalloc1(1, &coo));
  coo->n       = coo_n;
  coo->sf      = sf2;
  coo->sendlen = nleaves;
  coo->recvlen = nroots;
  coo->Annz    = Annz;
  coo->Bnnz    = Bnnz;
  coo->Annz2   = Annz2;
  coo->Bnnz2   = Bnnz2;
  coo->Atot1   = Atot1;
  coo->Atot2   = Atot2;
  coo->Btot1   = Btot1;
  coo->Btot2   = Btot2;
  coo->Ajmap1  = Ajmap1;
  coo->Aperm1  = Aperm1;
  coo->Bjmap1  = Bjmap1;
  coo->Bperm1  = Bperm1;
  coo->Aimap2  = Aimap2;
  coo->Ajmap2  = Ajmap2;
  coo->Aperm2  = Aperm2;
  coo->Bimap2  = Bimap2;
  coo->Bjmap2  = Bjmap2;
  coo->Bperm2  = Bperm2;
  coo->Cperm1  = Cperm1;
  // Allocate in preallocation. If not used, it has zero cost on host
  PetscCall(PetscMalloc2(coo->sendlen, &coo->sendbuf, coo->recvlen, &coo->recvbuf));
  PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &container));
  PetscCall(PetscContainerSetPointer(container, coo));
  PetscCall(PetscContainerSetUserDestroy(container, MatCOOStructDestroy_MPIAIJ));
  PetscCall(PetscObjectCompose((PetscObject)mat, "__PETSc_MatCOOStruct_Host", (PetscObject)container));
  PetscCall(PetscContainerDestroy(&container));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetValuesCOO_MPIAIJ(Mat mat, const PetscScalar v[], InsertMode imode)
{
  Mat_MPIAIJ          *mpiaij = (Mat_MPIAIJ *)mat->data;
  Mat                  A = mpiaij->A, B = mpiaij->B;
  PetscScalar         *Aa, *Ba;
  PetscScalar         *sendbuf, *recvbuf;
  const PetscCount    *Ajmap1, *Ajmap2, *Aimap2;
  const PetscCount    *Bjmap1, *Bjmap2, *Bimap2;
  const PetscCount    *Aperm1, *Aperm2, *Bperm1, *Bperm2;
  const PetscCount    *Cperm1;
  PetscContainer       container;
  MatCOOStruct_MPIAIJ *coo;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)mat, "__PETSc_MatCOOStruct_Host", (PetscObject *)&container));
  PetscCheck(container, PetscObjectComm((PetscObject)mat), PETSC_ERR_PLIB, "Not found MatCOOStruct on this matrix");
  PetscCall(PetscContainerGetPointer(container, (void **)&coo));
  sendbuf = coo->sendbuf;
  recvbuf = coo->recvbuf;
  Ajmap1  = coo->Ajmap1;
  Ajmap2  = coo->Ajmap2;
  Aimap2  = coo->Aimap2;
  Bjmap1  = coo->Bjmap1;
  Bjmap2  = coo->Bjmap2;
  Bimap2  = coo->Bimap2;
  Aperm1  = coo->Aperm1;
  Aperm2  = coo->Aperm2;
  Bperm1  = coo->Bperm1;
  Bperm2  = coo->Bperm2;
  Cperm1  = coo->Cperm1;

  PetscCall(MatSeqAIJGetArray(A, &Aa)); /* Might read and write matrix values */
  PetscCall(MatSeqAIJGetArray(B, &Ba));

  /* Pack entries to be sent to remote */
  for (PetscCount i = 0; i < coo->sendlen; i++) sendbuf[i] = v[Cperm1[i]];

  /* Send remote entries to their owner and overlap the communication with local computation */
  PetscCall(PetscSFReduceWithMemTypeBegin(coo->sf, MPIU_SCALAR, PETSC_MEMTYPE_HOST, sendbuf, PETSC_MEMTYPE_HOST, recvbuf, MPI_REPLACE));
  /* Add local entries to A and B */
  for (PetscCount i = 0; i < coo->Annz; i++) { /* All nonzeros in A are either zero'ed or added with a value (i.e., initialized) */
    PetscScalar sum = 0.0;                     /* Do partial summation first to improve numerical stability */
    for (PetscCount k = Ajmap1[i]; k < Ajmap1[i + 1]; k++) sum += v[Aperm1[k]];
    Aa[i] = (imode == INSERT_VALUES ? 0.0 : Aa[i]) + sum;
  }
  for (PetscCount i = 0; i < coo->Bnnz; i++) {
    PetscScalar sum = 0.0;
    for (PetscCount k = Bjmap1[i]; k < Bjmap1[i + 1]; k++) sum += v[Bperm1[k]];
    Ba[i] = (imode == INSERT_VALUES ? 0.0 : Ba[i]) + sum;
  }
  PetscCall(PetscSFReduceEnd(coo->sf, MPIU_SCALAR, sendbuf, recvbuf, MPI_REPLACE));

  /* Add received remote entries to A and B */
  for (PetscCount i = 0; i < coo->Annz2; i++) {
    for (PetscCount k = Ajmap2[i]; k < Ajmap2[i + 1]; k++) Aa[Aimap2[i]] += recvbuf[Aperm2[k]];
  }
  for (PetscCount i = 0; i < coo->Bnnz2; i++) {
    for (PetscCount k = Bjmap2[i]; k < Bjmap2[i + 1]; k++) Ba[Bimap2[i]] += recvbuf[Bperm2[k]];
  }
  PetscCall(MatSeqAIJRestoreArray(A, &Aa));
  PetscCall(MatSeqAIJRestoreArray(B, &Ba));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATMPIAIJ - MATMPIAIJ = "mpiaij" - A matrix type to be used for parallel sparse matrices.

   Options Database Keys:
. -mat_type mpiaij - sets the matrix type to `MATMPIAIJ` during a call to `MatSetFromOptions()`

   Level: beginner

   Notes:
   `MatSetValues()` may be called for this matrix type with a `NULL` argument for the numerical values,
    in this case the values associated with the rows and columns one passes in are set to zero
    in the matrix

    `MatSetOptions`(,`MAT_STRUCTURE_ONLY`,`PETSC_TRUE`) may be called for this matrix type. In this no
    space is allocated for the nonzero entries and any entries passed with `MatSetValues()` are ignored

.seealso: [](ch_matrices), `Mat`, `MATSEQAIJ`, `MATAIJ`, `MatCreateAIJ()`
M*/
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJ(Mat B)
{
  Mat_MPIAIJ *b;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)B), &size));

  PetscCall(PetscNew(&b));
  B->data       = (void *)b;
  B->ops[0]     = MatOps_Values;
  B->assembled  = PETSC_FALSE;
  B->insertmode = NOT_SET_VALUES;
  b->size       = size;

  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)B), &b->rank));

  /* build cache for off array entries formed */
  PetscCall(MatStashCreate_Private(PetscObjectComm((PetscObject)B), 1, &B->stash));

  b->donotstash  = PETSC_FALSE;
  b->colmap      = NULL;
  b->garray      = NULL;
  b->roworiented = PETSC_TRUE;

  /* stuff used for matrix vector multiply */
  b->lvec  = NULL;
  b->Mvctx = NULL;

  /* stuff for MatGetRow() */
  b->rowindices   = NULL;
  b->rowvalues    = NULL;
  b->getrowactive = PETSC_FALSE;

  /* flexible pointer used in CUSPARSE classes */
  b->spptr = NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMPIAIJSetUseScalableIncreaseOverlap_C", MatMPIAIJSetUseScalableIncreaseOverlap_MPIAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatStoreValues_C", MatStoreValues_MPIAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatRetrieveValues_C", MatRetrieveValues_MPIAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatIsTranspose_C", MatIsTranspose_MPIAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMPIAIJSetPreallocation_C", MatMPIAIJSetPreallocation_MPIAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatResetPreallocation_C", MatResetPreallocation_MPIAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMPIAIJSetPreallocationCSR_C", MatMPIAIJSetPreallocationCSR_MPIAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatDiagonalScaleLocal_C", MatDiagonalScaleLocal_MPIAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpiaij_mpiaijperm_C", MatConvert_MPIAIJ_MPIAIJPERM));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpiaij_mpiaijsell_C", MatConvert_MPIAIJ_MPIAIJSELL));
#if defined(PETSC_HAVE_CUDA)
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpiaij_mpiaijcusparse_C", MatConvert_MPIAIJ_MPIAIJCUSPARSE));
#endif
#if defined(PETSC_HAVE_HIP)
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpiaij_mpiaijhipsparse_C", MatConvert_MPIAIJ_MPIAIJHIPSPARSE));
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpiaij_mpiaijkokkos_C", MatConvert_MPIAIJ_MPIAIJKokkos));
#endif
#if defined(PETSC_HAVE_MKL_SPARSE)
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpiaij_mpiaijmkl_C", MatConvert_MPIAIJ_MPIAIJMKL));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpiaij_mpiaijcrl_C", MatConvert_MPIAIJ_MPIAIJCRL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpiaij_mpibaij_C", MatConvert_MPIAIJ_MPIBAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpiaij_mpisbaij_C", MatConvert_MPIAIJ_MPISBAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpiaij_mpidense_C", MatConvert_MPIAIJ_MPIDense));
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpiaij_elemental_C", MatConvert_MPIAIJ_Elemental));
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpiaij_scalapack_C", MatConvert_AIJ_ScaLAPACK));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpiaij_is_C", MatConvert_XAIJ_IS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpiaij_mpisell_C", MatConvert_MPIAIJ_MPISELL));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpiaij_hypre_C", MatConvert_AIJ_HYPRE));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_transpose_mpiaij_mpiaij_C", MatProductSetFromOptions_Transpose_AIJ_AIJ));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_is_mpiaij_C", MatProductSetFromOptions_IS_XAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_mpiaij_mpiaij_C", MatProductSetFromOptions_MPIAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSetPreallocationCOO_C", MatSetPreallocationCOO_MPIAIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSetValuesCOO_C", MatSetValuesCOO_MPIAIJ));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATMPIAIJ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateMPIAIJWithSplitArrays - creates a `MATMPIAIJ` matrix using arrays that contain the "diagonal"
  and "off-diagonal" part of the matrix in CSR format.

  Collective

  Input Parameters:
+ comm - MPI communicator
. m    - number of local rows (Cannot be `PETSC_DECIDE`)
. n    - This value should be the same as the local size used in creating the
         x vector for the matrix-vector product $y = Ax$. (or `PETSC_DECIDE` to have
         calculated if `N` is given) For square matrices `n` is almost always `m`.
. M    - number of global rows (or `PETSC_DETERMINE` to have calculated if `m` is given)
. N    - number of global columns (or `PETSC_DETERMINE` to have calculated if `n` is given)
. i    - row indices for "diagonal" portion of matrix; that is i[0] = 0, i[row] = i[row-1] + number of elements in that row of the matrix
. j    - column indices, which must be local, i.e., based off the start column of the diagonal portion
. a    - matrix values
. oi   - row indices for "off-diagonal" portion of matrix; that is oi[0] = 0, oi[row] = oi[row-1] + number of elements in that row of the matrix
. oj   - column indices, which must be global, representing global columns in the `MATMPIAIJ` matrix
- oa   - matrix values

  Output Parameter:
. mat - the matrix

  Level: advanced

  Notes:
  The `i`, `j`, and `a` arrays ARE NOT copied by this routine into the internal format used by PETSc (even in Fortran). The user
  must free the arrays once the matrix has been destroyed and not before.

  The `i` and `j` indices are 0 based

  See `MatCreateAIJ()` for the definition of "diagonal" and "off-diagonal" portion of the matrix

  This sets local rows and cannot be used to set off-processor values.

  Use of this routine is discouraged because it is inflexible and cumbersome to use. It is extremely rare that a
  legacy application natively assembles into exactly this split format. The code to do so is nontrivial and does
  not easily support in-place reassembly. It is recommended to use MatSetValues() (or a variant thereof) because
  the resulting assembly is easier to implement, will work with any matrix format, and the user does not have to
  keep track of the underlying array. Use `MatSetOption`(A,`MAT_NO_OFF_PROC_ENTRIES`,`PETSC_TRUE`) to disable all
  communication if it is known that only local entries will be set.

.seealso: [](ch_matrices), `Mat`, `MatCreate()`, `MatCreateSeqAIJ()`, `MatSetValues()`, `MatMPIAIJSetPreallocation()`, `MatMPIAIJSetPreallocationCSR()`,
          `MATMPIAIJ`, `MatCreateAIJ()`, `MatCreateMPIAIJWithArrays()`
@*/
PetscErrorCode MatCreateMPIAIJWithSplitArrays(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt i[], PetscInt j[], PetscScalar a[], PetscInt oi[], PetscInt oj[], PetscScalar oa[], Mat *mat)
{
  Mat_MPIAIJ *maij;

  PetscFunctionBegin;
  PetscCheck(m >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "local number of rows (m) cannot be PETSC_DECIDE, or negative");
  PetscCheck(i[0] == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "i (row indices) must start with 0");
  PetscCheck(oi[0] == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "oi (row indices) must start with 0");
  PetscCall(MatCreate(comm, mat));
  PetscCall(MatSetSizes(*mat, m, n, M, N));
  PetscCall(MatSetType(*mat, MATMPIAIJ));
  maij = (Mat_MPIAIJ *)(*mat)->data;

  (*mat)->preallocated = PETSC_TRUE;

  PetscCall(PetscLayoutSetUp((*mat)->rmap));
  PetscCall(PetscLayoutSetUp((*mat)->cmap));

  PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, m, n, i, j, a, &maij->A));
  PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, m, (*mat)->cmap->N, oi, oj, oa, &maij->B));

  PetscCall(MatSetOption(*mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(*mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_FALSE));
  PetscCall(MatSetOption(*mat, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  Mat       *mp;    /* intermediate products */
  PetscBool *mptmp; /* is the intermediate product temporary ? */
  PetscInt   cp;    /* number of intermediate products */

  /* support for MatGetBrowsOfAoCols_MPIAIJ for P_oth */
  PetscInt    *startsj_s, *startsj_r;
  PetscScalar *bufa;
  Mat          P_oth;

  /* may take advantage of merging product->B */
  Mat Bloc; /* B-local by merging diag and off-diag */

  /* cusparse does not have support to split between symbolic and numeric phases.
     When api_user is true, we don't need to update the numerical values
     of the temporary storage */
  PetscBool reusesym;

  /* support for COO values insertion */
  PetscScalar *coo_v, *coo_w; /* store on-process and off-process COO scalars, and used as MPI recv/send buffers respectively */
  PetscInt   **own;           /* own[i] points to address of on-process COO indices for Mat mp[i] */
  PetscInt   **off;           /* off[i] points to address of off-process COO indices for Mat mp[i] */
  PetscBool    hasoffproc;    /* if true, have off-process values insertion (i.e. AtB or PtAP) */
  PetscSF      sf;            /* used for non-local values insertion and memory malloc */
  PetscMemType mtype;

  /* customization */
  PetscBool abmerge;
  PetscBool P_oth_bind;
} MatMatMPIAIJBACKEND;

static PetscErrorCode MatDestroy_MatMatMPIAIJBACKEND(void *data)
{
  MatMatMPIAIJBACKEND *mmdata = (MatMatMPIAIJBACKEND *)data;
  PetscInt             i;

  PetscFunctionBegin;
  PetscCall(PetscFree2(mmdata->startsj_s, mmdata->startsj_r));
  PetscCall(PetscFree(mmdata->bufa));
  PetscCall(PetscSFFree(mmdata->sf, mmdata->mtype, mmdata->coo_v));
  PetscCall(PetscSFFree(mmdata->sf, mmdata->mtype, mmdata->coo_w));
  PetscCall(MatDestroy(&mmdata->P_oth));
  PetscCall(MatDestroy(&mmdata->Bloc));
  PetscCall(PetscSFDestroy(&mmdata->sf));
  for (i = 0; i < mmdata->cp; i++) PetscCall(MatDestroy(&mmdata->mp[i]));
  PetscCall(PetscFree2(mmdata->mp, mmdata->mptmp));
  PetscCall(PetscFree(mmdata->own[0]));
  PetscCall(PetscFree(mmdata->own));
  PetscCall(PetscFree(mmdata->off[0]));
  PetscCall(PetscFree(mmdata->off));
  PetscCall(PetscFree(mmdata));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Copy selected n entries with indices in idx[] of A to v[].
   If idx is NULL, copy the whole data array of A to v[]
 */
static PetscErrorCode MatSeqAIJCopySubArray(Mat A, PetscInt n, const PetscInt idx[], PetscScalar v[])
{
  PetscErrorCode (*f)(Mat, PetscInt, const PetscInt[], PetscScalar[]);

  PetscFunctionBegin;
  PetscCall(PetscObjectQueryFunction((PetscObject)A, "MatSeqAIJCopySubArray_C", &f));
  if (f) {
    PetscCall((*f)(A, n, idx, v));
  } else {
    const PetscScalar *vv;

    PetscCall(MatSeqAIJGetArrayRead(A, &vv));
    if (n && idx) {
      PetscScalar    *w  = v;
      const PetscInt *oi = idx;
      PetscInt        j;

      for (j = 0; j < n; j++) *w++ = vv[*oi++];
    } else {
      PetscCall(PetscArraycpy(v, vv, n));
    }
    PetscCall(MatSeqAIJRestoreArrayRead(A, &vv));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductNumeric_MPIAIJBACKEND(Mat C)
{
  MatMatMPIAIJBACKEND *mmdata;
  PetscInt             i, n_d, n_o;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(C->product->data, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Product data empty");
  mmdata = (MatMatMPIAIJBACKEND *)C->product->data;
  if (!mmdata->reusesym) { /* update temporary matrices */
    if (mmdata->P_oth) PetscCall(MatGetBrowsOfAoCols_MPIAIJ(C->product->A, C->product->B, MAT_REUSE_MATRIX, &mmdata->startsj_s, &mmdata->startsj_r, &mmdata->bufa, &mmdata->P_oth));
    if (mmdata->Bloc) PetscCall(MatMPIAIJGetLocalMatMerge(C->product->B, MAT_REUSE_MATRIX, NULL, &mmdata->Bloc));
  }
  mmdata->reusesym = PETSC_FALSE;

  for (i = 0; i < mmdata->cp; i++) {
    PetscCheck(mmdata->mp[i]->ops->productnumeric, PetscObjectComm((PetscObject)mmdata->mp[i]), PETSC_ERR_PLIB, "Missing numeric op for %s", MatProductTypes[mmdata->mp[i]->product->type]);
    PetscCall((*mmdata->mp[i]->ops->productnumeric)(mmdata->mp[i]));
  }
  for (i = 0, n_d = 0, n_o = 0; i < mmdata->cp; i++) {
    PetscInt noff;

    PetscCall(PetscIntCast(mmdata->off[i + 1] - mmdata->off[i], &noff));
    if (mmdata->mptmp[i]) continue;
    if (noff) {
      PetscInt nown;

      PetscCall(PetscIntCast(mmdata->own[i + 1] - mmdata->own[i], &nown));
      PetscCall(MatSeqAIJCopySubArray(mmdata->mp[i], noff, mmdata->off[i], mmdata->coo_w + n_o));
      PetscCall(MatSeqAIJCopySubArray(mmdata->mp[i], nown, mmdata->own[i], mmdata->coo_v + n_d));
      n_o += noff;
      n_d += nown;
    } else {
      Mat_SeqAIJ *mm = (Mat_SeqAIJ *)mmdata->mp[i]->data;

      PetscCall(MatSeqAIJCopySubArray(mmdata->mp[i], mm->nz, NULL, mmdata->coo_v + n_d));
      n_d += mm->nz;
    }
  }
  if (mmdata->hasoffproc) { /* offprocess insertion */
    PetscCall(PetscSFGatherBegin(mmdata->sf, MPIU_SCALAR, mmdata->coo_w, mmdata->coo_v + n_d));
    PetscCall(PetscSFGatherEnd(mmdata->sf, MPIU_SCALAR, mmdata->coo_w, mmdata->coo_v + n_d));
  }
  PetscCall(MatSetValuesCOO(C, mmdata->coo_v, INSERT_VALUES));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Support for Pt * A, A * P, or Pt * A * P */
#define MAX_NUMBER_INTERMEDIATE 4
PetscErrorCode MatProductSymbolic_MPIAIJBACKEND(Mat C)
{
  Mat_Product           *product = C->product;
  Mat                    A, P, mp[MAX_NUMBER_INTERMEDIATE]; /* A, P and a series of intermediate matrices */
  Mat_MPIAIJ            *a, *p;
  MatMatMPIAIJBACKEND   *mmdata;
  ISLocalToGlobalMapping P_oth_l2g = NULL;
  IS                     glob      = NULL;
  const char            *prefix;
  char                   pprefix[256];
  const PetscInt        *globidx, *P_oth_idx;
  PetscInt               i, j, cp, m, n, M, N, *coo_i, *coo_j;
  PetscCount             ncoo, ncoo_d, ncoo_o, ncoo_oown;
  PetscInt               cmapt[MAX_NUMBER_INTERMEDIATE], rmapt[MAX_NUMBER_INTERMEDIATE]; /* col/row map type for each Mat in mp[]. */
                                                                                         /* type-0: consecutive, start from 0; type-1: consecutive with */
                                                                                         /* a base offset; type-2: sparse with a local to global map table */
  const PetscInt *cmapa[MAX_NUMBER_INTERMEDIATE], *rmapa[MAX_NUMBER_INTERMEDIATE];       /* col/row local to global map array (table) for type-2 map type */

  MatProductType ptype;
  PetscBool      mptmp[MAX_NUMBER_INTERMEDIATE], hasoffproc = PETSC_FALSE, iscuda, iship, iskokk;
  PetscMPIInt    size;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(!product->data, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Product data not empty");
  ptype = product->type;
  if (product->A->symmetric == PETSC_BOOL3_TRUE && ptype == MATPRODUCT_AtB) {
    ptype                                          = MATPRODUCT_AB;
    product->symbolic_used_the_fact_A_is_symmetric = PETSC_TRUE;
  }
  switch (ptype) {
  case MATPRODUCT_AB:
    A          = product->A;
    P          = product->B;
    m          = A->rmap->n;
    n          = P->cmap->n;
    M          = A->rmap->N;
    N          = P->cmap->N;
    hasoffproc = PETSC_FALSE; /* will not scatter mat product values to other processes */
    break;
  case MATPRODUCT_AtB:
    P          = product->A;
    A          = product->B;
    m          = P->cmap->n;
    n          = A->cmap->n;
    M          = P->cmap->N;
    N          = A->cmap->N;
    hasoffproc = PETSC_TRUE;
    break;
  case MATPRODUCT_PtAP:
    A          = product->A;
    P          = product->B;
    m          = P->cmap->n;
    n          = P->cmap->n;
    M          = P->cmap->N;
    N          = P->cmap->N;
    hasoffproc = PETSC_TRUE;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Not for product type %s", MatProductTypes[ptype]);
  }
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)C), &size));
  if (size == 1) hasoffproc = PETSC_FALSE;

  /* defaults */
  for (i = 0; i < MAX_NUMBER_INTERMEDIATE; i++) {
    mp[i]    = NULL;
    mptmp[i] = PETSC_FALSE;
    rmapt[i] = -1;
    cmapt[i] = -1;
    rmapa[i] = NULL;
    cmapa[i] = NULL;
  }

  /* customization */
  PetscCall(PetscNew(&mmdata));
  mmdata->reusesym = product->api_user;
  if (ptype == MATPRODUCT_AB) {
    if (product->api_user) {
      PetscOptionsBegin(PetscObjectComm((PetscObject)C), ((PetscObject)C)->prefix, "MatMatMult", "Mat");
      PetscCall(PetscOptionsBool("-matmatmult_backend_mergeB", "Merge product->B local matrices", "MatMatMult", mmdata->abmerge, &mmdata->abmerge, NULL));
      PetscCall(PetscOptionsBool("-matmatmult_backend_pothbind", "Bind P_oth to CPU", "MatBindToCPU", mmdata->P_oth_bind, &mmdata->P_oth_bind, NULL));
      PetscOptionsEnd();
    } else {
      PetscOptionsBegin(PetscObjectComm((PetscObject)C), ((PetscObject)C)->prefix, "MatProduct_AB", "Mat");
      PetscCall(PetscOptionsBool("-mat_product_algorithm_backend_mergeB", "Merge product->B local matrices", "MatMatMult", mmdata->abmerge, &mmdata->abmerge, NULL));
      PetscCall(PetscOptionsBool("-mat_product_algorithm_backend_pothbind", "Bind P_oth to CPU", "MatBindToCPU", mmdata->P_oth_bind, &mmdata->P_oth_bind, NULL));
      PetscOptionsEnd();
    }
  } else if (ptype == MATPRODUCT_PtAP) {
    if (product->api_user) {
      PetscOptionsBegin(PetscObjectComm((PetscObject)C), ((PetscObject)C)->prefix, "MatPtAP", "Mat");
      PetscCall(PetscOptionsBool("-matptap_backend_pothbind", "Bind P_oth to CPU", "MatBindToCPU", mmdata->P_oth_bind, &mmdata->P_oth_bind, NULL));
      PetscOptionsEnd();
    } else {
      PetscOptionsBegin(PetscObjectComm((PetscObject)C), ((PetscObject)C)->prefix, "MatProduct_PtAP", "Mat");
      PetscCall(PetscOptionsBool("-mat_product_algorithm_backend_pothbind", "Bind P_oth to CPU", "MatBindToCPU", mmdata->P_oth_bind, &mmdata->P_oth_bind, NULL));
      PetscOptionsEnd();
    }
  }
  a = (Mat_MPIAIJ *)A->data;
  p = (Mat_MPIAIJ *)P->data;
  PetscCall(MatSetSizes(C, m, n, M, N));
  PetscCall(PetscLayoutSetUp(C->rmap));
  PetscCall(PetscLayoutSetUp(C->cmap));
  PetscCall(MatSetType(C, ((PetscObject)A)->type_name));
  PetscCall(MatGetOptionsPrefix(C, &prefix));

  cp = 0;
  switch (ptype) {
  case MATPRODUCT_AB: /* A * P */
    PetscCall(MatGetBrowsOfAoCols_MPIAIJ(A, P, MAT_INITIAL_MATRIX, &mmdata->startsj_s, &mmdata->startsj_r, &mmdata->bufa, &mmdata->P_oth));

    /* A_diag * P_local (merged or not) */
    if (mmdata->abmerge) { /* P's diagonal and off-diag blocks are merged to one matrix, then multiplied by A_diag */
      /* P is product->B */
      PetscCall(MatMPIAIJGetLocalMatMerge(P, MAT_INITIAL_MATRIX, &glob, &mmdata->Bloc));
      PetscCall(MatProductCreate(a->A, mmdata->Bloc, NULL, &mp[cp]));
      PetscCall(MatProductSetType(mp[cp], MATPRODUCT_AB));
      PetscCall(MatProductSetFill(mp[cp], product->fill));
      PetscCall(PetscSNPrintf(pprefix, sizeof(pprefix), "backend_p%" PetscInt_FMT "_", cp));
      PetscCall(MatSetOptionsPrefix(mp[cp], prefix));
      PetscCall(MatAppendOptionsPrefix(mp[cp], pprefix));
      mp[cp]->product->api_user = product->api_user;
      PetscCall(MatProductSetFromOptions(mp[cp]));
      PetscCall((*mp[cp]->ops->productsymbolic)(mp[cp]));
      PetscCall(ISGetIndices(glob, &globidx));
      rmapt[cp] = 1;
      cmapt[cp] = 2;
      cmapa[cp] = globidx;
      mptmp[cp] = PETSC_FALSE;
      cp++;
    } else { /* A_diag * P_diag and A_diag * P_off */
      PetscCall(MatProductCreate(a->A, p->A, NULL, &mp[cp]));
      PetscCall(MatProductSetType(mp[cp], MATPRODUCT_AB));
      PetscCall(MatProductSetFill(mp[cp], product->fill));
      PetscCall(PetscSNPrintf(pprefix, sizeof(pprefix), "backend_p%" PetscInt_FMT "_", cp));
      PetscCall(MatSetOptionsPrefix(mp[cp], prefix));
      PetscCall(MatAppendOptionsPrefix(mp[cp], pprefix));
      mp[cp]->product->api_user = product->api_user;
      PetscCall(MatProductSetFromOptions(mp[cp]));
      PetscCall((*mp[cp]->ops->productsymbolic)(mp[cp]));
      rmapt[cp] = 1;
      cmapt[cp] = 1;
      mptmp[cp] = PETSC_FALSE;
      cp++;
      PetscCall(MatProductCreate(a->A, p->B, NULL, &mp[cp]));
      PetscCall(MatProductSetType(mp[cp], MATPRODUCT_AB));
      PetscCall(MatProductSetFill(mp[cp], product->fill));
      PetscCall(PetscSNPrintf(pprefix, sizeof(pprefix), "backend_p%" PetscInt_FMT "_", cp));
      PetscCall(MatSetOptionsPrefix(mp[cp], prefix));
      PetscCall(MatAppendOptionsPrefix(mp[cp], pprefix));
      mp[cp]->product->api_user = product->api_user;
      PetscCall(MatProductSetFromOptions(mp[cp]));
      PetscCall((*mp[cp]->ops->productsymbolic)(mp[cp]));
      rmapt[cp] = 1;
      cmapt[cp] = 2;
      cmapa[cp] = p->garray;
      mptmp[cp] = PETSC_FALSE;
      cp++;
    }

    /* A_off * P_other */
    if (mmdata->P_oth) {
      PetscCall(MatSeqAIJCompactOutExtraColumns_SeqAIJ(mmdata->P_oth, &P_oth_l2g)); /* make P_oth use local col ids */
      PetscCall(ISLocalToGlobalMappingGetIndices(P_oth_l2g, &P_oth_idx));
      PetscCall(MatSetType(mmdata->P_oth, ((PetscObject)a->B)->type_name));
      PetscCall(MatBindToCPU(mmdata->P_oth, mmdata->P_oth_bind));
      PetscCall(MatProductCreate(a->B, mmdata->P_oth, NULL, &mp[cp]));
      PetscCall(MatProductSetType(mp[cp], MATPRODUCT_AB));
      PetscCall(MatProductSetFill(mp[cp], product->fill));
      PetscCall(PetscSNPrintf(pprefix, sizeof(pprefix), "backend_p%" PetscInt_FMT "_", cp));
      PetscCall(MatSetOptionsPrefix(mp[cp], prefix));
      PetscCall(MatAppendOptionsPrefix(mp[cp], pprefix));
      mp[cp]->product->api_user = product->api_user;
      PetscCall(MatProductSetFromOptions(mp[cp]));
      PetscCall((*mp[cp]->ops->productsymbolic)(mp[cp]));
      rmapt[cp] = 1;
      cmapt[cp] = 2;
      cmapa[cp] = P_oth_idx;
      mptmp[cp] = PETSC_FALSE;
      cp++;
    }
    break;

  case MATPRODUCT_AtB: /* (P^t * A): P_diag * A_loc + P_off * A_loc */
    /* A is product->B */
    PetscCall(MatMPIAIJGetLocalMatMerge(A, MAT_INITIAL_MATRIX, &glob, &mmdata->Bloc));
    if (A == P) { /* when A==P, we can take advantage of the already merged mmdata->Bloc */
      PetscCall(MatProductCreate(mmdata->Bloc, mmdata->Bloc, NULL, &mp[cp]));
      PetscCall(MatProductSetType(mp[cp], MATPRODUCT_AtB));
      PetscCall(MatProductSetFill(mp[cp], product->fill));
      PetscCall(PetscSNPrintf(pprefix, sizeof(pprefix), "backend_p%" PetscInt_FMT "_", cp));
      PetscCall(MatSetOptionsPrefix(mp[cp], prefix));
      PetscCall(MatAppendOptionsPrefix(mp[cp], pprefix));
      mp[cp]->product->api_user = product->api_user;
      PetscCall(MatProductSetFromOptions(mp[cp]));
      PetscCall((*mp[cp]->ops->productsymbolic)(mp[cp]));
      PetscCall(ISGetIndices(glob, &globidx));
      rmapt[cp] = 2;
      rmapa[cp] = globidx;
      cmapt[cp] = 2;
      cmapa[cp] = globidx;
      mptmp[cp] = PETSC_FALSE;
      cp++;
    } else {
      PetscCall(MatProductCreate(p->A, mmdata->Bloc, NULL, &mp[cp]));
      PetscCall(MatProductSetType(mp[cp], MATPRODUCT_AtB));
      PetscCall(MatProductSetFill(mp[cp], product->fill));
      PetscCall(PetscSNPrintf(pprefix, sizeof(pprefix), "backend_p%" PetscInt_FMT "_", cp));
      PetscCall(MatSetOptionsPrefix(mp[cp], prefix));
      PetscCall(MatAppendOptionsPrefix(mp[cp], pprefix));
      mp[cp]->product->api_user = product->api_user;
      PetscCall(MatProductSetFromOptions(mp[cp]));
      PetscCall((*mp[cp]->ops->productsymbolic)(mp[cp]));
      PetscCall(ISGetIndices(glob, &globidx));
      rmapt[cp] = 1;
      cmapt[cp] = 2;
      cmapa[cp] = globidx;
      mptmp[cp] = PETSC_FALSE;
      cp++;
      PetscCall(MatProductCreate(p->B, mmdata->Bloc, NULL, &mp[cp]));
      PetscCall(MatProductSetType(mp[cp], MATPRODUCT_AtB));
      PetscCall(MatProductSetFill(mp[cp], product->fill));
      PetscCall(PetscSNPrintf(pprefix, sizeof(pprefix), "backend_p%" PetscInt_FMT "_", cp));
      PetscCall(MatSetOptionsPrefix(mp[cp], prefix));
      PetscCall(MatAppendOptionsPrefix(mp[cp], pprefix));
      mp[cp]->product->api_user = product->api_user;
      PetscCall(MatProductSetFromOptions(mp[cp]));
      PetscCall((*mp[cp]->ops->productsymbolic)(mp[cp]));
      rmapt[cp] = 2;
      rmapa[cp] = p->garray;
      cmapt[cp] = 2;
      cmapa[cp] = globidx;
      mptmp[cp] = PETSC_FALSE;
      cp++;
    }
    break;
  case MATPRODUCT_PtAP:
    PetscCall(MatGetBrowsOfAoCols_MPIAIJ(A, P, MAT_INITIAL_MATRIX, &mmdata->startsj_s, &mmdata->startsj_r, &mmdata->bufa, &mmdata->P_oth));
    /* P is product->B */
    PetscCall(MatMPIAIJGetLocalMatMerge(P, MAT_INITIAL_MATRIX, &glob, &mmdata->Bloc));
    PetscCall(MatProductCreate(a->A, mmdata->Bloc, NULL, &mp[cp]));
    PetscCall(MatProductSetType(mp[cp], MATPRODUCT_PtAP));
    PetscCall(MatProductSetFill(mp[cp], product->fill));
    PetscCall(PetscSNPrintf(pprefix, sizeof(pprefix), "backend_p%" PetscInt_FMT "_", cp));
    PetscCall(MatSetOptionsPrefix(mp[cp], prefix));
    PetscCall(MatAppendOptionsPrefix(mp[cp], pprefix));
    mp[cp]->product->api_user = product->api_user;
    PetscCall(MatProductSetFromOptions(mp[cp]));
    PetscCall((*mp[cp]->ops->productsymbolic)(mp[cp]));
    PetscCall(ISGetIndices(glob, &globidx));
    rmapt[cp] = 2;
    rmapa[cp] = globidx;
    cmapt[cp] = 2;
    cmapa[cp] = globidx;
    mptmp[cp] = PETSC_FALSE;
    cp++;
    if (mmdata->P_oth) {
      PetscCall(MatSeqAIJCompactOutExtraColumns_SeqAIJ(mmdata->P_oth, &P_oth_l2g));
      PetscCall(ISLocalToGlobalMappingGetIndices(P_oth_l2g, &P_oth_idx));
      PetscCall(MatSetType(mmdata->P_oth, ((PetscObject)a->B)->type_name));
      PetscCall(MatBindToCPU(mmdata->P_oth, mmdata->P_oth_bind));
      PetscCall(MatProductCreate(a->B, mmdata->P_oth, NULL, &mp[cp]));
      PetscCall(MatProductSetType(mp[cp], MATPRODUCT_AB));
      PetscCall(MatProductSetFill(mp[cp], product->fill));
      PetscCall(PetscSNPrintf(pprefix, sizeof(pprefix), "backend_p%" PetscInt_FMT "_", cp));
      PetscCall(MatSetOptionsPrefix(mp[cp], prefix));
      PetscCall(MatAppendOptionsPrefix(mp[cp], pprefix));
      mp[cp]->product->api_user = product->api_user;
      PetscCall(MatProductSetFromOptions(mp[cp]));
      PetscCall((*mp[cp]->ops->productsymbolic)(mp[cp]));
      mptmp[cp] = PETSC_TRUE;
      cp++;
      PetscCall(MatProductCreate(mmdata->Bloc, mp[1], NULL, &mp[cp]));
      PetscCall(MatProductSetType(mp[cp], MATPRODUCT_AtB));
      PetscCall(MatProductSetFill(mp[cp], product->fill));
      PetscCall(PetscSNPrintf(pprefix, sizeof(pprefix), "backend_p%" PetscInt_FMT "_", cp));
      PetscCall(MatSetOptionsPrefix(mp[cp], prefix));
      PetscCall(MatAppendOptionsPrefix(mp[cp], pprefix));
      mp[cp]->product->api_user = product->api_user;
      PetscCall(MatProductSetFromOptions(mp[cp]));
      PetscCall((*mp[cp]->ops->productsymbolic)(mp[cp]));
      rmapt[cp] = 2;
      rmapa[cp] = globidx;
      cmapt[cp] = 2;
      cmapa[cp] = P_oth_idx;
      mptmp[cp] = PETSC_FALSE;
      cp++;
    }
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Not for product type %s", MatProductTypes[ptype]);
  }
  /* sanity check */
  if (size > 1)
    for (i = 0; i < cp; i++) PetscCheck(rmapt[i] != 2 || hasoffproc, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected offproc map type for product %" PetscInt_FMT, i);

  PetscCall(PetscMalloc2(cp, &mmdata->mp, cp, &mmdata->mptmp));
  for (i = 0; i < cp; i++) {
    mmdata->mp[i]    = mp[i];
    mmdata->mptmp[i] = mptmp[i];
  }
  mmdata->cp             = cp;
  C->product->data       = mmdata;
  C->product->destroy    = MatDestroy_MatMatMPIAIJBACKEND;
  C->ops->productnumeric = MatProductNumeric_MPIAIJBACKEND;

  /* memory type */
  mmdata->mtype = PETSC_MEMTYPE_HOST;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)C, &iscuda, MATSEQAIJCUSPARSE, MATMPIAIJCUSPARSE, ""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)C, &iship, MATSEQAIJHIPSPARSE, MATMPIAIJHIPSPARSE, ""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)C, &iskokk, MATSEQAIJKOKKOS, MATMPIAIJKOKKOS, ""));
  if (iscuda) mmdata->mtype = PETSC_MEMTYPE_CUDA;
  else if (iship) mmdata->mtype = PETSC_MEMTYPE_HIP;
  else if (iskokk) mmdata->mtype = PETSC_MEMTYPE_KOKKOS;

  /* prepare coo coordinates for values insertion */

  /* count total nonzeros of those intermediate seqaij Mats
    ncoo_d:    # of nonzeros of matrices that do not have offproc entries
    ncoo_o:    # of nonzeros (of matrices that might have offproc entries) that will be inserted to remote procs
    ncoo_oown: # of nonzeros (of matrices that might have offproc entries) that will be inserted locally
  */
  for (cp = 0, ncoo_d = 0, ncoo_o = 0, ncoo_oown = 0; cp < mmdata->cp; cp++) {
    Mat_SeqAIJ *mm = (Mat_SeqAIJ *)mp[cp]->data;
    if (mptmp[cp]) continue;
    if (rmapt[cp] == 2 && hasoffproc) { /* the rows need to be scatter to all processes (might include self) */
      const PetscInt *rmap = rmapa[cp];
      const PetscInt  mr   = mp[cp]->rmap->n;
      const PetscInt  rs   = C->rmap->rstart;
      const PetscInt  re   = C->rmap->rend;
      const PetscInt *ii   = mm->i;
      for (i = 0; i < mr; i++) {
        const PetscInt gr = rmap[i];
        const PetscInt nz = ii[i + 1] - ii[i];
        if (gr < rs || gr >= re) ncoo_o += nz; /* this row is offproc */
        else ncoo_oown += nz;                  /* this row is local */
      }
    } else ncoo_d += mm->nz;
  }

  /*
    ncoo: total number of nonzeros (including those inserted by remote procs) belonging to this proc

    ncoo = ncoo_d + ncoo_oown + ncoo2, which ncoo2 is number of nonzeros inserted to me by other procs.

    off[0] points to a big index array, which is shared by off[1,2,...]. Similarly, for own[0].

    off[p]: points to the segment for matrix mp[p], storing location of nonzeros that mp[p] will insert to others
    own[p]: points to the segment for matrix mp[p], storing location of nonzeros that mp[p] will insert locally
    so, off[p+1]-off[p] is the number of nonzeros that mp[p] will send to others.

    coo_i/j/v[]: [ncoo] row/col/val of nonzeros belonging to this proc.
    Ex. coo_i[]: the beginning part (of size ncoo_d + ncoo_oown) stores i of local nonzeros, and the remaining part stores i of nonzeros I will receive.
  */
  PetscCall(PetscCalloc1(mmdata->cp + 1, &mmdata->off)); /* +1 to make a csr-like data structure */
  PetscCall(PetscCalloc1(mmdata->cp + 1, &mmdata->own));

  /* gather (i,j) of nonzeros inserted by remote procs */
  if (hasoffproc) {
    PetscSF  msf;
    PetscInt ncoo2, *coo_i2, *coo_j2;

    PetscCall(PetscMalloc1(ncoo_o, &mmdata->off[0]));
    PetscCall(PetscMalloc1(ncoo_oown, &mmdata->own[0]));
    PetscCall(PetscMalloc2(ncoo_o, &coo_i, ncoo_o, &coo_j)); /* to collect (i,j) of entries to be sent to others */

    for (cp = 0, ncoo_o = 0; cp < mmdata->cp; cp++) {
      Mat_SeqAIJ *mm     = (Mat_SeqAIJ *)mp[cp]->data;
      PetscInt   *idxoff = mmdata->off[cp];
      PetscInt   *idxown = mmdata->own[cp];
      if (!mptmp[cp] && rmapt[cp] == 2) { /* row map is sparse */
        const PetscInt *rmap = rmapa[cp];
        const PetscInt *cmap = cmapa[cp];
        const PetscInt *ii   = mm->i;
        PetscInt       *coi  = coo_i + ncoo_o;
        PetscInt       *coj  = coo_j + ncoo_o;
        const PetscInt  mr   = mp[cp]->rmap->n;
        const PetscInt  rs   = C->rmap->rstart;
        const PetscInt  re   = C->rmap->rend;
        const PetscInt  cs   = C->cmap->rstart;
        for (i = 0; i < mr; i++) {
          const PetscInt *jj = mm->j + ii[i];
          const PetscInt  gr = rmap[i];
          const PetscInt  nz = ii[i + 1] - ii[i];
          if (gr < rs || gr >= re) { /* this is an offproc row */
            for (j = ii[i]; j < ii[i + 1]; j++) {
              *coi++    = gr;
              *idxoff++ = j;
            }
            if (!cmapt[cp]) { /* already global */
              for (j = 0; j < nz; j++) *coj++ = jj[j];
            } else if (cmapt[cp] == 1) { /* local to global for owned columns of C */
              for (j = 0; j < nz; j++) *coj++ = jj[j] + cs;
            } else { /* offdiag */
              for (j = 0; j < nz; j++) *coj++ = cmap[jj[j]];
            }
            ncoo_o += nz;
          } else { /* this is a local row */
            for (j = ii[i]; j < ii[i + 1]; j++) *idxown++ = j;
          }
        }
      }
      mmdata->off[cp + 1] = idxoff;
      mmdata->own[cp + 1] = idxown;
    }

    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)C), &mmdata->sf));
    PetscInt incoo_o;
    PetscCall(PetscIntCast(ncoo_o, &incoo_o));
    PetscCall(PetscSFSetGraphLayout(mmdata->sf, C->rmap, incoo_o /*nleaves*/, NULL /*ilocal*/, PETSC_OWN_POINTER, coo_i));
    PetscCall(PetscSFGetMultiSF(mmdata->sf, &msf));
    PetscCall(PetscSFGetGraph(msf, &ncoo2 /*nroots*/, NULL, NULL, NULL));
    ncoo = ncoo_d + ncoo_oown + ncoo2;
    PetscCall(PetscMalloc2(ncoo, &coo_i2, ncoo, &coo_j2));
    PetscCall(PetscSFGatherBegin(mmdata->sf, MPIU_INT, coo_i, coo_i2 + ncoo_d + ncoo_oown)); /* put (i,j) of remote nonzeros at back */
    PetscCall(PetscSFGatherEnd(mmdata->sf, MPIU_INT, coo_i, coo_i2 + ncoo_d + ncoo_oown));
    PetscCall(PetscSFGatherBegin(mmdata->sf, MPIU_INT, coo_j, coo_j2 + ncoo_d + ncoo_oown));
    PetscCall(PetscSFGatherEnd(mmdata->sf, MPIU_INT, coo_j, coo_j2 + ncoo_d + ncoo_oown));
    PetscCall(PetscFree2(coo_i, coo_j));
    /* allocate MPI send buffer to collect nonzero values to be sent to remote procs */
    PetscCall(PetscSFMalloc(mmdata->sf, mmdata->mtype, ncoo_o * sizeof(PetscScalar), (void **)&mmdata->coo_w));
    coo_i = coo_i2;
    coo_j = coo_j2;
  } else { /* no offproc values insertion */
    ncoo = ncoo_d;
    PetscCall(PetscMalloc2(ncoo, &coo_i, ncoo, &coo_j));

    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)C), &mmdata->sf));
    PetscCall(PetscSFSetGraph(mmdata->sf, 0, 0, NULL, PETSC_OWN_POINTER, NULL, PETSC_OWN_POINTER));
    PetscCall(PetscSFSetUp(mmdata->sf));
  }
  mmdata->hasoffproc = hasoffproc;

  /* gather (i,j) of nonzeros inserted locally */
  for (cp = 0, ncoo_d = 0; cp < mmdata->cp; cp++) {
    Mat_SeqAIJ     *mm   = (Mat_SeqAIJ *)mp[cp]->data;
    PetscInt       *coi  = coo_i + ncoo_d;
    PetscInt       *coj  = coo_j + ncoo_d;
    const PetscInt *jj   = mm->j;
    const PetscInt *ii   = mm->i;
    const PetscInt *cmap = cmapa[cp];
    const PetscInt *rmap = rmapa[cp];
    const PetscInt  mr   = mp[cp]->rmap->n;
    const PetscInt  rs   = C->rmap->rstart;
    const PetscInt  re   = C->rmap->rend;
    const PetscInt  cs   = C->cmap->rstart;

    if (mptmp[cp]) continue;
    if (rmapt[cp] == 1) { /* consecutive rows */
      /* fill coo_i */
      for (i = 0; i < mr; i++) {
        const PetscInt gr = i + rs;
        for (j = ii[i]; j < ii[i + 1]; j++) coi[j] = gr;
      }
      /* fill coo_j */
      if (!cmapt[cp]) { /* type-0, already global */
        PetscCall(PetscArraycpy(coj, jj, mm->nz));
      } else if (cmapt[cp] == 1) {                        /* type-1, local to global for consecutive columns of C */
        for (j = 0; j < mm->nz; j++) coj[j] = jj[j] + cs; /* lid + col start */
      } else {                                            /* type-2, local to global for sparse columns */
        for (j = 0; j < mm->nz; j++) coj[j] = cmap[jj[j]];
      }
      ncoo_d += mm->nz;
    } else if (rmapt[cp] == 2) { /* sparse rows */
      for (i = 0; i < mr; i++) {
        const PetscInt *jj = mm->j + ii[i];
        const PetscInt  gr = rmap[i];
        const PetscInt  nz = ii[i + 1] - ii[i];
        if (gr >= rs && gr < re) { /* local rows */
          for (j = ii[i]; j < ii[i + 1]; j++) *coi++ = gr;
          if (!cmapt[cp]) { /* type-0, already global */
            for (j = 0; j < nz; j++) *coj++ = jj[j];
          } else if (cmapt[cp] == 1) { /* local to global for owned columns of C */
            for (j = 0; j < nz; j++) *coj++ = jj[j] + cs;
          } else { /* type-2, local to global for sparse columns */
            for (j = 0; j < nz; j++) *coj++ = cmap[jj[j]];
          }
          ncoo_d += nz;
        }
      }
    }
  }
  if (glob) PetscCall(ISRestoreIndices(glob, &globidx));
  PetscCall(ISDestroy(&glob));
  if (P_oth_l2g) PetscCall(ISLocalToGlobalMappingRestoreIndices(P_oth_l2g, &P_oth_idx));
  PetscCall(ISLocalToGlobalMappingDestroy(&P_oth_l2g));
  /* allocate an array to store all nonzeros (inserted locally or remotely) belonging to this proc */
  PetscCall(PetscSFMalloc(mmdata->sf, mmdata->mtype, ncoo * sizeof(PetscScalar), (void **)&mmdata->coo_v));

  /* preallocate with COO data */
  PetscCall(MatSetPreallocationCOO(C, ncoo, coo_i, coo_j));
  PetscCall(PetscFree2(coo_i, coo_j));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatProductSetFromOptions_MPIAIJBACKEND(Mat mat)
{
  Mat_Product *product = mat->product;
#if defined(PETSC_HAVE_DEVICE)
  PetscBool match  = PETSC_FALSE;
  PetscBool usecpu = PETSC_FALSE;
#else
  PetscBool match = PETSC_TRUE;
#endif

  PetscFunctionBegin;
  MatCheckProduct(mat, 1);
#if defined(PETSC_HAVE_DEVICE)
  if (!product->A->boundtocpu && !product->B->boundtocpu) PetscCall(PetscObjectTypeCompare((PetscObject)product->B, ((PetscObject)product->A)->type_name, &match));
  if (match) { /* we can always fallback to the CPU if requested */
    switch (product->type) {
    case MATPRODUCT_AB:
      if (product->api_user) {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatMatMult", "Mat");
        PetscCall(PetscOptionsBool("-matmatmult_backend_cpu", "Use CPU code", "MatMatMult", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      } else {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatProduct_AB", "Mat");
        PetscCall(PetscOptionsBool("-mat_product_algorithm_backend_cpu", "Use CPU code", "MatMatMult", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      }
      break;
    case MATPRODUCT_AtB:
      if (product->api_user) {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatTransposeMatMult", "Mat");
        PetscCall(PetscOptionsBool("-mattransposematmult_backend_cpu", "Use CPU code", "MatTransposeMatMult", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      } else {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatProduct_AtB", "Mat");
        PetscCall(PetscOptionsBool("-mat_product_algorithm_backend_cpu", "Use CPU code", "MatTransposeMatMult", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      }
      break;
    case MATPRODUCT_PtAP:
      if (product->api_user) {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatPtAP", "Mat");
        PetscCall(PetscOptionsBool("-matptap_backend_cpu", "Use CPU code", "MatPtAP", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      } else {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatProduct_PtAP", "Mat");
        PetscCall(PetscOptionsBool("-mat_product_algorithm_backend_cpu", "Use CPU code", "MatPtAP", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      }
      break;
    default:
      break;
    }
    match = (PetscBool)!usecpu;
  }
#endif
  if (match) {
    switch (product->type) {
    case MATPRODUCT_AB:
    case MATPRODUCT_AtB:
    case MATPRODUCT_PtAP:
      mat->ops->productsymbolic = MatProductSymbolic_MPIAIJBACKEND;
      break;
    default:
      break;
    }
  }
  /* fallback to MPIAIJ ops */
  if (!mat->ops->productsymbolic) PetscCall(MatProductSetFromOptions_MPIAIJ(mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Produces a set of block column indices of the matrix row, one for each block represented in the original row

   n - the number of block indices in cc[]
   cc - the block indices (must be large enough to contain the indices)
*/
static inline PetscErrorCode MatCollapseRow(Mat Amat, PetscInt row, PetscInt bs, PetscInt *n, PetscInt *cc)
{
  PetscInt        cnt = -1, nidx, j;
  const PetscInt *idx;

  PetscFunctionBegin;
  PetscCall(MatGetRow(Amat, row, &nidx, &idx, NULL));
  if (nidx) {
    cnt     = 0;
    cc[cnt] = idx[0] / bs;
    for (j = 1; j < nidx; j++) {
      if (cc[cnt] < idx[j] / bs) cc[++cnt] = idx[j] / bs;
    }
  }
  PetscCall(MatRestoreRow(Amat, row, &nidx, &idx, NULL));
  *n = cnt + 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    Produces a set of block column indices of the matrix block row, one for each block represented in the original set of rows

    ncollapsed - the number of block indices
    collapsed - the block indices (must be large enough to contain the indices)
*/
static inline PetscErrorCode MatCollapseRows(Mat Amat, PetscInt start, PetscInt bs, PetscInt *w0, PetscInt *w1, PetscInt *w2, PetscInt *ncollapsed, PetscInt **collapsed)
{
  PetscInt i, nprev, *cprev = w0, ncur = 0, *ccur = w1, *merged = w2, *cprevtmp;

  PetscFunctionBegin;
  PetscCall(MatCollapseRow(Amat, start, bs, &nprev, cprev));
  for (i = start + 1; i < start + bs; i++) {
    PetscCall(MatCollapseRow(Amat, i, bs, &ncur, ccur));
    PetscCall(PetscMergeIntArray(nprev, cprev, ncur, ccur, &nprev, &merged));
    cprevtmp = cprev;
    cprev    = merged;
    merged   = cprevtmp;
  }
  *ncollapsed = nprev;
  if (collapsed) *collapsed = cprev;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
 MatCreateGraph_Simple_AIJ - create simple scalar matrix (graph) from potentially blocked matrix

 Input Parameter:
 . Amat - matrix
 - symmetrize - make the result symmetric
 + scale - scale with diagonal

 Output Parameter:
 . a_Gmat - output scalar graph >= 0

*/
PETSC_INTERN PetscErrorCode MatCreateGraph_Simple_AIJ(Mat Amat, PetscBool symmetrize, PetscBool scale, PetscReal filter, PetscInt index_size, PetscInt index[], Mat *a_Gmat)
{
  PetscInt  Istart, Iend, Ii, jj, kk, ncols, nloc, NN, MM, bs;
  MPI_Comm  comm;
  Mat       Gmat;
  PetscBool ismpiaij, isseqaij;
  Mat       a, b, c;
  MatType   jtype;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)Amat, &comm));
  PetscCall(MatGetOwnershipRange(Amat, &Istart, &Iend));
  PetscCall(MatGetSize(Amat, &MM, &NN));
  PetscCall(MatGetBlockSize(Amat, &bs));
  nloc = (Iend - Istart) / bs;

  PetscCall(PetscObjectBaseTypeCompare((PetscObject)Amat, MATSEQAIJ, &isseqaij));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)Amat, MATMPIAIJ, &ismpiaij));
  PetscCheck(isseqaij || ismpiaij, comm, PETSC_ERR_USER, "Require (MPI)AIJ matrix type");

  /* TODO GPU: these calls are potentially expensive if matrices are large and we want to use the GPU */
  /* A solution consists in providing a new API, MatAIJGetCollapsedAIJ, and each class can provide a fast
     implementation */
  if (bs > 1) {
    PetscCall(MatGetType(Amat, &jtype));
    PetscCall(MatCreate(comm, &Gmat));
    PetscCall(MatSetType(Gmat, jtype));
    PetscCall(MatSetSizes(Gmat, nloc, nloc, PETSC_DETERMINE, PETSC_DETERMINE));
    PetscCall(MatSetBlockSizes(Gmat, 1, 1));
    if (isseqaij || ((Mat_MPIAIJ *)Amat->data)->garray) {
      PetscInt  *d_nnz, *o_nnz;
      MatScalar *aa, val, *AA;
      PetscInt  *aj, *ai, *AJ, nc, nmax = 0;

      if (isseqaij) {
        a = Amat;
        b = NULL;
      } else {
        Mat_MPIAIJ *d = (Mat_MPIAIJ *)Amat->data;
        a             = d->A;
        b             = d->B;
      }
      PetscCall(PetscInfo(Amat, "New bs>1 Graph. nloc=%" PetscInt_FMT "\n", nloc));
      PetscCall(PetscMalloc2(nloc, &d_nnz, (isseqaij ? 0 : nloc), &o_nnz));
      for (c = a, kk = 0; c && kk < 2; c = b, kk++) {
        PetscInt       *nnz = (c == a) ? d_nnz : o_nnz;
        const PetscInt *cols1, *cols2;

        for (PetscInt brow = 0, nc1, nc2, ok = 1; brow < nloc * bs; brow += bs) { // block rows
          PetscCall(MatGetRow(c, brow, &nc2, &cols2, NULL));
          nnz[brow / bs] = nc2 / bs;
          if (nc2 % bs) ok = 0;
          if (nnz[brow / bs] > nmax) nmax = nnz[brow / bs];
          for (PetscInt ii = 1; ii < bs; ii++) { // check for non-dense blocks
            PetscCall(MatGetRow(c, brow + ii, &nc1, &cols1, NULL));
            if (nc1 != nc2) ok = 0;
            else {
              for (PetscInt jj = 0; jj < nc1 && ok == 1; jj++) {
                if (cols1[jj] != cols2[jj]) ok = 0;
                if (cols1[jj] % bs != jj % bs) ok = 0;
              }
            }
            PetscCall(MatRestoreRow(c, brow + ii, &nc1, &cols1, NULL));
          }
          PetscCall(MatRestoreRow(c, brow, &nc2, &cols2, NULL));
          if (!ok) {
            PetscCall(PetscFree2(d_nnz, o_nnz));
            PetscCall(PetscInfo(Amat, "Found sparse blocks - revert to slow method\n"));
            goto old_bs;
          }
        }
      }
      PetscCall(MatSeqAIJSetPreallocation(Gmat, 0, d_nnz));
      PetscCall(MatMPIAIJSetPreallocation(Gmat, 0, d_nnz, 0, o_nnz));
      PetscCall(PetscFree2(d_nnz, o_nnz));
      PetscCall(PetscMalloc2(nmax, &AA, nmax, &AJ));
      // diag
      for (PetscInt brow = 0, n, grow; brow < nloc * bs; brow += bs) { // block rows
        Mat_SeqAIJ *aseq = (Mat_SeqAIJ *)a->data;

        ai = aseq->i;
        n  = ai[brow + 1] - ai[brow];
        aj = aseq->j + ai[brow];
        for (PetscInt k = 0; k < n; k += bs) {   // block columns
          AJ[k / bs] = aj[k] / bs + Istart / bs; // diag starts at (Istart,Istart)
          val        = 0;
          if (index_size == 0) {
            for (PetscInt ii = 0; ii < bs; ii++) { // rows in block
              aa = aseq->a + ai[brow + ii] + k;
              for (PetscInt jj = 0; jj < bs; jj++) {    // columns in block
                val += PetscAbs(PetscRealPart(aa[jj])); // a sort of norm
              }
            }
          } else {                                            // use (index,index) value if provided
            for (PetscInt iii = 0; iii < index_size; iii++) { // rows in block
              PetscInt ii = index[iii];
              aa          = aseq->a + ai[brow + ii] + k;
              for (PetscInt jjj = 0; jjj < index_size; jjj++) { // columns in block
                PetscInt jj = index[jjj];
                val += PetscAbs(PetscRealPart(aa[jj]));
              }
            }
          }
          PetscAssert(k / bs < nmax, comm, PETSC_ERR_USER, "k / bs (%d) >= nmax (%d)", (int)(k / bs), (int)nmax);
          AA[k / bs] = val;
        }
        grow = Istart / bs + brow / bs;
        PetscCall(MatSetValues(Gmat, 1, &grow, n / bs, AJ, AA, ADD_VALUES));
      }
      // off-diag
      if (ismpiaij) {
        Mat_MPIAIJ        *aij = (Mat_MPIAIJ *)Amat->data;
        const PetscScalar *vals;
        const PetscInt    *cols, *garray = aij->garray;

        PetscCheck(garray, PETSC_COMM_SELF, PETSC_ERR_USER, "No garray ?");
        for (PetscInt brow = 0, grow; brow < nloc * bs; brow += bs) { // block rows
          PetscCall(MatGetRow(b, brow, &ncols, &cols, NULL));
          for (PetscInt k = 0, cidx = 0; k < ncols; k += bs, cidx++) {
            PetscAssert(k / bs < nmax, comm, PETSC_ERR_USER, "k / bs >= nmax");
            AA[k / bs] = 0;
            AJ[cidx]   = garray[cols[k]] / bs;
          }
          nc = ncols / bs;
          PetscCall(MatRestoreRow(b, brow, &ncols, &cols, NULL));
          if (index_size == 0) {
            for (PetscInt ii = 0; ii < bs; ii++) { // rows in block
              PetscCall(MatGetRow(b, brow + ii, &ncols, &cols, &vals));
              for (PetscInt k = 0; k < ncols; k += bs) {
                for (PetscInt jj = 0; jj < bs; jj++) { // cols in block
                  PetscAssert(k / bs < nmax, comm, PETSC_ERR_USER, "k / bs (%d) >= nmax (%d)", (int)(k / bs), (int)nmax);
                  AA[k / bs] += PetscAbs(PetscRealPart(vals[k + jj]));
                }
              }
              PetscCall(MatRestoreRow(b, brow + ii, &ncols, &cols, &vals));
            }
          } else {                                            // use (index,index) value if provided
            for (PetscInt iii = 0; iii < index_size; iii++) { // rows in block
              PetscInt ii = index[iii];
              PetscCall(MatGetRow(b, brow + ii, &ncols, &cols, &vals));
              for (PetscInt k = 0; k < ncols; k += bs) {
                for (PetscInt jjj = 0; jjj < index_size; jjj++) { // cols in block
                  PetscInt jj = index[jjj];
                  AA[k / bs] += PetscAbs(PetscRealPart(vals[k + jj]));
                }
              }
              PetscCall(MatRestoreRow(b, brow + ii, &ncols, &cols, &vals));
            }
          }
          grow = Istart / bs + brow / bs;
          PetscCall(MatSetValues(Gmat, 1, &grow, nc, AJ, AA, ADD_VALUES));
        }
      }
      PetscCall(MatAssemblyBegin(Gmat, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(Gmat, MAT_FINAL_ASSEMBLY));
      PetscCall(PetscFree2(AA, AJ));
    } else {
      const PetscScalar *vals;
      const PetscInt    *idx;
      PetscInt          *d_nnz, *o_nnz, *w0, *w1, *w2;
    old_bs:
      /*
       Determine the preallocation needed for the scalar matrix derived from the vector matrix.
       */
      PetscCall(PetscInfo(Amat, "OLD bs>1 CreateGraph\n"));
      PetscCall(PetscMalloc2(nloc, &d_nnz, (isseqaij ? 0 : nloc), &o_nnz));
      if (isseqaij) {
        PetscInt max_d_nnz;

        /*
         Determine exact preallocation count for (sequential) scalar matrix
         */
        PetscCall(MatSeqAIJGetMaxRowNonzeros(Amat, &max_d_nnz));
        max_d_nnz = PetscMin(nloc, bs * max_d_nnz);
        PetscCall(PetscMalloc3(max_d_nnz, &w0, max_d_nnz, &w1, max_d_nnz, &w2));
        for (Ii = 0, jj = 0; Ii < Iend; Ii += bs, jj++) PetscCall(MatCollapseRows(Amat, Ii, bs, w0, w1, w2, &d_nnz[jj], NULL));
        PetscCall(PetscFree3(w0, w1, w2));
      } else if (ismpiaij) {
        Mat             Daij, Oaij;
        const PetscInt *garray;
        PetscInt        max_d_nnz;

        PetscCall(MatMPIAIJGetSeqAIJ(Amat, &Daij, &Oaij, &garray));
        /*
         Determine exact preallocation count for diagonal block portion of scalar matrix
         */
        PetscCall(MatSeqAIJGetMaxRowNonzeros(Daij, &max_d_nnz));
        max_d_nnz = PetscMin(nloc, bs * max_d_nnz);
        PetscCall(PetscMalloc3(max_d_nnz, &w0, max_d_nnz, &w1, max_d_nnz, &w2));
        for (Ii = 0, jj = 0; Ii < Iend - Istart; Ii += bs, jj++) PetscCall(MatCollapseRows(Daij, Ii, bs, w0, w1, w2, &d_nnz[jj], NULL));
        PetscCall(PetscFree3(w0, w1, w2));
        /*
         Over estimate (usually grossly over), preallocation count for off-diagonal portion of scalar matrix
         */
        for (Ii = 0, jj = 0; Ii < Iend - Istart; Ii += bs, jj++) {
          o_nnz[jj] = 0;
          for (kk = 0; kk < bs; kk++) { /* rows that get collapsed to a single row */
            PetscCall(MatGetRow(Oaij, Ii + kk, &ncols, NULL, NULL));
            o_nnz[jj] += ncols;
            PetscCall(MatRestoreRow(Oaij, Ii + kk, &ncols, NULL, NULL));
          }
          if (o_nnz[jj] > (NN / bs - nloc)) o_nnz[jj] = NN / bs - nloc;
        }
      } else SETERRQ(comm, PETSC_ERR_USER, "Require AIJ matrix type");
      /* get scalar copy (norms) of matrix */
      PetscCall(MatSeqAIJSetPreallocation(Gmat, 0, d_nnz));
      PetscCall(MatMPIAIJSetPreallocation(Gmat, 0, d_nnz, 0, o_nnz));
      PetscCall(PetscFree2(d_nnz, o_nnz));
      for (Ii = Istart; Ii < Iend; Ii++) {
        PetscInt dest_row = Ii / bs;

        PetscCall(MatGetRow(Amat, Ii, &ncols, &idx, &vals));
        for (jj = 0; jj < ncols; jj++) {
          PetscInt    dest_col = idx[jj] / bs;
          PetscScalar sv       = PetscAbs(PetscRealPart(vals[jj]));

          PetscCall(MatSetValues(Gmat, 1, &dest_row, 1, &dest_col, &sv, ADD_VALUES));
        }
        PetscCall(MatRestoreRow(Amat, Ii, &ncols, &idx, &vals));
      }
      PetscCall(MatAssemblyBegin(Gmat, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(Gmat, MAT_FINAL_ASSEMBLY));
    }
  } else {
    if (symmetrize || filter >= 0 || scale) PetscCall(MatDuplicate(Amat, MAT_COPY_VALUES, &Gmat));
    else {
      Gmat = Amat;
      PetscCall(PetscObjectReference((PetscObject)Gmat));
    }
    if (isseqaij) {
      a = Gmat;
      b = NULL;
    } else {
      Mat_MPIAIJ *d = (Mat_MPIAIJ *)Gmat->data;
      a             = d->A;
      b             = d->B;
    }
    if (filter >= 0 || scale) {
      /* take absolute value of each entry */
      for (c = a, kk = 0; c && kk < 2; c = b, kk++) {
        MatInfo      info;
        PetscScalar *avals;

        PetscCall(MatGetInfo(c, MAT_LOCAL, &info));
        PetscCall(MatSeqAIJGetArray(c, &avals));
        for (int jj = 0; jj < info.nz_used; jj++) avals[jj] = PetscAbsScalar(avals[jj]);
        PetscCall(MatSeqAIJRestoreArray(c, &avals));
      }
    }
  }
  if (symmetrize) {
    PetscBool isset, issym;

    PetscCall(MatIsSymmetricKnown(Amat, &isset, &issym));
    if (!isset || !issym) {
      Mat matTrans;

      PetscCall(MatTranspose(Gmat, MAT_INITIAL_MATRIX, &matTrans));
      PetscCall(MatAXPY(Gmat, 1.0, matTrans, Gmat->structurally_symmetric == PETSC_BOOL3_TRUE ? SAME_NONZERO_PATTERN : DIFFERENT_NONZERO_PATTERN));
      PetscCall(MatDestroy(&matTrans));
    }
    PetscCall(MatSetOption(Gmat, MAT_SYMMETRIC, PETSC_TRUE));
  } else if (Amat != Gmat) PetscCall(MatPropagateSymmetryOptions(Amat, Gmat));
  if (scale) {
    /* scale c for all diagonal values = 1 or -1 */
    Vec diag;

    PetscCall(MatCreateVecs(Gmat, &diag, NULL));
    PetscCall(MatGetDiagonal(Gmat, diag));
    PetscCall(VecReciprocal(diag));
    PetscCall(VecSqrtAbs(diag));
    PetscCall(MatDiagonalScale(Gmat, diag, diag));
    PetscCall(VecDestroy(&diag));
  }
  PetscCall(MatViewFromOptions(Gmat, NULL, "-mat_graph_view"));
  if (filter >= 0) {
    PetscCall(MatFilter(Gmat, filter, PETSC_TRUE, PETSC_TRUE));
    PetscCall(MatViewFromOptions(Gmat, NULL, "-mat_filter_graph_view"));
  }
  *a_Gmat = Gmat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    Special version for direct calls from Fortran
*/

/* Change these macros so can be used in void function */
/* Identical to PetscCallVoid, except it assigns to *_ierr */
#undef PetscCall
#define PetscCall(...) \
  do { \
    PetscErrorCode ierr_msv_mpiaij = __VA_ARGS__; \
    if (PetscUnlikely(ierr_msv_mpiaij)) { \
      *_ierr = PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr_msv_mpiaij, PETSC_ERROR_REPEAT, " "); \
      return; \
    } \
  } while (0)

#undef SETERRQ
#define SETERRQ(comm, ierr, ...) \
  do { \
    *_ierr = PetscError(comm, __LINE__, PETSC_FUNCTION_NAME, __FILE__, ierr, PETSC_ERROR_INITIAL, __VA_ARGS__); \
    return; \
  } while (0)

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define matsetvaluesmpiaij_ MATSETVALUESMPIAIJ
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define matsetvaluesmpiaij_ matsetvaluesmpiaij
#else
#endif
PETSC_EXTERN void matsetvaluesmpiaij_(Mat *mmat, PetscInt *mm, const PetscInt im[], PetscInt *mn, const PetscInt in[], const PetscScalar v[], InsertMode *maddv, PetscErrorCode *_ierr)
{
  Mat         mat = *mmat;
  PetscInt    m = *mm, n = *mn;
  InsertMode  addv = *maddv;
  Mat_MPIAIJ *aij  = (Mat_MPIAIJ *)mat->data;
  PetscScalar value;

  MatCheckPreallocated(mat, 1);
  if (mat->insertmode == NOT_SET_VALUES) mat->insertmode = addv;
  else PetscCheck(mat->insertmode == addv, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Cannot mix add values and insert values");
  {
    PetscInt  i, j, rstart = mat->rmap->rstart, rend = mat->rmap->rend;
    PetscInt  cstart = mat->cmap->rstart, cend = mat->cmap->rend, row, col;
    PetscBool roworiented = aij->roworiented;

    /* Some Variables required in the macro */
    Mat         A     = aij->A;
    Mat_SeqAIJ *a     = (Mat_SeqAIJ *)A->data;
    PetscInt   *aimax = a->imax, *ai = a->i, *ailen = a->ilen, *aj = a->j;
    MatScalar  *aa;
    PetscBool   ignorezeroentries = ((a->ignorezeroentries && (addv == ADD_VALUES)) ? PETSC_TRUE : PETSC_FALSE);
    Mat         B                 = aij->B;
    Mat_SeqAIJ *b                 = (Mat_SeqAIJ *)B->data;
    PetscInt   *bimax = b->imax, *bi = b->i, *bilen = b->ilen, *bj = b->j, bm = aij->B->rmap->n, am = aij->A->rmap->n;
    MatScalar  *ba;
    /* This variable below is only for the PETSC_HAVE_VIENNACL or PETSC_HAVE_CUDA cases, but we define it in all cases because we
     * cannot use "#if defined" inside a macro. */
    PETSC_UNUSED PetscBool inserted = PETSC_FALSE;

    PetscInt  *rp1, *rp2, ii, nrow1, nrow2, _i, rmax1, rmax2, N, low1, high1, low2, high2, t, lastcol1, lastcol2;
    PetscInt   nonew = a->nonew;
    MatScalar *ap1, *ap2;

    PetscFunctionBegin;
    PetscCall(MatSeqAIJGetArray(A, &aa));
    PetscCall(MatSeqAIJGetArray(B, &ba));
    for (i = 0; i < m; i++) {
      if (im[i] < 0) continue;
      PetscCheck(im[i] < mat->rmap->N, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT, im[i], mat->rmap->N - 1);
      if (im[i] >= rstart && im[i] < rend) {
        row      = im[i] - rstart;
        lastcol1 = -1;
        rp1      = aj + ai[row];
        ap1      = aa + ai[row];
        rmax1    = aimax[row];
        nrow1    = ailen[row];
        low1     = 0;
        high1    = nrow1;
        lastcol2 = -1;
        rp2      = bj + bi[row];
        ap2      = ba + bi[row];
        rmax2    = bimax[row];
        nrow2    = bilen[row];
        low2     = 0;
        high2    = nrow2;

        for (j = 0; j < n; j++) {
          if (roworiented) value = v[i * n + j];
          else value = v[i + j * m];
          if (ignorezeroentries && value == 0.0 && (addv == ADD_VALUES) && im[i] != in[j]) continue;
          if (in[j] >= cstart && in[j] < cend) {
            col = in[j] - cstart;
            MatSetValues_SeqAIJ_A_Private(row, col, value, addv, im[i], in[j]);
          } else if (in[j] < 0) continue;
          else if (PetscUnlikelyDebug(in[j] >= mat->cmap->N)) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Column too large: col %" PetscInt_FMT " max %" PetscInt_FMT, in[j], mat->cmap->N - 1);
          } else {
            if (mat->was_assembled) {
              if (!aij->colmap) PetscCall(MatCreateColmap_MPIAIJ_Private(mat));
#if defined(PETSC_USE_CTABLE)
              PetscCall(PetscHMapIGetWithDefault(aij->colmap, in[j] + 1, 0, &col));
              col--;
#else
              col = aij->colmap[in[j]] - 1;
#endif
              if (col < 0 && !((Mat_SeqAIJ *)aij->A->data)->nonew) {
                PetscCall(MatDisAssemble_MPIAIJ(mat));
                col = in[j];
                /* Reinitialize the variables required by MatSetValues_SeqAIJ_B_Private() */
                B        = aij->B;
                b        = (Mat_SeqAIJ *)B->data;
                bimax    = b->imax;
                bi       = b->i;
                bilen    = b->ilen;
                bj       = b->j;
                rp2      = bj + bi[row];
                ap2      = ba + bi[row];
                rmax2    = bimax[row];
                nrow2    = bilen[row];
                low2     = 0;
                high2    = nrow2;
                bm       = aij->B->rmap->n;
                ba       = b->a;
                inserted = PETSC_FALSE;
              }
            } else col = in[j];
            MatSetValues_SeqAIJ_B_Private(row, col, value, addv, im[i], in[j]);
          }
        }
      } else if (!aij->donotstash) {
        if (roworiented) {
          PetscCall(MatStashValuesRow_Private(&mat->stash, im[i], n, in, v + i * n, (PetscBool)(ignorezeroentries && (addv == ADD_VALUES))));
        } else {
          PetscCall(MatStashValuesCol_Private(&mat->stash, im[i], n, in, v + i, m, (PetscBool)(ignorezeroentries && (addv == ADD_VALUES))));
        }
      }
    }
    PetscCall(MatSeqAIJRestoreArray(A, &aa));
    PetscCall(MatSeqAIJRestoreArray(B, &ba));
  }
  PetscFunctionReturnVoid();
}

/* Undefining these here since they were redefined from their original definition above! No
 * other PETSc functions should be defined past this point, as it is impossible to recover the
 * original definitions */
#undef PetscCall
#undef SETERRQ
