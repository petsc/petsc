#include <petsc_kokkos.hpp>
#include <petscvec_kokkos.hpp>
#include <petscmat_kokkos.hpp>
#include <petscpkg_version.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/sfimpl.h>
#include <petsc/private/kokkosimpl.hpp>
#include <petscsys.h>

#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
#include <KokkosSparse_CrsMatrix.hpp>

// To suppress compiler warnings:
// /path/include/KokkosSparse_spmv_bsrmatrix_tpl_spec_decl.hpp:434:63:
// warning: 'cusparseStatus_t cusparseDbsrmm(cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
// cusparseOperation_t, int, int, int, int, const double*, cusparseMatDescr_t, const double*, const int*, const int*,
// int, const double*, int, const double*, double*, int)' is deprecated: please use cusparseSpMM instead [-Wdeprecated-declarations]
#define DISABLE_CUSPARSE_DEPRECATED
#include <KokkosSparse_spmv.hpp>

#include <KokkosSparse_spiluk.hpp>
#include <KokkosSparse_sptrsv.hpp>
#include <KokkosSparse_spgemm.hpp>
#include <KokkosSparse_spadd.hpp>
#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_InverseLU_Decl.hpp>

#include <../src/mat/impls/aij/seq/kokkos/aijkok.hpp>

#if PETSC_PKG_KOKKOS_KERNELS_VERSION_GE(3, 7, 0)
  #include <KokkosSparse_Utils.hpp>
using KokkosSparse::sort_crs_matrix;
using KokkosSparse::Impl::transpose_matrix;
#else
  #include <KokkosKernels_Sorting.hpp>
using KokkosKernels::sort_crs_matrix;
using KokkosKernels::Impl::transpose_matrix;
#endif

#if PETSC_PKG_KOKKOS_KERNELS_VERSION_GE(4, 6, 0)
using KokkosSparse::spiluk_symbolic;
using KokkosSparse::spiluk_numeric;
using KokkosSparse::sptrsv_symbolic;
using KokkosSparse::sptrsv_solve;
using KokkosSparse::Experimental::SPTRSVAlgorithm;
using KokkosSparse::Experimental::SPILUKAlgorithm;
#else
using KokkosSparse::Experimental::spiluk_symbolic;
using KokkosSparse::Experimental::spiluk_numeric;
using KokkosSparse::Experimental::sptrsv_symbolic;
using KokkosSparse::Experimental::sptrsv_solve;
using KokkosSparse::Experimental::SPTRSVAlgorithm;
using KokkosSparse::Experimental::SPILUKAlgorithm;
#endif

static PetscErrorCode MatSetOps_SeqAIJKokkos(Mat); /* Forward declaration */

/* MatAssemblyEnd_SeqAIJKokkos() happens when we finalized nonzeros of the matrix, either after
   we assembled the matrix on host, or after we directly produced the matrix data on device (ex., through MatMatMult).
   In the latter case, it is important to set a_dual's sync state correctly.
 */
static PetscErrorCode MatAssemblyEnd_SeqAIJKokkos(Mat A, MatAssemblyType mode)
{
  Mat_SeqAIJ       *aijseq;
  Mat_SeqAIJKokkos *aijkok;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatAssemblyEnd_SeqAIJ(A, mode));

  aijseq = static_cast<Mat_SeqAIJ *>(A->data);
  aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);

  /* If aijkok does not exist, we just copy i, j to device.
     If aijkok already exists, but the device's nonzero pattern does not match with the host's, we assume the latest data is on host.
     In both cases, we build a new aijkok structure.
  */
  if (!aijkok || aijkok->nonzerostate != A->nonzerostate) { /* aijkok might not exist yet or nonzero pattern has changed */
    if (aijkok && aijkok->host_aij_allocated_by_kokkos) {   /* Avoid accidentally freeing much needed a,i,j on host when deleting aijkok */
      PetscCall(PetscShmgetAllocateArray(aijkok->nrows() + 1, sizeof(PetscInt), (void **)&aijseq->i));
      PetscCall(PetscShmgetAllocateArray(aijkok->nnz(), sizeof(PetscInt), (void **)&aijseq->j));
      PetscCall(PetscShmgetAllocateArray(aijkok->nnz(), sizeof(PetscInt), (void **)&aijseq->a));
      PetscCall(PetscArraycpy(aijseq->i, aijkok->i_host_data(), aijkok->nrows() + 1));
      PetscCall(PetscArraycpy(aijseq->j, aijkok->j_host_data(), aijkok->nnz()));
      PetscCall(PetscArraycpy(aijseq->a, aijkok->a_host_data(), aijkok->nnz()));
      aijseq->free_a  = PETSC_TRUE;
      aijseq->free_ij = PETSC_TRUE;
      /* This arises from MatCreateSeqAIJKokkosWithKokkosCsrMatrix() used in MatMatMult, where
         we have the CsrMatrix on device first and then copy to host, followed by
         MatSetMPIAIJWithSplitSeqAIJ() with garray = NULL.
         One could improve it by not using NULL garray.
      */
    }
    delete aijkok;
    aijkok   = new Mat_SeqAIJKokkos(A->rmap->n, A->cmap->n, aijseq, A->nonzerostate, PETSC_FALSE /*don't copy mat values to device*/);
    A->spptr = aijkok;
  } else if (A->rmap->n && aijkok->diag_dual.extent(0) == 0) { // MatProduct might directly produce AIJ on device, but not the diag.
    MatRowMapKokkosViewHost diag_h(aijseq->diag, A->rmap->n);
    auto                    diag_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), diag_h);
    aijkok->diag_dual              = MatRowMapKokkosDualView(diag_d, diag_h);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Sync CSR data to device if not yet */
PETSC_INTERN PetscErrorCode MatSeqAIJKokkosSyncDevice(Mat A)
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCheck(A->factortype == MAT_FACTOR_NONE, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Can't sync factorized matrix from host to device");
  PetscCheck(aijkok, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Unexpected NULL (Mat_SeqAIJKokkos*)A->spptr");
  if (aijkok->a_dual.need_sync_device()) {
    PetscCall(KokkosDualViewSyncDevice(aijkok->a_dual, PetscGetKokkosExecutionSpace()));
    aijkok->transpose_updated = PETSC_FALSE; /* values of the transpose is out-of-date */
    aijkok->hermitian_updated = PETSC_FALSE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Mark the CSR data on device as modified */
PETSC_INTERN PetscErrorCode MatSeqAIJKokkosModifyDevice(Mat A)
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCheck(A->factortype == MAT_FACTOR_NONE, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Not supported for factorized matries");
  aijkok->a_dual.clear_sync_state();
  aijkok->a_dual.modify_device();
  aijkok->transpose_updated = PETSC_FALSE;
  aijkok->hermitian_updated = PETSC_FALSE;
  PetscCall(MatSeqAIJInvalidateDiagonal(A));
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJKokkosSyncHost(Mat A)
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
  auto              exec   = PetscGetKokkosExecutionSpace();

  PetscFunctionBegin;
  PetscCheckTypeName(A, MATSEQAIJKOKKOS);
  /* We do not expect one needs factors on host  */
  PetscCheck(A->factortype == MAT_FACTOR_NONE, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Can't sync factorized matrix from device to host");
  PetscCheck(aijkok, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Missing AIJKOK");
  PetscCall(KokkosDualViewSyncHost(aijkok->a_dual, exec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJGetArray_SeqAIJKokkos(Mat A, PetscScalar *array[])
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);

  PetscFunctionBegin;
  /* aijkok contains valid pointers only if the host's nonzerostate matches with the device's.
    Calling MatSeqAIJSetPreallocation() or MatSetValues() on host, where aijseq->{i,j,a} might be
    reallocated, will lead to stale {i,j,a}_dual in aijkok. In both operations, the hosts's nonzerostate
    must have been updated. The stale aijkok will be rebuilt during MatAssemblyEnd.
  */
  if (aijkok && A->nonzerostate == aijkok->nonzerostate) {
    PetscCall(KokkosDualViewSyncHost(aijkok->a_dual, PetscGetKokkosExecutionSpace()));
    *array = aijkok->a_dual.view_host().data();
  } else { /* Happens when calling MatSetValues on a newly created matrix */
    *array = static_cast<Mat_SeqAIJ *>(A->data)->a;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJRestoreArray_SeqAIJKokkos(Mat A, PetscScalar *array[])
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);

  PetscFunctionBegin;
  if (aijkok && A->nonzerostate == aijkok->nonzerostate) aijkok->a_dual.modify_host();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJGetArrayRead_SeqAIJKokkos(Mat A, const PetscScalar *array[])
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);

  PetscFunctionBegin;
  if (aijkok && A->nonzerostate == aijkok->nonzerostate) {
    PetscCall(KokkosDualViewSyncHost(aijkok->a_dual, PetscGetKokkosExecutionSpace()));
    *array = aijkok->a_dual.view_host().data();
  } else {
    *array = static_cast<Mat_SeqAIJ *>(A->data)->a;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJRestoreArrayRead_SeqAIJKokkos(Mat A, const PetscScalar *array[])
{
  PetscFunctionBegin;
  *array = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJGetArrayWrite_SeqAIJKokkos(Mat A, PetscScalar *array[])
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);

  PetscFunctionBegin;
  if (aijkok && A->nonzerostate == aijkok->nonzerostate) {
    *array = aijkok->a_dual.view_host().data();
  } else { /* Ex. happens with MatZeroEntries on a preallocated but not assembled matrix */
    *array = static_cast<Mat_SeqAIJ *>(A->data)->a;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJRestoreArrayWrite_SeqAIJKokkos(Mat A, PetscScalar *array[])
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);

  PetscFunctionBegin;
  if (aijkok && A->nonzerostate == aijkok->nonzerostate) {
    aijkok->a_dual.clear_sync_state();
    aijkok->a_dual.modify_host();
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJGetCSRAndMemType_SeqAIJKokkos(Mat A, const PetscInt **i, const PetscInt **j, PetscScalar **a, PetscMemType *mtype)
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);

  PetscFunctionBegin;
  PetscCheck(aijkok != NULL, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONGSTATE, "aijkok is NULL");

  if (i) *i = aijkok->i_device_data();
  if (j) *j = aijkok->j_device_data();
  if (a) {
    PetscCall(KokkosDualViewSyncDevice(aijkok->a_dual, PetscGetKokkosExecutionSpace()));
    *a = aijkok->a_device_data();
  }
  if (mtype) *mtype = PETSC_MEMTYPE_KOKKOS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetCurrentMemType_SeqAIJKokkos(PETSC_UNUSED Mat A, PetscMemType *mtype)
{
  PetscFunctionBegin;
  *mtype = PETSC_MEMTYPE_KOKKOS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Generate the sparsity pattern of a MatSeqAIJKokkos matrix's transpose on device.

  Input Parameter:
.  A       - the MATSEQAIJKOKKOS matrix

  Output Parameters:
+  perm_d - the permutation array on device, which connects Ta(i) = Aa(perm(i))
-  T_d    - the transpose on device, whose value array is allocated but not initialized
*/
static PetscErrorCode MatSeqAIJKokkosGenerateTransposeStructure(Mat A, MatRowMapKokkosView &perm_d, KokkosCsrMatrix &T_d)
{
  Mat_SeqAIJ             *aseq = static_cast<Mat_SeqAIJ *>(A->data);
  PetscInt                nz = aseq->nz, m = A->rmap->N, n = A->cmap->n;
  const PetscInt         *Ai = aseq->i, *Aj = aseq->j;
  MatRowMapKokkosViewHost Ti_h(NoInit("Ti"), n + 1);
  MatRowMapType          *Ti = Ti_h.data();
  MatColIdxKokkosViewHost Tj_h(NoInit("Tj"), nz);
  MatRowMapKokkosViewHost perm_h(NoInit("permutation"), nz);
  PetscInt               *Tj   = Tj_h.data();
  PetscInt               *perm = perm_h.data();
  PetscInt               *offset;

  PetscFunctionBegin;
  // Populate Ti
  PetscCallCXX(Kokkos::deep_copy(Ti_h, 0));
  Ti++;
  for (PetscInt i = 0; i < nz; i++) Ti[Aj[i]]++;
  Ti--;
  for (PetscInt i = 0; i < n; i++) Ti[i + 1] += Ti[i];

  // Populate Tj and the permutation array
  PetscCall(PetscCalloc1(n, &offset)); // offset in each T row to fill in its column indices
  for (PetscInt i = 0; i < m; i++) {
    for (PetscInt j = Ai[i]; j < Ai[i + 1]; j++) { // A's (i,j) is T's (j,i)
      PetscInt r    = Aj[j];                       // row r of T
      PetscInt disp = Ti[r] + offset[r];

      Tj[disp]   = i; // col i of T
      perm[disp] = j;
      offset[r]++;
    }
  }
  PetscCall(PetscFree(offset));

  // Sort each row of T, along with the permutation array
  for (PetscInt i = 0; i < n; i++) PetscCall(PetscSortIntWithArray(Ti[i + 1] - Ti[i], Tj + Ti[i], perm + Ti[i]));

  // Output perm and T on device
  auto Ti_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), Ti_h);
  auto Tj_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), Tj_h);
  PetscCallCXX(T_d = KokkosCsrMatrix("csrmatT", n, m, nz, MatScalarKokkosView("Ta", nz), Ti_d, Tj_d));
  PetscCallCXX(perm_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), perm_h));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Generate the transpose on device and cache it internally
// Note: KK transpose_matrix() does not have support symbolic/numeric transpose, so we do it on our own
PETSC_INTERN PetscErrorCode MatSeqAIJKokkosGenerateTranspose_Private(Mat A, KokkosCsrMatrix *csrmatT)
{
  Mat_SeqAIJ       *aseq = static_cast<Mat_SeqAIJ *>(A->data);
  Mat_SeqAIJKokkos *akok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
  PetscInt          nz = aseq->nz, m = A->rmap->N, n = A->cmap->n;
  KokkosCsrMatrix  &T = akok->csrmatT;

  PetscFunctionBegin;
  PetscCheck(akok, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Unexpected NULL (Mat_SeqAIJKokkos*)A->spptr");
  PetscCall(KokkosDualViewSyncDevice(akok->a_dual, PetscGetKokkosExecutionSpace())); // Sync A's values since we are going to access them on device

  const auto &Aa = akok->a_dual.view_device();

  if (A->symmetric == PETSC_BOOL3_TRUE) {
    *csrmatT = akok->csrmat;
  } else {
    // See if we already have a cached transpose and its value is up to date
    if (T.numRows() == n && T.numCols() == m) {  // this indicates csrmatT had been generated before, otherwise T has 0 rows/cols after construction
      if (!akok->transpose_updated) {            // if the value is out of date, update the cached version
        const auto &perm = akok->transpose_perm; // get the permutation array
        auto       &Ta   = T.values;

        PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, nz), KOKKOS_LAMBDA(const PetscInt i) { Ta(i) = Aa(perm(i)); }));
      }
    } else { // Generate T of size n x m for the first time
      MatRowMapKokkosView perm;

      PetscCall(MatSeqAIJKokkosGenerateTransposeStructure(A, perm, T));
      akok->transpose_perm = perm; // cache the perm in this matrix for reuse
      PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, nz), KOKKOS_LAMBDA(const PetscInt i) { T.values(i) = Aa(perm(i)); }));
    }
    akok->transpose_updated = PETSC_TRUE;
    *csrmatT                = akok->csrmatT;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Generate the Hermitian on device and cache it internally
static PetscErrorCode MatSeqAIJKokkosGenerateHermitian_Private(Mat A, KokkosCsrMatrix *csrmatH)
{
  Mat_SeqAIJ       *aseq = static_cast<Mat_SeqAIJ *>(A->data);
  Mat_SeqAIJKokkos *akok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
  PetscInt          nz = aseq->nz, m = A->rmap->N, n = A->cmap->n;
  KokkosCsrMatrix  &T = akok->csrmatH;

  PetscFunctionBegin;
  PetscCheck(akok, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Unexpected NULL (Mat_SeqAIJKokkos*)A->spptr");
  PetscCall(KokkosDualViewSyncDevice(akok->a_dual, PetscGetKokkosExecutionSpace())); // Sync A's values since we are going to access them on device

  const auto &Aa = akok->a_dual.view_device();

  if (A->hermitian == PETSC_BOOL3_TRUE) {
    *csrmatH = akok->csrmat;
  } else {
    // See if we already have a cached hermitian and its value is up to date
    if (T.numRows() == n && T.numCols() == m) {  // this indicates csrmatT had been generated before, otherwise T has 0 rows/cols after construction
      if (!akok->hermitian_updated) {            // if the value is out of date, update the cached version
        const auto &perm = akok->transpose_perm; // get the permutation array
        auto       &Ta   = T.values;

        PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, nz), KOKKOS_LAMBDA(const PetscInt i) { Ta(i) = PetscConj(Aa(perm(i))); }));
      }
    } else { // Generate T of size n x m for the first time
      MatRowMapKokkosView perm;

      PetscCall(MatSeqAIJKokkosGenerateTransposeStructure(A, perm, T));
      akok->transpose_perm = perm; // cache the perm in this matrix for reuse
      PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, nz), KOKKOS_LAMBDA(const PetscInt i) { T.values(i) = PetscConj(Aa(perm(i))); }));
    }
    akok->hermitian_updated = PETSC_TRUE;
    *csrmatH                = akok->csrmatH;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y = A x */
static PetscErrorCode MatMult_SeqAIJKokkos(Mat A, Vec xx, Vec yy)
{
  Mat_SeqAIJKokkos          *aijkok;
  ConstPetscScalarKokkosView xv;
  PetscScalarKokkosView      yv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  PetscCall(VecGetKokkosView(xx, &xv));
  PetscCall(VecGetKokkosViewWrite(yy, &yv));
  aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
  PetscCallCXX(KokkosSparse::spmv(PetscGetKokkosExecutionSpace(), "N", 1.0 /*alpha*/, aijkok->csrmat, xv, 0.0 /*beta*/, yv)); /* y = alpha A x + beta y */
  PetscCall(VecRestoreKokkosView(xx, &xv));
  PetscCall(VecRestoreKokkosViewWrite(yy, &yv));
  /* 2.0*nnz - numRows seems more accurate here but assumes there are no zero-rows. So a little sloppy here. */
  PetscCall(PetscLogGpuFlops(2.0 * aijkok->csrmat.nnz()));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y = A^T x */
static PetscErrorCode MatMultTranspose_SeqAIJKokkos(Mat A, Vec xx, Vec yy)
{
  Mat_SeqAIJKokkos          *aijkok;
  const char                *mode;
  ConstPetscScalarKokkosView xv;
  PetscScalarKokkosView      yv;
  KokkosCsrMatrix            csrmat;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  PetscCall(VecGetKokkosView(xx, &xv));
  PetscCall(VecGetKokkosViewWrite(yy, &yv));
  if (A->form_explicit_transpose) {
    PetscCall(MatSeqAIJKokkosGenerateTranspose_Private(A, &csrmat));
    mode = "N";
  } else {
    aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
    csrmat = aijkok->csrmat;
    mode   = "T";
  }
  PetscCallCXX(KokkosSparse::spmv(PetscGetKokkosExecutionSpace(), mode, 1.0 /*alpha*/, csrmat, xv, 0.0 /*beta*/, yv)); /* y = alpha A^T x + beta y */
  PetscCall(VecRestoreKokkosView(xx, &xv));
  PetscCall(VecRestoreKokkosViewWrite(yy, &yv));
  PetscCall(PetscLogGpuFlops(2.0 * csrmat.nnz()));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* y = A^H x */
static PetscErrorCode MatMultHermitianTranspose_SeqAIJKokkos(Mat A, Vec xx, Vec yy)
{
  Mat_SeqAIJKokkos          *aijkok;
  const char                *mode;
  ConstPetscScalarKokkosView xv;
  PetscScalarKokkosView      yv;
  KokkosCsrMatrix            csrmat;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  PetscCall(VecGetKokkosView(xx, &xv));
  PetscCall(VecGetKokkosViewWrite(yy, &yv));
  if (A->form_explicit_transpose) {
    PetscCall(MatSeqAIJKokkosGenerateHermitian_Private(A, &csrmat));
    mode = "N";
  } else {
    aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
    csrmat = aijkok->csrmat;
    mode   = "C";
  }
  PetscCallCXX(KokkosSparse::spmv(PetscGetKokkosExecutionSpace(), mode, 1.0 /*alpha*/, csrmat, xv, 0.0 /*beta*/, yv)); /* y = alpha A^H x + beta y */
  PetscCall(VecRestoreKokkosView(xx, &xv));
  PetscCall(VecRestoreKokkosViewWrite(yy, &yv));
  PetscCall(PetscLogGpuFlops(2.0 * csrmat.nnz()));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* z = A x + y */
static PetscErrorCode MatMultAdd_SeqAIJKokkos(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqAIJKokkos          *aijkok;
  ConstPetscScalarKokkosView xv;
  PetscScalarKokkosView      zv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  if (zz != yy) PetscCall(VecCopy(yy, zz)); // depending on yy's sync flags, zz might get its latest data on host
  PetscCall(VecGetKokkosView(xx, &xv));
  PetscCall(VecGetKokkosView(zz, &zv)); // do after VecCopy(yy, zz) to get the latest data on device
  aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
  PetscCallCXX(KokkosSparse::spmv(PetscGetKokkosExecutionSpace(), "N", 1.0 /*alpha*/, aijkok->csrmat, xv, 1.0 /*beta*/, zv)); /* z = alpha A x + beta z */
  PetscCall(VecRestoreKokkosView(xx, &xv));
  PetscCall(VecRestoreKokkosView(zz, &zv));
  PetscCall(PetscLogGpuFlops(2.0 * aijkok->csrmat.nnz()));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* z = A^T x + y */
static PetscErrorCode MatMultTransposeAdd_SeqAIJKokkos(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqAIJKokkos          *aijkok;
  const char                *mode;
  ConstPetscScalarKokkosView xv;
  PetscScalarKokkosView      zv;
  KokkosCsrMatrix            csrmat;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  if (zz != yy) PetscCall(VecCopy(yy, zz));
  PetscCall(VecGetKokkosView(xx, &xv));
  PetscCall(VecGetKokkosView(zz, &zv));
  if (A->form_explicit_transpose) {
    PetscCall(MatSeqAIJKokkosGenerateTranspose_Private(A, &csrmat));
    mode = "N";
  } else {
    aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
    csrmat = aijkok->csrmat;
    mode   = "T";
  }
  PetscCallCXX(KokkosSparse::spmv(PetscGetKokkosExecutionSpace(), mode, 1.0 /*alpha*/, csrmat, xv, 1.0 /*beta*/, zv)); /* z = alpha A^T x + beta z */
  PetscCall(VecRestoreKokkosView(xx, &xv));
  PetscCall(VecRestoreKokkosView(zz, &zv));
  PetscCall(PetscLogGpuFlops(2.0 * csrmat.nnz()));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* z = A^H x + y */
static PetscErrorCode MatMultHermitianTransposeAdd_SeqAIJKokkos(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqAIJKokkos          *aijkok;
  const char                *mode;
  ConstPetscScalarKokkosView xv;
  PetscScalarKokkosView      zv;
  KokkosCsrMatrix            csrmat;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  if (zz != yy) PetscCall(VecCopy(yy, zz));
  PetscCall(VecGetKokkosView(xx, &xv));
  PetscCall(VecGetKokkosView(zz, &zv));
  if (A->form_explicit_transpose) {
    PetscCall(MatSeqAIJKokkosGenerateHermitian_Private(A, &csrmat));
    mode = "N";
  } else {
    aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
    csrmat = aijkok->csrmat;
    mode   = "C";
  }
  PetscCallCXX(KokkosSparse::spmv(PetscGetKokkosExecutionSpace(), mode, 1.0 /*alpha*/, csrmat, xv, 1.0 /*beta*/, zv)); /* z = alpha A^H x + beta z */
  PetscCall(VecRestoreKokkosView(xx, &xv));
  PetscCall(VecRestoreKokkosView(zz, &zv));
  PetscCall(PetscLogGpuFlops(2.0 * csrmat.nnz()));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetOption_SeqAIJKokkos(Mat A, MatOption op, PetscBool flg)
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);

  PetscFunctionBegin;
  switch (op) {
  case MAT_FORM_EXPLICIT_TRANSPOSE:
    /* need to destroy the transpose matrix if present to prevent from logic errors if flg is set to true later */
    if (A->form_explicit_transpose && !flg && aijkok) PetscCall(aijkok->DestroyMatTranspose());
    A->form_explicit_transpose = flg;
    break;
  default:
    PetscCall(MatSetOption_SeqAIJ(A, op, flg));
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Depending on reuse, either build a new mat, or use the existing mat */
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJKokkos(Mat A, MatType mtype, MatReuse reuse, Mat *newmat)
{
  Mat_SeqAIJ *aseq;

  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  if (reuse == MAT_INITIAL_MATRIX) { /* Build a brand new mat */
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, newmat));
    PetscCall(MatSetType(*newmat, mtype));
  } else if (reuse == MAT_REUSE_MATRIX) {                 /* Reuse the mat created before */
    PetscCall(MatCopy(A, *newmat, SAME_NONZERO_PATTERN)); /* newmat is already a SeqAIJKokkos */
  } else if (reuse == MAT_INPLACE_MATRIX) {               /* newmat is A */
    PetscCheck(A == *newmat, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "A != *newmat with MAT_INPLACE_MATRIX");
    PetscCall(PetscFree(A->defaultvectype));
    PetscCall(PetscStrallocpy(VECKOKKOS, &A->defaultvectype)); /* Allocate and copy the string */
    PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATSEQAIJKOKKOS));
    PetscCall(MatSetOps_SeqAIJKokkos(A));
    aseq = static_cast<Mat_SeqAIJ *>(A->data);
    if (A->assembled) { /* Copy i, j (but not values) to device for an assembled matrix if not yet */
      PetscCheck(!A->spptr, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Expect NULL (Mat_SeqAIJKokkos*)A->spptr");
      A->spptr = new Mat_SeqAIJKokkos(A->rmap->n, A->cmap->n, aseq, A->nonzerostate, PETSC_FALSE);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* MatDuplicate always creates a new matrix. MatDuplicate can be called either on an assembled matrix or
   an unassembled matrix, even though MAT_COPY_VALUES is not allowed for unassembled matrix.
 */
static PetscErrorCode MatDuplicate_SeqAIJKokkos(Mat A, MatDuplicateOption dupOption, Mat *B)
{
  Mat_SeqAIJ       *bseq;
  Mat_SeqAIJKokkos *akok = static_cast<Mat_SeqAIJKokkos *>(A->spptr), *bkok;
  Mat               mat;

  PetscFunctionBegin;
  /* Do not copy values on host as A's latest values might be on device. We don't want to do sync blindly */
  PetscCall(MatDuplicate_SeqAIJ(A, MAT_DO_NOT_COPY_VALUES, B));
  mat = *B;
  if (A->assembled) {
    bseq = static_cast<Mat_SeqAIJ *>(mat->data);
    bkok = new Mat_SeqAIJKokkos(mat->rmap->n, mat->cmap->n, bseq, mat->nonzerostate, PETSC_FALSE);
    bkok->a_dual.clear_sync_state(); /* Clear B's sync state as it will be decided below */
    /* Now copy values to B if needed */
    if (dupOption == MAT_COPY_VALUES) {
      if (akok->a_dual.need_sync_device()) {
        Kokkos::deep_copy(bkok->a_dual.view_host(), akok->a_dual.view_host());
        bkok->a_dual.modify_host();
      } else { /* If device has the latest data, we only copy data on device */
        Kokkos::deep_copy(bkok->a_dual.view_device(), akok->a_dual.view_device());
        bkok->a_dual.modify_device();
      }
    } else { /* MAT_DO_NOT_COPY_VALUES or MAT_SHARE_NONZERO_PATTERN. B's values should be zeroed */
      /* B's values on host should be already zeroed by MatDuplicate_SeqAIJ() */
      bkok->a_dual.modify_host();
    }
    mat->spptr = bkok;
  }

  PetscCall(PetscFree(mat->defaultvectype));
  PetscCall(PetscStrallocpy(VECKOKKOS, &mat->defaultvectype)); /* Allocate and copy the string */
  PetscCall(PetscObjectChangeTypeName((PetscObject)mat, MATSEQAIJKOKKOS));
  PetscCall(MatSetOps_SeqAIJKokkos(mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatTranspose_SeqAIJKokkos(Mat A, MatReuse reuse, Mat *B)
{
  Mat               At;
  KokkosCsrMatrix   internT;
  Mat_SeqAIJKokkos *atkok, *bkok;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) PetscCall(MatTransposeCheckNonzeroState_Private(A, *B));
  PetscCall(MatSeqAIJKokkosGenerateTranspose_Private(A, &internT)); /* Generate a transpose internally */
  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_INPLACE_MATRIX) {
    /* Deep copy internT, as we want to isolate the internal transpose */
    PetscCallCXX(atkok = new Mat_SeqAIJKokkos(KokkosCsrMatrix("csrmat", internT)));
    PetscCall(MatCreateSeqAIJKokkosWithCSRMatrix(PetscObjectComm((PetscObject)A), atkok, &At));
    if (reuse == MAT_INITIAL_MATRIX) *B = At;
    else PetscCall(MatHeaderReplace(A, &At)); /* Replace A with At inplace */
  } else {                                    /* MAT_REUSE_MATRIX, just need to copy values to B on device */
    if ((*B)->assembled) {
      bkok = static_cast<Mat_SeqAIJKokkos *>((*B)->spptr);
      PetscCallCXX(Kokkos::deep_copy(bkok->a_dual.view_device(), internT.values));
      PetscCall(MatSeqAIJKokkosModifyDevice(*B));
    } else if ((*B)->preallocated) { /* It is ok for B to be only preallocated, as needed in MatTranspose_MPIAIJ */
      Mat_SeqAIJ             *bseq = static_cast<Mat_SeqAIJ *>((*B)->data);
      MatScalarKokkosViewHost a_h(bseq->a, internT.nnz()); /* bseq->nz = 0 if unassembled */
      MatColIdxKokkosViewHost j_h(bseq->j, internT.nnz());
      PetscCallCXX(Kokkos::deep_copy(a_h, internT.values));
      PetscCallCXX(Kokkos::deep_copy(j_h, internT.graph.entries));
    } else SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONGSTATE, "B must be assembled or preallocated");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_SeqAIJKokkos(Mat A)
{
  Mat_SeqAIJKokkos *aijkok;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
    delete aijkok;
  } else {
    delete static_cast<Mat_SeqAIJKokkosTriFactors *>(A->spptr);
  }
  A->spptr = NULL;
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatFactorGetSolverType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetPreallocationCOO_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetValuesCOO_C", NULL));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaijkokkos_hypre_C", NULL));
#endif
  PetscCall(MatDestroy_SeqAIJ(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATSEQAIJKOKKOS - MATAIJKOKKOS = "(seq)aijkokkos" - A matrix type to be used for sparse matrices with Kokkos

   A matrix type using Kokkos-Kernels CrsMatrix type for portability across different device types

   Options Database Key:
.  -mat_type aijkokkos - sets the matrix type to `MATSEQAIJKOKKOS` during a call to `MatSetFromOptions()`

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatCreateSeqAIJKokkos()`, `MATMPIAIJKOKKOS`
M*/
PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJKokkos(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(MatCreate_SeqAIJ(A));
  PetscCall(MatConvert_SeqAIJ_SeqAIJKokkos(A, MATSEQAIJKOKKOS, MAT_INPLACE_MATRIX, &A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Merge A, B into a matrix C. A is put before B. C's size would be A->rmap->n by (A->cmap->n + B->cmap->n) */
PetscErrorCode MatSeqAIJKokkosMergeMats(Mat A, Mat B, MatReuse reuse, Mat *C)
{
  Mat_SeqAIJ         *a, *b;
  Mat_SeqAIJKokkos   *akok, *bkok, *ckok;
  MatScalarKokkosView aa, ba, ca;
  MatRowMapKokkosView ai, bi, ci;
  MatColIdxKokkosView aj, bj, cj;
  PetscInt            m, n, nnz, aN;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(B, MAT_CLASSID, 2);
  PetscAssertPointer(C, 4);
  PetscCheckTypeName(A, MATSEQAIJKOKKOS);
  PetscCheckTypeName(B, MATSEQAIJKOKKOS);
  PetscCheck(A->rmap->n == B->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid number or rows %" PetscInt_FMT " != %" PetscInt_FMT, A->rmap->n, B->rmap->n);
  PetscCheck(reuse != MAT_INPLACE_MATRIX, PETSC_COMM_SELF, PETSC_ERR_SUP, "MAT_INPLACE_MATRIX not supported");

  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  PetscCall(MatSeqAIJKokkosSyncDevice(B));
  a    = static_cast<Mat_SeqAIJ *>(A->data);
  b    = static_cast<Mat_SeqAIJ *>(B->data);
  akok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
  bkok = static_cast<Mat_SeqAIJKokkos *>(B->spptr);
  aa   = akok->a_dual.view_device();
  ai   = akok->i_dual.view_device();
  ba   = bkok->a_dual.view_device();
  bi   = bkok->i_dual.view_device();
  m    = A->rmap->n; /* M, N and nnz of C */
  n    = A->cmap->n + B->cmap->n;
  nnz  = a->nz + b->nz;
  aN   = A->cmap->n; /* N of A */
  if (reuse == MAT_INITIAL_MATRIX) {
    aj      = akok->j_dual.view_device();
    bj      = bkok->j_dual.view_device();
    auto ca = MatScalarKokkosView("a", aa.extent(0) + ba.extent(0));
    auto ci = MatRowMapKokkosView("i", ai.extent(0));
    auto cj = MatColIdxKokkosView("j", aj.extent(0) + bj.extent(0));

    /* Concatenate A and B in parallel using Kokkos hierarchical parallelism */
    Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), m, Kokkos::AUTO()), KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {
        PetscInt i       = t.league_rank(); /* row i */
        PetscInt coffset = ai(i) + bi(i), alen = ai(i + 1) - ai(i), blen = bi(i + 1) - bi(i);

        Kokkos::single(Kokkos::PerTeam(t), [=]() { /* this side effect only happens once per whole team */
                                                   ci(i) = coffset;
                                                   if (i == m - 1) ci(m) = ai(m) + bi(m);
        });

        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, alen + blen), [&](PetscInt k) {
          if (k < alen) {
            ca(coffset + k) = aa(ai(i) + k);
            cj(coffset + k) = aj(ai(i) + k);
          } else {
            ca(coffset + k) = ba(bi(i) + k - alen);
            cj(coffset + k) = bj(bi(i) + k - alen) + aN; /* Entries in B get new column indices in C */
          }
        });
      });
    PetscCallCXX(ckok = new Mat_SeqAIJKokkos(m, n, nnz, ci, cj, ca));
    PetscCall(MatCreateSeqAIJKokkosWithCSRMatrix(PETSC_COMM_SELF, ckok, C));
  } else if (reuse == MAT_REUSE_MATRIX) {
    PetscValidHeaderSpecific(*C, MAT_CLASSID, 4);
    PetscCheckTypeName(*C, MATSEQAIJKOKKOS);
    ckok = static_cast<Mat_SeqAIJKokkos *>((*C)->spptr);
    ca   = ckok->a_dual.view_device();
    ci   = ckok->i_dual.view_device();

    Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), m, Kokkos::AUTO()), KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {
        PetscInt i    = t.league_rank(); /* row i */
        PetscInt alen = ai(i + 1) - ai(i), blen = bi(i + 1) - bi(i);
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, alen + blen), [&](PetscInt k) {
          if (k < alen) ca(ci(i) + k) = aa(ai(i) + k);
          else ca(ci(i) + k) = ba(bi(i) + k - alen);
        });
      });
    PetscCall(MatSeqAIJKokkosModifyDevice(*C));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductDataDestroy_SeqAIJKokkos(void *pdata)
{
  PetscFunctionBegin;
  delete static_cast<MatProductData_SeqAIJKokkos *>(pdata);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductNumeric_SeqAIJKokkos_SeqAIJKokkos(Mat C)
{
  Mat_Product                 *product = C->product;
  Mat                          A, B;
  bool                         transA, transB; /* use bool, since KK needs this type */
  Mat_SeqAIJKokkos            *akok, *bkok, *ckok;
  Mat_SeqAIJ                  *c;
  MatProductData_SeqAIJKokkos *pdata;
  KokkosCsrMatrix              csrmatA, csrmatB;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(C->product->data, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Product data empty");
  pdata = static_cast<MatProductData_SeqAIJKokkos *>(C->product->data);

  // See if numeric has already been done in symbolic (e.g., user calls MatMatMult(A,B,MAT_INITIAL_MATRIX,..,C)).
  // If yes, skip the numeric, but reset the flag so that next time when user calls MatMatMult(E,F,MAT_REUSE_MATRIX,..,C),
  // we still do numeric.
  if (pdata->reusesym) { // numeric reuses results from symbolic
    pdata->reusesym = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  switch (product->type) {
  case MATPRODUCT_AB:
    transA = false;
    transB = false;
    break;
  case MATPRODUCT_AtB:
    transA = true;
    transB = false;
    break;
  case MATPRODUCT_ABt:
    transA = false;
    transB = true;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Unsupported product type %s", MatProductTypes[product->type]);
  }

  A = product->A;
  B = product->B;
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  PetscCall(MatSeqAIJKokkosSyncDevice(B));
  akok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
  bkok = static_cast<Mat_SeqAIJKokkos *>(B->spptr);
  ckok = static_cast<Mat_SeqAIJKokkos *>(C->spptr);

  PetscCheck(ckok, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Device data structure spptr is empty");

  csrmatA = akok->csrmat;
  csrmatB = bkok->csrmat;

  /* TODO: Once KK spgemm implements transpose, we can get rid of the explicit transpose here */
  if (transA) {
    PetscCall(MatSeqAIJKokkosGenerateTranspose_Private(A, &csrmatA));
    transA = false;
  }

  if (transB) {
    PetscCall(MatSeqAIJKokkosGenerateTranspose_Private(B, &csrmatB));
    transB = false;
  }
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCXX(KokkosSparse::spgemm_numeric(pdata->kh, csrmatA, transA, csrmatB, transB, ckok->csrmat));
#if PETSC_PKG_KOKKOS_KERNELS_VERSION_LT(4, 0, 0)
  auto spgemmHandle = pdata->kh.get_spgemm_handle();
  if (spgemmHandle->get_sort_option() != 1) PetscCallCXX(sort_crs_matrix(ckok->csrmat)); /* without sort, mat_tests-ex62_14_seqaijkokkos fails */
#endif

  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(MatSeqAIJKokkosModifyDevice(C));
  /* shorter version of MatAssemblyEnd_SeqAIJ */
  c = (Mat_SeqAIJ *)C->data;
  PetscCall(PetscInfo(C, "Matrix size: %" PetscInt_FMT " X %" PetscInt_FMT "; storage space: 0 unneeded,%" PetscInt_FMT " used\n", C->rmap->n, C->cmap->n, c->nz));
  PetscCall(PetscInfo(C, "Number of mallocs during MatSetValues() is 0\n"));
  PetscCall(PetscInfo(C, "Maximum nonzeros in any row is %" PetscInt_FMT "\n", c->rmax));
  c->reallocs         = 0;
  C->info.mallocs     = 0;
  C->info.nz_unneeded = 0;
  C->assembled = C->was_assembled = PETSC_TRUE;
  C->num_ass++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_SeqAIJKokkos_SeqAIJKokkos(Mat C)
{
  Mat_Product                 *product = C->product;
  MatProductType               ptype;
  Mat                          A, B;
  bool                         transA, transB;
  Mat_SeqAIJKokkos            *akok, *bkok, *ckok;
  MatProductData_SeqAIJKokkos *pdata;
  MPI_Comm                     comm;
  KokkosCsrMatrix              csrmatA, csrmatB, csrmatC;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCall(PetscObjectGetComm((PetscObject)C, &comm));
  PetscCheck(!product->data, comm, PETSC_ERR_PLIB, "Product data not empty");
  A = product->A;
  B = product->B;
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  PetscCall(MatSeqAIJKokkosSyncDevice(B));
  akok    = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
  bkok    = static_cast<Mat_SeqAIJKokkos *>(B->spptr);
  csrmatA = akok->csrmat;
  csrmatB = bkok->csrmat;

  ptype = product->type;
  // Take advantage of the symmetry if true
  if (A->symmetric == PETSC_BOOL3_TRUE && ptype == MATPRODUCT_AtB) {
    ptype                                          = MATPRODUCT_AB;
    product->symbolic_used_the_fact_A_is_symmetric = PETSC_TRUE;
  }
  if (B->symmetric == PETSC_BOOL3_TRUE && ptype == MATPRODUCT_ABt) {
    ptype                                          = MATPRODUCT_AB;
    product->symbolic_used_the_fact_B_is_symmetric = PETSC_TRUE;
  }

  switch (ptype) {
  case MATPRODUCT_AB:
    transA = false;
    transB = false;
    PetscCall(MatSetBlockSizesFromMats(C, A, B));
    break;
  case MATPRODUCT_AtB:
    transA = true;
    transB = false;
    if (A->cmap->bs > 0) PetscCall(PetscLayoutSetBlockSize(C->rmap, A->cmap->bs));
    if (B->cmap->bs > 0) PetscCall(PetscLayoutSetBlockSize(C->cmap, B->cmap->bs));
    break;
  case MATPRODUCT_ABt:
    transA = false;
    transB = true;
    if (A->rmap->bs > 0) PetscCall(PetscLayoutSetBlockSize(C->rmap, A->rmap->bs));
    if (B->rmap->bs > 0) PetscCall(PetscLayoutSetBlockSize(C->cmap, B->rmap->bs));
    break;
  default:
    SETERRQ(comm, PETSC_ERR_PLIB, "Unsupported product type %s", MatProductTypes[product->type]);
  }
  PetscCallCXX(product->data = pdata = new MatProductData_SeqAIJKokkos());
  pdata->reusesym = product->api_user;

  /* TODO: add command line options to select spgemm algorithms */
  auto spgemm_alg = KokkosSparse::SPGEMMAlgorithm::SPGEMM_DEFAULT; /* default alg is TPL if enabled, otherwise KK */

  /* CUDA-10.2's spgemm has bugs. We prefer the SpGEMMreuse APIs introduced in cuda-11.4 */
#if defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE)
  #if PETSC_PKG_CUDA_VERSION_LT(11, 4, 0)
  spgemm_alg = KokkosSparse::SPGEMMAlgorithm::SPGEMM_KK;
  #endif
#endif
  PetscCallCXX(pdata->kh.create_spgemm_handle(spgemm_alg));

  PetscCall(PetscLogGpuTimeBegin());
  /* TODO: Get rid of the explicit transpose once KK-spgemm implements the transpose option */
  if (transA) {
    PetscCall(MatSeqAIJKokkosGenerateTranspose_Private(A, &csrmatA));
    transA = false;
  }

  if (transB) {
    PetscCall(MatSeqAIJKokkosGenerateTranspose_Private(B, &csrmatB));
    transB = false;
  }

  PetscCallCXX(KokkosSparse::spgemm_symbolic(pdata->kh, csrmatA, transA, csrmatB, transB, csrmatC));
  /* spgemm_symbolic() only populates C's rowmap, but not C's column indices.
    So we have to do a fake spgemm_numeric() here to get csrmatC.j_d setup, before
    calling new Mat_SeqAIJKokkos().
    TODO: Remove the fake spgemm_numeric() after KK fixed this problem.
  */
  PetscCallCXX(KokkosSparse::spgemm_numeric(pdata->kh, csrmatA, transA, csrmatB, transB, csrmatC));
#if PETSC_PKG_KOKKOS_KERNELS_VERSION_LT(4, 0, 0)
  /* Query if KK outputs a sorted matrix. If not, we need to sort it */
  auto spgemmHandle = pdata->kh.get_spgemm_handle();
  if (spgemmHandle->get_sort_option() != 1) PetscCallCXX(sort_crs_matrix(csrmatC)); /* sort_option defaults to -1 in KK!*/
#endif
  PetscCall(PetscLogGpuTimeEnd());

  PetscCallCXX(ckok = new Mat_SeqAIJKokkos(csrmatC));
  PetscCall(MatSetSeqAIJKokkosWithCSRMatrix(C, ckok));
  C->product->destroy = MatProductDataDestroy_SeqAIJKokkos;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* handles sparse matrix matrix ops */
static PetscErrorCode MatProductSetFromOptions_SeqAIJKokkos(Mat mat)
{
  Mat_Product *product = mat->product;
  PetscBool    Biskok = PETSC_FALSE, Ciskok = PETSC_TRUE;

  PetscFunctionBegin;
  MatCheckProduct(mat, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)product->B, MATSEQAIJKOKKOS, &Biskok));
  if (product->type == MATPRODUCT_ABC) PetscCall(PetscObjectTypeCompare((PetscObject)product->C, MATSEQAIJKOKKOS, &Ciskok));
  if (Biskok && Ciskok) {
    switch (product->type) {
    case MATPRODUCT_AB:
    case MATPRODUCT_AtB:
    case MATPRODUCT_ABt:
      mat->ops->productsymbolic = MatProductSymbolic_SeqAIJKokkos_SeqAIJKokkos;
      break;
    case MATPRODUCT_PtAP:
    case MATPRODUCT_RARt:
    case MATPRODUCT_ABC:
      mat->ops->productsymbolic = MatProductSymbolic_ABC_Basic;
      break;
    default:
      break;
    }
  } else { /* fallback for AIJ */
    PetscCall(MatProductSetFromOptions_SeqAIJ(mat));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatScale_SeqAIJKokkos(Mat A, PetscScalar a)
{
  Mat_SeqAIJKokkos *aijkok;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
  KokkosBlas::scal(PetscGetKokkosExecutionSpace(), aijkok->a_dual.view_device(), a, aijkok->a_dual.view_device());
  PetscCall(MatSeqAIJKokkosModifyDevice(A));
  PetscCall(PetscLogGpuFlops(aijkok->a_dual.extent(0)));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// add a to A's diagonal (if A is square) or main diagonal (if A is rectangular)
static PetscErrorCode MatShift_SeqAIJKokkos(Mat A, PetscScalar a)
{
  Mat_SeqAIJ *aijseq = static_cast<Mat_SeqAIJ *>(A->data);

  PetscFunctionBegin;
  if (A->assembled && aijseq->diagonaldense) { // no missing diagonals
    PetscInt n = PetscMin(A->rmap->n, A->cmap->n);

    PetscCall(PetscLogGpuTimeBegin());
    PetscCall(MatSeqAIJKokkosSyncDevice(A));
    const auto  aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
    const auto &Aa     = aijkok->a_dual.view_device();
    const auto &Adiag  = aijkok->diag_dual.view_device();
    PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, n), KOKKOS_LAMBDA(const PetscInt i) { Aa(Adiag(i)) += a; }));
    PetscCall(MatSeqAIJKokkosModifyDevice(A));
    PetscCall(PetscLogGpuFlops(n));
    PetscCall(PetscLogGpuTimeEnd());
  } else { // need reassembly, very slow!
    PetscCall(MatShift_Basic(A, a));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDiagonalSet_SeqAIJKokkos(Mat Y, Vec D, InsertMode is)
{
  Mat_SeqAIJ *aijseq = static_cast<Mat_SeqAIJ *>(Y->data);

  PetscFunctionBegin;
  if (Y->assembled && aijseq->diagonaldense) { // no missing diagonals
    ConstPetscScalarKokkosView dv;
    PetscInt                   n, nv;

    PetscCall(PetscLogGpuTimeBegin());
    PetscCall(MatSeqAIJKokkosSyncDevice(Y));
    PetscCall(VecGetKokkosView(D, &dv));
    PetscCall(VecGetLocalSize(D, &nv));
    n = PetscMin(Y->rmap->n, Y->cmap->n);
    PetscCheck(n == nv, PetscObjectComm((PetscObject)Y), PETSC_ERR_ARG_SIZ, "Matrix size and vector size do not match");

    const auto  aijkok = static_cast<Mat_SeqAIJKokkos *>(Y->spptr);
    const auto &Aa     = aijkok->a_dual.view_device();
    const auto &Adiag  = aijkok->diag_dual.view_device();
    PetscCallCXX(Kokkos::parallel_for(
      Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, n), KOKKOS_LAMBDA(const PetscInt i) {
        if (is == INSERT_VALUES) Aa(Adiag(i)) = dv(i);
        else Aa(Adiag(i)) += dv(i);
      }));
    PetscCall(VecRestoreKokkosView(D, &dv));
    PetscCall(MatSeqAIJKokkosModifyDevice(Y));
    PetscCall(PetscLogGpuFlops(n));
    PetscCall(PetscLogGpuTimeEnd());
  } else { // need reassembly, very slow!
    PetscCall(MatDiagonalSet_Default(Y, D, is));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDiagonalScale_SeqAIJKokkos(Mat A, Vec ll, Vec rr)
{
  Mat_SeqAIJ                *aijseq = static_cast<Mat_SeqAIJ *>(A->data);
  PetscInt                   m = A->rmap->n, n = A->cmap->n, nz = aijseq->nz;
  ConstPetscScalarKokkosView lv, rv;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  const auto  aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
  const auto &Aa     = aijkok->a_dual.view_device();
  const auto &Ai     = aijkok->i_dual.view_device();
  const auto &Aj     = aijkok->j_dual.view_device();
  if (ll) {
    PetscCall(VecGetLocalSize(ll, &m));
    PetscCheck(m == A->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Left scaling vector wrong length");
    PetscCall(VecGetKokkosView(ll, &lv));
    PetscCallCXX(Kokkos::parallel_for( // for each row
      Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), m, Kokkos::AUTO()), KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {
        PetscInt i   = t.league_rank(); // row i
        PetscInt len = Ai(i + 1) - Ai(i);
        // scale entries on the row
        Kokkos::parallel_for(Kokkos::TeamThreadRange(t, len), [&](PetscInt j) { Aa(Ai(i) + j) *= lv(i); });
      }));
    PetscCall(VecRestoreKokkosView(ll, &lv));
    PetscCall(PetscLogGpuFlops(nz));
  }
  if (rr) {
    PetscCall(VecGetLocalSize(rr, &n));
    PetscCheck(n == A->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Right scaling vector wrong length");
    PetscCall(VecGetKokkosView(rr, &rv));
    PetscCallCXX(Kokkos::parallel_for( // for each nonzero
      Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, nz), KOKKOS_LAMBDA(const PetscInt k) { Aa(k) *= rv(Aj(k)); }));
    PetscCall(VecRestoreKokkosView(rr, &lv));
    PetscCall(PetscLogGpuFlops(nz));
  }
  PetscCall(MatSeqAIJKokkosModifyDevice(A));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroEntries_SeqAIJKokkos(Mat A)
{
  Mat_SeqAIJKokkos *aijkok;

  PetscFunctionBegin;
  aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
  if (aijkok) { /* Only zero the device if data is already there */
    KokkosBlas::fill(PetscGetKokkosExecutionSpace(), aijkok->a_dual.view_device(), 0.0);
    PetscCall(MatSeqAIJKokkosModifyDevice(A));
  } else { /* Might be preallocated but not assembled */
    PetscCall(MatZeroEntries_SeqAIJ(A));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetDiagonal_SeqAIJKokkos(Mat A, Vec x)
{
  Mat_SeqAIJKokkos     *aijkok;
  PetscInt              n;
  PetscScalarKokkosView xv;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(x, &n));
  PetscCheck(n == A->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Nonconforming matrix and vector");
  PetscCheck(A->factortype == MAT_FACTOR_NONE, PETSC_COMM_SELF, PETSC_ERR_SUP, "MatGetDiagonal_SeqAIJKokkos not supported on factored matrices");

  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);

  const auto &Aa    = aijkok->a_dual.view_device();
  const auto &Ai    = aijkok->i_dual.view_device();
  const auto &Adiag = aijkok->diag_dual.view_device();

  PetscCall(VecGetKokkosViewWrite(x, &xv));
  Kokkos::parallel_for(
    Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, n), KOKKOS_LAMBDA(const PetscInt i) {
      if (Adiag(i) < Ai(i + 1)) xv(i) = Aa(Adiag(i));
      else xv(i) = 0;
    });
  PetscCall(VecRestoreKokkosViewWrite(x, &xv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Get a Kokkos View from a mat of type MatSeqAIJKokkos */
PetscErrorCode MatSeqAIJGetKokkosView(Mat A, ConstMatScalarKokkosView *kv)
{
  Mat_SeqAIJKokkos *aijkok;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(kv, 2);
  PetscCheckTypeName(A, MATSEQAIJKOKKOS);
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
  *kv    = aijkok->a_dual.view_device();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqAIJRestoreKokkosView(Mat A, ConstMatScalarKokkosView *kv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(kv, 2);
  PetscCheckTypeName(A, MATSEQAIJKOKKOS);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqAIJGetKokkosView(Mat A, MatScalarKokkosView *kv)
{
  Mat_SeqAIJKokkos *aijkok;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(kv, 2);
  PetscCheckTypeName(A, MATSEQAIJKOKKOS);
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
  *kv    = aijkok->a_dual.view_device();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqAIJRestoreKokkosView(Mat A, MatScalarKokkosView *kv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(kv, 2);
  PetscCheckTypeName(A, MATSEQAIJKOKKOS);
  PetscCall(MatSeqAIJKokkosModifyDevice(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqAIJGetKokkosViewWrite(Mat A, MatScalarKokkosView *kv)
{
  Mat_SeqAIJKokkos *aijkok;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(kv, 2);
  PetscCheckTypeName(A, MATSEQAIJKOKKOS);
  aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
  *kv    = aijkok->a_dual.view_device();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSeqAIJRestoreKokkosViewWrite(Mat A, MatScalarKokkosView *kv)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(kv, 2);
  PetscCheckTypeName(A, MATSEQAIJKOKKOS);
  PetscCall(MatSeqAIJKokkosModifyDevice(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCreateSeqAIJKokkosWithKokkosViews(MPI_Comm comm, PetscInt m, PetscInt n, Kokkos::View<PetscInt *> &i_d, Kokkos::View<PetscInt *> &j_d, Kokkos::View<PetscScalar *> &a_d, Mat *A)
{
  Mat_SeqAIJKokkos *akok;

  PetscFunctionBegin;
  PetscCallCXX(akok = new Mat_SeqAIJKokkos(m, n, j_d.extent(0), i_d, j_d, a_d));
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSeqAIJKokkosWithCSRMatrix(*A, akok));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Computes Y += alpha X */
static PetscErrorCode MatAXPY_SeqAIJKokkos(Mat Y, PetscScalar alpha, Mat X, MatStructure pattern)
{
  Mat_SeqAIJ              *x = (Mat_SeqAIJ *)X->data, *y = (Mat_SeqAIJ *)Y->data;
  Mat_SeqAIJKokkos        *xkok, *ykok, *zkok;
  ConstMatScalarKokkosView Xa;
  MatScalarKokkosView      Ya;
  auto                     exec = PetscGetKokkosExecutionSpace();

  PetscFunctionBegin;
  PetscCheckTypeName(Y, MATSEQAIJKOKKOS);
  PetscCheckTypeName(X, MATSEQAIJKOKKOS);
  PetscCall(MatSeqAIJKokkosSyncDevice(Y));
  PetscCall(MatSeqAIJKokkosSyncDevice(X));
  PetscCall(PetscLogGpuTimeBegin());

  if (pattern != SAME_NONZERO_PATTERN && x->nz == y->nz) {
    PetscBool e;
    PetscCall(PetscArraycmp(x->i, y->i, Y->rmap->n + 1, &e));
    if (e) {
      PetscCall(PetscArraycmp(x->j, y->j, y->nz, &e));
      if (e) pattern = SAME_NONZERO_PATTERN;
    }
  }

  /* cusparseDcsrgeam2() computes C = alpha A + beta B. If one knew sparsity pattern of C, one can skip
    cusparseScsrgeam2_bufferSizeExt() / cusparseXcsrgeam2Nnz(), and directly call cusparseScsrgeam2().
    If X is SUBSET_NONZERO_PATTERN of Y, we could take advantage of this cusparse feature. However,
    KokkosSparse::spadd(alpha,A,beta,B,C) has symbolic and numeric phases, MatAXPY does not.
  */
  ykok = static_cast<Mat_SeqAIJKokkos *>(Y->spptr);
  xkok = static_cast<Mat_SeqAIJKokkos *>(X->spptr);
  Xa   = xkok->a_dual.view_device();
  Ya   = ykok->a_dual.view_device();

  if (pattern == SAME_NONZERO_PATTERN) {
    KokkosBlas::axpy(exec, alpha, Xa, Ya);
    PetscCall(MatSeqAIJKokkosModifyDevice(Y));
  } else if (pattern == SUBSET_NONZERO_PATTERN) {
    MatRowMapKokkosView Xi = xkok->i_dual.view_device(), Yi = ykok->i_dual.view_device();
    MatColIdxKokkosView Xj = xkok->j_dual.view_device(), Yj = ykok->j_dual.view_device();

    Kokkos::parallel_for(
      Kokkos::TeamPolicy<>(exec, Y->rmap->n, 1), KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {
        PetscInt i = t.league_rank(); // row i
        Kokkos::single(Kokkos::PerTeam(t), [=]() {
          // Only one thread works in a team
          PetscInt p, q = Yi(i);
          for (p = Xi(i); p < Xi(i + 1); p++) {          // For each nonzero on row i of X,
            while (Xj(p) != Yj(q) && q < Yi(i + 1)) q++; // find the matching nonzero on row i of Y.
            if (Xj(p) == Yj(q)) {                        // Found it
              Ya(q) += alpha * Xa(p);
              q++;
            } else {
            // If not found, it indicates the input is wrong (X is not a SUBSET_NONZERO_PATTERN of Y).
            // Just insert a NaN at the beginning of row i if it is not empty, to make the result wrong.
#if PETSC_PKG_KOKKOS_VERSION_GE(3, 7, 0)
              if (Yi(i) != Yi(i + 1)) Ya(Yi(i)) = Kokkos::ArithTraits<PetscScalar>::nan();
#else
              if (Yi(i) != Yi(i + 1)) Ya(Yi(i)) = Kokkos::Experimental::nan("1");
#endif
            }
          }
        });
      });
    PetscCall(MatSeqAIJKokkosModifyDevice(Y));
  } else { // different nonzero patterns
    Mat             Z;
    KokkosCsrMatrix zcsr;
    KernelHandle    kh;
    kh.create_spadd_handle(true); // X, Y are sorted
    KokkosSparse::spadd_symbolic(&kh, xkok->csrmat, ykok->csrmat, zcsr);
    KokkosSparse::spadd_numeric(&kh, alpha, xkok->csrmat, (PetscScalar)1.0, ykok->csrmat, zcsr);
    zkok = new Mat_SeqAIJKokkos(zcsr);
    PetscCall(MatCreateSeqAIJKokkosWithCSRMatrix(PETSC_COMM_SELF, zkok, &Z));
    PetscCall(MatHeaderReplace(Y, &Z));
    kh.destroy_spadd_handle();
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(xkok->a_dual.extent(0) * 2)); // Because we scaled X and then added it to Y
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct MatCOOStruct_SeqAIJKokkos {
  PetscCount           n;
  PetscCount           Atot;
  PetscInt             nz;
  PetscCountKokkosView jmap;
  PetscCountKokkosView perm;

  MatCOOStruct_SeqAIJKokkos(const MatCOOStruct_SeqAIJ *coo_h)
  {
    nz   = coo_h->nz;
    n    = coo_h->n;
    Atot = coo_h->Atot;
    jmap = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(coo_h->jmap, nz + 1));
    perm = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), PetscCountKokkosViewHost(coo_h->perm, Atot));
  }
};

static PetscErrorCode MatCOOStructDestroy_SeqAIJKokkos(void **data)
{
  PetscFunctionBegin;
  PetscCallCXX(delete static_cast<MatCOOStruct_SeqAIJKokkos *>(*data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetPreallocationCOO_SeqAIJKokkos(Mat mat, PetscCount coo_n, PetscInt coo_i[], PetscInt coo_j[])
{
  Mat_SeqAIJKokkos          *akok;
  Mat_SeqAIJ                *aseq;
  PetscContainer             container_h;
  MatCOOStruct_SeqAIJ       *coo_h;
  MatCOOStruct_SeqAIJKokkos *coo_d;

  PetscFunctionBegin;
  PetscCall(MatSetPreallocationCOO_SeqAIJ(mat, coo_n, coo_i, coo_j));
  aseq = static_cast<Mat_SeqAIJ *>(mat->data);
  akok = static_cast<Mat_SeqAIJKokkos *>(mat->spptr);
  delete akok;
  mat->spptr = akok = new Mat_SeqAIJKokkos(mat->rmap->n, mat->cmap->n, aseq, mat->nonzerostate + 1, PETSC_FALSE);
  PetscCall(MatZeroEntries_SeqAIJKokkos(mat));

  // Copy the COO struct to device
  PetscCall(PetscObjectQuery((PetscObject)mat, "__PETSc_MatCOOStruct_Host", (PetscObject *)&container_h));
  PetscCall(PetscContainerGetPointer(container_h, (void **)&coo_h));
  PetscCallCXX(coo_d = new MatCOOStruct_SeqAIJKokkos(coo_h));

  // Put the COO struct in a container and then attach that to the matrix
  PetscCall(PetscObjectContainerCompose((PetscObject)mat, "__PETSc_MatCOOStruct_Device", coo_d, MatCOOStructDestroy_SeqAIJKokkos));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetValuesCOO_SeqAIJKokkos(Mat A, const PetscScalar v[], InsertMode imode)
{
  MatScalarKokkosView        Aa;
  ConstMatScalarKokkosView   kv;
  PetscMemType               memtype;
  PetscContainer             container;
  MatCOOStruct_SeqAIJKokkos *coo;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)A, "__PETSc_MatCOOStruct_Device", (PetscObject *)&container));
  PetscCall(PetscContainerGetPointer(container, (void **)&coo));

  const auto &n    = coo->n;
  const auto &Annz = coo->nz;
  const auto &jmap = coo->jmap;
  const auto &perm = coo->perm;

  PetscCall(PetscGetMemType(v, &memtype));
  if (PetscMemTypeHost(memtype)) { /* If user gave v[] in host, we might need to copy it to device if any */
    kv = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), ConstMatScalarKokkosViewHost(v, n));
  } else {
    kv = ConstMatScalarKokkosView(v, n); /* Directly use v[]'s memory */
  }

  if (imode == INSERT_VALUES) PetscCall(MatSeqAIJGetKokkosViewWrite(A, &Aa)); /* write matrix values */
  else PetscCall(MatSeqAIJGetKokkosView(A, &Aa));                             /* read & write matrix values */

  PetscCall(PetscLogGpuTimeBegin());
  Kokkos::parallel_for(
    Kokkos::RangePolicy<>(PetscGetKokkosExecutionSpace(), 0, Annz), KOKKOS_LAMBDA(const PetscCount i) {
      PetscScalar sum = 0.0;
      for (PetscCount k = jmap(i); k < jmap(i + 1); k++) sum += kv(perm(k));
      Aa(i) = (imode == INSERT_VALUES ? 0.0 : Aa(i)) + sum;
    });
  PetscCall(PetscLogGpuTimeEnd());

  if (imode == INSERT_VALUES) PetscCall(MatSeqAIJRestoreKokkosViewWrite(A, &Aa));
  else PetscCall(MatSeqAIJRestoreKokkosView(A, &Aa));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetOps_SeqAIJKokkos(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;

  PetscFunctionBegin;
  A->offloadmask = PETSC_OFFLOAD_KOKKOS; /* We do not really use this flag */
  A->boundtocpu  = PETSC_FALSE;

  A->ops->assemblyend               = MatAssemblyEnd_SeqAIJKokkos;
  A->ops->destroy                   = MatDestroy_SeqAIJKokkos;
  A->ops->duplicate                 = MatDuplicate_SeqAIJKokkos;
  A->ops->axpy                      = MatAXPY_SeqAIJKokkos;
  A->ops->scale                     = MatScale_SeqAIJKokkos;
  A->ops->zeroentries               = MatZeroEntries_SeqAIJKokkos;
  A->ops->productsetfromoptions     = MatProductSetFromOptions_SeqAIJKokkos;
  A->ops->mult                      = MatMult_SeqAIJKokkos;
  A->ops->multadd                   = MatMultAdd_SeqAIJKokkos;
  A->ops->multtranspose             = MatMultTranspose_SeqAIJKokkos;
  A->ops->multtransposeadd          = MatMultTransposeAdd_SeqAIJKokkos;
  A->ops->multhermitiantranspose    = MatMultHermitianTranspose_SeqAIJKokkos;
  A->ops->multhermitiantransposeadd = MatMultHermitianTransposeAdd_SeqAIJKokkos;
  A->ops->productnumeric            = MatProductNumeric_SeqAIJKokkos_SeqAIJKokkos;
  A->ops->transpose                 = MatTranspose_SeqAIJKokkos;
  A->ops->setoption                 = MatSetOption_SeqAIJKokkos;
  A->ops->getdiagonal               = MatGetDiagonal_SeqAIJKokkos;
  A->ops->shift                     = MatShift_SeqAIJKokkos;
  A->ops->diagonalset               = MatDiagonalSet_SeqAIJKokkos;
  A->ops->diagonalscale             = MatDiagonalScale_SeqAIJKokkos;
  A->ops->getcurrentmemtype         = MatGetCurrentMemType_SeqAIJKokkos;
  a->ops->getarray                  = MatSeqAIJGetArray_SeqAIJKokkos;
  a->ops->restorearray              = MatSeqAIJRestoreArray_SeqAIJKokkos;
  a->ops->getarrayread              = MatSeqAIJGetArrayRead_SeqAIJKokkos;
  a->ops->restorearrayread          = MatSeqAIJRestoreArrayRead_SeqAIJKokkos;
  a->ops->getarraywrite             = MatSeqAIJGetArrayWrite_SeqAIJKokkos;
  a->ops->restorearraywrite         = MatSeqAIJRestoreArrayWrite_SeqAIJKokkos;
  a->ops->getcsrandmemtype          = MatSeqAIJGetCSRAndMemType_SeqAIJKokkos;

  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetPreallocationCOO_C", MatSetPreallocationCOO_SeqAIJKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetValuesCOO_C", MatSetValuesCOO_SeqAIJKokkos));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqaijkokkos_hypre_C", MatConvert_AIJ_HYPRE));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Extract the (prescribled) diagonal blocks of the matrix and then invert them

  Input Parameters:
+  A       - the MATSEQAIJKOKKOS matrix
.  bs      - block sizes in 'csr' format, i.e., the i-th block has size bs(i+1) - bs(i)
.  bs2     - square of block sizes in 'csr' format, i.e., the i-th block should be stored at offset bs2(i) in diagVal[]
.  blkMap  - map row ids to block ids, i.e., row i belongs to the block blkMap(i)
-  work    - a pre-allocated work buffer (as big as diagVal) for use by this routine

  Output Parameter:
.  diagVal - the (pre-allocated) buffer to store the inverted blocks (each block is stored in column-major order)
*/
PETSC_INTERN PetscErrorCode MatInvertVariableBlockDiagonal_SeqAIJKokkos(Mat A, const PetscIntKokkosView &bs, const PetscIntKokkosView &bs2, const PetscIntKokkosView &blkMap, PetscScalarKokkosView &work, PetscScalarKokkosView &diagVal)
{
  Mat_SeqAIJKokkos *akok    = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
  PetscInt          nblocks = bs.extent(0) - 1;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJKokkosSyncDevice(A)); // Since we'll access A's value on device

  // Pull out the diagonal blocks of the matrix and then invert the blocks
  auto Aa    = akok->a_dual.view_device();
  auto Ai    = akok->i_dual.view_device();
  auto Aj    = akok->j_dual.view_device();
  auto Adiag = akok->diag_dual.view_device();
  // TODO: how to tune the team size?
#if defined(KOKKOS_ENABLE_UNIFIED_MEMORY)
  auto ts = Kokkos::AUTO();
#else
  auto ts = 16; // improved performance 30% over Kokkos::AUTO() with CUDA, but failed with "Kokkos::abort: Requested Team Size is too large!" on CPUs
#endif
  PetscCallCXX(Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(PetscGetKokkosExecutionSpace(), nblocks, ts), KOKKOS_LAMBDA(const KokkosTeamMemberType &teamMember) {
      const PetscInt bid    = teamMember.league_rank();                                                   // block id
      const PetscInt rstart = bs(bid);                                                                    // this block starts from this row
      const PetscInt m      = bs(bid + 1) - bs(bid);                                                      // size of this block
      const auto    &B      = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft>(&diagVal(bs2(bid)), m, m); // column-major order
      const auto    &W      = PetscScalarKokkosView(&work(bs2(bid)), m * m);

      Kokkos::parallel_for(Kokkos::TeamThreadRange(teamMember, m), [=](const PetscInt &r) { // r-th row in B
        PetscInt i = rstart + r;                                                            // i-th row in A

        if (Ai(i) <= Adiag(i) && Adiag(i) < Ai(i + 1)) { // if the diagonal exists (common case)
          PetscInt first = Adiag(i) - r;                 // we start to check nonzeros from here along this row

          for (PetscInt c = 0; c < m; c++) {                   // walk n steps to see what column indices we will meet
            if (first + c < Ai(i) || first + c >= Ai(i + 1)) { // this entry (first+c) is out of range of this row, in other words, its value is zero
              B(r, c) = 0.0;
            } else if (Aj(first + c) == rstart + c) { // this entry is right on the (rstart+c) column
              B(r, c) = Aa(first + c);
            } else { // this entry does not show up in the CSR
              B(r, c) = 0.0;
            }
          }
        } else { // rare case that the diagonal does not exist
          const PetscInt begin = Ai(i);
          const PetscInt end   = Ai(i + 1);
          for (PetscInt c = 0; c < m; c++) B(r, c) = 0.0;
          for (PetscInt j = begin; j < end; j++) { // scan the whole row; could use binary search but this is a rare case so we did not.
            if (rstart <= Aj(j) && Aj(j) < rstart + m) B(r, Aj(j) - rstart) = Aa(j);
            else if (Aj(j) >= rstart + m) break;
          }
        }
      });

      // LU-decompose B (w/o pivoting) and then invert B
      KokkosBatched::TeamLU<KokkosTeamMemberType, KokkosBatched::Algo::LU::Unblocked>::invoke(teamMember, B, 0.0);
      KokkosBatched::TeamInverseLU<KokkosTeamMemberType, KokkosBatched::Algo::InverseLU::Unblocked>::invoke(teamMember, B, W);
    }));
  // PetscLogGpuFlops() is done in the caller PCSetUp_VPBJacobi_Kokkos as we don't want to compute the flops in kernels
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSetSeqAIJKokkosWithCSRMatrix(Mat A, Mat_SeqAIJKokkos *akok)
{
  Mat_SeqAIJ *aseq;
  PetscInt    i, m, n;
  auto        exec = PetscGetKokkosExecutionSpace();

  PetscFunctionBegin;
  PetscCheck(!A->spptr, PETSC_COMM_SELF, PETSC_ERR_PLIB, "A->spptr is supposed to be empty");

  m = akok->nrows();
  n = akok->ncols();
  PetscCall(MatSetSizes(A, m, n, m, n));
  PetscCall(MatSetType(A, MATSEQAIJKOKKOS));

  /* Set up data structures of A as a MATSEQAIJ */
  PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(A, MAT_SKIP_ALLOCATION, NULL));
  aseq = (Mat_SeqAIJ *)A->data;

  PetscCall(KokkosDualViewSyncHost(akok->i_dual, exec)); /* We always need sync'ed i, j on host */
  PetscCall(KokkosDualViewSyncHost(akok->j_dual, exec));

  aseq->i       = akok->i_host_data();
  aseq->j       = akok->j_host_data();
  aseq->a       = akok->a_host_data();
  aseq->nonew   = -1; /*this indicates that inserting a new value in the matrix that generates a new nonzero is an error*/
  aseq->free_a  = PETSC_FALSE;
  aseq->free_ij = PETSC_FALSE;
  aseq->nz      = akok->nnz();
  aseq->maxnz   = aseq->nz;

  PetscCall(PetscMalloc1(m, &aseq->imax));
  PetscCall(PetscMalloc1(m, &aseq->ilen));
  for (i = 0; i < m; i++) aseq->ilen[i] = aseq->imax[i] = aseq->i[i + 1] - aseq->i[i];

  /* It is critical to set the nonzerostate, as we use it to check if sparsity pattern (hence data) has changed on host in MatAssemblyEnd */
  akok->nonzerostate = A->nonzerostate;
  A->spptr           = akok; /* Set A->spptr before MatAssembly so that A->spptr won't be allocated again there */
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSeqAIJKokkosGetKokkosCsrMatrix(Mat A, KokkosCsrMatrix *csr)
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJKokkosSyncDevice(A));
  *csr = static_cast<Mat_SeqAIJKokkos *>(A->spptr)->csrmat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatCreateSeqAIJKokkosWithKokkosCsrMatrix(MPI_Comm comm, KokkosCsrMatrix csr, Mat *A)
{
  Mat_SeqAIJKokkos *akok;

  PetscFunctionBegin;
  PetscCallCXX(akok = new Mat_SeqAIJKokkos(csr));
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSeqAIJKokkosWithCSRMatrix(*A, akok));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Crete a SEQAIJKOKKOS matrix with a Mat_SeqAIJKokkos data structure

   Note we have names like MatSeqAIJSetPreallocationCSR, so I use capitalized CSR
 */
PETSC_INTERN PetscErrorCode MatCreateSeqAIJKokkosWithCSRMatrix(MPI_Comm comm, Mat_SeqAIJKokkos *akok, Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSeqAIJKokkosWithCSRMatrix(*A, akok));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatCreateSeqAIJKokkos - Creates a sparse matrix in `MATSEQAIJKOKKOS` (compressed row) format
  (the default parallel PETSc format). This matrix will ultimately be handled by
  Kokkos for calculations.

  Collective

  Input Parameters:
+ comm - MPI communicator, set to `PETSC_COMM_SELF`
. m    - number of rows
. n    - number of columns
. nz   - number of nonzeros per row (same for all rows), ignored if `nnz` is provided
- nnz  - array containing the number of nonzeros in the various rows (possibly different for each row) or `NULL`

  Output Parameter:
. A - the matrix

  Level: intermediate

  Notes:
  It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()`,
  MatXXXXSetPreallocation() paradgm instead of this routine directly.
  [MatXXXXSetPreallocation() is, for example, `MatSeqAIJSetPreallocation()`]

  The AIJ format, also called
  compressed row storage, is fully compatible with standard Fortran
  storage.  That is, the stored row and column indices can begin at
  either one (as in Fortran) or zero.

  Specify the preallocated storage with either `nz` or `nnz` (not both).
  Set `nz` = `PETSC_DEFAULT` and `nnz` = `NULL` for PETSc to control dynamic memory
  allocation.

.seealso: [](ch_matrices), `Mat`, `MatCreate()`, `MatCreateAIJ()`, `MatSetValues()`, `MatSeqAIJSetColumnIndices()`, `MatCreateSeqAIJWithArrays()`
@*/
PetscErrorCode MatCreateSeqAIJKokkos(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt nz, const PetscInt nnz[], Mat *A)
{
  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, m, n, m, n));
  PetscCall(MatSetType(*A, MATSEQAIJKOKKOS));
  PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(*A, nz, (PetscInt *)nnz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// After matrix numeric factorization, there are still steps to do before triangular solve can be called.
// For example, for transpose solve, we might need to compute the transpose matrices if the solver does not support it (such as KK, while cusparse does).
// In cusparse, one has to call cusparseSpSV_analysis() with updated triangular matrix values before calling cusparseSpSV_solve().
// Simiarily, in KK sptrsv_symbolic() has to be called before sptrsv_solve(). We put these steps in MatSeqAIJKokkos{Transpose}SolveCheck.
static PetscErrorCode MatSeqAIJKokkosSolveCheck(Mat A)
{
  Mat_SeqAIJKokkosTriFactors *factors   = (Mat_SeqAIJKokkosTriFactors *)A->spptr;
  const PetscBool             has_lower = factors->iL_d.extent(0) ? PETSC_TRUE : PETSC_FALSE; // false with Choleksy
  const PetscBool             has_upper = factors->iU_d.extent(0) ? PETSC_TRUE : PETSC_FALSE; // true with LU and Choleksy

  PetscFunctionBegin;
  if (!factors->sptrsv_symbolic_completed) { // If sptrsv_symbolic was not called yet
    if (has_upper) PetscCallCXX(sptrsv_symbolic(&factors->khU, factors->iU_d, factors->jU_d, factors->aU_d));
    if (has_lower) PetscCallCXX(sptrsv_symbolic(&factors->khL, factors->iL_d, factors->jL_d, factors->aL_d));
    factors->sptrsv_symbolic_completed = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqAIJKokkosTransposeSolveCheck(Mat A)
{
  const PetscInt              n         = A->rmap->n;
  Mat_SeqAIJKokkosTriFactors *factors   = (Mat_SeqAIJKokkosTriFactors *)A->spptr;
  const PetscBool             has_lower = factors->iL_d.extent(0) ? PETSC_TRUE : PETSC_FALSE; // false with Choleksy
  const PetscBool             has_upper = factors->iU_d.extent(0) ? PETSC_TRUE : PETSC_FALSE; // true with LU or Choleksy

  PetscFunctionBegin;
  if (!factors->transpose_updated) {
    if (has_upper) {
      if (!factors->iUt_d.extent(0)) {                                 // Allocate Ut on device if not yet
        factors->iUt_d = MatRowMapKokkosView("factors->iUt_d", n + 1); // KK requires this view to be initialized to 0 to call transpose_matrix
        factors->jUt_d = MatColIdxKokkosView(NoInit("factors->jUt_d"), factors->jU_d.extent(0));
        factors->aUt_d = MatScalarKokkosView(NoInit("factors->aUt_d"), factors->aU_d.extent(0));
      }

      if (factors->iU_h.extent(0)) { // If U is on host (factorization was done on host), we also compute the transpose on host
        if (!factors->U) {
          Mat_SeqAIJ *seq;

          PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, n, n, factors->iU_h.data(), factors->jU_h.data(), factors->aU_h.data(), &factors->U));
          PetscCall(MatTranspose(factors->U, MAT_INITIAL_MATRIX, &factors->Ut));

          seq            = static_cast<Mat_SeqAIJ *>(factors->Ut->data);
          factors->iUt_h = MatRowMapKokkosViewHost(seq->i, n + 1);
          factors->jUt_h = MatColIdxKokkosViewHost(seq->j, seq->nz);
          factors->aUt_h = MatScalarKokkosViewHost(seq->a, seq->nz);
        } else {
          PetscCall(MatTranspose(factors->U, MAT_REUSE_MATRIX, &factors->Ut)); // Matrix Ut' data is aliased with {i, j, a}Ut_h
        }
        // Copy Ut from host to device
        PetscCallCXX(Kokkos::deep_copy(factors->iUt_d, factors->iUt_h));
        PetscCallCXX(Kokkos::deep_copy(factors->jUt_d, factors->jUt_h));
        PetscCallCXX(Kokkos::deep_copy(factors->aUt_d, factors->aUt_h));
      } else { // If U was computed on device, we also compute the transpose there
        // TODO: KK transpose_matrix() does not sort column indices, however cusparse requires sorted indices. We have to sort the indices, until KK provides finer control options.
        PetscCallCXX(transpose_matrix<ConstMatRowMapKokkosView, ConstMatColIdxKokkosView, ConstMatScalarKokkosView, MatRowMapKokkosView, MatColIdxKokkosView, MatScalarKokkosView, MatRowMapKokkosView, DefaultExecutionSpace>(n, n, factors->iU_d,
                                                                                                                                                                                                                               factors->jU_d, factors->aU_d,
                                                                                                                                                                                                                               factors->iUt_d, factors->jUt_d,
                                                                                                                                                                                                                               factors->aUt_d));
        PetscCallCXX(sort_crs_matrix<DefaultExecutionSpace, MatRowMapKokkosView, MatColIdxKokkosView, MatScalarKokkosView>(factors->iUt_d, factors->jUt_d, factors->aUt_d));
      }
      PetscCallCXX(sptrsv_symbolic(&factors->khUt, factors->iUt_d, factors->jUt_d, factors->aUt_d));
    }

    // do the same for L with LU
    if (has_lower) {
      if (!factors->iLt_d.extent(0)) {                                 // Allocate Lt on device if not yet
        factors->iLt_d = MatRowMapKokkosView("factors->iLt_d", n + 1); // KK requires this view to be initialized to 0 to call transpose_matrix
        factors->jLt_d = MatColIdxKokkosView(NoInit("factors->jLt_d"), factors->jL_d.extent(0));
        factors->aLt_d = MatScalarKokkosView(NoInit("factors->aLt_d"), factors->aL_d.extent(0));
      }

      if (factors->iL_h.extent(0)) { // If L is on host, we also compute the transpose on host
        if (!factors->L) {
          Mat_SeqAIJ *seq;

          PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, n, n, factors->iL_h.data(), factors->jL_h.data(), factors->aL_h.data(), &factors->L));
          PetscCall(MatTranspose(factors->L, MAT_INITIAL_MATRIX, &factors->Lt));

          seq            = static_cast<Mat_SeqAIJ *>(factors->Lt->data);
          factors->iLt_h = MatRowMapKokkosViewHost(seq->i, n + 1);
          factors->jLt_h = MatColIdxKokkosViewHost(seq->j, seq->nz);
          factors->aLt_h = MatScalarKokkosViewHost(seq->a, seq->nz);
        } else {
          PetscCall(MatTranspose(factors->L, MAT_REUSE_MATRIX, &factors->Lt)); // Matrix Lt' data is aliased with {i, j, a}Lt_h
        }
        // Copy Lt from host to device
        PetscCallCXX(Kokkos::deep_copy(factors->iLt_d, factors->iLt_h));
        PetscCallCXX(Kokkos::deep_copy(factors->jLt_d, factors->jLt_h));
        PetscCallCXX(Kokkos::deep_copy(factors->aLt_d, factors->aLt_h));
      } else { // If L was computed on device, we also compute the transpose there
        // TODO: KK transpose_matrix() does not sort column indices, however cusparse requires sorted indices. We have to sort the indices, until KK provides finer control options.
        PetscCallCXX(transpose_matrix<ConstMatRowMapKokkosView, ConstMatColIdxKokkosView, ConstMatScalarKokkosView, MatRowMapKokkosView, MatColIdxKokkosView, MatScalarKokkosView, MatRowMapKokkosView, DefaultExecutionSpace>(n, n, factors->iL_d,
                                                                                                                                                                                                                               factors->jL_d, factors->aL_d,
                                                                                                                                                                                                                               factors->iLt_d, factors->jLt_d,
                                                                                                                                                                                                                               factors->aLt_d));
        PetscCallCXX(sort_crs_matrix<DefaultExecutionSpace, MatRowMapKokkosView, MatColIdxKokkosView, MatScalarKokkosView>(factors->iLt_d, factors->jLt_d, factors->aLt_d));
      }
      PetscCallCXX(sptrsv_symbolic(&factors->khLt, factors->iLt_d, factors->jLt_d, factors->aLt_d));
    }

    factors->transpose_updated = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Solve Ax = b, with RAR = U^T D U, where R is the row (and col) permutation matrix on A.
// R is represented by rowperm in factors. If R is identity (i.e, no reordering), then rowperm is empty.
static PetscErrorCode MatSolve_SeqAIJKokkos_Cholesky(Mat A, Vec bb, Vec xx)
{
  auto                        exec    = PetscGetKokkosExecutionSpace();
  Mat_SeqAIJKokkosTriFactors *factors = (Mat_SeqAIJKokkosTriFactors *)A->spptr;
  PetscInt                    m       = A->rmap->n;
  PetscScalarKokkosView       D       = factors->D_d;
  PetscScalarKokkosView       X, Y, B; // alias
  ConstPetscScalarKokkosView  b;
  PetscScalarKokkosView       x;
  PetscIntKokkosView         &rowperm  = factors->rowperm;
  PetscBool                   identity = rowperm.extent(0) ? PETSC_FALSE : PETSC_TRUE;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosSolveCheck(A));          // for UX = T
  PetscCall(MatSeqAIJKokkosTransposeSolveCheck(A)); // for U^T Y = B
  PetscCall(VecGetKokkosView(bb, &b));
  PetscCall(VecGetKokkosViewWrite(xx, &x));

  // Solve U^T Y = B
  if (identity) { // Reorder b with the row permutation
    B = PetscScalarKokkosView(const_cast<PetscScalar *>(b.data()), b.extent(0));
    Y = factors->workVector;
  } else {
    B = factors->workVector;
    PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, m), KOKKOS_LAMBDA(const PetscInt i) { B(i) = b(rowperm(i)); }));
    Y = x;
  }
  PetscCallCXX(sptrsv_solve(exec, &factors->khUt, factors->iUt_d, factors->jUt_d, factors->aUt_d, B, Y));

  // Solve diag(D) Y' = Y.
  // Actually just do Y' = Y*D since D is already inverted in MatCholeskyFactorNumeric_SeqAIJ(). It is basically a vector element-wise multiplication.
  PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, m), KOKKOS_LAMBDA(const PetscInt i) { Y(i) = Y(i) * D(i); }));

  // Solve UX = Y
  if (identity) {
    X = x;
  } else {
    X = factors->workVector; // B is not needed anymore
  }
  PetscCallCXX(sptrsv_solve(exec, &factors->khU, factors->iU_d, factors->jU_d, factors->aU_d, Y, X));

  // Reorder X with the inverse column (row) permutation
  if (!identity) PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, m), KOKKOS_LAMBDA(const PetscInt i) { x(rowperm(i)) = X(i); }));

  PetscCall(VecRestoreKokkosView(bb, &b));
  PetscCall(VecRestoreKokkosViewWrite(xx, &x));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Solve Ax = b, with RAC = LU, where R and C are row and col permutation matrices on A respectively.
// R and C are represented by rowperm and colperm in factors.
// If R or C is identity (i.e, no reordering), then rowperm or colperm is empty.
static PetscErrorCode MatSolve_SeqAIJKokkos_LU(Mat A, Vec bb, Vec xx)
{
  auto                        exec    = PetscGetKokkosExecutionSpace();
  Mat_SeqAIJKokkosTriFactors *factors = (Mat_SeqAIJKokkosTriFactors *)A->spptr;
  PetscInt                    m       = A->rmap->n;
  PetscScalarKokkosView       X, Y, B; // alias
  ConstPetscScalarKokkosView  b;
  PetscScalarKokkosView       x;
  PetscIntKokkosView         &rowperm      = factors->rowperm;
  PetscIntKokkosView         &colperm      = factors->colperm;
  PetscBool                   row_identity = rowperm.extent(0) ? PETSC_FALSE : PETSC_TRUE;
  PetscBool                   col_identity = colperm.extent(0) ? PETSC_FALSE : PETSC_TRUE;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosSolveCheck(A));
  PetscCall(VecGetKokkosView(bb, &b));
  PetscCall(VecGetKokkosViewWrite(xx, &x));

  // Solve L Y = B (i.e., L (U C^- x) = R b).  R b indicates applying the row permutation on b.
  if (row_identity) {
    B = PetscScalarKokkosView(const_cast<PetscScalar *>(b.data()), b.extent(0));
    Y = factors->workVector;
  } else {
    B = factors->workVector;
    PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, m), KOKKOS_LAMBDA(const PetscInt i) { B(i) = b(rowperm(i)); }));
    Y = x;
  }
  PetscCallCXX(sptrsv_solve(exec, &factors->khL, factors->iL_d, factors->jL_d, factors->aL_d, B, Y));

  // Solve U C^- x = Y
  if (col_identity) {
    X = x;
  } else {
    X = factors->workVector;
  }
  PetscCallCXX(sptrsv_solve(exec, &factors->khU, factors->iU_d, factors->jU_d, factors->aU_d, Y, X));

  // x = C X; Reorder X with the inverse col permutation
  if (!col_identity) PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, m), KOKKOS_LAMBDA(const PetscInt i) { x(colperm(i)) = X(i); }));

  PetscCall(VecRestoreKokkosView(bb, &b));
  PetscCall(VecRestoreKokkosViewWrite(xx, &x));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Solve A^T x = b, with RAC = LU, where R and C are row and col permutation matrices on A respectively.
// R and C are represented by rowperm and colperm in factors.
// If R or C is identity (i.e, no reordering), then rowperm or colperm is empty.
// A = R^-1 L U C^-1, so A^T = C^-T U^T L^T R^-T. But since C^- = C^T, R^- = R^T, we have A^T = C U^T L^T R.
static PetscErrorCode MatSolveTranspose_SeqAIJKokkos_LU(Mat A, Vec bb, Vec xx)
{
  auto                        exec    = PetscGetKokkosExecutionSpace();
  Mat_SeqAIJKokkosTriFactors *factors = (Mat_SeqAIJKokkosTriFactors *)A->spptr;
  PetscInt                    m       = A->rmap->n;
  PetscScalarKokkosView       X, Y, B; // alias
  ConstPetscScalarKokkosView  b;
  PetscScalarKokkosView       x;
  PetscIntKokkosView         &rowperm      = factors->rowperm;
  PetscIntKokkosView         &colperm      = factors->colperm;
  PetscBool                   row_identity = rowperm.extent(0) ? PETSC_FALSE : PETSC_TRUE;
  PetscBool                   col_identity = colperm.extent(0) ? PETSC_FALSE : PETSC_TRUE;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCall(MatSeqAIJKokkosTransposeSolveCheck(A)); // Update L^T, U^T if needed, and do sptrsv symbolic for L^T, U^T
  PetscCall(VecGetKokkosView(bb, &b));
  PetscCall(VecGetKokkosViewWrite(xx, &x));

  // Solve U^T Y = B (i.e., U^T (L^T R x) = C^- b).  Note C^- b = C^T b, which means applying the column permutation on b.
  if (col_identity) { // Reorder b with the col permutation
    B = PetscScalarKokkosView(const_cast<PetscScalar *>(b.data()), b.extent(0));
    Y = factors->workVector;
  } else {
    B = factors->workVector;
    PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, m), KOKKOS_LAMBDA(const PetscInt i) { B(i) = b(colperm(i)); }));
    Y = x;
  }
  PetscCallCXX(sptrsv_solve(exec, &factors->khUt, factors->iUt_d, factors->jUt_d, factors->aUt_d, B, Y));

  // Solve L^T X = Y
  if (row_identity) {
    X = x;
  } else {
    X = factors->workVector;
  }
  PetscCallCXX(sptrsv_solve(exec, &factors->khLt, factors->iLt_d, factors->jLt_d, factors->aLt_d, Y, X));

  // x = R^- X = R^T X; Reorder X with the inverse row permutation
  if (!row_identity) PetscCallCXX(Kokkos::parallel_for(Kokkos::RangePolicy<>(exec, 0, m), KOKKOS_LAMBDA(const PetscInt i) { x(rowperm(i)) = X(i); }));

  PetscCall(VecRestoreKokkosView(bb, &b));
  PetscCall(VecRestoreKokkosViewWrite(xx, &x));
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorNumeric_SeqAIJKokkos(Mat B, Mat A, const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJKokkosSyncHost(A));
  PetscCall(MatLUFactorNumeric_SeqAIJ(B, A, info));

  if (!info->solveonhost) { // if solve on host, then we don't need to copy L, U to device
    Mat_SeqAIJKokkosTriFactors *factors = (Mat_SeqAIJKokkosTriFactors *)B->spptr;
    Mat_SeqAIJ                 *b       = static_cast<Mat_SeqAIJ *>(B->data);
    const PetscInt             *Bi = b->i, *Bj = b->j, *Bdiag = b->diag;
    const MatScalar            *Ba = b->a;
    PetscInt                    m = B->rmap->n, n = B->cmap->n;

    if (factors->iL_h.extent(0) == 0) { // Allocate memory and copy the L, U structure for the first time
      // Allocate memory and copy the structure
      factors->iL_h = MatRowMapKokkosViewHost(NoInit("iL_h"), m + 1);
      factors->jL_h = MatColIdxKokkosViewHost(NoInit("jL_h"), (Bi[m] - Bi[0]) + m); // + the diagonal entries
      factors->aL_h = MatScalarKokkosViewHost(NoInit("aL_h"), (Bi[m] - Bi[0]) + m);
      factors->iU_h = MatRowMapKokkosViewHost(NoInit("iU_h"), m + 1);
      factors->jU_h = MatColIdxKokkosViewHost(NoInit("jU_h"), (Bdiag[0] - Bdiag[m]));
      factors->aU_h = MatScalarKokkosViewHost(NoInit("aU_h"), (Bdiag[0] - Bdiag[m]));

      PetscInt *Li = factors->iL_h.data();
      PetscInt *Lj = factors->jL_h.data();
      PetscInt *Ui = factors->iU_h.data();
      PetscInt *Uj = factors->jU_h.data();

      Li[0] = Ui[0] = 0;
      for (PetscInt i = 0; i < m; i++) {
        PetscInt llen = Bi[i + 1] - Bi[i];       // exclusive of the diagonal entry
        PetscInt ulen = Bdiag[i] - Bdiag[i + 1]; // inclusive of the diagonal entry

        PetscCall(PetscArraycpy(Lj + Li[i], Bj + Bi[i], llen)); // entries of L on the left of the diagonal
        Lj[Li[i] + llen] = i;                                   // diagonal entry of L

        Uj[Ui[i]] = i;                                                             // diagonal entry of U
        PetscCall(PetscArraycpy(Uj + Ui[i] + 1, Bj + Bdiag[i + 1] + 1, ulen - 1)); // entries of U on  the right of the diagonal

        Li[i + 1] = Li[i] + llen + 1;
        Ui[i + 1] = Ui[i] + ulen;
      }

      factors->iL_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), factors->iL_h);
      factors->jL_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), factors->jL_h);
      factors->iU_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), factors->iU_h);
      factors->jU_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), factors->jU_h);
      factors->aL_d = Kokkos::create_mirror_view(DefaultMemorySpace(), factors->aL_h);
      factors->aU_d = Kokkos::create_mirror_view(DefaultMemorySpace(), factors->aU_h);

      // Copy row/col permutation to device
      IS        rowperm = ((Mat_SeqAIJ *)B->data)->row;
      PetscBool row_identity;
      PetscCall(ISIdentity(rowperm, &row_identity));
      if (!row_identity) {
        const PetscInt *ip;

        PetscCall(ISGetIndices(rowperm, &ip));
        factors->rowperm = PetscIntKokkosView(NoInit("rowperm"), m);
        PetscCallCXX(Kokkos::deep_copy(factors->rowperm, PetscIntKokkosViewHost(const_cast<PetscInt *>(ip), m)));
        PetscCall(ISRestoreIndices(rowperm, &ip));
        PetscCall(PetscLogCpuToGpu(m * sizeof(PetscInt)));
      }

      IS        colperm = ((Mat_SeqAIJ *)B->data)->col;
      PetscBool col_identity;
      PetscCall(ISIdentity(colperm, &col_identity));
      if (!col_identity) {
        const PetscInt *ip;

        PetscCall(ISGetIndices(colperm, &ip));
        factors->colperm = PetscIntKokkosView(NoInit("colperm"), n);
        PetscCallCXX(Kokkos::deep_copy(factors->colperm, PetscIntKokkosViewHost(const_cast<PetscInt *>(ip), n)));
        PetscCall(ISRestoreIndices(colperm, &ip));
        PetscCall(PetscLogCpuToGpu(n * sizeof(PetscInt)));
      }

      /* Create sptrsv handles for L, U and their transpose */
#if defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE)
      auto sptrsv_alg = SPTRSVAlgorithm::SPTRSV_CUSPARSE;
#else
      auto sptrsv_alg = SPTRSVAlgorithm::SEQLVLSCHD_TP1;
#endif
      factors->khL.create_sptrsv_handle(sptrsv_alg, m, true /* L is lower tri */);
      factors->khU.create_sptrsv_handle(sptrsv_alg, m, false /* U is not lower tri */);
      factors->khLt.create_sptrsv_handle(sptrsv_alg, m, false /* L^T is not lower tri */);
      factors->khUt.create_sptrsv_handle(sptrsv_alg, m, true /* U^T is lower tri */);
    }

    // Copy the value
    for (PetscInt i = 0; i < m; i++) {
      PetscInt        llen = Bi[i + 1] - Bi[i];
      PetscInt        ulen = Bdiag[i] - Bdiag[i + 1];
      const PetscInt *Li   = factors->iL_h.data();
      const PetscInt *Ui   = factors->iU_h.data();

      PetscScalar *La = factors->aL_h.data();
      PetscScalar *Ua = factors->aU_h.data();

      PetscCall(PetscArraycpy(La + Li[i], Ba + Bi[i], llen)); // entries of L
      La[Li[i] + llen] = 1.0;                                 // diagonal entry

      Ua[Ui[i]] = 1.0 / Ba[Bdiag[i]];                                            // diagonal entry
      PetscCall(PetscArraycpy(Ua + Ui[i] + 1, Ba + Bdiag[i + 1] + 1, ulen - 1)); // entries of U
    }

    PetscCallCXX(Kokkos::deep_copy(factors->aL_d, factors->aL_h));
    PetscCallCXX(Kokkos::deep_copy(factors->aU_d, factors->aU_h));
    // Once the factors' values have changed, we need to update their transpose and redo sptrsv symbolic
    factors->transpose_updated         = PETSC_FALSE;
    factors->sptrsv_symbolic_completed = PETSC_FALSE;

    B->ops->solve          = MatSolve_SeqAIJKokkos_LU;
    B->ops->solvetranspose = MatSolveTranspose_SeqAIJKokkos_LU;
  }

  B->ops->matsolve          = NULL;
  B->ops->matsolvetranspose = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatILUFactorNumeric_SeqAIJKokkos_ILU0(Mat B, Mat A, const MatFactorInfo *info)
{
  Mat_SeqAIJKokkos           *aijkok   = (Mat_SeqAIJKokkos *)A->spptr;
  Mat_SeqAIJKokkosTriFactors *factors  = (Mat_SeqAIJKokkosTriFactors *)B->spptr;
  PetscInt                    fill_lev = info->levels;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  PetscCheck(!info->factoronhost, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "MatFactorInfo.factoronhost should be false");
  PetscCall(MatSeqAIJKokkosSyncDevice(A));

  auto a_d = aijkok->a_dual.view_device();
  auto i_d = aijkok->i_dual.view_device();
  auto j_d = aijkok->j_dual.view_device();

  PetscCallCXX(spiluk_numeric(&factors->kh, fill_lev, i_d, j_d, a_d, factors->iL_d, factors->jL_d, factors->aL_d, factors->iU_d, factors->jU_d, factors->aU_d));

  B->assembled              = PETSC_TRUE;
  B->preallocated           = PETSC_TRUE;
  B->ops->solve             = MatSolve_SeqAIJKokkos_LU;
  B->ops->solvetranspose    = MatSolveTranspose_SeqAIJKokkos_LU;
  B->ops->matsolve          = NULL;
  B->ops->matsolvetranspose = NULL;

  /* Once the factors' value changed, we need to update their transpose and sptrsv handle */
  factors->transpose_updated         = PETSC_FALSE;
  factors->sptrsv_symbolic_completed = PETSC_FALSE;
  /* TODO: log flops, but how to know that? */
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Use KK's spiluk_symbolic() to do ILU0 symbolic factorization, with no row/col reordering
static PetscErrorCode MatILUFactorSymbolic_SeqAIJKokkos_ILU0(Mat B, Mat A, IS, IS, const MatFactorInfo *info)
{
  Mat_SeqAIJKokkos           *aijkok;
  Mat_SeqAIJ                 *b;
  Mat_SeqAIJKokkosTriFactors *factors  = (Mat_SeqAIJKokkosTriFactors *)B->spptr;
  PetscInt                    fill_lev = info->levels;
  PetscInt                    nnzA     = ((Mat_SeqAIJ *)A->data)->nz, nnzL, nnzU;
  PetscInt                    n        = A->rmap->n;

  PetscFunctionBegin;
  PetscCheck(!info->factoronhost, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "MatFactorInfo's factoronhost should be false as we are doing it on device right now");
  PetscCall(MatSeqAIJKokkosSyncDevice(A));

  /* Create a spiluk handle and then do symbolic factorization */
  nnzL = nnzU = PetscRealIntMultTruncate(info->fill, nnzA);
  factors->kh.create_spiluk_handle(SPILUKAlgorithm::SEQLVLSCHD_TP1, n, nnzL, nnzU);

  auto spiluk_handle = factors->kh.get_spiluk_handle();

  Kokkos::realloc(factors->iL_d, n + 1); /* Free old arrays and realloc */
  Kokkos::realloc(factors->jL_d, spiluk_handle->get_nnzL());
  Kokkos::realloc(factors->iU_d, n + 1);
  Kokkos::realloc(factors->jU_d, spiluk_handle->get_nnzU());

  aijkok   = (Mat_SeqAIJKokkos *)A->spptr;
  auto i_d = aijkok->i_dual.view_device();
  auto j_d = aijkok->j_dual.view_device();
  PetscCallCXX(spiluk_symbolic(&factors->kh, fill_lev, i_d, j_d, factors->iL_d, factors->jL_d, factors->iU_d, factors->jU_d));
  /* TODO: if spiluk_symbolic is asynchronous, do we need to sync before calling get_nnzL()? */

  Kokkos::resize(factors->jL_d, spiluk_handle->get_nnzL()); /* Shrink or expand, and retain old value */
  Kokkos::resize(factors->jU_d, spiluk_handle->get_nnzU());
  Kokkos::realloc(factors->aL_d, spiluk_handle->get_nnzL()); /* No need to retain old value */
  Kokkos::realloc(factors->aU_d, spiluk_handle->get_nnzU());

  /* TODO: add options to select sptrsv algorithms */
  /* Create sptrsv handles for L, U and their transpose */
#if defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE)
  auto sptrsv_alg = SPTRSVAlgorithm::SPTRSV_CUSPARSE;
#else
  auto sptrsv_alg = SPTRSVAlgorithm::SEQLVLSCHD_TP1;
#endif

  factors->khL.create_sptrsv_handle(sptrsv_alg, n, true /* L is lower tri */);
  factors->khU.create_sptrsv_handle(sptrsv_alg, n, false /* U is not lower tri */);
  factors->khLt.create_sptrsv_handle(sptrsv_alg, n, false /* L^T is not lower tri */);
  factors->khUt.create_sptrsv_handle(sptrsv_alg, n, true /* U^T is lower tri */);

  /* Fill fields of the factor matrix B */
  PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(B, MAT_SKIP_ALLOCATION, NULL));
  b     = (Mat_SeqAIJ *)B->data;
  b->nz = b->maxnz          = spiluk_handle->get_nnzL() + spiluk_handle->get_nnzU();
  B->info.fill_ratio_given  = info->fill;
  B->info.fill_ratio_needed = nnzA > 0 ? ((PetscReal)b->nz) / ((PetscReal)nnzA) : 1.0;

  B->ops->lufactornumeric = MatILUFactorNumeric_SeqAIJKokkos_ILU0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorSymbolic_SeqAIJKokkos(Mat B, Mat A, IS isrow, IS iscol, const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCall(MatLUFactorSymbolic_SeqAIJ(B, A, isrow, iscol, info));
  PetscCheck(!B->spptr, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Expected a NULL spptr");
  PetscCallCXX(B->spptr = new Mat_SeqAIJKokkosTriFactors(B->rmap->n));
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJKokkos;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatILUFactorSymbolic_SeqAIJKokkos(Mat B, Mat A, IS isrow, IS iscol, const MatFactorInfo *info)
{
  PetscBool row_identity = PETSC_FALSE, col_identity = PETSC_FALSE;

  PetscFunctionBegin;
  if (!info->factoronhost) {
    PetscCall(ISIdentity(isrow, &row_identity));
    PetscCall(ISIdentity(iscol, &col_identity));
  }

  PetscCheck(!B->spptr, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Expected a NULL spptr");
  PetscCallCXX(B->spptr = new Mat_SeqAIJKokkosTriFactors(B->rmap->n));

  if (!info->factoronhost && !info->levels && row_identity && col_identity) { // if level 0 and no reordering
    PetscCall(MatILUFactorSymbolic_SeqAIJKokkos_ILU0(B, A, isrow, iscol, info));
  } else {
    PetscCall(MatILUFactorSymbolic_SeqAIJ(B, A, isrow, iscol, info)); // otherwise, use PETSc's ILU on host
    B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJKokkos;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactorNumeric_SeqAIJKokkos(Mat B, Mat A, const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscCall(MatSeqAIJKokkosSyncHost(A));
  PetscCall(MatCholeskyFactorNumeric_SeqAIJ(B, A, info));

  if (!info->solveonhost) { // if solve on host, then we don't need to copy L, U to device
    Mat_SeqAIJKokkosTriFactors *factors = (Mat_SeqAIJKokkosTriFactors *)B->spptr;
    Mat_SeqAIJ                 *b       = static_cast<Mat_SeqAIJ *>(B->data);
    const PetscInt             *Bi = b->i, *Bj = b->j, *Bdiag = b->diag;
    const MatScalar            *Ba = b->a;
    PetscInt                    m  = B->rmap->n;

    if (factors->iU_h.extent(0) == 0) { // First time of numeric factorization
      // Allocate memory and copy the structure
      factors->iU_h = PetscIntKokkosViewHost(const_cast<PetscInt *>(Bi), m + 1); // wrap Bi as iU_h
      factors->jU_h = MatColIdxKokkosViewHost(NoInit("jU_h"), Bi[m]);
      factors->aU_h = MatScalarKokkosViewHost(NoInit("aU_h"), Bi[m]);
      factors->D_h  = MatScalarKokkosViewHost(NoInit("D_h"), m);
      factors->aU_d = Kokkos::create_mirror_view(DefaultMemorySpace(), factors->aU_h);
      factors->D_d  = Kokkos::create_mirror_view(DefaultMemorySpace(), factors->D_h);

      // Build jU_h from the skewed Aj
      PetscInt *Uj = factors->jU_h.data();
      for (PetscInt i = 0; i < m; i++) {
        PetscInt ulen = Bi[i + 1] - Bi[i];
        Uj[Bi[i]]     = i;                                              // diagonal entry
        PetscCall(PetscArraycpy(Uj + Bi[i] + 1, Bj + Bi[i], ulen - 1)); // entries of U on the right of the diagonal
      }

      // Copy iU, jU to device
      PetscCallCXX(factors->iU_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), factors->iU_h));
      PetscCallCXX(factors->jU_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), factors->jU_h));

      // Copy row/col permutation to device
      IS        rowperm = ((Mat_SeqAIJ *)B->data)->row;
      PetscBool row_identity;
      PetscCall(ISIdentity(rowperm, &row_identity));
      if (!row_identity) {
        const PetscInt *ip;

        PetscCall(ISGetIndices(rowperm, &ip));
        PetscCallCXX(factors->rowperm = PetscIntKokkosView(NoInit("rowperm"), m));
        PetscCallCXX(Kokkos::deep_copy(factors->rowperm, PetscIntKokkosViewHost(const_cast<PetscInt *>(ip), m)));
        PetscCall(ISRestoreIndices(rowperm, &ip));
        PetscCall(PetscLogCpuToGpu(m * sizeof(PetscInt)));
      }

      // Create sptrsv handles for U and U^T
#if defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE)
      auto sptrsv_alg = SPTRSVAlgorithm::SPTRSV_CUSPARSE;
#else
      auto sptrsv_alg = SPTRSVAlgorithm::SEQLVLSCHD_TP1;
#endif
      factors->khU.create_sptrsv_handle(sptrsv_alg, m, false /* U is not lower tri */);
      factors->khUt.create_sptrsv_handle(sptrsv_alg, m, true /* U^T is lower tri */);
    }
    // These pointers were set MatCholeskyFactorNumeric_SeqAIJ(), so we always need to update them
    B->ops->solve          = MatSolve_SeqAIJKokkos_Cholesky;
    B->ops->solvetranspose = MatSolve_SeqAIJKokkos_Cholesky;

    // Copy the value
    PetscScalar *Ua = factors->aU_h.data();
    PetscScalar *D  = factors->D_h.data();
    for (PetscInt i = 0; i < m; i++) {
      D[i]      = Ba[Bdiag[i]];     // actually Aa[Adiag[i]] is the inverse of the diagonal
      Ua[Bi[i]] = (PetscScalar)1.0; // set the unit diagonal for U
      for (PetscInt k = 0; k < Bi[i + 1] - Bi[i] - 1; k++) Ua[Bi[i] + 1 + k] = -Ba[Bi[i] + k];
    }
    PetscCallCXX(Kokkos::deep_copy(factors->aU_d, factors->aU_h));
    PetscCallCXX(Kokkos::deep_copy(factors->D_d, factors->D_h));

    factors->sptrsv_symbolic_completed = PETSC_FALSE; // When numeric value changed, we must do these again
    factors->transpose_updated         = PETSC_FALSE;
  }

  B->ops->matsolve          = NULL;
  B->ops->matsolvetranspose = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatICCFactorSymbolic_SeqAIJKokkos(Mat B, Mat A, IS perm, const MatFactorInfo *info)
{
  PetscFunctionBegin;
  if (info->solveonhost) {
    // If solve on host, we have to change the type, as eventually we need to call MatSolve_SeqSBAIJ_1_NaturalOrdering() etc.
    PetscCall(MatSetType(B, MATSEQSBAIJ));
    PetscCall(MatSeqSBAIJSetPreallocation(B, 1, MAT_SKIP_ALLOCATION, NULL));
  }

  PetscCall(MatICCFactorSymbolic_SeqAIJ(B, A, perm, info));

  if (!info->solveonhost) {
    // If solve on device, B is still a MATSEQAIJKOKKOS, so we are good to allocate B->spptr
    PetscCheck(!B->spptr, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Expected a NULL spptr");
    PetscCallCXX(B->spptr = new Mat_SeqAIJKokkosTriFactors(B->rmap->n));
    B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJKokkos;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactorSymbolic_SeqAIJKokkos(Mat B, Mat A, IS perm, const MatFactorInfo *info)
{
  PetscFunctionBegin;
  if (info->solveonhost) {
    // If solve on host, we have to change the type, as eventually we need to call MatSolve_SeqSBAIJ_1_NaturalOrdering() etc.
    PetscCall(MatSetType(B, MATSEQSBAIJ));
    PetscCall(MatSeqSBAIJSetPreallocation(B, 1, MAT_SKIP_ALLOCATION, NULL));
  }

  PetscCall(MatCholeskyFactorSymbolic_SeqAIJ(B, A, perm, info)); // it sets B's two ISes ((Mat_SeqAIJ*)B->data)->{row, col} to perm

  if (!info->solveonhost) {
    // If solve on device, B is still a MATSEQAIJKOKKOS, so we are good to allocate B->spptr
    PetscCheck(!B->spptr, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Expected a NULL spptr");
    PetscCallCXX(B->spptr = new Mat_SeqAIJKokkosTriFactors(B->rmap->n));
    B->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJKokkos;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// The _Kokkos suffix means we will use Kokkos as a solver for the SeqAIJKokkos matrix
static PetscErrorCode MatFactorGetSolverType_SeqAIJKokkos_Kokkos(Mat A, MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERKOKKOS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  MATSOLVERKOKKOS = "Kokkos" - A matrix solver type providing triangular solvers for sequential matrices
  on a single GPU of type, `MATSEQAIJKOKKOS`, `MATAIJKOKKOS`.

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `PCFactorSetMatSolverType()`, `MatSolverType`, `MatCreateSeqAIJKokkos()`, `MATAIJKOKKOS`
M*/
PETSC_EXTERN PetscErrorCode MatGetFactor_SeqAIJKokkos_Kokkos(Mat A, MatFactorType ftype, Mat *B) /* MatGetFactor_<MatType>_<MatSolverType> */
{
  PetscInt n = A->rmap->n;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCall(MatCreate(comm, B));
  PetscCall(MatSetSizes(*B, n, n, n, n));
  PetscCall(MatSetBlockSizesFromMats(*B, A, A));
  (*B)->factortype = ftype;
  PetscCall(MatSetType(*B, MATSEQAIJKOKKOS));
  PetscCall(MatSeqAIJSetPreallocation(*B, MAT_SKIP_ALLOCATION, NULL));
  PetscCheck(!(*B)->spptr, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Expected a NULL spptr");

  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU || ftype == MAT_FACTOR_ILUDT) {
    (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJKokkos;
    (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_SeqAIJKokkos;
    PetscCall(PetscStrallocpy(MATORDERINGND, (char **)&(*B)->preferredordering[MAT_FACTOR_LU]));
    PetscCall(PetscStrallocpy(MATORDERINGNATURAL, (char **)&(*B)->preferredordering[MAT_FACTOR_ILU]));
    PetscCall(PetscStrallocpy(MATORDERINGNATURAL, (char **)&(*B)->preferredordering[MAT_FACTOR_ILUDT]));
  } else if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    (*B)->ops->iccfactorsymbolic      = MatICCFactorSymbolic_SeqAIJKokkos;
    (*B)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqAIJKokkos;
    PetscCall(PetscStrallocpy(MATORDERINGND, (char **)&(*B)->preferredordering[MAT_FACTOR_CHOLESKY]));
    PetscCall(PetscStrallocpy(MATORDERINGNATURAL, (char **)&(*B)->preferredordering[MAT_FACTOR_ICC]));
  } else SETERRQ(comm, PETSC_ERR_SUP, "MatFactorType %s is not supported by MatType SeqAIJKokkos", MatFactorTypes[ftype]);

  // The factorization can use the ordering provided in MatLUFactorSymbolic(), MatCholeskyFactorSymbolic() etc, though we do it on host
  (*B)->canuseordering = PETSC_TRUE;
  PetscCall(PetscObjectComposeFunction((PetscObject)*B, "MatFactorGetSolverType_C", MatFactorGetSolverType_SeqAIJKokkos_Kokkos));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSolverTypeRegister_Kokkos(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERKOKKOS, MATSEQAIJKOKKOS, MAT_FACTOR_LU, MatGetFactor_SeqAIJKokkos_Kokkos));
  PetscCall(MatSolverTypeRegister(MATSOLVERKOKKOS, MATSEQAIJKOKKOS, MAT_FACTOR_CHOLESKY, MatGetFactor_SeqAIJKokkos_Kokkos));
  PetscCall(MatSolverTypeRegister(MATSOLVERKOKKOS, MATSEQAIJKOKKOS, MAT_FACTOR_ILU, MatGetFactor_SeqAIJKokkos_Kokkos));
  PetscCall(MatSolverTypeRegister(MATSOLVERKOKKOS, MATSEQAIJKOKKOS, MAT_FACTOR_ICC, MatGetFactor_SeqAIJKokkos_Kokkos));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Utility to print out a KokkosCsrMatrix for debugging */
PETSC_INTERN PetscErrorCode PrintCsrMatrix(const KokkosCsrMatrix &csrmat)
{
  const auto        &iv = Kokkos::create_mirror_view_and_copy(HostMirrorMemorySpace(), csrmat.graph.row_map);
  const auto        &jv = Kokkos::create_mirror_view_and_copy(HostMirrorMemorySpace(), csrmat.graph.entries);
  const auto        &av = Kokkos::create_mirror_view_and_copy(HostMirrorMemorySpace(), csrmat.values);
  const PetscInt    *i  = iv.data();
  const PetscInt    *j  = jv.data();
  const PetscScalar *a  = av.data();
  PetscInt           m = csrmat.numRows(), n = csrmat.numCols(), nnz = csrmat.nnz();

  PetscFunctionBegin;
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "%" PetscInt_FMT " x %" PetscInt_FMT " SeqAIJKokkos, with %" PetscInt_FMT " nonzeros\n", m, n, nnz));
  for (PetscInt k = 0; k < m; k++) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "%" PetscInt_FMT ": ", k));
    for (PetscInt p = i[k]; p < i[k + 1]; p++) PetscCall(PetscPrintf(PETSC_COMM_SELF, "%" PetscInt_FMT "(%.1f), ", j[p], (double)PetscRealPart(a[p])));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
