static const char help[] = "Test CUDA/HIP AIJ, SELL, and dense operations and array copies on the current device context.\n";

#include <petscmat.h>
#include <petscdevice.h>
#include <petsc/private/deviceimpl.h>
#if PetscDefined(HAVE_CUDA)
  #include <petscdevice_cuda.h>
#endif
#if PetscDefined(HAVE_HIP)
  #include <petscdevice_hip.h>
#endif

static PetscErrorCode WriteValues(Mat A, PetscDeviceType type, PetscDeviceContext dctx, const PetscScalar input[], PetscBool delay)
{
  PetscScalar *a;
  void        *stream;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextGetStreamHandle(dctx, &stream));
#if PetscDefined(HAVE_CUDA)
  if (type == PETSC_DEVICE_CUDA) {
    PetscCall(MatSeqAIJCUSPARSEGetArray(A, &a));
    if (delay) PetscCall(PetscDeviceContextDelay(dctx, 0.05));
    PetscCallCUDA(cudaMemcpyAsync(a, input, 4 * sizeof(*a), cudaMemcpyHostToDevice, *(cudaStream_t *)stream));
    PetscCall(MatSeqAIJCUSPARSERestoreArray(A, &a));
  }
#endif
#if PetscDefined(HAVE_HIP)
  if (type == PETSC_DEVICE_HIP) {
    PetscCall(MatSeqAIJHIPSPARSEGetArray(A, &a));
    if (delay) PetscCall(PetscDeviceContextDelay(dctx, 0.05));
    PetscCallHIP(hipMemcpyAsync(a, input, 4 * sizeof(*a), hipMemcpyHostToDevice, *(hipStream_t *)stream));
    PetscCall(MatSeqAIJHIPSPARSERestoreArray(A, &a));
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestCopies(Mat A, PetscDeviceType type, PetscDeviceContext dctx, const PetscScalar input[])
{
  PetscErrorCode (*copy)(Mat, PetscInt, const PetscInt[], PetscScalar[]) = NULL;
  const PetscInt     idx[]                                               = {3, 1, 0, 2};
  const PetscScalar *a;
  PetscScalar       *host, *device;
  PetscStreamType    streamtype;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectQueryFunction((PetscObject)A, "MatSeqAIJCopySubArray_C", &copy));
  PetscCheck(copy, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing device subarray copy implementation");
  PetscCall(PetscDeviceContextGetStreamType(dctx, &streamtype));
  PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, 4, PETSC_DECIDE, &host));
  PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_DEVICE, 4, sizeof(PetscScalar), &device));

  PetscCall(MatZeroEntries(A));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(WriteValues(A, type, dctx, input, PETSC_TRUE));
  PetscCall(MatSeqAIJGetArrayRead(A, &a));
  for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == input[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Host readback overtook the device write at entry %" PetscInt_FMT, i);
  PetscCall(MatSeqAIJRestoreArrayRead(A, &a));

  for (PetscInt d = 0; d < 2; ++d) {
    for (PetscInt indexed = 0; indexed < 2; ++indexed) {
      PetscCall(MatZeroEntries(A));
      PetscCall(PetscDeviceContextSynchronize(dctx));
      PetscCall(WriteValues(A, type, dctx, input, PETSC_TRUE));
      PetscCall((*copy)(A, 4, indexed ? idx : NULL, d ? device : host));
      if (streamtype == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
        PetscBool idle;

        PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
        PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Subarray copy returned with work pending on a barrier context");
      }
      if (d) {
        PetscCall(PetscDeviceArrayCopy(dctx, host, device, 4));
        PetscCall(PetscDeviceContextSynchronize(dctx));
      }
      for (PetscInt i = 0; i < 4; ++i)
        PetscCheck(host[i] == input[indexed ? idx[i] : i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Subarray copy (device %" PetscInt_FMT ", indexed %" PetscInt_FMT ") overtook the device write at entry %" PetscInt_FMT, d, indexed, i);
      PetscCall((*copy)(A, 0, NULL, d ? device : host));
    }
  }
  PetscCall(PetscDeviceFree(dctx, device));
  PetscCall(PetscDeviceFree(dctx, host));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestGetDiagonal(Mat A, PetscDeviceType type, PetscDeviceContext dctx, const PetscScalar input[])
{
  Vec                diag;
  const PetscScalar *a;
  PetscStreamType    streamtype;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextGetStreamType(dctx, &streamtype));
  PetscCall(MatCreateVecs(A, &diag, NULL));
  // Clear A so a stale diagonal differs from the values written after the delay.
  PetscCall(MatZeroEntries(A));
  // Warm the allocation and kernel before delaying the matrix values.
  PetscCall(MatGetDiagonal(A, diag));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(WriteValues(A, type, dctx, input, PETSC_TRUE));
  PetscCall(MatGetDiagonal(A, diag));
  if (streamtype == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
    PetscBool idle;

    PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
    PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatGetDiagonal() returned with work pending on a barrier context");
  }
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(VecGetArrayRead(diag, &a));
  for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == input[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect diagonal entry %" PetscInt_FMT, i);
  PetscCall(VecRestoreArrayRead(diag, &a));
  PetscCall(VecDestroy(&diag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestDiagonalScale(Mat A, PetscDeviceType type, PetscDeviceContext dctx, const PetscScalar input[])
{
  Vec                left, right;
  const PetscScalar *a;
  PetscStreamType    streamtype;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextGetStreamType(dctx, &streamtype));
  PetscCall(MatCreateVecs(A, &right, &left));
  PetscCall(VecSet(left, 2));
  PetscCall(VecSet(right, 3));
  for (PetscInt mode = 0; mode < 3; ++mode) {
    PetscScalar factor = mode == 0 ? 2 : (mode == 1 ? 3 : 6);
    Vec         ll = mode == 1 ? NULL : left, rr = mode == 0 ? NULL : right;

    // Warm the scaling kernels before delaying the matrix values.
    PetscCall(MatDiagonalScale(A, ll, rr));
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCall(WriteValues(A, type, dctx, input, PETSC_TRUE));
    PetscCall(MatDiagonalScale(A, ll, rr));
    if (streamtype == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
      PetscBool idle;

      PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
      PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatDiagonalScale() mode %" PetscInt_FMT " returned with work pending on a barrier context", mode);
    }
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCall(MatSeqAIJGetArrayRead(A, &a));
    for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == factor * input[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatDiagonalScale() mode %" PetscInt_FMT ": incorrect entry %" PetscInt_FMT, mode, i);
    PetscCall(MatSeqAIJRestoreArrayRead(A, &a));
  }
  PetscCall(VecDestroy(&left));
  PetscCall(VecDestroy(&right));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestGetIJ(PetscDeviceType type, PetscDeviceContext dctx)
{
  Mat             A;
  const PetscInt *i, *j;
  PetscInt       *rows, *cols;
  PetscBool       idle;
  void           *stream;

  PetscFunctionBeginUser;
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, 4, 4, 1, NULL, &A));
  PetscCall(MatSetValue(A, 1, 2, 7, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatConvert(A, type == PETSC_DEVICE_CUDA ? MATSEQAIJCUSPARSE : MATSEQAIJHIPSPARSE, MAT_INPLACE_MATRIX, &A));
  PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, 5, PETSC_DECIDE, &rows));
  PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, 1, PETSC_DECIDE, &cols));
  PetscCall(PetscDeviceContextGetStreamHandle(dctx, &stream));
  // Build the compressed device matrix before testing the full row-offset upload.
#if PetscDefined(HAVE_CUDA)
  if (type == PETSC_DEVICE_CUDA) {
    PetscCall(MatSeqAIJCUSPARSEGetIJ(A, PETSC_TRUE, &i, &j));
    PetscCall(MatSeqAIJCUSPARSERestoreIJ(A, PETSC_TRUE, &i, &j));
  }
#endif
#if PetscDefined(HAVE_HIP)
  if (type == PETSC_DEVICE_HIP) {
    PetscCall(MatSeqAIJHIPSPARSEGetIJ(A, PETSC_TRUE, &i, &j));
    PetscCall(MatSeqAIJHIPSPARSERestoreIJ(A, PETSC_TRUE, &i, &j));
  }
#endif
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(PetscDeviceContextDelay(dctx, 0.2));
#if PetscDefined(HAVE_CUDA)
  if (type == PETSC_DEVICE_CUDA) PetscCall(MatSeqAIJCUSPARSEGetIJ(A, PETSC_FALSE, &i, &j));
#endif
#if PetscDefined(HAVE_HIP)
  if (type == PETSC_DEVICE_HIP) PetscCall(MatSeqAIJHIPSPARSEGetIJ(A, PETSC_FALSE, &i, &j));
#endif
  PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
  PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatSeqAIJGetIJ() returned before its current-context upload completed");
#if PetscDefined(HAVE_CUDA)
  if (type == PETSC_DEVICE_CUDA) {
    PetscCallCUDA(cudaMemcpyAsync(rows, i, 5 * sizeof(*rows), cudaMemcpyDeviceToHost, *(cudaStream_t *)stream));
    PetscCallCUDA(cudaMemcpyAsync(cols, j, sizeof(*cols), cudaMemcpyDeviceToHost, *(cudaStream_t *)stream));
  }
#endif
#if PetscDefined(HAVE_HIP)
  if (type == PETSC_DEVICE_HIP) {
    PetscCallHIP(hipMemcpyAsync(rows, i, 5 * sizeof(*rows), hipMemcpyDeviceToHost, *(hipStream_t *)stream));
    PetscCallHIP(hipMemcpyAsync(cols, j, sizeof(*cols), hipMemcpyDeviceToHost, *(hipStream_t *)stream));
  }
#endif
  PetscCall(PetscDeviceContextSynchronize(dctx));
  for (PetscInt k = 0; k < 5; ++k) PetscCheck(rows[k] == (k > 1), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect row offset at entry %" PetscInt_FMT, k);
  PetscCheck(cols[0] == 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect column index");
#if PetscDefined(HAVE_CUDA)
  if (type == PETSC_DEVICE_CUDA) PetscCall(MatSeqAIJCUSPARSERestoreIJ(A, PETSC_FALSE, &i, &j));
#endif
#if PetscDefined(HAVE_HIP)
  if (type == PETSC_DEVICE_HIP) PetscCall(MatSeqAIJHIPSPARSERestoreIJ(A, PETSC_FALSE, &i, &j));
#endif
  PetscCall(PetscDeviceFree(dctx, cols));
  PetscCall(PetscDeviceFree(dctx, rows));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SparsePointerMode(PetscDeviceType type, PetscDeviceContext dctx, PetscBool set, PetscBool device_mode)
{
  PetscFunctionBeginUser;
#if PetscDefined(HAVE_CUDA)
  if (type == PETSC_DEVICE_CUDA) {
    cusparseHandle_t      handle;
    cusparsePointerMode_t mode = device_mode ? CUSPARSE_POINTER_MODE_DEVICE : CUSPARSE_POINTER_MODE_HOST;

    PetscCall(PetscDeviceContextGetSPARSEHandle_Internal(dctx, &handle));
    if (set) PetscCallCUSPARSE(cusparseSetPointerMode(handle, mode));
    else {
      cusparsePointerMode_t actual;

      PetscCallCUSPARSE(cusparseGetPointerMode(handle, &actual));
      PetscCheck(actual == mode, PETSC_COMM_SELF, PETSC_ERR_PLIB, "cuSPARSE pointer mode was not restored");
    }
  }
#endif
#if PetscDefined(HAVE_HIP)
  if (type == PETSC_DEVICE_HIP) {
    hipsparseHandle_t      handle;
    hipsparsePointerMode_t mode = device_mode ? HIPSPARSE_POINTER_MODE_DEVICE : HIPSPARSE_POINTER_MODE_HOST;

    PetscCall(PetscDeviceContextGetSPARSEHandle_Internal(dctx, &handle));
    if (set) PetscCallHIPSPARSE(hipsparseSetPointerMode(handle, mode));
    else {
      hipsparsePointerMode_t actual;

      PetscCallHIPSPARSE(hipsparseGetPointerMode(handle, &actual));
      PetscCheck(actual == mode, PETSC_COMM_SELF, PETSC_ERR_PLIB, "hipSPARSE pointer mode was not restored");
    }
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestSparseOperations(Mat A, PetscDeviceType type, PetscDeviceContext dctx, const PetscScalar input[])
{
  const PetscScalar  ones[] = {1, 1, 1, 1};
  const PetscScalar *a;
  PetscStreamType    streamtype;
  PetscBool          idle;
  Vec                x, y;
  Mat                B, C;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextGetStreamType(dctx, &streamtype));
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(VecSet(x, 1));
  PetscCall(SparsePointerMode(type, dctx, PETSC_TRUE, PETSC_FALSE));
  PetscCall(WriteValues(A, type, dctx, ones, PETSC_FALSE));
  PetscCall(MatMult(A, x, y));
  PetscCall(SparsePointerMode(type, dctx, PETSC_FALSE, PETSC_FALSE));
  PetscCall(PetscDeviceContextSynchronize(dctx));

  PetscCall(WriteValues(A, type, dctx, input, PETSC_TRUE));
  PetscCall(MatMult(A, x, y));
  if (streamtype == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
    PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
    PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatMult() returned with work pending on a barrier context");
  }
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(VecGetArrayRead(y, &a));
  for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == input[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatMult() returned an incorrect entry %" PetscInt_FMT, i);
  PetscCall(VecRestoreArrayRead(y, &a));

  PetscCall(MatSetOption(A, MAT_FORM_EXPLICIT_TRANSPOSE, PETSC_TRUE));
  PetscCall(MatMultTranspose(A, x, y));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(VecSet(x, 0));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(PetscDeviceContextDelay(dctx, 0.05));
  PetscCall(VecSet(x, 2));
  PetscCall(MatMultTranspose(A, x, y));
  if (streamtype == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
    PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
    PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatMultTranspose() returned with work pending on a barrier context");
  }
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(VecGetArrayRead(y, &a));
  for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == 2 * input[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatMultTranspose() returned an incorrect entry %" PetscInt_FMT, i);
  PetscCall(VecRestoreArrayRead(y, &a));
  PetscCall(WriteValues(A, type, dctx, ones, PETSC_TRUE));
  PetscCall(MatMultTranspose(A, x, y));
  if (streamtype == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
    PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
    PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatMultTranspose() returned with a stale transpose update pending on a barrier context");
  }
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(VecGetArrayRead(y, &a));
  for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatMultTranspose() used stale matrix values at entry %" PetscInt_FMT, i);
  PetscCall(VecRestoreArrayRead(y, &a));
  PetscCall(MatSetOption(A, MAT_FORM_EXPLICIT_TRANSPOSE, PETSC_FALSE));
  PetscCall(VecSet(x, 1));
  PetscCall(WriteValues(A, type, dctx, input, PETSC_FALSE));

  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  PetscCall(WriteValues(B, type, dctx, ones, PETSC_FALSE));
  PetscCall(MatMatMult(A, B, MAT_INITIAL_MATRIX, 1.0, &C));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(WriteValues(B, type, dctx, input, PETSC_TRUE));
  PetscCall(MatMatMult(A, B, MAT_REUSE_MATRIX, 1.0, &C));
  PetscCall(SparsePointerMode(type, dctx, PETSC_FALSE, PETSC_FALSE));
  if (streamtype == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
    PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
    PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatMatMult() returned with work pending on a barrier context");
  }
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(MatSeqAIJGetArrayRead(C, &a));
  for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == input[i] * input[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatMatMult() returned an incorrect entry %" PetscInt_FMT, i);
  PetscCall(MatSeqAIJRestoreArrayRead(C, &a));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&B));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestSparseSolve(PetscDeviceType type, PetscDeviceContext saved, PetscDeviceContext dctx)
{
  Mat                A, F;
  IS                 perm;
  Vec                b, x;
  MatFactorInfo      info;
  PetscStreamType    streamtype;
  PetscBool          idle;
  const PetscScalar *a;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextSetCurrentContext(saved));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, 4, 4, 1, NULL, &A));
  for (PetscInt i = 0; i < 4; ++i) PetscCall(MatSetValue(A, i, i, 2, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatConvert(A, type == PETSC_DEVICE_CUDA ? MATSEQAIJCUSPARSE : MATSEQAIJHIPSPARSE, MAT_INPLACE_MATRIX, &A));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, 4, 0, 1, &perm));
  PetscCall(MatCreateVecs(A, &b, &x));
  PetscCall(MatFactorInfoInitialize(&info));
  PetscCall(PetscDeviceContextGetStreamType(dctx, &streamtype));
  for (PetscInt kind = 0; kind < 2; ++kind) {
    MatFactorType factortype = kind ? MAT_FACTOR_CHOLESKY : MAT_FACTOR_LU;

    PetscCall(PetscDeviceContextSetCurrentContext(saved));
    PetscCall(MatGetFactor(A, type == PETSC_DEVICE_CUDA ? MATSOLVERCUSPARSE : MATSOLVERHIPSPARSE, factortype, &F));
    PetscCall(SparsePointerMode(type, saved, PETSC_TRUE, PETSC_TRUE));
    if (kind) {
      PetscCall(MatCholeskyFactorSymbolic(F, A, perm, &info));
      PetscCall(MatCholeskyFactorNumeric(F, A, &info));
    } else {
      PetscCall(MatLUFactorSymbolic(F, A, perm, perm, &info));
      PetscCall(MatLUFactorNumeric(F, A, &info));
    }
    PetscCall(SparsePointerMode(type, saved, PETSC_FALSE, PETSC_TRUE));
    PetscCall(SparsePointerMode(type, saved, PETSC_TRUE, PETSC_FALSE));
    PetscCall(PetscDeviceContextSynchronize(saved));
    PetscCall(PetscDeviceContextSetCurrentContext(dctx));
    for (PetscInt solve = 0; solve < 3; ++solve) {
      PetscCall(VecSet(b, 0));
      PetscCall(VecSet(x, 0));
      PetscCall(PetscDeviceContextSynchronize(dctx));
      PetscCall(PetscDeviceContextDelay(dctx, 0.05));
      PetscCall(VecSet(b, 2));
      PetscCall(SparsePointerMode(type, dctx, PETSC_TRUE, PETSC_TRUE));
      if (solve == 1) PetscCall(MatSolveTranspose(F, b, x));
      else PetscCall(MatSolve(F, b, x));
      PetscCall(SparsePointerMode(type, dctx, PETSC_FALSE, PETSC_TRUE));
      PetscCall(SparsePointerMode(type, dctx, PETSC_TRUE, PETSC_FALSE));
      if (streamtype == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
        PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
        PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatSolve() returned with work pending on a barrier context");
      }
      PetscCall(PetscDeviceContextSynchronize(dctx));
      PetscCall(VecGetArrayRead(x, &a));
      for (PetscInt i = 0; i < 4; ++i) PetscCheck(PetscAbsScalar(a[i] - 1) < 100 * PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatSolve() returned an incorrect entry %" PetscInt_FMT, i);
      PetscCall(VecRestoreArrayRead(x, &a));
    }
    PetscCall(MatDestroy(&F));
  }
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(ISDestroy(&perm));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode WriteVector(Vec x, PetscDeviceType type, PetscDeviceContext dctx, const PetscScalar input[])
{
  PetscScalar *a;
  PetscInt     n;
  PetscMemType mtype;
  void        *stream;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetArrayWriteAndMemType(x, &a, &mtype));
  PetscCheck(PetscMemTypeDevice(mtype), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected a device array");
  PetscCall(PetscDeviceContextGetStreamHandle(dctx, &stream));
  PetscCall(PetscDeviceContextDelay(dctx, 0.05));
#if PetscDefined(HAVE_CUDA)
  if (type == PETSC_DEVICE_CUDA) PetscCallCUDA(cudaMemcpyAsync(a, input, n * sizeof(*a), cudaMemcpyHostToDevice, *(cudaStream_t *)stream));
#endif
#if PetscDefined(HAVE_HIP)
  if (type == PETSC_DEVICE_HIP) PetscCallHIP(hipMemcpyAsync(a, input, n * sizeof(*a), hipMemcpyHostToDevice, *(hipStream_t *)stream));
#endif
  PetscCall(VecRestoreArrayWriteAndMemType(x, &a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckBarrier(PetscDeviceContext dctx)
{
  PetscStreamType streamtype;
  PetscBool       idle;

  PetscFunctionBeginUser;
  PetscCall(PetscDeviceContextGetStreamType(dctx, &streamtype));
  if (streamtype == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
    PetscCall(PetscDeviceContextQueryIdle(dctx, &idle));
    PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Operation returned with work pending on a barrier context");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestSELL(PetscDeviceType type, PetscDeviceContext dctx)
{
  Mat                A, B;
  Vec                x, y, z;
  PetscScalar       *input;
  const PetscScalar *a;
  const PetscScalar  expected[] = {3, 3, 5, 9, 5, 8, 10};
#if PetscDefined(USE_LOG) && PetscDefined(HAVE_DEVICE)
  PetscLogEvent      event;
  PetscEventPerfInfo before, after;
  MatInfo            info;
#endif

  PetscFunctionBeginUser;
#if PetscDefined(USE_LOG) && PetscDefined(HAVE_DEVICE)
  PetscCall(PetscLogDefaultBegin());
  PetscCall(PetscLogEventRegister("SELLUpdate", MAT_CLASSID, &event));
#endif
  PetscCall(PetscDeviceMalloc(dctx, PETSC_MEMTYPE_HOST, 32, PETSC_DECIDE, &input));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  for (PetscInt i = 0; i < 32; ++i) input[i] = i + 1;
  PetscCall(MatCreate(PETSC_COMM_SELF, &A));
  PetscCall(MatSetSizes(A, 32, 32, 32, 32));
  PetscCall(MatSetType(A, MATSEQSELL));
  PetscCall(MatSetUp(A));
  for (PetscInt i = 0; i < 32; ++i) PetscCall(MatSetValue(A, i, i, 2, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatConvert(A, type == PETSC_DEVICE_CUDA ? MATSEQSELLCUDA : MATSEQSELLHIP, MAT_INPLACE_MATRIX, &A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(VecDuplicate(y, &z));
  PetscCall(VecSet(x, 1));
  PetscCall(VecSet(y, 1));
  // Destroying a duplicate before its first upload must release its backend data.
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  PetscCall(MatDestroy(&B));
  // MatMultAdd() must also work before any multiply has initialized the device arrays.
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  PetscCall(MatSetFromOptions(B));
  for (size_t stage = 0; stage < PETSC_STATIC_ARRAY_LENGTH(expected); ++stage) {
    Mat C = stage == 0 ? A : B;

#if PetscDefined(USE_LOG) && PetscDefined(HAVE_DEVICE)
    PetscCall(PetscLogEventGetPerfInfo(PETSC_DETERMINE, event, &before));
    PetscCall(PetscLogEventBegin(event, C, 0, 0, 0));
#endif
    if (stage == 2) {
      PetscCall(MatScale(A, 2));
      PetscCall(MatCopy(A, B, SAME_NONZERO_PATTERN));
      PetscCall(MatScale(A, 0.5));
    } else if (stage == 3) {
      PetscCall(MatSetOption(C, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));
      PetscCall(MatStoreValues(C));
      PetscCall(MatScale(C, 2));
    } else if (stage == 4) PetscCall(MatRetrieveValues(C));
    else if (stage == 5) {
      PetscCall(MatSetOption(C, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE));
      PetscCall(MatSetOption(C, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
      for (PetscInt i = 0; i < 32; ++i) PetscCall(MatSetValue(C, i, (i + 1) % 32, 3, INSERT_VALUES));
      PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));
    } else if (stage == 6) {
      for (PetscInt i = 0; i < 32; ++i) PetscCall(MatSetValue(C, i, i, 6, INSERT_VALUES));
      PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));
    }
    PetscCall(MatMultAdd(C, x, y, z));
    PetscCall(CheckBarrier(dctx));
    PetscCall(PetscDeviceContextSynchronize(dctx));
#if PetscDefined(USE_LOG) && PetscDefined(HAVE_DEVICE)
    PetscCall(PetscLogEventEnd(event, C, 0, 0, 0));
    if (stage == 2 || stage == 3 || stage == 4 || stage == 6) {
      PetscCall(PetscLogEventGetPerfInfo(PETSC_DETERMINE, event, &after));
      PetscCall(MatGetInfo(C, MAT_LOCAL, &info));
      PetscCheck(after.CpuToGpuSize - before.CpuToGpuSize == info.nz_used * sizeof(PetscScalar), PETSC_COMM_SELF, PETSC_ERR_PLIB, "SELL values-only update stage %zu uploaded %g bytes instead of %g", stage, after.CpuToGpuSize - before.CpuToGpuSize, info.nz_used * sizeof(PetscScalar));
    }
#endif
    PetscCall(VecGetArrayRead(z, &a));
    for (PetscInt i = 0; i < 32; ++i) PetscCheck(a[i] == expected[stage], PETSC_COMM_SELF, PETSC_ERR_PLIB, "SELL MatMultAdd() stage %zu returned an incorrect entry %" PetscInt_FMT, stage, i);
    PetscCall(VecRestoreArrayRead(z, &a));
  }
  PetscCall(MatDestroy(&B));
  PetscCall(MatMult(A, x, z));
  PetscCall(PetscDeviceContextSynchronize(dctx));

  for (PetscInt op = 0; op < 4; ++op) {
    PetscCall(VecSet(x, 0));
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCall(WriteVector(x, type, dctx, input));
    if (op == 1) PetscCall(MatMultAdd(A, x, y, z));
    else PetscCall(MatMult(A, x, z));
    PetscCall(CheckBarrier(dctx));
    if (op == 2) {
      // Upload new values after a queued multiply has consumed the old values.
      for (PetscInt i = 0; i < 32; ++i) PetscCall(MatSetValue(A, i, i, 3, INSERT_VALUES));
      PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    } else if (op == 3) {
      PetscCall(PetscDeviceContextDelay(dctx, 0.05));
      PetscCall(MatZeroEntries(A));
      PetscCall(CheckBarrier(dctx));
    }
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCall(VecGetArrayRead(z, &a));
    for (PetscInt i = 0; i < 32; ++i) PetscCheck(a[i] == (op == 3 ? 3 : 2) * input[i] + (op == 1), PETSC_COMM_SELF, PETSC_ERR_PLIB, "SELL operation %" PetscInt_FMT " returned an incorrect entry %" PetscInt_FMT, op, i);
    PetscCall(VecRestoreArrayRead(z, &a));
  }
  PetscCall(MatMult(A, x, z));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(VecGetArrayRead(z, &a));
  for (PetscInt i = 0; i < 32; ++i) PetscCheck(a[i] == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "SELL zeroing left a nonzero entry %" PetscInt_FMT, i);
  PetscCall(VecRestoreArrayRead(z, &a));
  // An empty matrix still needs device slice offsets before values-only updates.
  PetscCall(MatCreateSeqSELL(PETSC_COMM_SELF, 32, 32, 0, NULL, &B));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatConvert(B, type == PETSC_DEVICE_CUDA ? MATSEQSELLCUDA : MATSEQSELLHIP, MAT_INPLACE_MATRIX, &B));
  PetscCall(MatSetFromOptions(B));
  for (PetscInt stage = 0; stage < 3; ++stage) {
    if (stage == 1) PetscCall(MatScale(B, 2));
    else if (stage == 2) {
      PetscCall(MatSetOption(B, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
      for (PetscInt i = 0; i < 32; ++i) PetscCall(MatSetValue(B, i, i, 2, INSERT_VALUES));
      PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
    }
    PetscCall(MatMult(B, x, z));
    PetscCall(CheckBarrier(dctx));
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCall(VecGetArrayRead(z, &a));
    for (PetscInt i = 0; i < 32; ++i) PetscCheck(a[i] == (stage == 2 ? 2 * input[i] : 0), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Initially empty SELL matrix stage %" PetscInt_FMT " returned an incorrect entry %" PetscInt_FMT, stage, i);
    PetscCall(VecRestoreArrayRead(z, &a));
  }
  PetscCall(MatDestroy(&B));
  PetscCall(VecDestroy(&z));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscDeviceFree(dctx, input));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestDense(MPI_Comm comm, PetscDeviceType type, PetscDeviceContext dctx, const PetscScalar input[])
{
  PetscErrorCode (*mult)(Mat, Vec, Vec, PetscInt, PetscInt) = NULL;
  Mat                A;
  Vec                x, y;
  PetscScalar       *a;
  const PetscScalar *b;
  PetscMPIInt        rank, size;
  PetscInt           n;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, 4, rank ? 0 : 4, PETSC_DECIDE, 4));
  PetscCall(MatSetType(A, type == PETSC_DEVICE_CUDA ? MATDENSECUDA : MATDENSEHIP));
  PetscCall(MatSetUp(A));
  PetscCall(MatDenseGetArrayWrite(A, &a));
  for (PetscInt j = 0; j < 4; ++j)
    for (PetscInt i = 0; i < 4; ++i) a[4 * j + i] = i == j ? 2 : 0;
  PetscCall(MatDenseRestoreArrayWrite(A, &a));
  PetscCall(MatCreateVecs(A, &y, &x));
  PetscCall(VecSet(x, 1));
  PetscCall(VecSet(y, 0));
  PetscCall(PetscObjectQueryFunction((PetscObject)A, "MatMultHermitianTransposeColumnRange_C", &mult));
  PetscCheck(mult, comm, PETSC_ERR_PLIB, "Missing dense column-range multiply implementation");
  PetscCall((*mult)(A, x, y, 1, 3));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(WriteVector(y, type, dctx, input));
  PetscCall((*mult)(A, x, y, 1, 3));
  PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscCall(VecGetLocalSize(y, &n));
  PetscCall(VecGetArrayRead(y, &b));
  for (PetscInt i = 0; i < n; ++i) PetscCheck(b[i] == ((i == 1 || i == 2) ? 2 * size : input[i]), comm, PETSC_ERR_PLIB, "Dense column-range multiply returned an incorrect entry %" PetscInt_FMT, i);
  PetscCall(VecRestoreArrayRead(y, &b));
  PetscCall(MatDenseGetArrayRead(A, &b));
  for (PetscInt j = 0; j < 4; ++j)
    for (PetscInt i = 0; i < 4; ++i) PetscCheck(b[4 * j + i] == (i == j ? 2 : 0), comm, PETSC_ERR_PLIB, "Incorrect dense array entry");
  PetscCall(MatDenseRestoreArrayRead(A, &b));
  if (size == 1) {
    Mat B, X;
    IS  perm;

    PetscCall(ISCreateStride(comm, 4, 0, 1, &perm));
    PetscCall(MatLUFactor(A, perm, perm, NULL));
    PetscCall(MatCreateSeqDense(comm, 4, 2, NULL, &B));
    PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &X));
    PetscCall(MatDenseGetArrayWrite(B, &a));
    for (PetscInt j = 0; j < 2; ++j)
      for (PetscInt i = 0; i < 4; ++i) a[4 * j + i] = (j + 1) * input[i];
    PetscCall(MatDenseRestoreArrayWrite(B, &a));
    // Host output requires a temporary device array on the solve's stream.
    PetscCall(MatMatSolve(A, B, X));
    PetscCall(PetscDeviceContextSynchronize(dctx));
    // Change the right-hand side so a stale solution differs from the expected one.
    PetscCall(MatScale(B, 2));
    PetscCall(PetscDeviceContextDelay(dctx, 0.05));
    PetscCall(MatMatSolve(A, B, X));
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCall(MatDenseGetArrayRead(X, &b));
    for (PetscInt j = 0; j < 2; ++j)
      for (PetscInt i = 0; i < 4; ++i) PetscCheck(PetscAbsScalar(b[4 * j + i] - (j + 1) * input[i]) < PETSC_SMALL, comm, PETSC_ERR_PLIB, "Incorrect dense solve entry");
    PetscCall(MatDenseRestoreArrayRead(X, &b));
    PetscCall(MatDestroy(&X));
    PetscCall(MatDestroy(&B));
    PetscCall(ISDestroy(&perm));
  }
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if !PetscDefined(USE_COMPLEX)
static PetscErrorCode TestSELLChunks(PetscDeviceType type)
{
  Mat                A, B, C;
  Vec                x, y, z, x_ref, y_ref;
  const PetscScalar *a, *a_ref;
  PetscReal          avgwidth;
  PetscInt           maxwidth;

  PetscFunctionBeginUser;
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, 512, 512, 0, NULL, &A));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  for (PetscInt i = 0; i < 512; ++i) PetscCall(MatSetValue(A, i, i, 2, INSERT_VALUES));
  // One wide slice makes the default kernel selection use the chunk map.
  for (PetscInt j = 1; j < 256; ++j) PetscCall(MatSetValue(A, 0, j, 1, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatConvert(A, type == PETSC_DEVICE_CUDA ? MATSEQSELLCUDA : MATSEQSELLHIP, MAT_INITIAL_MATRIX, &B));
  PetscCall(MatSeqSELLGetMaxSliceWidth(B, &maxwidth));
  PetscCall(MatSeqSELLGetAvgSliceWidth(B, &avgwidth));
  PetscCheck(maxwidth / avgwidth > 12 && maxwidth / avgwidth / 512 > 0.001, PETSC_COMM_SELF, PETSC_ERR_PLIB, "SELL regression must select the chunked kernel");
  PetscCall(MatDuplicate(B, MAT_COPY_VALUES, &C));
  PetscCall(MatDestroy(&B));
  PetscCall(MatSetOption(C, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatCreateVecs(C, &x, &y));
  PetscCall(VecDuplicate(y, &z));
  PetscCall(MatCreateVecs(A, &x_ref, &y_ref));
  PetscCall(VecSet(x, 1));
  PetscCall(VecSet(x_ref, 1));
  for (PetscInt stage = 0; stage < 3; ++stage) {
    if (stage) {
      PetscCall(MatSetValue(A, 16, 17, stage, INSERT_VALUES));
      PetscCall(MatSetValue(C, 16, 17, stage, INSERT_VALUES));
      PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));
    }
    PetscCall(MatMult(A, x_ref, y_ref));
    PetscCall(MatMult(C, x, y));
    PetscCall(MatMultAdd(C, x, y, z));
    PetscCall(VecGetArrayRead(y_ref, &a_ref));
    for (PetscInt add = 0; add < 2; ++add) {
      PetscCall(VecGetArrayRead(add ? z : y, &a));
      for (PetscInt i = 0; i < 512; ++i) PetscCheck(a[i] == (add + 1) * a_ref[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "SELL chunked multiply stage %" PetscInt_FMT " add %" PetscInt_FMT " returned an incorrect entry %" PetscInt_FMT, stage, add, i);
      PetscCall(VecRestoreArrayRead(add ? z : y, &a));
    }
    PetscCall(VecRestoreArrayRead(y_ref, &a_ref));
  }
  PetscCall(VecDestroy(&x_ref));
  PetscCall(VecDestroy(&y_ref));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&z));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&C));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode TestSparseSolves(PetscDeviceType type)
{
  Mat                A, F;
  Vec                b, x;
  IS                 perm;
  MatFactorInfo      info;
  const PetscScalar *a;

  PetscFunctionBeginUser;
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, 4, 4, 1, NULL, &A));
  for (PetscInt i = 0; i < 4; ++i) PetscCall(MatSetValue(A, i, i, 2, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatConvert(A, type == PETSC_DEVICE_CUDA ? MATSEQAIJCUSPARSE : MATSEQAIJHIPSPARSE, MAT_INPLACE_MATRIX, &A));
  PetscCall(MatCreateVecs(A, &b, &x));
  PetscCall(VecSet(b, 2));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, 4, 0, 1, &perm));
  PetscCall(MatFactorInfoInitialize(&info));
  for (PetscInt kind = 0; kind < 2; ++kind) {
    PetscCall(MatGetFactor(A, type == PETSC_DEVICE_CUDA ? MATSOLVERCUSPARSE : MATSOLVERHIPSPARSE, kind ? MAT_FACTOR_CHOLESKY : MAT_FACTOR_LU, &F));
    if (kind) {
      PetscCall(MatCholeskyFactorSymbolic(F, A, perm, &info));
      PetscCall(MatCholeskyFactorNumeric(F, A, &info));
    } else {
      PetscCall(MatLUFactorSymbolic(F, A, perm, perm, &info));
      PetscCall(MatLUFactorNumeric(F, A, &info));
    }
    for (PetscInt transpose = 0; transpose < 2; ++transpose) {
      if (transpose) PetscCall(MatSolveTranspose(F, b, x));
      else PetscCall(MatSolve(F, b, x));
      PetscCall(VecGetArrayRead(x, &a));
      for (PetscInt i = 0; i < 4; ++i) PetscCheck(PetscAbsScalar(a[i] - 1) <= 10 * PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect sparse solve at entry %" PetscInt_FMT, i);
      PetscCall(VecRestoreArrayRead(x, &a));
    }
    PetscCall(MatDestroy(&F));
  }
  PetscCall(ISDestroy(&perm));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestTransposeUpdate(PetscDeviceType type)
{
  Mat                A;
  Vec                x, y;
  PetscScalar       *values;
  const PetscScalar *a;
  const PetscScalar  ones[] = {1, 1, 1, 1};

  PetscFunctionBeginUser;
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, 4, 4, 1, NULL, &A));
  for (PetscInt i = 0; i < 4; ++i) PetscCall(MatSetValue(A, i, i, 2, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatConvert(A, type == PETSC_DEVICE_CUDA ? MATSEQAIJCUSPARSE : MATSEQAIJHIPSPARSE, MAT_INPLACE_MATRIX, &A));
  PetscCall(MatSetOption(A, MAT_FORM_EXPLICIT_TRANSPOSE, PETSC_TRUE));
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(VecSet(x, 2));
  PetscCall(MatMultTranspose(A, x, y));
#if PetscDefined(HAVE_CUDA)
  if (type == PETSC_DEVICE_CUDA) {
    PetscCall(MatSeqAIJCUSPARSEGetArray(A, &values));
    PetscCallCUDA(cudaMemcpy(values, ones, sizeof(ones), cudaMemcpyHostToDevice));
    PetscCall(MatSeqAIJCUSPARSERestoreArray(A, &values));
  }
#endif
#if PetscDefined(HAVE_HIP)
  if (type == PETSC_DEVICE_HIP) {
    PetscCall(MatSeqAIJHIPSPARSEGetArray(A, &values));
    PetscCallHIP(hipMemcpy(values, ones, sizeof(ones), hipMemcpyHostToDevice));
    PetscCall(MatSeqAIJHIPSPARSERestoreArray(A, &values));
  }
#endif
  PetscCall(MatMultTranspose(A, x, y));
  PetscCall(VecGetArrayRead(y, &a));
  for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Transpose multiply used stale values at entry %" PetscInt_FMT, i);
  PetscCall(VecRestoreArrayRead(y, &a));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat                   A, B;
  PetscDeviceContext    saved, current;
  PetscDevice           device;
  PetscDeviceType       type;
  PetscScalar          *input;
  const PetscScalar    *a;
  const PetscStreamType streams[] = {PETSC_STREAM_NONBLOCKING, PETSC_STREAM_NONBLOCKING_WITH_BARRIER};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscDeviceContextGetCurrentContext(&saved));
  PetscCall(PetscDeviceContextGetDevice(saved, &device));
  PetscCall(PetscDeviceGetType(device, &type));
#if !PetscDefined(USE_COMPLEX)
  PetscCall(TestSELLChunks(type));
#endif
  PetscCall(TestSparseSolves(type));
  PetscCall(TestTransposeUpdate(type));
  PetscCall(PetscDeviceMalloc(saved, PETSC_MEMTYPE_HOST, 4, PETSC_DECIDE, &input));
  PetscCall(PetscDeviceContextSynchronize(saved));
  for (PetscInt i = 0; i < 4; ++i) input[i] = i + 1;
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, 4, 4, 1, NULL, &A));
  for (PetscInt i = 0; i < 4; ++i) PetscCall(MatSetValue(A, i, i, 1, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatConvert(A, type == PETSC_DEVICE_CUDA ? MATSEQAIJCUSPARSE : MATSEQAIJHIPSPARSE, MAT_INPLACE_MATRIX, &A));
  PetscCall(WriteValues(A, type, saved, input, PETSC_FALSE));
  PetscCall(PetscDeviceContextSynchronize(saved));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  PetscCall(PetscDeviceContextSynchronize(saved));

  for (size_t k = 0; k < PETSC_STATIC_ARRAY_LENGTH(streams); ++k) {
    PetscCall(PetscDeviceContextDuplicate(saved, &current));
    PetscCall(PetscDeviceContextSetStreamType(current, streams[k]));
    PetscCall(PetscDeviceContextSetUp(current));
    PetscCall(PetscDeviceContextSetCurrentContext(current));
    // Warm the zeroing operation before delaying the input write.
    PetscCall(MatZeroEntries(A));
    PetscCall(PetscDeviceContextSynchronize(current));
    PetscCall(WriteValues(A, type, current, input, PETSC_TRUE));
    PetscCall(MatZeroEntries(A));
    PetscCall(PetscDeviceContextSynchronize(current));
    PetscCall(MatSeqAIJGetArrayRead(A, &a));
    for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stream type %s: CSR entry %" PetscInt_FMT " is %g instead of zero", PetscStreamTypes[streams[k]], i, (double)PetscRealPart(a[i]));
    PetscCall(MatSeqAIJRestoreArrayRead(A, &a));

    // Warm the BLAS handle and scaling kernel before checking completion on return.
    PetscCall(MatScale(A, 2));
    PetscCall(PetscDeviceContextSynchronize(current));
    PetscCall(WriteValues(A, type, current, input, PETSC_TRUE));
    PetscCall(MatScale(A, 2));
    if (streams[k] == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
      PetscBool idle;

      PetscCall(PetscDeviceContextQueryIdle(current, &idle));
      PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatScale() returned with work pending on a barrier context");
    }
    PetscCall(PetscDeviceContextSynchronize(current));
    PetscCall(MatSeqAIJGetArrayRead(A, &a));
    for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == 2 * input[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stream type %s: incorrectly scaled CSR entry %" PetscInt_FMT, PetscStreamTypes[streams[k]], i);
    PetscCall(MatSeqAIJRestoreArrayRead(A, &a));

    // Warm AXPY before delaying the update of its source matrix.
    PetscCall(MatAXPY(A, 2, B, SAME_NONZERO_PATTERN));
    // Clear B so an AXPY that overtakes the delayed write produces a wrong result.
    PetscCall(MatZeroEntries(B));
    PetscCall(PetscDeviceContextSynchronize(current));
    PetscCall(WriteValues(A, type, current, input, PETSC_FALSE));
    PetscCall(WriteValues(B, type, current, input, PETSC_TRUE));
    PetscCall(MatAXPY(A, 2, B, SAME_NONZERO_PATTERN));
    if (streams[k] == PETSC_STREAM_NONBLOCKING_WITH_BARRIER) {
      PetscBool idle;

      PetscCall(PetscDeviceContextQueryIdle(current, &idle));
      PetscCheck(idle, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatAXPY() returned with work pending on a barrier context");
    }
    PetscCall(PetscDeviceContextSynchronize(current));
    PetscCall(MatSeqAIJGetArrayRead(A, &a));
    for (PetscInt i = 0; i < 4; ++i) PetscCheck(a[i] == 3 * input[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stream type %s: incorrect CSR entry %" PetscInt_FMT " after MatAXPY()", PetscStreamTypes[streams[k]], i);
    PetscCall(MatSeqAIJRestoreArrayRead(A, &a));
    PetscCall(TestCopies(A, type, current, input));
    PetscCall(TestGetDiagonal(A, type, current, input));
    PetscCall(TestDiagonalScale(A, type, current, input));
    PetscCall(TestGetIJ(type, current));
    PetscCall(TestSparseOperations(A, type, current, input));
    PetscCall(TestSparseSolve(type, saved, current));
    PetscCall(TestSELL(type, current));
    PetscCall(TestDense(PETSC_COMM_SELF, type, current, input));
    PetscCall(TestDense(PETSC_COMM_WORLD, type, current, input));
    PetscCall(PetscDeviceContextSetCurrentContext(saved));
    PetscCall(PetscDeviceContextDestroy(&current));
  }

  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscDeviceFree(saved, input));
  PetscCall(PetscDeviceContextSynchronize(saved));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/empty.out
    nsize: {{1 2}}

    test:
      suffix: cuda
      requires: cuda
      args: -mat_sell_spmv_cuda_kernel 1

    test:
      suffix: hip
      requires: hip
      args: -mat_sell_spmv_hip_kernel 1

TEST*/
