static const char help[] = "Test CUDA/HIP matrix storage and value updates.\n";

#include <petscmat.h>
#include <petscdevice.h>
#if PetscDefined(HAVE_CUDA)
  #include <petscdevice_cuda.h>
#endif
#if PetscDefined(HAVE_HIP)
  #include <petscdevice_hip.h>
#endif

static PetscErrorCode TestSELL(PetscDeviceType type, PetscDeviceContext dctx)
{
  Mat                A, B;
  Vec                x, y, z;
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
  PetscCall(PetscDeviceContextSynchronize(dctx));
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
  PetscCall(VecDestroy(&z));
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
  PetscDeviceContext dctx;
  PetscDeviceType    type;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscDeviceContextGetCurrentContext(&dctx));
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &type));
  PetscCall(TestSELL(type, dctx));
#if !PetscDefined(USE_COMPLEX)
  PetscCall(TestSELLChunks(type));
#endif
  PetscCall(TestSparseSolves(type));
  PetscCall(TestTransposeUpdate(type));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    nsize: {{1 2}}
    output_file: output/empty.out
    test:
      suffix: cuda
      requires: cuda
      args: -default_device_type cuda -mat_sell_spmv_cuda_kernel 1
    test:
      suffix: hip
      requires: hip
      args: -default_device_type hip -mat_sell_spmv_hip_kernel 1

TEST*/
