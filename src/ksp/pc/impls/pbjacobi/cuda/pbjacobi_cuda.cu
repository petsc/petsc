#include <petscdevice_cuda.h>
#include <petsc/private/petsclegacycupmblas.h>
#include <../src/ksp/pc/impls/pbjacobi/pbjacobi.h>

#if PETSC_PKG_CUDA_VERSION_LT(11, 7, 0)
__global__ static void MatMultBatched(PetscInt bs, PetscInt mbs, const PetscScalar *A, const PetscScalar *x, PetscScalar *y, PetscBool transpose)
{
  const PetscInt gridSize = gridDim.x * blockDim.x;
  PetscInt       row      = blockIdx.x * blockDim.x + threadIdx.x;
  const PetscInt bs2      = bs * bs;

  /* One row per thread. The blocks are stored in column-major order */
  for (; row < bs * mbs; row += gridSize) {
    const PetscScalar *Ap, *xp;
    PetscScalar       *yp;
    PetscInt           i, j, k;

    k  = row / bs;                               /* k-th block */
    i  = row % bs;                               /* this thread deals with i-th row of the block */
    Ap = &A[bs2 * k + i * (transpose ? bs : 1)]; /* Ap points to the first entry of i-th row */
    xp = &x[bs * k];
    yp = &y[bs * k];
    /* multiply i-th row (column) with x */
    yp[i] = 0.0;
    for (j = 0; j < bs; j++) {
      yp[i] += Ap[0] * xp[j];
      Ap += (transpose ? 1 : bs); /* block is in column major order */
    }
  }
}
#endif

static PetscErrorCode PCApplyOrTranspose_PBJacobi_CUDA(PC pc, Vec x, Vec y, cublasOperation_t op)
{
  const PetscScalar *xx;
  PetscScalar       *yy;
  cublasHandle_t     handle;
  PC_PBJacobi       *jac = (PC_PBJacobi *)pc->data;
  const PetscScalar *A   = (const PetscScalar *)jac->spptr;
  const PetscInt     bs = jac->bs, mbs = jac->mbs;

  PetscFunctionBegin;
  PetscCall(VecCUDAGetArrayRead(x, &xx));
  PetscCall(VecCUDAGetArrayWrite(y, &yy));
  PetscCall(PetscCUBLASGetHandle(&handle));
  PetscCallCUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST)); /* alpha, beta are on host */

#if PETSC_PKG_CUDA_VERSION_GE(11, 7, 0)
  /* y = alpha op(A) x + beta y */
  const PetscScalar alpha = 1.0, beta = 0.0;
  PetscCallCUBLAS(cublasXgemvStridedBatched(handle, op, bs, bs, &alpha, A, bs, bs * bs, xx, 1, bs, &beta, yy, 1, bs, mbs));
#else
  PetscInt gridSize = PetscMin((bs * mbs + 255) / 256, 2147483647); /* <= 2^31-1 */
  MatMultBatched<<<gridSize, 256>>>(bs, mbs, A, xx, yy, (op == CUBLAS_OP_T ? PETSC_TRUE : PETSC_FALSE));
  PetscCallCUDA(cudaGetLastError());
#endif
  PetscCall(VecCUDARestoreArrayRead(x, &xx));
  PetscCall(VecCUDARestoreArrayWrite(y, &yy));
  PetscCall(PetscLogGpuFlops(bs * bs * mbs * 2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_PBJacobi_CUDA(PC pc, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(PCApplyOrTranspose_PBJacobi_CUDA(pc, x, y, CUBLAS_OP_N)); // No transpose
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyTranspose_PBJacobi_CUDA(PC pc, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(PCApplyOrTranspose_PBJacobi_CUDA(pc, x, y, CUBLAS_OP_T)); // Transpose
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_PBJacobi_CUDA(PC pc)
{
  PC_PBJacobi *jac = (PC_PBJacobi *)pc->data;

  PetscFunctionBegin;
  PetscCallCUDA(cudaFree(jac->spptr));
  PetscCall(PCDestroy_PBJacobi(pc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PCSetUp_PBJacobi_CUDA(PC pc)
{
  PC_PBJacobi *jac = (PC_PBJacobi *)pc->data;
  size_t       size;

  PetscFunctionBegin;
  PetscCall(PCSetUp_PBJacobi_Host(pc)); /* Compute the inverse on host now. Might worth doing it on device directly */
  size = sizeof(PetscScalar) * jac->bs * jac->bs * jac->mbs;

  /* PBJacobi_CUDA is simple so that we use jac->spptr as if it is diag_d */
  if (!jac->spptr) PetscCallCUDAVoid(cudaMalloc(&jac->spptr, size));
  PetscCallCUDAVoid(cudaMemcpy(jac->spptr, jac->diag, size, cudaMemcpyHostToDevice));
  PetscCall(PetscLogCpuToGpu(size));

  pc->ops->apply          = PCApply_PBJacobi_CUDA;
  pc->ops->applytranspose = PCApplyTranspose_PBJacobi_CUDA;
  pc->ops->destroy        = PCDestroy_PBJacobi_CUDA;
  PetscFunctionReturn(PETSC_SUCCESS);
}
