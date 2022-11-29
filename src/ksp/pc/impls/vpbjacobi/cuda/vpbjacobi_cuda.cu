#include <petscdevice_cuda.h>
#include <../src/ksp/pc/impls/vpbjacobi/vpbjacobi.h>

/* A class that manages helper arrays assisting parallel PCApply() with CUDA */
struct PC_VPBJacobi_CUDA {
  /* Cache the old sizes to check if we need realloc */
  PetscInt n;       /* number of rows of the local matrix */
  PetscInt nblocks; /* number of point blocks */
  PetscInt nsize;   /* sum of sizes of the point blocks */

  /* Helper arrays that are pre-computed on host and then copied to device.
    bs:     [nblocks+1], "csr" version of bsizes[], with bs[0] = 0, bs[nblocks] = n.
    bs2:    [nblocks+1], "csr" version of squares of bsizes[], with bs2[0] = 0, bs2[nblocks] = nsize.
    matIdx: [n], row i of the local matrix belongs to the matIdx_d[i] block
  */
  PetscInt *bs_h, *bs2_h, *matIdx_h;
  PetscInt *bs_d, *bs2_d, *matIdx_d;

  MatScalar *diag_d; /* [nsize], store inverse of the point blocks on device */

  PC_VPBJacobi_CUDA(PetscInt n, PetscInt nblocks, PetscInt nsize, const PetscInt *bsizes, MatScalar *diag_h) : n(n), nblocks(nblocks), nsize(nsize)
  {
    /* malloc memory on host and device, and then update */
    PetscCallVoid(PetscMalloc3(nblocks + 1, &bs_h, nblocks + 1, &bs2_h, n, &matIdx_h));
    PetscCallCUDAVoid(cudaMalloc(&bs_d, sizeof(PetscInt) * (nblocks + 1)));
    PetscCallCUDAVoid(cudaMalloc(&bs2_d, sizeof(PetscInt) * (nblocks + 1)));
    PetscCallCUDAVoid(cudaMalloc(&matIdx_d, sizeof(PetscInt) * n));
    PetscCallCUDAVoid(cudaMalloc(&diag_d, sizeof(MatScalar) * nsize));
    PetscCallVoid(UpdateOffsetsOnDevice(bsizes, diag_h));
  }

  PetscErrorCode UpdateOffsetsOnDevice(const PetscInt *bsizes, MatScalar *diag_h)
  {
    PetscFunctionBegin;
    PetscCall(ComputeOffsetsOnHost(bsizes));
    PetscCallCUDA(cudaMemcpy(bs_d, bs_h, sizeof(PetscInt) * (nblocks + 1), cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(bs2_d, bs2_h, sizeof(PetscInt) * (nblocks + 1), cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(matIdx_d, matIdx_h, sizeof(PetscInt) * n, cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(diag_d, diag_h, sizeof(MatScalar) * nsize, cudaMemcpyHostToDevice));
    PetscCall(PetscLogCpuToGpu(sizeof(PetscInt) * (2 * nblocks + 2 + n) + sizeof(MatScalar) * nsize));
    PetscFunctionReturn(0);
  }

  ~PC_VPBJacobi_CUDA()
  {
    PetscCallVoid(PetscFree3(bs_h, bs2_h, matIdx_h));
    PetscCallCUDAVoid(cudaFree(bs_d));
    PetscCallCUDAVoid(cudaFree(bs2_d));
    PetscCallCUDAVoid(cudaFree(matIdx_d));
    PetscCallCUDAVoid(cudaFree(diag_d));
  }

private:
  PetscErrorCode ComputeOffsetsOnHost(const PetscInt *bsizes)
  {
    PetscFunctionBegin;
    bs_h[0] = bs2_h[0] = 0;
    for (PetscInt i = 0; i < nblocks; i++) {
      bs_h[i + 1]  = bs_h[i] + bsizes[i];
      bs2_h[i + 1] = bs2_h[i] + bsizes[i] * bsizes[i];
      for (PetscInt j = 0; j < bsizes[i]; j++) matIdx_h[bs_h[i] + j] = i;
    }
    PetscFunctionReturn(0);
  }
};

/* Like cublasDgemvBatched() but with variable-sized blocks

  Input Parameters:
+ n       - number of rows of the local matrix
. bs      - [nblocks+1], prefix sum of bsizes[]
. bs2     - [nblocks+1], prefix sum of squares of bsizes[]
. matIdx  - [n], store block/matrix index for each row
. A       - blocks of the matrix back to back in column-major order
- x       - the input vector

  Output Parameter:
. y - the output vector
*/
__global__ static void MatMultBatched(PetscInt n, const PetscInt *bs, const PetscInt *bs2, const PetscInt *matIdx, const MatScalar *A, const PetscScalar *x, PetscScalar *y)
{
  const PetscInt gridSize = gridDim.x * blockDim.x;
  PetscInt       tid      = blockIdx.x * blockDim.x + threadIdx.x;
  PetscInt       i, j, k, m;

  /* One row per thread. The blocks/matrices are stored in column-major order */
  for (; tid < n; tid += gridSize) {
    k = matIdx[tid];       /* k-th block */
    m = bs[k + 1] - bs[k]; /* block size of the k-th block */
    i = tid - bs[k];       /* i-th row of the block */
    A += bs2[k] + i;       /* advance A to the first entry of i-th row */
    x += bs[k];
    y += bs[k];

    y[i] = 0.0;
    for (j = 0; j < m; j++) {
      y[i] += A[0] * x[j];
      A += m;
    }
  }
}

static PetscErrorCode PCApply_VPBJacobi_CUDA(PC pc, Vec x, Vec y)
{
  PC_VPBJacobi      *jac   = (PC_VPBJacobi *)pc->data;
  PC_VPBJacobi_CUDA *pcuda = static_cast<PC_VPBJacobi_CUDA *>(jac->spptr);
  const PetscScalar *xx;
  PetscScalar       *yy;
  PetscInt           n;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuTimeBegin());
  if (PetscDefined(USE_DEBUG)) {
    PetscBool isCuda;
    PetscCall(PetscObjectTypeCompareAny((PetscObject)x, &isCuda, VECSEQCUDA, VECMPICUDA, ""));
    if (isCuda) PetscCall(PetscObjectTypeCompareAny((PetscObject)y, &isCuda, VECSEQCUDA, VECMPICUDA, ""));
    PetscCheck(isCuda, PETSC_COMM_SELF, PETSC_ERR_SUP, "PC: applying a CUDA pmat to non-cuda vectors");
  }

  PetscCall(MatGetLocalSize(pc->pmat, &n, NULL));
  if (n) {
    PetscInt gridSize = PetscMin((n + 255) / 256, 2147483647); /* <= 2^31-1 */
    PetscCall(VecCUDAGetArrayRead(x, &xx));
    PetscCall(VecCUDAGetArrayWrite(y, &yy));
    MatMultBatched<<<gridSize, 256>>>(n, pcuda->bs_d, pcuda->bs2_d, pcuda->matIdx_d, pcuda->diag_d, xx, yy);
    PetscCallCUDA(cudaGetLastError());
    PetscCall(VecCUDARestoreArrayRead(x, &xx));
    PetscCall(VecCUDARestoreArrayWrite(y, &yy));
  }
  PetscCall(PetscLogGpuFlops(pcuda->nsize * 2)); /* FMA on entries in all blocks */
  PetscCall(PetscLogGpuTimeEnd());
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_VPBJacobi_CUDA(PC pc)
{
  PC_VPBJacobi *jac = (PC_VPBJacobi *)pc->data;

  PetscFunctionBegin;
  PetscCallCXX(delete static_cast<PC_VPBJacobi_CUDA *>(jac->spptr));
  PCDestroy_VPBJacobi(pc);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PCSetUp_VPBJacobi_CUDA(PC pc)
{
  PC_VPBJacobi      *jac   = (PC_VPBJacobi *)pc->data;
  PC_VPBJacobi_CUDA *pcuda = static_cast<PC_VPBJacobi_CUDA *>(jac->spptr);
  PetscInt           i, n, nblocks, nsize = 0;
  const PetscInt    *bsizes;

  PetscFunctionBegin;
  PetscCall(PCSetUp_VPBJacobi_Host(pc)); /* Compute the inverse on host now. Might worth doing it on device directly */
  PetscCall(MatGetVariableBlockSizes(pc->pmat, &nblocks, &bsizes));
  for (i = 0; i < nblocks; i++) nsize += bsizes[i] * bsizes[i];
  PetscCall(MatGetLocalSize(pc->pmat, &n, NULL));

  /* If one calls MatSetVariableBlockSizes() multiple times and sizes have been changed (is it allowed?), we delete the old and rebuild anyway */
  if (pcuda && (pcuda->n != n || pcuda->nblocks != nblocks || pcuda->nsize != nsize)) {
    PetscCallCXX(delete pcuda);
    pcuda = nullptr;
  }

  if (!pcuda) { /* allocate the struct along with the helper arrays from the scatch */
    PetscCallCXX(jac->spptr = new PC_VPBJacobi_CUDA(n, nblocks, nsize, bsizes, jac->diag));
  } else { /* update the value only */
    PetscCall(pcuda->UpdateOffsetsOnDevice(bsizes, jac->diag));
  }

  pc->ops->apply   = PCApply_VPBJacobi_CUDA;
  pc->ops->destroy = PCDestroy_VPBJacobi_CUDA;
  PetscFunctionReturn(0);
}
