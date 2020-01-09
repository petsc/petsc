#include <cuda_runtime.h>

#include <petscdevice_cuda.h>
#include <../src/mat/impls/sell/seq/sell.h> /*I   "petscmat.h"  I*/

typedef struct {
  PetscInt  *colidx; /* column index */
  MatScalar *val;
  PetscInt  *sliidx;
  PetscInt   nonzerostate;
} Mat_SeqSELLCUDA;

static PetscErrorCode MatSeqSELLCUDA_Destroy(Mat_SeqSELLCUDA **cudastruct)
{
  PetscFunctionBegin;
  if (*cudastruct) {
    if ((*cudastruct)->colidx) { PetscCallCUDA(cudaFree((*cudastruct)->colidx)); }
    if ((*cudastruct)->val) { PetscCallCUDA(cudaFree((*cudastruct)->val)); }
    if ((*cudastruct)->sliidx) { PetscCallCUDA(cudaFree((*cudastruct)->sliidx)); }
    PetscCall(PetscFree(*cudastruct));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqSELLCUDACopyToGPU(Mat A)
{
  Mat_SeqSELLCUDA *cudastruct = (Mat_SeqSELLCUDA *)A->spptr;
  Mat_SeqSELL     *a          = (Mat_SeqSELL *)A->data;

  PetscFunctionBegin;
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    PetscCall(PetscLogEventBegin(MAT_CUDACopyToGPU, A, 0, 0, 0));
    if (A->assembled && A->nonzerostate == cudastruct->nonzerostate) {
      /* copy values only */
      PetscCallCUDA(cudaMemcpy(cudastruct->val, a->val, a->sliidx[a->totalslices] * sizeof(MatScalar), cudaMemcpyHostToDevice));
      PetscCall(PetscLogCpuToGpu(a->sliidx[a->totalslices] * (sizeof(MatScalar))));
    } else {
      if (cudastruct->colidx) { PetscCallCUDA(cudaFree(cudastruct->colidx)); }
      if (cudastruct->val) { PetscCallCUDA(cudaFree(cudastruct->val)); }
      if (cudastruct->sliidx) { PetscCallCUDA(cudaFree(cudastruct->sliidx)); }
      PetscCallCUDA(cudaMalloc((void **)&(cudastruct->sliidx), (a->totalslices + 1) * sizeof(PetscInt)));
      PetscCallCUDA(cudaMalloc((void **)&(cudastruct->colidx), a->maxallocmat * sizeof(PetscInt)));
      PetscCallCUDA(cudaMalloc((void **)&(cudastruct->val), a->maxallocmat * sizeof(MatScalar)));
      /* copy values, nz or maxallocmat? */
      PetscCallCUDA(cudaMemcpy(cudastruct->sliidx, a->sliidx, (a->totalslices + 1) * sizeof(PetscInt), cudaMemcpyHostToDevice));
      PetscCallCUDA(cudaMemcpy(cudastruct->colidx, a->colidx, a->sliidx[a->totalslices] * sizeof(PetscInt), cudaMemcpyHostToDevice));
      PetscCallCUDA(cudaMemcpy(cudastruct->val, a->val, a->sliidx[a->totalslices] * sizeof(MatScalar), cudaMemcpyHostToDevice));
      PetscCall(PetscLogCpuToGpu(a->sliidx[a->totalslices] * (sizeof(MatScalar) + sizeof(PetscInt)) + (a->totalslices + 1) * sizeof(PetscInt)));
      cudastruct->nonzerostate = A->nonzerostate;
    }
    PetscCallCUDA(WaitForCUDA());
    PetscCall(PetscLogEventEnd(MAT_CUDACopyToGPU, A, 0, 0, 0));
    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

__global__ void matmult_seqsell_basic_kernel(PetscInt nrows, PetscInt totalslices, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, PetscScalar *y)
{
  PetscInt  i, row, slice_id, row_in_slice;
  MatScalar sum;
  /* one thread per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / SLICE_HEIGHT;
    row_in_slice = row % SLICE_HEIGHT;
    if (slice_id < totalslices) {
      sum = 0.0;
      for (i = sliidx[slice_id] + row_in_slice; i < sliidx[slice_id + 1]; i += SLICE_HEIGHT) sum += aval[i] * x[acolidx[i]];
      if (slice_id == totalslices - 1 && nrows % SLICE_HEIGHT) { /* if last slice has padding rows */
        if (row_in_slice < (nrows % SLICE_HEIGHT)) y[row] = sum;
      } else {
        y[row] = sum;
      }
    }
  }
}

__global__ void matmultadd_seqsell_basic_kernel(PetscInt nrows, PetscInt totalslices, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, const PetscScalar *y, PetscScalar *z)
{
  PetscInt  i, row, slice_id, row_in_slice;
  MatScalar sum;
  /* one thread per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / SLICE_HEIGHT;
    row_in_slice = row % SLICE_HEIGHT;
    if (slice_id < totalslices) {
      sum = 0.0;
      for (i = sliidx[slice_id] + row_in_slice; i < sliidx[slice_id + 1]; i += SLICE_HEIGHT) sum += aval[i] * x[acolidx[i]];
      if (slice_id == totalslices - 1 && nrows % SLICE_HEIGHT) { /* if last slice has padding rows */
        if (row_in_slice < (nrows % SLICE_HEIGHT)) z[row] = y[row] + sum;
      } else {
        z[row] = y[row] + sum;
      }
    }
  }
}

__global__ void matmult_seqsell_tiled_kernel(PetscInt nrows, PetscInt totalslices, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, PetscScalar *y)
{
  __shared__ MatScalar shared[256];
  PetscInt             i, row, slice_id, row_in_slice;
  /* one thread per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / SLICE_HEIGHT;
    row_in_slice = row % SLICE_HEIGHT;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + SLICE_HEIGHT * threadIdx.y; i < sliidx[slice_id + 1]; i += SLICE_HEIGHT * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
    if (blockDim.y > 4) {
      __syncthreads();
      if (threadIdx.y < 4) { shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 4) * blockDim.x + threadIdx.x]; }
    }
    if (blockDim.y > 2) {
      __syncthreads();
      if (threadIdx.y < 2) { shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 2) * blockDim.x + threadIdx.x]; }
    }
    if (blockDim.y > 1) {
      __syncthreads();
      if (threadIdx.y < 1) { shared[threadIdx.x] += shared[blockDim.x + threadIdx.x]; }
    }
    if (threadIdx.y < 1) {
      if (slice_id == totalslices - 1 && nrows % SLICE_HEIGHT) { /* if last slice has padding rows */
        if (row_in_slice < (nrows % SLICE_HEIGHT)) y[row] = shared[threadIdx.x];
      } else {
        y[row] = shared[threadIdx.x];
      }
    }
  }
}

__global__ void matmultadd_seqsell_tiled_kernel(PetscInt nrows, PetscInt totalslices, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, const PetscScalar *y, PetscScalar *z)
{
  __shared__ MatScalar shared[256];
  PetscInt             i, row, slice_id, row_in_slice;
  /* one thread per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / SLICE_HEIGHT;
    row_in_slice = row % SLICE_HEIGHT;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + SLICE_HEIGHT * threadIdx.y; i < sliidx[slice_id + 1]; i += SLICE_HEIGHT * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
    if (blockDim.y > 4) {
      __syncthreads();
      if (threadIdx.y < 4) { shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 4) * blockDim.x + threadIdx.x]; }
    }
    if (blockDim.y > 2) {
      __syncthreads();
      if (threadIdx.y < 2) { shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 2) * blockDim.x + threadIdx.x]; }
    }
    if (blockDim.y > 1) {
      __syncthreads();
      if (threadIdx.y < 1) { shared[threadIdx.x] += shared[blockDim.x + threadIdx.x]; }
    }
    if (threadIdx.y < 1) {
      if (slice_id == totalslices - 1 && nrows % SLICE_HEIGHT) { /* if last slice has padding rows */
        if (row_in_slice < (nrows % SLICE_HEIGHT)) z[row] = y[row] + shared[threadIdx.x];
      } else {
        z[row] = y[row] + shared[threadIdx.x];
      }
    }
  }
}

PetscErrorCode MatMult_SeqSELLCUDA(Mat A, Vec xx, Vec yy)
{
  Mat_SeqSELL       *a          = (Mat_SeqSELL *)A->data;
  Mat_SeqSELLCUDA   *cudastruct = (Mat_SeqSELLCUDA *)A->spptr;
  PetscScalar       *y;
  const PetscScalar *x;
  PetscInt           totalslices = a->totalslices, nrows = A->rmap->n;
  MatScalar         *aval;
  PetscInt          *acolidx;
  PetscInt          *sliidx;
  PetscInt           nblocks, blocksize = 256;

  PetscFunctionBegin;
  PetscCall(MatSeqSELLCUDACopyToGPU(A));
  /* cudastruct may not be available until MatSeqSELLCUDACopyToGPU() is called */
  aval    = cudastruct->val;
  acolidx = cudastruct->colidx;
  sliidx  = cudastruct->sliidx;

  PetscCall(VecCUDAGetArrayRead(xx, &x));
  PetscCall(VecCUDAGetArrayWrite(yy, &y));
  PetscCall(PetscLogGpuTimeBegin());
  nblocks = (nrows + blocksize - 1) / blocksize;
  if (nblocks >= 80) {
    matmult_seqsell_basic_kernel<<<nblocks, blocksize>>>(nrows, totalslices, acolidx, aval, sliidx, x, y);
  } else {
    PetscInt avg_width;
    dim3     block1(256, 1), block2(128, 2), block4(64, 4), block8(32, 8);
    avg_width = a->sliidx[a->totalslices] / (SLICE_HEIGHT * a->totalslices);
    if (avg_width > 64) {
      matmult_seqsell_tiled_kernel<<<nblocks * 8, block8>>>(nrows, totalslices, acolidx, aval, sliidx, x, y);
    } else if (avg_width > 32) {
      matmult_seqsell_tiled_kernel<<<nblocks * 4, block4>>>(nrows, totalslices, acolidx, aval, sliidx, x, y);
    } else if (avg_width > 16) {
      matmult_seqsell_tiled_kernel<<<nblocks * 2, block2>>>(nrows, totalslices, acolidx, aval, sliidx, x, y);
    } else {
      matmult_seqsell_tiled_kernel<<<nblocks, block1>>>(nrows, totalslices, acolidx, aval, sliidx, x, y);
    }
  }
  PetscCallCUDA(WaitForCUDA());
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(VecCUDARestoreArrayRead(xx, &x));
  PetscCall(VecCUDARestoreArrayWrite(yy, &y));
  PetscCall(PetscLogGpuFlops(2.0 * a->nz - a->nonzerorowcnt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultAdd_SeqSELLCUDA(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqSELL       *a          = (Mat_SeqSELL *)A->data;
  Mat_SeqSELLCUDA   *cudastruct = (Mat_SeqSELLCUDA *)A->spptr;
  PetscScalar       *z;
  const PetscScalar *y, *x;
  PetscInt           totalslices = a->totalslices, nrows = A->rmap->n;
  MatScalar         *aval    = cudastruct->val;
  PetscInt          *acolidx = cudastruct->colidx;
  PetscInt          *sliidx  = cudastruct->sliidx;

  PetscFunctionBegin;
  PetscCall(MatSeqSELLCUDACopyToGPU(A));
  if (a->nz) {
    PetscInt nblocks, blocksize = 256;
    PetscCall(VecCUDAGetArrayRead(xx, &x));
    PetscCall(VecCUDAGetArrayRead(yy, &y));
    PetscCall(VecCUDAGetArrayWrite(zz, &z));
    PetscCall(PetscLogGpuTimeBegin());
    nblocks = (nrows + blocksize - 1) / blocksize;
    if (nblocks >= 80) {
      matmultadd_seqsell_basic_kernel<<<nblocks, blocksize>>>(nrows, totalslices, acolidx, aval, sliidx, x, y, z);
    } else {
      PetscInt avg_width;
      dim3     block1(256, 1), block2(128, 2), block4(64, 4), block8(32, 8);
      avg_width = a->sliidx[a->totalslices] / (SLICE_HEIGHT * a->totalslices);
      if (avg_width > 64) {
        matmultadd_seqsell_tiled_kernel<<<nblocks * 8, block8>>>(nrows, totalslices, acolidx, aval, sliidx, x, y, z);
      } else if (avg_width > 32) {
        matmultadd_seqsell_tiled_kernel<<<nblocks * 4, block4>>>(nrows, totalslices, acolidx, aval, sliidx, x, y, z);
      } else if (avg_width > 16) {
        matmultadd_seqsell_tiled_kernel<<<nblocks * 2, block2>>>(nrows, totalslices, acolidx, aval, sliidx, x, y, z);
      } else {
        matmultadd_seqsell_tiled_kernel<<<nblocks, block1>>>(nrows, totalslices, acolidx, aval, sliidx, x, y, z);
      }
    }
    PetscCallCUDA(WaitForCUDA());
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(VecCUDARestoreArrayRead(xx, &x));
    PetscCall(VecCUDARestoreArrayRead(yy, &y));
    PetscCall(VecCUDARestoreArrayWrite(zz, &z));
    PetscCall(PetscLogGpuFlops(2.0 * a->nz));
  } else {
    PetscCall(VecCopy(yy, zz));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_SeqSELLCUDA(Mat A, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "SeqSELLCUDA options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAssemblyEnd_SeqSELLCUDA(Mat A, MatAssemblyType mode)
{
  PetscFunctionBegin;
  PetscCall(MatAssemblyEnd_SeqSELL(A, mode));
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(PETSC_SUCCESS);
  if (A->factortype == MAT_FACTOR_NONE) { PetscCall(MatSeqSELLCUDACopyToGPU(A)); }
  A->ops->mult    = MatMult_SeqSELLCUDA;
  A->ops->multadd = MatMultAdd_SeqSELLCUDA;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_SeqSELLCUDA(Mat A)
{
  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    if (A->offloadmask != PETSC_OFFLOAD_UNALLOCATED) { PetscCall(MatSeqSELLCUDA_Destroy((Mat_SeqSELLCUDA **)&A->spptr)); }
  }
  PetscCall(MatDestroy_SeqSELL(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqSELL_SeqSELLCUDA(Mat);
static PetscErrorCode       MatDuplicate_SeqSELLCUDA(Mat A, MatDuplicateOption cpvalues, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatDuplicate_SeqSELL(A, cpvalues, B));
  PetscCall(MatConvert_SeqSELL_SeqSELLCUDA(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode MatConvert_SeqSELL_SeqSELLCUDA(Mat B)
{
  Mat_SeqSELLCUDA *cudastruct;

  PetscFunctionBegin;
  PetscCall(PetscFree(B->defaultvectype));
  PetscCall(PetscStrallocpy(VECCUDA, &B->defaultvectype));

  if (!B->spptr) {
    if (B->factortype == MAT_FACTOR_NONE) {
      PetscCall(PetscNew(&cudastruct));
      B->spptr = cudastruct;
    }
  }

  B->ops->assemblyend    = MatAssemblyEnd_SeqSELLCUDA;
  B->ops->destroy        = MatDestroy_SeqSELLCUDA;
  B->ops->setfromoptions = MatSetFromOptions_SeqSELLCUDA;
  B->ops->mult           = MatMult_SeqSELLCUDA;
  B->ops->multadd        = MatMultAdd_SeqSELLCUDA;
  B->ops->duplicate      = MatDuplicate_SeqSELLCUDA;

  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQSELLCUDA));
  B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqSELLCUDA(Mat B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate_SeqSELL(B));
  PetscCall(MatConvert_SeqSELL_SeqSELLCUDA(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
