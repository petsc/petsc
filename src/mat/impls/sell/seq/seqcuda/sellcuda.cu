#include <petscdevice_cuda.h>
#include <petsc/private/cupmatomics.hpp>
#include <../src/mat/impls/sell/seq/sell.h> /*I   "petscmat.h"  I*/

#define SLICE_HEIGHT 16

typedef struct {
  PetscInt   maxallocmat;
  PetscInt   totalentries;
  PetscInt  *colidx; /* column index array, device pointer */
  MatScalar *val;    /* value array, device pointer */
  PetscInt   totalslices;
  PetscInt  *sliidx; /* slice index array, device pointer */
  PetscInt   nonzerostate;
  PetscInt   kernelchoice;
  PetscInt   blocky;
  PetscInt   chunksperblock;
  PetscInt   totalchunks;
  PetscInt  *chunk_slice_map; /* starting slice for each chunk, device pointer */
} Mat_SeqSELLCUDA;

static PetscErrorCode MatSeqSELLCUDA_Destroy(Mat_SeqSELLCUDA **cudastruct)
{
  PetscFunctionBegin;
  if (*cudastruct) {
    if ((*cudastruct)->colidx) PetscCallCUDA(cudaFree((*cudastruct)->colidx));
    if ((*cudastruct)->val) PetscCallCUDA(cudaFree((*cudastruct)->val));
    if ((*cudastruct)->sliidx) PetscCallCUDA(cudaFree((*cudastruct)->sliidx));
    if ((*cudastruct)->chunk_slice_map) PetscCallCUDA(cudaFree((*cudastruct)->chunk_slice_map));
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
      if (cudastruct->colidx) PetscCallCUDA(cudaFree(cudastruct->colidx));
      if (cudastruct->val) PetscCallCUDA(cudaFree(cudastruct->val));
      if (cudastruct->sliidx) PetscCallCUDA(cudaFree(cudastruct->sliidx));
      if (cudastruct->chunk_slice_map) PetscCallCUDA(cudaFree(cudastruct->chunk_slice_map));
      cudastruct->maxallocmat  = a->maxallocmat;
      cudastruct->totalentries = a->sliidx[a->totalslices];
      cudastruct->totalslices  = a->totalslices;
      cudastruct->totalchunks  = a->totalchunks;
      PetscCallCUDA(cudaMalloc((void **)&cudastruct->colidx, a->maxallocmat * sizeof(*cudastruct->colidx)));
      PetscCallCUDA(cudaMalloc((void **)&cudastruct->val, a->maxallocmat * sizeof(*cudastruct->val)));
      /* copy values, nz or maxallocmat? */
      PetscCallCUDA(cudaMemcpy(cudastruct->colidx, a->colidx, a->sliidx[a->totalslices] * sizeof(*a->colidx), cudaMemcpyHostToDevice));
      PetscCallCUDA(cudaMemcpy(cudastruct->val, a->val, a->sliidx[a->totalslices] * sizeof(*a->val), cudaMemcpyHostToDevice));

      PetscCallCUDA(cudaMalloc((void **)&cudastruct->sliidx, (a->totalslices + 1) * sizeof(*cudastruct->sliidx)));
      PetscCallCUDA(cudaMemcpy(cudastruct->sliidx, a->sliidx, (a->totalslices + 1) * sizeof(*a->sliidx), cudaMemcpyHostToDevice));
      PetscCallCUDA(cudaMalloc((void **)&cudastruct->chunk_slice_map, a->totalchunks * sizeof(*cudastruct->chunk_slice_map)));
      PetscCallCUDA(cudaMemcpy(cudastruct->chunk_slice_map, a->chunk_slice_map, a->totalchunks * sizeof(*a->chunk_slice_map), cudaMemcpyHostToDevice));
      PetscCall(PetscLogCpuToGpu(a->sliidx[a->totalslices] * (sizeof(MatScalar) + sizeof(PetscInt)) + (a->totalslices + 1 + a->totalchunks) * sizeof(PetscInt)));
    }
    PetscCallCUDA(WaitForCUDA());
    PetscCall(PetscLogEventEnd(MAT_CUDACopyToGPU, A, 0, 0, 0));
    A->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static __global__ void matmult_seqsell_basic_kernel(PetscInt nrows, PetscInt sliceheight, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, PetscScalar *y)
{
  PetscInt  i, row, slice_id, row_in_slice;
  MatScalar sum;
  /* one thread per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / sliceheight;
    row_in_slice = row % sliceheight;
    sum          = 0.0;
    for (i = sliidx[slice_id] + row_in_slice; i < sliidx[slice_id + 1]; i += sliceheight) sum += aval[i] * x[acolidx[i]];
    y[row] = sum;
  }
}

static __global__ void matmultadd_seqsell_basic_kernel(PetscInt nrows, PetscInt sliceheight, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, const PetscScalar *y, PetscScalar *z)
{
  PetscInt  i, row, slice_id, row_in_slice;
  MatScalar sum;
  /* one thread per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / sliceheight;
    row_in_slice = row % sliceheight;
    sum          = 0.0;
    for (i = sliidx[slice_id] + row_in_slice; i < sliidx[slice_id + 1]; i += sliceheight) sum += aval[i] * x[acolidx[i]];
    z[row] = y[row] + sum;
  }
}

#if !defined(PETSC_USE_COMPLEX)
/* use 1 block per slice, suitable for large slice width */
template <int BLOCKY>
__global__ void matmult_seqsell_tiled_kernel9(PetscInt nrows, PetscInt sliceheight, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, PetscScalar *y)
{
  __shared__ MatScalar shared[32][BLOCKY];
  PetscInt             i, row, slice_id = blockIdx.x;
  int                  tid = threadIdx.x + threadIdx.y * 32;
  /* transposed index */
  int         tidx = tid % BLOCKY;
  int         tidy = tid / BLOCKY;
  PetscScalar t    = 0.0;

  row = slice_id * sliceheight + threadIdx.x % sliceheight;
  if (row < nrows) {
    for (i = sliidx[slice_id] + threadIdx.x + 32 * threadIdx.y; i < sliidx[slice_id + 1]; i += 32 * BLOCKY) t += aval[i] * x[acolidx[i]];
  }
  #pragma unroll
  for (int offset = 16; offset >= sliceheight; offset /= 2) t += __shfl_down_sync(0xffffffff, t, offset);
  /* transpose layout to reduce each row using warp shfl */
  if (threadIdx.x < sliceheight) shared[threadIdx.x][threadIdx.y] = t;
  __syncthreads();
  if (tidy < sliceheight) t = shared[tidy][tidx];
  #pragma unroll
  for (int offset = BLOCKY / 2; offset > 0; offset /= 2) t += __shfl_down_sync(0xffffffff, t, offset, BLOCKY);
  if (tidx == 0 && tidy < sliceheight) shared[0][tidy] = t;
  __syncthreads();
  if (row < nrows && threadIdx.y == 0 && threadIdx.x < sliceheight) y[row] = shared[0][threadIdx.x];
}

/* use 1 block per slice, suitable for large slice width */
template <int BLOCKY>
__global__ void matmultadd_seqsell_tiled_kernel9(PetscInt nrows, PetscInt sliceheight, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, const PetscScalar *y, PetscScalar *z)
{
  __shared__ MatScalar shared[32][BLOCKY];
  PetscInt             i, row, slice_id = blockIdx.x;
  int                  tid = threadIdx.x + threadIdx.y * 32;
  /* transposed index */
  int         tidx = tid % BLOCKY;
  int         tidy = tid / BLOCKY;
  PetscScalar t    = 0.0;

  row = slice_id * sliceheight + threadIdx.x % sliceheight;
  if (row < nrows) {
    for (i = sliidx[slice_id] + threadIdx.x + 32 * threadIdx.y; i < sliidx[slice_id + 1]; i += 32 * BLOCKY) t += aval[i] * x[acolidx[i]];
  }
  #pragma unroll
  for (int offset = 16; offset >= sliceheight; offset /= 2) t += __shfl_down_sync(0xffffffff, t, offset);
  /* transpose layout to reduce each row using warp shfl */
  if (threadIdx.x < sliceheight) shared[threadIdx.x][threadIdx.y] = t;
  __syncthreads();
  if (tidy < sliceheight) t = shared[tidy][tidx];
  #pragma unroll
  for (int offset = BLOCKY / 2; offset > 0; offset /= 2) t += __shfl_down_sync(0xffffffff, t, offset, BLOCKY);
  if (tidx == 0 && tidy < sliceheight) shared[0][tidy] = t;
  __syncthreads();
  if (row < nrows && threadIdx.y == 0 && threadIdx.x < sliceheight) z[row] = y[row] + shared[0][threadIdx.x];
}

template <int BLOCKY>
__device__ __forceinline__ static bool segment_scan(PetscInt flag[], MatScalar shared[], PetscScalar *val)
{
  bool head = true;
  #pragma unroll
  for (int i = 1; i < BLOCKY * 2; i <<= 1) {
    int halfwarpid                         = threadIdx.y * 2 + threadIdx.x / 16;
    shared[threadIdx.x + threadIdx.y * 32] = 0;
    if (halfwarpid >= i && flag[halfwarpid - i] == flag[halfwarpid]) {
      shared[threadIdx.x + threadIdx.y * 32] = *val;
      if (i == 1) head = false;
    }
    __syncthreads();
    if (halfwarpid < BLOCKY * 2 - i) *val += shared[threadIdx.x + threadIdx.y * 32 + i * 16];
    __syncthreads();
  }
  return head;
}

/* load-balancing version. Chunksize is equal to the number of threads per block */
template <int BLOCKY>
__global__ void matmult_seqsell_tiled_kernel8(PetscInt nrows, PetscInt sliceheight, PetscInt chunksperblock, PetscInt totalchunks, const PetscInt *chunk_slice_map, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, PetscScalar *y)
{
  __shared__ MatScalar shared[BLOCKY * 32];
  PetscInt             gid, row, start_slice, cid;
  PetscScalar          t = 0.0;
  AtomicAdd<MatScalar> atomAdd;
  /* zero out y */
  for (int iter = 0; iter < 1 + (nrows - 1) / (gridDim.x * 32 * BLOCKY); iter++) {
    gid = gridDim.x * 32 * BLOCKY * iter + blockIdx.x * BLOCKY * 32 + threadIdx.y * 32 + threadIdx.x;
    if (gid < nrows) y[gid] = 0.0;
  }
  for (int iter = 0; iter < chunksperblock; iter++) {
    cid = blockIdx.x * chunksperblock + iter; /* chunk id */
    if (cid < totalchunks) {
      start_slice = chunk_slice_map[cid]; /* starting slice at each iteration */
      gid         = cid * BLOCKY * 32 + threadIdx.y * 32 + threadIdx.x;
      if ((cid + 1) * BLOCKY * 32 > sliidx[start_slice + 1]) { /* this iteration covers more than one slice */
        __shared__ PetscInt flag[BLOCKY * 2];
        bool                write;
        PetscInt            slice_id = start_slice, totalslices = PetscCeilIntMacro(nrows, sliceheight), totalentries = sliidx[totalslices];
        /* find out the slice that this element belongs to */
        while (gid < totalentries && gid >= sliidx[slice_id + 1]) slice_id++;
        if (threadIdx.x % 16 == 0) flag[threadIdx.y * 2 + threadIdx.x / 16] = slice_id;
        row = slice_id * sliceheight + threadIdx.x % sliceheight;
        if (row < nrows && gid < totalentries) t = aval[gid] * x[acolidx[gid]];
        __syncthreads();
        write = segment_scan<BLOCKY>(flag, shared, &t);
        if (row < nrows && gid < totalentries && write) atomAdd(y[row], t);
        t = 0.0;
      } else { /* this iteration covers only one slice */
        row = start_slice * sliceheight + threadIdx.x % sliceheight;
        if (row < nrows) t += aval[gid] * x[acolidx[gid]];
        if (iter == chunksperblock - 1 || (cid + 2) * BLOCKY * 32 > sliidx[start_slice + 1]) { /* last iteration or next iteration covers more than one slice */
          int tid = threadIdx.x + threadIdx.y * 32, tidx = tid % BLOCKY, tidy = tid / BLOCKY;
  /* reduction and write to output vector */
  #pragma unroll
          for (int offset = 16; offset >= sliceheight; offset /= 2) t += __shfl_down_sync(0xffffffff, t, offset);
          /* transpose layout to reduce each row using warp shfl */
          if (threadIdx.x < sliceheight) shared[threadIdx.x * BLOCKY + threadIdx.y] = t; /* shared[threadIdx.x][threadIdx.y] = t */
          __syncthreads();
          if (tidy < sliceheight) t = shared[tidy * BLOCKY + tidx]; /* shared[tidy][tidx] */
  #pragma unroll
          for (int offset = BLOCKY / 2; offset > 0; offset /= 2) t += __shfl_down_sync(0xffffffff, t, offset, BLOCKY);
          if (tidx == 0 && tidy < sliceheight) shared[tidy] = t; /* shared[0][tidy] = t */
          __syncthreads();
          if (row < nrows && threadIdx.y == 0 && threadIdx.x < sliceheight) atomAdd(y[row], shared[threadIdx.x]); /* shared[0][threadIdx.x] */
          t = 0.0;
        }
      }
    }
  }
}

/* load-balancing version. Chunksize is equal to the number of threads per block */
template <int BLOCKY>
__global__ void matmultadd_seqsell_tiled_kernel8(PetscInt nrows, PetscInt sliceheight, PetscInt chunksperblock, PetscInt totalchunks, const PetscInt *chunk_slice_map, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, const PetscScalar *y, PetscScalar *z)
{
  __shared__ MatScalar shared[BLOCKY * 32];
  PetscInt             gid, row, start_slice, cid;
  PetscScalar          t = 0.0;
  AtomicAdd<MatScalar> atomAdd;
  /* copy y to z */
  for (int iter = 0; iter < 1 + (nrows - 1) / (gridDim.x * 32 * BLOCKY); iter++) {
    gid = gridDim.x * 32 * BLOCKY * iter + blockIdx.x * BLOCKY * 32 + threadIdx.y * 32 + threadIdx.x;
    if (gid < nrows) z[gid] = y[gid];
  }
  for (int iter = 0; iter < chunksperblock; iter++) {
    cid = blockIdx.x * chunksperblock + iter; /* chunk id */
    if (cid < totalchunks) {
      start_slice = chunk_slice_map[cid]; /* starting slice at each iteration */
      gid         = cid * BLOCKY * 32 + threadIdx.y * 32 + threadIdx.x;
      if ((cid + 1) * BLOCKY * 32 > sliidx[start_slice + 1]) { /* this iteration covers more than one slice */
        __shared__ PetscInt flag[BLOCKY * 2];
        bool                write;
        PetscInt            slice_id = start_slice, totalslices = PetscCeilIntMacro(nrows, sliceheight), totalentries = sliidx[totalslices];
        /* find out the slice that this element belongs to */
        while (gid < totalentries && gid >= sliidx[slice_id + 1]) slice_id++;
        if (threadIdx.x % 16 == 0) flag[threadIdx.y * 2 + threadIdx.x / 16] = slice_id;
        row = slice_id * sliceheight + threadIdx.x % sliceheight;
        if (row < nrows && gid < totalentries) t = aval[gid] * x[acolidx[gid]];
        __syncthreads();
        write = segment_scan<BLOCKY>(flag, shared, &t);
        if (row < nrows && gid < totalentries && write) atomAdd(z[row], t);
        t = 0.0;
      } else { /* this iteration covers only one slice */
        row = start_slice * sliceheight + threadIdx.x % sliceheight;
        if (row < nrows) t += aval[gid] * x[acolidx[gid]];
        if (iter == chunksperblock - 1 || (cid + 2) * BLOCKY * 32 > sliidx[start_slice + 1]) { /* last iteration or next iteration covers more than one slice */
          int tid = threadIdx.x + threadIdx.y * 32, tidx = tid % BLOCKY, tidy = tid / BLOCKY;
  /* reduction and write to output vector */
  #pragma unroll
          for (int offset = 16; offset >= sliceheight; offset /= 2) t += __shfl_down_sync(0xffffffff, t, offset);
          /* transpose layout to reduce each row using warp shfl */
          if (threadIdx.x < sliceheight) shared[threadIdx.x * BLOCKY + threadIdx.y] = t; /* shared[threadIdx.x][threadIdx.y] = t */
          __syncthreads();
          if (tidy < sliceheight) t = shared[tidy * BLOCKY + tidx]; /* shared[tidy][tidx] */
  #pragma unroll
          for (int offset = BLOCKY / 2; offset > 0; offset /= 2) t += __shfl_down_sync(0xffffffff, t, offset, BLOCKY);
          if (tidx == 0 && tidy < sliceheight) shared[tidy] = t; /* shared[0][tidy] = t */
          __syncthreads();
          if (row < nrows && threadIdx.y == 0 && threadIdx.x < sliceheight) atomAdd(z[row], shared[threadIdx.x]); /* shared[0][threadIdx.x] */
          t = 0.0;
        }
      }
    }
  }
}

/* use 1 warp per slice, suitable for small slice width */
static __global__ void matmult_seqsell_tiled_kernel7(PetscInt nrows, PetscInt sliceheight, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, PetscScalar *y)
{
  PetscInt i, row, slice_id;
  slice_id = blockIdx.x * blockDim.y + threadIdx.y;
  row      = slice_id * sliceheight + threadIdx.x % sliceheight;
  double t = 0.0;
  if (row < nrows) {
    for (i = sliidx[slice_id] + threadIdx.x; i < sliidx[slice_id + 1]; i += 32) t += aval[i] * x[acolidx[i]];
  }
  #pragma unroll
  for (int offset = 16; offset >= sliceheight; offset /= 2) t += __shfl_down_sync(0xffffffff, t, offset);
  if (row < nrows && threadIdx.x < sliceheight) y[row] = t;
}

/* use 1 warp per slice, suitable for small slice width */
static __global__ void matmultadd_seqsell_tiled_kernel7(PetscInt nrows, PetscInt sliceheight, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, const PetscScalar *y, PetscScalar *z)
{
  PetscInt i, row, slice_id;
  slice_id = blockIdx.x * blockDim.y + threadIdx.y;
  row      = slice_id * sliceheight + threadIdx.x % sliceheight;
  double t = 0.0;
  if (row < nrows) {
    for (i = sliidx[slice_id] + threadIdx.x; i < sliidx[slice_id + 1]; i += 32) t += aval[i] * x[acolidx[i]];
  }
  #pragma unroll
  for (int offset = 16; offset >= sliceheight; offset /= 2) t += __shfl_down_sync(0xffffffff, t, offset);
  if (row < nrows && threadIdx.x < sliceheight) z[row] = y[row] + t;
}
#endif

/***********  Kernel 2-6  are tied to slice height 16. They are kept only for performance comparison  **********/

static __global__ void matmult_seqsell_tiled_kernel6(PetscInt nrows, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, PetscScalar *y)
{
  __shared__ MatScalar shared[512];
  PetscInt             i, row, slice_id, row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / SLICE_HEIGHT;
    row_in_slice = row % SLICE_HEIGHT;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + SLICE_HEIGHT * threadIdx.y; i < sliidx[slice_id + 1]; i += SLICE_HEIGHT * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 16) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 16) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 8) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 8) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 4) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 4) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 2) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 2) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x + threadIdx.x];
      y[row] = shared[threadIdx.x];
    }
  }
}

static __global__ void matmult_seqsell_tiled_kernel5(PetscInt nrows, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, PetscScalar *y)
{
  __shared__ MatScalar shared[512];
  PetscInt             i, row, slice_id, row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / SLICE_HEIGHT;
    row_in_slice = row % SLICE_HEIGHT;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + SLICE_HEIGHT * threadIdx.y; i < sliidx[slice_id + 1]; i += SLICE_HEIGHT * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 8) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 8) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 4) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 4) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 2) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 2) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x + threadIdx.x];
      y[row] = shared[threadIdx.x];
    }
  }
}

static __global__ void matmult_seqsell_tiled_kernel4(PetscInt nrows, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, PetscScalar *y)
{
  __shared__ MatScalar shared[512];
  PetscInt             i, row, slice_id, row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / SLICE_HEIGHT;
    row_in_slice = row % SLICE_HEIGHT;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + SLICE_HEIGHT * threadIdx.y; i < sliidx[slice_id + 1]; i += SLICE_HEIGHT * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 4) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 4) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 2) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 2) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x + threadIdx.x];
      y[row] = shared[threadIdx.x];
    }
  }
}

static __global__ void matmult_seqsell_tiled_kernel3(PetscInt nrows, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, PetscScalar *y)
{
  __shared__ MatScalar shared[512];
  PetscInt             i, row, slice_id, row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / SLICE_HEIGHT;
    row_in_slice = row % SLICE_HEIGHT;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + SLICE_HEIGHT * threadIdx.y; i < sliidx[slice_id + 1]; i += SLICE_HEIGHT * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 2) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 2) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x + threadIdx.x];
      y[row] = shared[threadIdx.x];
    }
  }
}

static __global__ void matmult_seqsell_tiled_kernel2(PetscInt nrows, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, PetscScalar *y)
{
  __shared__ MatScalar shared[512];
  PetscInt             i, row, slice_id, row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / SLICE_HEIGHT;
    row_in_slice = row % SLICE_HEIGHT;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + SLICE_HEIGHT * threadIdx.y; i < sliidx[slice_id + 1]; i += SLICE_HEIGHT * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x + threadIdx.x];
      y[row] = shared[threadIdx.x];
    }
  }
}

static __global__ void matmultadd_seqsell_tiled_kernel6(PetscInt nrows, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, const PetscScalar *y, PetscScalar *z)
{
  __shared__ MatScalar shared[512];
  PetscInt             i, row, slice_id, row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / SLICE_HEIGHT;
    row_in_slice = row % SLICE_HEIGHT;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + SLICE_HEIGHT * threadIdx.y; i < sliidx[slice_id + 1]; i += SLICE_HEIGHT * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 16) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 16) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 8) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 8) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 4) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 4) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 2) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 2) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x + threadIdx.x];
      z[row] = y[row] + shared[threadIdx.x];
    }
  }
}

static __global__ void matmultadd_seqsell_tiled_kernel5(PetscInt nrows, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, const PetscScalar *y, PetscScalar *z)
{
  __shared__ MatScalar shared[512];
  PetscInt             i, row, slice_id, row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / SLICE_HEIGHT;
    row_in_slice = row % SLICE_HEIGHT;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + SLICE_HEIGHT * threadIdx.y; i < sliidx[slice_id + 1]; i += SLICE_HEIGHT * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 8) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 8) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 4) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 4) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 2) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 2) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x + threadIdx.x];
      z[row] = y[row] + shared[threadIdx.x];
    }
  }
}

static __global__ void matmultadd_seqsell_tiled_kernel4(PetscInt nrows, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, const PetscScalar *y, PetscScalar *z)
{
  __shared__ MatScalar shared[512];
  PetscInt             i, row, slice_id, row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / SLICE_HEIGHT;
    row_in_slice = row % SLICE_HEIGHT;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + SLICE_HEIGHT * threadIdx.y; i < sliidx[slice_id + 1]; i += SLICE_HEIGHT * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 4) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 4) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 2) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 2) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x + threadIdx.x];
      z[row] = y[row] + shared[threadIdx.x];
    }
  }
}

static __global__ void matmultadd_seqsell_tiled_kernel3(PetscInt nrows, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, const PetscScalar *y, PetscScalar *z)
{
  __shared__ MatScalar shared[512];
  PetscInt             i, row, slice_id, row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / SLICE_HEIGHT;
    row_in_slice = row % SLICE_HEIGHT;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + SLICE_HEIGHT * threadIdx.y; i < sliidx[slice_id + 1]; i += SLICE_HEIGHT * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 2) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 2) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x + threadIdx.x];
      z[row] = y[row] + shared[threadIdx.x];
    }
  }
}

static __global__ void matmultadd_seqsell_tiled_kernel2(PetscInt nrows, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, const PetscScalar *y, PetscScalar *z)
{
  __shared__ MatScalar shared[512];
  PetscInt             i, row, slice_id, row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / SLICE_HEIGHT;
    row_in_slice = row % SLICE_HEIGHT;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + SLICE_HEIGHT * threadIdx.y; i < sliidx[slice_id + 1]; i += SLICE_HEIGHT * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x + threadIdx.x];
      z[row] = y[row] + shared[threadIdx.x];
    }
  }
}

static PetscErrorCode MatMult_SeqSELLCUDA(Mat A, Vec xx, Vec yy)
{
  Mat_SeqSELL       *a          = (Mat_SeqSELL *)A->data;
  Mat_SeqSELLCUDA   *cudastruct = (Mat_SeqSELLCUDA *)A->spptr;
  PetscScalar       *y;
  const PetscScalar *x;
  PetscInt           nrows = A->rmap->n, sliceheight = a->sliceheight;
  MatScalar         *aval;
  PetscInt          *acolidx;
  PetscInt          *sliidx;
  PetscInt           nblocks, blocksize = 512; /* blocksize must be multiple of SLICE_HEIGHT*32 */
  dim3               block2(256, 2), block4(128, 4), block8(64, 8), block16(32, 16), block32(16, 32);
#if !defined(PETSC_USE_COMPLEX)
  PetscInt  chunksperblock, nchunks, *chunk_slice_map;
  PetscReal maxoveravg;
#endif

  PetscFunctionBegin;
  PetscCheck(32 % sliceheight == 0, PETSC_COMM_SELF, PETSC_ERR_SUP, "The kernel requires a slice height be a divisor of 32, but the input matrix has a slice height of %" PetscInt_FMT, sliceheight);
  PetscCheck(!(cudastruct->kernelchoice >= 2 && cudastruct->kernelchoice <= 6 && sliceheight != SLICE_HEIGHT), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Kernel choices {2-6} requires the slice height of the matrix be 16, but the current slice height is %" PetscInt_FMT, sliceheight);
  PetscCall(MatSeqSELLCUDACopyToGPU(A));
  /* cudastruct may not be available until MatSeqSELLCUDACopyToGPU() is called */
  aval    = cudastruct->val;
  acolidx = cudastruct->colidx;
  sliidx  = cudastruct->sliidx;

  PetscCall(VecCUDAGetArrayRead(xx, &x));
  PetscCall(VecCUDAGetArrayWrite(yy, &y));
  PetscCall(PetscLogGpuTimeBegin());

  switch (cudastruct->kernelchoice) {
#if !defined(PETSC_USE_COMPLEX)
  case 9:
    nblocks = 1 + (nrows - 1) / sliceheight;
    if (cudastruct->blocky == 2) {
      matmult_seqsell_tiled_kernel9<2><<<nblocks, dim3(32, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    } else if (cudastruct->blocky == 4) {
      matmult_seqsell_tiled_kernel9<4><<<nblocks, dim3(32, 4)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    } else if (cudastruct->blocky == 8) {
      matmult_seqsell_tiled_kernel9<8><<<nblocks, dim3(32, 8)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    } else if (cudastruct->blocky == 16) {
      matmult_seqsell_tiled_kernel9<16><<<nblocks, dim3(32, 16)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    } else if (cudastruct->blocky == 32) {
      matmult_seqsell_tiled_kernel9<32><<<nblocks, dim3(32, 32)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    } else {
      matmult_seqsell_tiled_kernel9<2><<<nblocks, dim3(32, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    }
    break;
  case 7:
    nblocks = 1 + (nrows - 1) / (2 * sliceheight);
    if (cudastruct->blocky == 2) {
      matmult_seqsell_tiled_kernel7<<<nblocks, dim3(32, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    } else if (cudastruct->blocky == 4) {
      matmult_seqsell_tiled_kernel7<<<nblocks, dim3(32, 4)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    } else if (cudastruct->blocky == 8) {
      matmult_seqsell_tiled_kernel7<<<nblocks, dim3(32, 8)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    } else if (cudastruct->blocky == 16) {
      matmult_seqsell_tiled_kernel7<<<nblocks, dim3(32, 16)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    } else if (cudastruct->blocky == 32) {
      matmult_seqsell_tiled_kernel7<<<nblocks, dim3(32, 32)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    } else {
      matmult_seqsell_tiled_kernel7<<<nblocks, dim3(32, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    }
    break;
#endif
  case 6:
    nblocks = 1 + (nrows - 1) / (blocksize / 32); /* 1 slice per block if blocksize=512 */
    matmult_seqsell_tiled_kernel6<<<nblocks, block32>>>(nrows, acolidx, aval, sliidx, x, y);
    break;
  case 5:
    nblocks = 1 + (nrows - 1) / (blocksize / 16); /* 2 slices per block if blocksize=512*/
    matmult_seqsell_tiled_kernel5<<<nblocks, block16>>>(nrows, acolidx, aval, sliidx, x, y);
    break;
  case 4:
    nblocks = 1 + (nrows - 1) / (blocksize / 8); /* 4 slices per block if blocksize=512 */
    matmult_seqsell_tiled_kernel4<<<nblocks, block8>>>(nrows, acolidx, aval, sliidx, x, y);
    break;
  case 3:
    nblocks = 1 + (nrows - 1) / (blocksize / 4); /* 8 slices per block if blocksize=512 */
    matmult_seqsell_tiled_kernel3<<<nblocks, block4>>>(nrows, acolidx, aval, sliidx, x, y);
    break;
  case 2: /* 16 slices per block if blocksize=512 */
    nblocks = 1 + (nrows - 1) / (blocksize / 2);
    matmult_seqsell_tiled_kernel2<<<nblocks, block2>>>(nrows, acolidx, aval, sliidx, x, y);
    break;
  case 1: /* 32 slices per block if blocksize=512 */
    nblocks = 1 + (nrows - 1) / blocksize;
    matmult_seqsell_basic_kernel<<<nblocks, blocksize>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    break;
#if !defined(PETSC_USE_COMPLEX)
  case 0:
    maxoveravg = a->maxslicewidth / a->avgslicewidth;
    if (maxoveravg > 12.0 && maxoveravg / nrows > 0.001) { /* important threshold */
      /* each block handles approximately one slice */
      PetscInt blocky = a->chunksize / 32;
      nchunks         = cudastruct->totalchunks;
      chunksperblock  = cudastruct->chunksperblock ? cudastruct->chunksperblock : 1 + (cudastruct->totalentries / cudastruct->totalslices - 1) / a->chunksize;
      nblocks         = 1 + (nchunks - 1) / chunksperblock;
      chunk_slice_map = cudastruct->chunk_slice_map;
      if (blocky == 2) {
        matmult_seqsell_tiled_kernel8<2><<<nblocks, dim3(32, 2)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y);
      } else if (blocky == 4) {
        matmult_seqsell_tiled_kernel8<4><<<nblocks, dim3(32, 4)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y);
      } else if (blocky == 8) {
        matmult_seqsell_tiled_kernel8<8><<<nblocks, dim3(32, 8)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y);
      } else if (blocky == 16) {
        matmult_seqsell_tiled_kernel8<16><<<nblocks, dim3(32, 16)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y);
      } else if (blocky == 32) {
        matmult_seqsell_tiled_kernel8<32><<<nblocks, dim3(32, 32)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y);
      } else {
        matmult_seqsell_tiled_kernel8<2><<<nblocks, dim3(32, 2)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y);
      }
    } else {
      PetscInt avgslicesize = sliceheight * a->avgslicewidth;
      if (avgslicesize <= 432) {
        if (sliceheight * a->maxslicewidth < 2048 && nrows > 100000) {
          nblocks = 1 + (nrows - 1) / (2 * sliceheight); /* two slices per block */
          matmult_seqsell_tiled_kernel7<<<nblocks, dim3(32, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
        } else {
          nblocks = 1 + (nrows - 1) / sliceheight;
          matmult_seqsell_tiled_kernel9<2><<<nblocks, dim3(32, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
        }
      } else if (avgslicesize <= 2400) {
        nblocks = 1 + (nrows - 1) / sliceheight;
        matmult_seqsell_tiled_kernel9<8><<<nblocks, dim3(32, 8)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
      } else {
        nblocks = 1 + (nrows - 1) / sliceheight;
        matmult_seqsell_tiled_kernel9<16><<<nblocks, dim3(32, 16)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
      }
    }
    break;
#endif
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "unsupported kernel choice %" PetscInt_FMT " for MatMult_SeqSELLCUDA.", cudastruct->kernelchoice);
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(VecCUDARestoreArrayRead(xx, &x));
  PetscCall(VecCUDARestoreArrayWrite(yy, &y));
  PetscCall(PetscLogGpuFlops(2.0 * a->nz - a->nonzerorowcnt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultAdd_SeqSELLCUDA(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqSELL       *a          = (Mat_SeqSELL *)A->data;
  Mat_SeqSELLCUDA   *cudastruct = (Mat_SeqSELLCUDA *)A->spptr;
  PetscScalar       *z;
  const PetscScalar *y, *x;
  PetscInt           nrows = A->rmap->n, sliceheight = a->sliceheight;
  MatScalar         *aval    = cudastruct->val;
  PetscInt          *acolidx = cudastruct->colidx;
  PetscInt          *sliidx  = cudastruct->sliidx;
#if !defined(PETSC_USE_COMPLEX)
  PetscReal maxoveravg;
  PetscInt  chunksperblock, nchunks, *chunk_slice_map;
  PetscInt  blocky = cudastruct->blocky;
#endif

  PetscFunctionBegin;
  PetscCheck(32 % sliceheight == 0, PETSC_COMM_SELF, PETSC_ERR_SUP, "The kernel requires a slice height be a divisor of 32, but the input matrix has a slice height of %" PetscInt_FMT, sliceheight);
  PetscCheck(!(cudastruct->kernelchoice >= 2 && cudastruct->kernelchoice <= 6 && sliceheight != SLICE_HEIGHT), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Kernel choices {2-6} requires the slice height of the matrix be 16, but the current slice height is %" PetscInt_FMT, sliceheight);
  PetscCall(MatSeqSELLCUDACopyToGPU(A));
  if (a->nz) {
    PetscInt nblocks, blocksize = 512;
    dim3     block2(256, 2), block4(128, 4), block8(64, 8), block16(32, 16), block32(16, 32);
    PetscCall(VecCUDAGetArrayRead(xx, &x));
    PetscCall(VecCUDAGetArrayRead(yy, &y));
    PetscCall(VecCUDAGetArrayWrite(zz, &z));
    PetscCall(PetscLogGpuTimeBegin());

    switch (cudastruct->kernelchoice) {
#if !defined(PETSC_USE_COMPLEX)
    case 9:
      nblocks = 1 + (nrows - 1) / sliceheight;
      if (blocky == 2) {
        matmultadd_seqsell_tiled_kernel9<2><<<nblocks, dim3(32, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 4) {
        matmultadd_seqsell_tiled_kernel9<4><<<nblocks, dim3(32, 4)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 8) {
        matmultadd_seqsell_tiled_kernel9<8><<<nblocks, dim3(32, 8)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 16) {
        matmultadd_seqsell_tiled_kernel9<16><<<nblocks, dim3(32, 16)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 32) {
        matmultadd_seqsell_tiled_kernel9<32><<<nblocks, dim3(32, 32)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      } else {
        matmultadd_seqsell_tiled_kernel9<2><<<nblocks, dim3(32, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      }
      break;
    case 8:
      /* each block handles approximately one slice */
      nchunks         = cudastruct->totalchunks;
      blocky          = a->chunksize / 32;
      chunksperblock  = cudastruct->chunksperblock ? cudastruct->chunksperblock : 1 + (cudastruct->totalentries / cudastruct->totalslices - 1) / a->chunksize;
      nblocks         = 1 + (nchunks - 1) / chunksperblock;
      chunk_slice_map = cudastruct->chunk_slice_map;
      if (blocky == 2) {
        matmultadd_seqsell_tiled_kernel8<2><<<nblocks, dim3(32, 2)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 4) {
        matmultadd_seqsell_tiled_kernel8<4><<<nblocks, dim3(32, 4)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 8) {
        matmultadd_seqsell_tiled_kernel8<8><<<nblocks, dim3(32, 8)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 16) {
        matmultadd_seqsell_tiled_kernel8<16><<<nblocks, dim3(32, 16)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 32) {
        matmultadd_seqsell_tiled_kernel8<32><<<nblocks, dim3(32, 32)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
      } else {
        matmultadd_seqsell_tiled_kernel8<2><<<nblocks, dim3(32, 2)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
      }
      break;
    case 7:
      nblocks = 1 + (nrows - 1) / (2 * sliceheight);
      if (blocky == 2) {
        matmultadd_seqsell_tiled_kernel7<<<nblocks, dim3(32, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 4) {
        matmultadd_seqsell_tiled_kernel7<<<nblocks, dim3(32, 4)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 8) {
        matmultadd_seqsell_tiled_kernel7<<<nblocks, dim3(32, 8)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 16) {
        matmultadd_seqsell_tiled_kernel7<<<nblocks, dim3(32, 16)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 32) {
        matmultadd_seqsell_tiled_kernel7<<<nblocks, dim3(32, 32)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      } else {
        matmultadd_seqsell_tiled_kernel7<<<nblocks, dim3(32, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      }
      break;
#endif
    case 6:
      nblocks = 1 + (nrows - 1) / (blocksize / 32);
      matmultadd_seqsell_tiled_kernel6<<<nblocks, block32>>>(nrows, acolidx, aval, sliidx, x, y, z);
      break;
    case 5:
      nblocks = 1 + (nrows - 1) / (blocksize / 16);
      matmultadd_seqsell_tiled_kernel5<<<nblocks, block16>>>(nrows, acolidx, aval, sliidx, x, y, z);
      break;
    case 4:
      nblocks = 1 + (nrows - 1) / (blocksize / 8);
      matmultadd_seqsell_tiled_kernel4<<<nblocks, block8>>>(nrows, acolidx, aval, sliidx, x, y, z);
      break;
    case 3:
      nblocks = 1 + (nrows - 1) / (blocksize / 4);
      matmultadd_seqsell_tiled_kernel3<<<nblocks, block4>>>(nrows, acolidx, aval, sliidx, x, y, z);
      break;
    case 2:
      nblocks = 1 + (nrows - 1) / (blocksize / 2);
      matmultadd_seqsell_tiled_kernel2<<<nblocks, block2>>>(nrows, acolidx, aval, sliidx, x, y, z);
      break;
    case 1:
      nblocks = 1 + (nrows - 1) / blocksize;
      matmultadd_seqsell_basic_kernel<<<nblocks, blocksize>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      break;
#if !defined(PETSC_USE_COMPLEX)
    case 0:
      maxoveravg = a->maxslicewidth / a->avgslicewidth;
      if (maxoveravg > 12.0 && maxoveravg / nrows > 0.001) { /* important threshold */
        /* each block handles approximately one slice */
        nchunks         = cudastruct->totalchunks;
        blocky          = a->chunksize / 32;
        chunksperblock  = cudastruct->chunksperblock ? cudastruct->chunksperblock : 1 + (cudastruct->totalentries / cudastruct->totalslices - 1) / a->chunksize;
        nblocks         = 1 + (nchunks - 1) / chunksperblock;
        chunk_slice_map = cudastruct->chunk_slice_map;
        if (blocky == 2) {
          matmultadd_seqsell_tiled_kernel8<2><<<nblocks, dim3(32, 2)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
        } else if (blocky == 4) {
          matmultadd_seqsell_tiled_kernel8<4><<<nblocks, dim3(32, 4)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
        } else if (blocky == 8) {
          matmultadd_seqsell_tiled_kernel8<8><<<nblocks, dim3(32, 8)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
        } else if (blocky == 16) {
          matmultadd_seqsell_tiled_kernel8<16><<<nblocks, dim3(32, 16)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
        } else if (blocky == 32) {
          matmultadd_seqsell_tiled_kernel8<32><<<nblocks, dim3(32, 32)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
        } else {
          matmultadd_seqsell_tiled_kernel8<2><<<nblocks, dim3(32, 2)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
        }
      } else {
        PetscInt avgslicesize = sliceheight * a->avgslicewidth;
        if (avgslicesize <= 432) {
          if (sliceheight * a->maxslicewidth < 2048 && nrows > 100000) {
            nblocks = 1 + (nrows - 1) / (2 * sliceheight); /* two slices per block */
            matmultadd_seqsell_tiled_kernel7<<<nblocks, dim3(32, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
          } else {
            nblocks = 1 + (nrows - 1) / sliceheight;
            matmultadd_seqsell_tiled_kernel9<2><<<nblocks, dim3(32, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
          }
        } else if (avgslicesize <= 2400) {
          nblocks = 1 + (nrows - 1) / sliceheight;
          matmultadd_seqsell_tiled_kernel9<8><<<nblocks, dim3(32, 8)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
        } else {
          nblocks = 1 + (nrows - 1) / sliceheight;
          matmultadd_seqsell_tiled_kernel9<16><<<nblocks, dim3(32, 16)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
        }
      }
      break;
#endif
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "unsupported kernel choice %" PetscInt_FMT " for MatMult_SeqSELLCUDA.", cudastruct->kernelchoice);
    }
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

static PetscErrorCode MatSetFromOptions_SeqSELLCUDA(Mat A, PetscOptionItems PetscOptionsObject)
{
  Mat_SeqSELLCUDA *cudastruct = (Mat_SeqSELLCUDA *)A->spptr;
  PetscInt         kernel, blocky;
  PetscBool        flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "SeqSELLCUDA options");
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mat_sell_spmv_cuda_blocky", &blocky, &flg));
  if (flg) {
    PetscCheck(blocky == 2 || blocky == 4 || blocky == 8 || blocky == 16 || blocky == 32, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported blocky: %" PetscInt_FMT " it should be in {2,4,8,16,32}", blocky);
    cudastruct->blocky = blocky;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mat_sell_spmv_cuda_kernel", &kernel, &flg));
  if (flg) {
    PetscCheck(kernel >= 0 && kernel <= 9, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Wrong kernel choice: %" PetscInt_FMT " it should be in [0,9]", kernel);
    cudastruct->kernelchoice = kernel;
    if (kernel == 8) PetscCall(PetscOptionsGetInt(NULL, NULL, "-mat_sell_spmv_cuda_chunksperblock", &cudastruct->chunksperblock, &flg));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatAssemblyEnd_SpMV_Preprocessing_Private(Mat A)
{
  Mat_SeqSELL *a = (Mat_SeqSELL *)A->data;

  PetscFunctionBegin;
  PetscCall(MatSeqSELLGetAvgSliceWidth(A, &a->avgslicewidth));
  PetscCall(MatSeqSELLGetMaxSliceWidth(A, &a->maxslicewidth));
  PetscCall(MatSeqSELLGetFillRatio(A, &a->fillratio));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAssemblyEnd_SeqSELLCUDA(Mat A, MatAssemblyType mode)
{
  PetscFunctionBegin;
  PetscCall(MatAssemblyEnd_SeqSELL(A, mode));
  PetscCall(MatAssemblyEnd_SpMV_Preprocessing_Private(A));
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(PETSC_SUCCESS);
  if (A->factortype == MAT_FACTOR_NONE) PetscCall(MatSeqSELLCUDACopyToGPU(A));
  A->ops->mult    = MatMult_SeqSELLCUDA;
  A->ops->multadd = MatMultAdd_SeqSELLCUDA;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroEntries_SeqSELLCUDA(Mat A)
{
  PetscBool    both = PETSC_FALSE;
  Mat_SeqSELL *a    = (Mat_SeqSELL *)A->data;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    Mat_SeqSELLCUDA *cudastruct = (Mat_SeqSELLCUDA *)A->spptr;
    if (cudastruct->val) {
      both = PETSC_TRUE;
      PetscCallCUDA(cudaMemset(cudastruct->val, 0, a->sliidx[a->totalslices] * sizeof(*cudastruct->val)));
    }
  }
  PetscCall(PetscArrayzero(a->val, a->sliidx[a->totalslices]));
  PetscCall(MatSeqSELLInvalidateDiagonal(A));
  if (both) A->offloadmask = PETSC_OFFLOAD_BOTH;
  else A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_SeqSELLCUDA(Mat A)
{
  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE && A->offloadmask != PETSC_OFFLOAD_UNALLOCATED) PetscCall(MatSeqSELLCUDA_Destroy((Mat_SeqSELLCUDA **)&A->spptr));
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

PETSC_INTERN PetscErrorCode MatConvert_SeqSELL_SeqSELLCUDA(Mat B)
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
  B->ops->zeroentries    = MatZeroEntries_SeqSELLCUDA;

  /* No need to assemble SeqSELL, but need to do the preprocessing for SpMV */
  PetscCall(MatAssemblyEnd_SpMV_Preprocessing_Private(B));

  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQSELLCUDA));
  B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  MATSEQSELLCUDA - MATSELLCUDA = "(seq)sellcuda" - A matrix type to be used for sparse matrices on NVIDIA GPUs.

  Options Database Keys:
+  -mat_type seqsellcuda - sets the matrix type to "seqsellcuda" during a call to `MatSetFromOptions()`
.  -mat_sell_spmv_cuda_kernel - selects a spmv kernel for MatSELLCUDA
-  -mat_sell_spmv_cuda_blocky - sets the y dimension of the block size of the spmv kernels. These kernels use a 2D block with the x dimension being 32

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MATSELLCUDA`
M*/

PETSC_EXTERN PetscErrorCode MatCreate_SeqSELLCUDA(Mat B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate_SeqSELL(B));
  PetscCall(MatConvert_SeqSELL_SeqSELLCUDA(B));
  PetscCall(MatSetFromOptions(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
