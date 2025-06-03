#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

#include <petscdevice_hip.h>
#include <petsc/private/cupmatomics.hpp>
#include <../src/mat/impls/sell/seq/sell.h> /*I   "petscmat.h"  I*/

#define WARP_SIZE 64

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
} Mat_SeqSELLHIP;

static PetscErrorCode MatSeqSELLHIP_Destroy(Mat_SeqSELLHIP **hipstruct)
{
  PetscFunctionBegin;
  if (*hipstruct) {
    if ((*hipstruct)->colidx) PetscCallHIP(hipFree((*hipstruct)->colidx));
    if ((*hipstruct)->val) PetscCallHIP(hipFree((*hipstruct)->val));
    if ((*hipstruct)->sliidx) PetscCallHIP(hipFree((*hipstruct)->sliidx));
    if ((*hipstruct)->chunk_slice_map) PetscCallHIP(hipFree((*hipstruct)->chunk_slice_map));
    PetscCall(PetscFree(*hipstruct));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSeqSELLHIPCopyToGPU(Mat A)
{
  Mat_SeqSELLHIP *hipstruct = (Mat_SeqSELLHIP *)A->spptr;
  Mat_SeqSELL    *a         = (Mat_SeqSELL *)A->data;

  PetscFunctionBegin;
  if (A->offloadmask == PETSC_OFFLOAD_UNALLOCATED || A->offloadmask == PETSC_OFFLOAD_CPU) {
    PetscCall(PetscLogEventBegin(MAT_HIPCopyToGPU, A, 0, 0, 0));
    if (A->assembled && A->nonzerostate == hipstruct->nonzerostate) {
      /* copy values only */
      PetscCallHIP(hipMemcpy(hipstruct->val, a->val, a->sliidx[a->totalslices] * sizeof(MatScalar), hipMemcpyHostToDevice));
      PetscCall(PetscLogCpuToGpu(a->sliidx[a->totalslices] * (sizeof(MatScalar))));
    } else {
      if (hipstruct->colidx) PetscCallHIP(hipFree(hipstruct->colidx));
      if (hipstruct->val) PetscCallHIP(hipFree(hipstruct->val));
      if (hipstruct->sliidx) PetscCallHIP(hipFree(hipstruct->sliidx));
      if (hipstruct->chunk_slice_map) PetscCallHIP(hipFree(hipstruct->chunk_slice_map));
      hipstruct->maxallocmat  = a->maxallocmat;
      hipstruct->totalentries = a->sliidx[a->totalslices];
      hipstruct->totalslices  = a->totalslices;
      hipstruct->totalchunks  = a->totalchunks;
      PetscCallHIP(hipMalloc((void **)&hipstruct->colidx, a->maxallocmat * sizeof(*hipstruct->colidx)));
      PetscCallHIP(hipMalloc((void **)&hipstruct->val, a->maxallocmat * sizeof(*hipstruct->val)));
      /* copy values, nz or maxallocmat? */
      PetscCallHIP(hipMemcpy(hipstruct->colidx, a->colidx, a->sliidx[a->totalslices] * sizeof(*a->colidx), hipMemcpyHostToDevice));
      PetscCallHIP(hipMemcpy(hipstruct->val, a->val, a->sliidx[a->totalslices] * sizeof(*a->val), hipMemcpyHostToDevice));

      PetscCallHIP(hipMalloc((void **)&hipstruct->sliidx, (a->totalslices + 1) * sizeof(*hipstruct->sliidx)));
      PetscCallHIP(hipMemcpy(hipstruct->sliidx, a->sliidx, (a->totalslices + 1) * sizeof(*a->sliidx), hipMemcpyHostToDevice));
      PetscCallHIP(hipMalloc((void **)&hipstruct->chunk_slice_map, a->totalchunks * sizeof(*hipstruct->chunk_slice_map)));
      PetscCallHIP(hipMemcpy(hipstruct->chunk_slice_map, a->chunk_slice_map, a->totalchunks * sizeof(*a->chunk_slice_map), hipMemcpyHostToDevice));
      PetscCall(PetscLogCpuToGpu(a->sliidx[a->totalslices] * (sizeof(MatScalar) + sizeof(PetscInt)) + (a->totalslices + 1 + a->totalchunks) * sizeof(PetscInt)));
    }
    PetscCallHIP(WaitForHIP());
    PetscCall(PetscLogEventEnd(MAT_HIPCopyToGPU, A, 0, 0, 0));
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
PETSC_PRAGMA_DIAGNOSTIC_IGNORED_BEGIN("-Wpass-failed")
/* use 1 block per slice, suitable for large slice width */
template <int BLOCKY>
__global__ void matmult_seqsell_tiled_kernel9(PetscInt nrows, PetscInt sliceheight, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, PetscScalar *y)
{
  __shared__ MatScalar shared[WARP_SIZE][BLOCKY];
  PetscInt             i, row, slice_id = blockIdx.x;
  int                  tid = threadIdx.x + threadIdx.y * WARP_SIZE;
  /* transposed index */
  int         tidx = tid % BLOCKY;
  int         tidy = tid / BLOCKY;
  PetscScalar t    = 0.0;

  row = slice_id * sliceheight + threadIdx.x % sliceheight;
  if (row < nrows) {
    for (i = sliidx[slice_id] + threadIdx.x + WARP_SIZE * threadIdx.y; i < sliidx[slice_id + 1]; i += WARP_SIZE * BLOCKY) t += aval[i] * x[acolidx[i]];
  }
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset >= sliceheight; offset /= 2) t += __shfl_down(t, offset);
  /* transpose layout to reduce each row using warp shfl */
  if (threadIdx.x < sliceheight) shared[threadIdx.x][threadIdx.y] = t;
  __syncthreads();
  if (tidy < sliceheight) t = shared[tidy][tidx];
  #pragma unroll
  for (int offset = BLOCKY / 2; offset > 0; offset /= 2) t += __shfl_down(t, offset, BLOCKY);
  if (tidx == 0 && tidy < sliceheight) shared[0][tidy] = t;
  __syncthreads();
  if (row < nrows && threadIdx.y == 0 && threadIdx.x < sliceheight) y[row] = shared[0][threadIdx.x];
}

/* use 1 block per slice, suitable for large slice width */
template <int BLOCKY>
__global__ void matmultadd_seqsell_tiled_kernel9(PetscInt nrows, PetscInt sliceheight, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, const PetscScalar *y, PetscScalar *z)
{
  __shared__ MatScalar shared[WARP_SIZE][BLOCKY];
  PetscInt             i, row, slice_id = blockIdx.x;
  int                  tid = threadIdx.x + threadIdx.y * WARP_SIZE;
  /* transposed index */
  int         tidx = tid % BLOCKY;
  int         tidy = tid / BLOCKY;
  PetscScalar t    = 0.0;

  row = slice_id * sliceheight + threadIdx.x % sliceheight;
  if (row < nrows) {
    for (i = sliidx[slice_id] + threadIdx.x + WARP_SIZE * threadIdx.y; i < sliidx[slice_id + 1]; i += WARP_SIZE * BLOCKY) t += aval[i] * x[acolidx[i]];
  }
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset >= sliceheight; offset /= 2) t += __shfl_down(t, offset);
  /* transpose layout to reduce each row using warp shfl */
  if (threadIdx.x < sliceheight) shared[threadIdx.x][threadIdx.y] = t;
  __syncthreads();
  if (tidy < sliceheight) t = shared[tidy][tidx];
  #pragma unroll
  for (int offset = BLOCKY / 2; offset > 0; offset /= 2) t += __shfl_down(t, offset, BLOCKY);
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
    int halfwarpid                                = threadIdx.y * 2 + threadIdx.x / (WARP_SIZE / 2);
    shared[threadIdx.x + threadIdx.y * WARP_SIZE] = 0;
    if (halfwarpid >= i && flag[halfwarpid - i] == flag[halfwarpid]) {
      shared[threadIdx.x + threadIdx.y * WARP_SIZE] = *val;
      if (i == 1) head = false;
    }
    __syncthreads();
    if (halfwarpid < BLOCKY * 2 - i) *val += shared[threadIdx.x + threadIdx.y * WARP_SIZE + i * WARP_SIZE];
    __syncthreads();
  }
  return head;
}

/* load-balancing version. Chunksize is equal to the number of threads per block */
template <int BLOCKY>
__global__ void matmult_seqsell_tiled_kernel8(PetscInt nrows, PetscInt sliceheight, PetscInt chunksperblock, PetscInt totalchunks, const PetscInt *chunk_slice_map, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, PetscScalar *y)
{
  __shared__ MatScalar shared[BLOCKY * WARP_SIZE];
  PetscInt             gid, row, start_slice, cid;
  PetscScalar          t = 0.0;
  AtomicAdd<MatScalar> atomAdd;
  /* zero out y */
  for (int iter = 0; iter < 1 + (nrows - 1) / (gridDim.x * WARP_SIZE * BLOCKY); iter++) {
    gid = gridDim.x * WARP_SIZE * BLOCKY * iter + blockIdx.x * BLOCKY * WARP_SIZE + threadIdx.y * WARP_SIZE + threadIdx.x;
    if (gid < nrows) y[gid] = 0.0;
  }
  for (int iter = 0; iter < chunksperblock; iter++) {
    cid = blockIdx.x * chunksperblock + iter; /* chunk id */
    if (cid < totalchunks) {
      start_slice = chunk_slice_map[cid]; /* starting slice at each iteration */
      gid         = cid * BLOCKY * WARP_SIZE + threadIdx.y * WARP_SIZE + threadIdx.x;
      if ((cid + 1) * BLOCKY * WARP_SIZE > sliidx[start_slice + 1]) { /* this iteration covers more than one slice */
        __shared__ PetscInt flag[BLOCKY * 2];
        bool                write;
        PetscInt            slice_id = start_slice, totalslices = PetscCeilIntMacro(nrows, sliceheight), totalentries = sliidx[totalslices];
        /* find out the slice that this element belongs to */
        while (gid < totalentries && gid >= sliidx[slice_id + 1]) slice_id++;
        if (threadIdx.x % (WARP_SIZE / 2) == 0) flag[threadIdx.y * 2 + threadIdx.x / (WARP_SIZE / 2)] = slice_id;
        row = slice_id * sliceheight + threadIdx.x % sliceheight;
        if (row < nrows && gid < totalentries) t = aval[gid] * x[acolidx[gid]];
        __syncthreads();
        write = segment_scan<BLOCKY>(flag, shared, &t);
        if (row < nrows && gid < totalentries && write) atomAdd(y[row], t);
        t = 0.0;
      } else { /* this iteration covers only one slice */
        row = start_slice * sliceheight + threadIdx.x % sliceheight;
        if (row < nrows) t += aval[gid] * x[acolidx[gid]];
        if (iter == chunksperblock - 1 || (cid + 2) * BLOCKY * WARP_SIZE > sliidx[start_slice + 1]) { /* last iteration or next iteration covers more than one slice */
          int tid = threadIdx.x + threadIdx.y * WARP_SIZE, tidx = tid % BLOCKY, tidy = tid / BLOCKY;
  /* reduction and write to output vector */
  #pragma unroll
          for (int offset = WARP_SIZE / 2; offset >= sliceheight; offset /= 2) t += __shfl_down(t, offset);
          /* transpose layout to reduce each row using warp shfl */
          if (threadIdx.x < sliceheight) shared[threadIdx.x * BLOCKY + threadIdx.y] = t; /* shared[threadIdx.x][threadIdx.y] = t */
          __syncthreads();
          if (tidy < sliceheight) t = shared[tidy * BLOCKY + tidx]; /* shared[tidy][tidx] */
  #pragma unroll
          for (int offset = BLOCKY / 2; offset > 0; offset /= 2) t += __shfl_down(t, offset, BLOCKY);
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
  __shared__ MatScalar shared[BLOCKY * WARP_SIZE];
  PetscInt             gid, row, start_slice, cid;
  PetscScalar          t = 0.0;
  AtomicAdd<MatScalar> atomAdd;
  /* copy y to z */
  for (int iter = 0; iter < 1 + (nrows - 1) / (gridDim.x * WARP_SIZE * BLOCKY); iter++) {
    gid = gridDim.x * WARP_SIZE * BLOCKY * iter + blockIdx.x * BLOCKY * WARP_SIZE + threadIdx.y * WARP_SIZE + threadIdx.x;
    if (gid < nrows) z[gid] = y[gid];
  }
  for (int iter = 0; iter < chunksperblock; iter++) {
    cid = blockIdx.x * chunksperblock + iter; /* chunk id */
    if (cid < totalchunks) {
      start_slice = chunk_slice_map[cid]; /* starting slice at each iteration */
      gid         = cid * BLOCKY * WARP_SIZE + threadIdx.y * WARP_SIZE + threadIdx.x;
      if ((cid + 1) * BLOCKY * WARP_SIZE > sliidx[start_slice + 1]) { /* this iteration covers more than one slice */
        __shared__ PetscInt flag[BLOCKY * 2];
        bool                write;
        PetscInt            slice_id = start_slice, totalslices = PetscCeilIntMacro(nrows, sliceheight), totalentries = sliidx[totalslices];
        /* find out the slice that this element belongs to */
        while (gid < totalentries && gid >= sliidx[slice_id + 1]) slice_id++;
        if (threadIdx.x % (WARP_SIZE / 2) == 0) flag[threadIdx.y * 2 + threadIdx.x / (WARP_SIZE / 2)] = slice_id;
        row = slice_id * sliceheight + threadIdx.x % sliceheight;
        if (row < nrows && gid < totalentries) t = aval[gid] * x[acolidx[gid]];
        __syncthreads();
        write = segment_scan<BLOCKY>(flag, shared, &t);
        if (row < nrows && gid < totalentries && write) atomAdd(z[row], t);
        t = 0.0;
      } else { /* this iteration covers only one slice */
        row = start_slice * sliceheight + threadIdx.x % sliceheight;
        if (row < nrows) t += aval[gid] * x[acolidx[gid]];
        if (iter == chunksperblock - 1 || (cid + 2) * BLOCKY * WARP_SIZE > sliidx[start_slice + 1]) { /* last iteration or next iteration covers more than one slice */
          int tid = threadIdx.x + threadIdx.y * WARP_SIZE, tidx = tid % BLOCKY, tidy = tid / BLOCKY;
  /* reduction and write to output vector */
  #pragma unroll
          for (int offset = WARP_SIZE / 2; offset >= sliceheight; offset /= 2) t += __shfl_down(t, offset);
          /* transpose layout to reduce each row using warp shfl */
          if (threadIdx.x < sliceheight) shared[threadIdx.x * BLOCKY + threadIdx.y] = t; /* shared[threadIdx.x][threadIdx.y] = t */
          __syncthreads();
          if (tidy < sliceheight) t = shared[tidy * BLOCKY + tidx]; /* shared[tidy][tidx] */
  #pragma unroll
          for (int offset = BLOCKY / 2; offset > 0; offset /= 2) t += __shfl_down(t, offset, BLOCKY);
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
    for (i = sliidx[slice_id] + threadIdx.x; i < sliidx[slice_id + 1]; i += WARP_SIZE) t += aval[i] * x[acolidx[i]];
  }
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset >= sliceheight; offset /= 2) t += __shfl_down(t, offset);
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
    for (i = sliidx[slice_id] + threadIdx.x; i < sliidx[slice_id + 1]; i += WARP_SIZE) t += aval[i] * x[acolidx[i]];
  }
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset >= sliceheight; offset /= 2) t += __shfl_down(t, offset);
  if (row < nrows && threadIdx.x < sliceheight) z[row] = y[row] + t;
}
PETSC_PRAGMA_DIAGNOSTIC_IGNORED_END()
#endif

/***********  Kernel 2-6 require a slice height smaller than 512, 256, 128, 64, 32, espectively. They are kept only for performance comparison  **********/

static __global__ void matmult_seqsell_tiled_kernel6(PetscInt nrows, PetscInt sliceheight, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, PetscScalar *y)
{
  __shared__ MatScalar shared[32 * 16];
  PetscInt             i, row, slice_id, row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / sliceheight;
    row_in_slice = row % sliceheight;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + sliceheight * threadIdx.y; i < sliidx[slice_id + 1]; i += sliceheight * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
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

static __global__ void matmult_seqsell_tiled_kernel5(PetscInt nrows, PetscInt sliceheight, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, PetscScalar *y)
{
  __shared__ MatScalar shared[32 * 16];
  PetscInt             i, row, slice_id, row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / sliceheight;
    row_in_slice = row % sliceheight;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + sliceheight * threadIdx.y; i < sliidx[slice_id + 1]; i += sliceheight * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
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

static __global__ void matmult_seqsell_tiled_kernel4(PetscInt nrows, PetscInt sliceheight, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, PetscScalar *y)
{
  __shared__ MatScalar shared[32 * 16];
  PetscInt             i, row, slice_id, row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / sliceheight;
    row_in_slice = row % sliceheight;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + sliceheight * threadIdx.y; i < sliidx[slice_id + 1]; i += sliceheight * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
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

static __global__ void matmult_seqsell_tiled_kernel3(PetscInt nrows, PetscInt sliceheight, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, PetscScalar *y)
{
  __shared__ MatScalar shared[32 * 16];
  PetscInt             i, row, slice_id, row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / sliceheight;
    row_in_slice = row % sliceheight;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + sliceheight * threadIdx.y; i < sliidx[slice_id + 1]; i += sliceheight * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 2) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 2) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x + threadIdx.x];
      y[row] = shared[threadIdx.x];
    }
  }
}

static __global__ void matmult_seqsell_tiled_kernel2(PetscInt nrows, PetscInt sliceheight, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, PetscScalar *y)
{
  __shared__ MatScalar shared[32 * 16];
  PetscInt             i, row, slice_id, row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / sliceheight;
    row_in_slice = row % sliceheight;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + sliceheight * threadIdx.y; i < sliidx[slice_id + 1]; i += sliceheight * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x + threadIdx.x];
      y[row] = shared[threadIdx.x];
    }
  }
}

static __global__ void matmultadd_seqsell_tiled_kernel6(PetscInt nrows, PetscInt sliceheight, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, const PetscScalar *y, PetscScalar *z)
{
  __shared__ MatScalar shared[32 * 16];
  PetscInt             i, row, slice_id, row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / sliceheight;
    row_in_slice = row % sliceheight;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + sliceheight * threadIdx.y; i < sliidx[slice_id + 1]; i += sliceheight * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
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

static __global__ void matmultadd_seqsell_tiled_kernel5(PetscInt nrows, PetscInt sliceheight, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, const PetscScalar *y, PetscScalar *z)
{
  __shared__ MatScalar shared[32 * 16];
  PetscInt             i, row, slice_id, row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / sliceheight;
    row_in_slice = row % sliceheight;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + sliceheight * threadIdx.y; i < sliidx[slice_id + 1]; i += sliceheight * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
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

static __global__ void matmultadd_seqsell_tiled_kernel4(PetscInt nrows, PetscInt sliceheight, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, const PetscScalar *y, PetscScalar *z)
{
  __shared__ MatScalar shared[32 * 16];
  PetscInt             i, row, slice_id, row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / sliceheight;
    row_in_slice = row % sliceheight;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + sliceheight * threadIdx.y; i < sliidx[slice_id + 1]; i += sliceheight * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
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

static __global__ void matmultadd_seqsell_tiled_kernel3(PetscInt nrows, PetscInt sliceheight, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, const PetscScalar *y, PetscScalar *z)
{
  __shared__ MatScalar shared[32 * 16];
  PetscInt             i, row, slice_id, row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / sliceheight;
    row_in_slice = row % sliceheight;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + sliceheight * threadIdx.y; i < sliidx[slice_id + 1]; i += sliceheight * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 2) shared[threadIdx.y * blockDim.x + threadIdx.x] += shared[(threadIdx.y + 2) * blockDim.x + threadIdx.x];
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x + threadIdx.x];
      z[row] = y[row] + shared[threadIdx.x];
    }
  }
}

static __global__ void matmultadd_seqsell_tiled_kernel2(PetscInt nrows, PetscInt sliceheight, const PetscInt *acolidx, const MatScalar *aval, const PetscInt *sliidx, const PetscScalar *x, const PetscScalar *y, PetscScalar *z)
{
  __shared__ MatScalar shared[32 * 16];
  PetscInt             i, row, slice_id, row_in_slice;
  /* multiple threads per row. */
  row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < nrows) {
    slice_id     = row / sliceheight;
    row_in_slice = row % sliceheight;

    shared[threadIdx.y * blockDim.x + threadIdx.x] = 0.0;
    for (i = sliidx[slice_id] + row_in_slice + sliceheight * threadIdx.y; i < sliidx[slice_id + 1]; i += sliceheight * blockDim.y) shared[threadIdx.y * blockDim.x + threadIdx.x] += aval[i] * x[acolidx[i]];
    __syncthreads();
    if (threadIdx.y < 1) {
      shared[threadIdx.x] += shared[blockDim.x + threadIdx.x];
      z[row] = y[row] + shared[threadIdx.x];
    }
  }
}

static PetscErrorCode MatMult_SeqSELLHIP(Mat A, Vec xx, Vec yy)
{
  Mat_SeqSELL       *a         = (Mat_SeqSELL *)A->data;
  Mat_SeqSELLHIP    *hipstruct = (Mat_SeqSELLHIP *)A->spptr;
  PetscScalar       *y;
  const PetscScalar *x;
  PetscInt           nrows = A->rmap->n, sliceheight = a->sliceheight;
  MatScalar         *aval;
  PetscInt          *acolidx;
  PetscInt          *sliidx;
  PetscInt           nblocks, blocksize = 512; /* blocksize is fixed to be 512 */
  dim3               block2(256, 2), block4(128, 4), block8(64, 8), block16(32, 16), block32(16, 32);
#if !defined(PETSC_USE_COMPLEX)
  PetscInt  chunksperblock, nchunks, *chunk_slice_map;
  PetscReal maxoveravg;
#endif

  PetscFunctionBegin;
  PetscCheck(WARP_SIZE % sliceheight == 0, PETSC_COMM_SELF, PETSC_ERR_SUP, "The kernel requires a slice height be a divisor of WARP_SIZE, but the input matrix has a slice height of %" PetscInt_FMT, sliceheight);
  PetscCheck(!(hipstruct->kernelchoice >= 2 && hipstruct->kernelchoice <= 6 && sliceheight > 32), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Kernel choices {2-6} requires the slice height of the matrix be less than 32, but the current slice height is %" PetscInt_FMT, sliceheight);
  PetscCall(MatSeqSELLHIPCopyToGPU(A));
  /* hipstruct may not be available until MatSeqSELLHIPCopyToGPU() is called */
  aval    = hipstruct->val;
  acolidx = hipstruct->colidx;
  sliidx  = hipstruct->sliidx;

  PetscCall(VecHIPGetArrayRead(xx, &x));
  PetscCall(VecHIPGetArrayWrite(yy, &y));
  PetscCall(PetscLogGpuTimeBegin());

  switch (hipstruct->kernelchoice) {
#if !defined(PETSC_USE_COMPLEX)
  case 9: /* 1 slice per block */
    nblocks = 1 + (nrows - 1) / sliceheight;
    if (hipstruct->blocky == 2) {
      matmult_seqsell_tiled_kernel9<2><<<nblocks, dim3(WARP_SIZE, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    } else if (hipstruct->blocky == 4) {
      matmult_seqsell_tiled_kernel9<4><<<nblocks, dim3(WARP_SIZE, 4)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    } else if (hipstruct->blocky == 8) {
      matmult_seqsell_tiled_kernel9<8><<<nblocks, dim3(WARP_SIZE, 8)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    } else if (hipstruct->blocky == 16) {
      matmult_seqsell_tiled_kernel9<16><<<nblocks, dim3(WARP_SIZE, 16)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    } else {
      matmult_seqsell_tiled_kernel9<2><<<nblocks, dim3(WARP_SIZE, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    }
    break;
  case 7: /* each block handles blocky slices */
    nblocks = 1 + (nrows - 1) / (hipstruct->blocky * sliceheight);
    if (hipstruct->blocky == 2) {
      matmult_seqsell_tiled_kernel7<<<nblocks, dim3(WARP_SIZE, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    } else if (hipstruct->blocky == 4) {
      matmult_seqsell_tiled_kernel7<<<nblocks, dim3(WARP_SIZE, 4)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    } else if (hipstruct->blocky == 8) {
      matmult_seqsell_tiled_kernel7<<<nblocks, dim3(WARP_SIZE, 8)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    } else if (hipstruct->blocky == 16) {
      matmult_seqsell_tiled_kernel7<<<nblocks, dim3(WARP_SIZE, 16)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    } else {
      nblocks = 1 + (nrows - 1) / (2 * sliceheight);
      matmult_seqsell_tiled_kernel7<<<nblocks, dim3(WARP_SIZE, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    }
    break;
#endif
  case 6:
    nblocks = 1 + (nrows - 1) / (blocksize / 32); /* 1 slice per block if sliceheight=32 */
    matmult_seqsell_tiled_kernel6<<<nblocks, block32>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    break;
  case 5:
    nblocks = 1 + (nrows - 1) / (blocksize / 16); /* 2 slices per block if sliceheight=32*/
    matmult_seqsell_tiled_kernel5<<<nblocks, block16>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    break;
  case 4:
    nblocks = 1 + (nrows - 1) / (blocksize / 8); /* 4 slices per block if sliceheight=32 */
    matmult_seqsell_tiled_kernel4<<<nblocks, block8>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    break;
  case 3:
    nblocks = 1 + (nrows - 1) / (blocksize / 4); /* 8 slices per block if sliceheight=32 */
    matmult_seqsell_tiled_kernel3<<<nblocks, block4>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    break;
  case 2: /* 16 slices per block if sliceheight=32 */
    nblocks = 1 + (nrows - 1) / (blocksize / 2);
    matmult_seqsell_tiled_kernel2<<<nblocks, block2>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    break;
  case 1: /* 32 slices per block if sliceheight=32 */
    nblocks = 1 + (nrows - 1) / blocksize;
    matmult_seqsell_basic_kernel<<<nblocks, blocksize>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
    break;
#if !defined(PETSC_USE_COMPLEX)
  case 0:
    maxoveravg = a->maxslicewidth / a->avgslicewidth;
    if (maxoveravg > 12.0 && maxoveravg / nrows > 0.001) { /* important threshold */
      /* each block handles approximately one slice */
      PetscInt blocky = a->chunksize / 32;
      nchunks         = hipstruct->totalchunks;
      chunksperblock  = hipstruct->chunksperblock ? hipstruct->chunksperblock : 1 + (hipstruct->totalentries / hipstruct->totalslices - 1) / a->chunksize;
      nblocks         = 1 + (nchunks - 1) / chunksperblock;
      chunk_slice_map = hipstruct->chunk_slice_map;
      if (blocky == 2) {
        matmult_seqsell_tiled_kernel8<2><<<nblocks, dim3(WARP_SIZE, 2)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y);
      } else if (blocky == 4) {
        matmult_seqsell_tiled_kernel8<4><<<nblocks, dim3(WARP_SIZE, 4)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y);
      } else if (blocky == 8) {
        matmult_seqsell_tiled_kernel8<8><<<nblocks, dim3(WARP_SIZE, 8)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y);
      } else if (blocky == 16) {
        matmult_seqsell_tiled_kernel8<16><<<nblocks, dim3(WARP_SIZE, 16)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y);
      } else {
        matmult_seqsell_tiled_kernel8<2><<<nblocks, dim3(WARP_SIZE, 2)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y);
      }
    } else {
      PetscInt avgslicesize = sliceheight * a->avgslicewidth;
      if (avgslicesize <= 432) {
        if (sliceheight * a->maxslicewidth < 2048 && nrows > 100000) {
          nblocks = 1 + (nrows - 1) / (2 * sliceheight); /* two slices per block */
          matmult_seqsell_tiled_kernel7<<<nblocks, dim3(WARP_SIZE, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
        } else {
          nblocks = 1 + (nrows - 1) / sliceheight;
          matmult_seqsell_tiled_kernel9<2><<<nblocks, dim3(WARP_SIZE, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
        }
      } else if (avgslicesize <= 2400) {
        nblocks = 1 + (nrows - 1) / sliceheight;
        matmult_seqsell_tiled_kernel9<8><<<nblocks, dim3(WARP_SIZE, 8)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
      } else {
        nblocks = 1 + (nrows - 1) / sliceheight;
        matmult_seqsell_tiled_kernel9<16><<<nblocks, dim3(WARP_SIZE, 16)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y);
      }
    }
    break;
#endif
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "unsupported kernel choice %" PetscInt_FMT " for MatMult_SeqSELLHIP.", hipstruct->kernelchoice);
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(VecHIPRestoreArrayRead(xx, &x));
  PetscCall(VecHIPRestoreArrayWrite(yy, &y));
  PetscCall(PetscLogGpuFlops(2.0 * a->nz - a->nonzerorowcnt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultAdd_SeqSELLHIP(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqSELL       *a         = (Mat_SeqSELL *)A->data;
  Mat_SeqSELLHIP    *hipstruct = (Mat_SeqSELLHIP *)A->spptr;
  PetscScalar       *z;
  const PetscScalar *y, *x;
  PetscInt           nrows = A->rmap->n, sliceheight = a->sliceheight;
  MatScalar         *aval    = hipstruct->val;
  PetscInt          *acolidx = hipstruct->colidx;
  PetscInt          *sliidx  = hipstruct->sliidx;
#if !defined(PETSC_USE_COMPLEX)
  PetscReal maxoveravg;
  PetscInt  chunksperblock, nchunks, *chunk_slice_map;
  PetscInt  blocky = hipstruct->blocky;
#endif

  PetscFunctionBegin;
  PetscCheck(WARP_SIZE % sliceheight == 0, PETSC_COMM_SELF, PETSC_ERR_SUP, "The kernel requires a slice height be a divisor of WARP_SIZE, but the input matrix has a slice height of %" PetscInt_FMT, sliceheight);
  PetscCheck(!(hipstruct->kernelchoice >= 2 && hipstruct->kernelchoice <= 6 && sliceheight != sliceheight), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Kernel choices {2-6} requires the slice height of the matrix be 16, but the current slice height is %" PetscInt_FMT, sliceheight);
  PetscCall(MatSeqSELLHIPCopyToGPU(A));
  if (a->nz) {
    PetscInt nblocks, blocksize = 512;
    dim3     block2(256, 2), block4(128, 4), block8(64, 8), block16(32, 16), block32(16, 32);
    PetscCall(VecHIPGetArrayRead(xx, &x));
    PetscCall(VecHIPGetArrayRead(yy, &y));
    PetscCall(VecHIPGetArrayWrite(zz, &z));
    PetscCall(PetscLogGpuTimeBegin());

    switch (hipstruct->kernelchoice) {
#if !defined(PETSC_USE_COMPLEX)
    case 9:
      nblocks = 1 + (nrows - 1) / sliceheight;
      if (blocky == 2) {
        matmultadd_seqsell_tiled_kernel9<2><<<nblocks, dim3(WARP_SIZE, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 4) {
        matmultadd_seqsell_tiled_kernel9<4><<<nblocks, dim3(WARP_SIZE, 4)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 8) {
        matmultadd_seqsell_tiled_kernel9<8><<<nblocks, dim3(WARP_SIZE, 8)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 16) {
        matmultadd_seqsell_tiled_kernel9<16><<<nblocks, dim3(WARP_SIZE, 16)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      } else {
        matmultadd_seqsell_tiled_kernel9<2><<<nblocks, dim3(WARP_SIZE, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      }
      break;
    case 8:
      /* each block handles approximately one slice */
      nchunks         = hipstruct->totalchunks;
      blocky          = a->chunksize / 32;
      chunksperblock  = hipstruct->chunksperblock ? hipstruct->chunksperblock : 1 + (hipstruct->totalentries / hipstruct->totalslices - 1) / a->chunksize;
      nblocks         = 1 + (nchunks - 1) / chunksperblock;
      chunk_slice_map = hipstruct->chunk_slice_map;
      if (blocky == 2) {
        matmultadd_seqsell_tiled_kernel8<2><<<nblocks, dim3(WARP_SIZE, 2)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 4) {
        matmultadd_seqsell_tiled_kernel8<4><<<nblocks, dim3(WARP_SIZE, 4)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 8) {
        matmultadd_seqsell_tiled_kernel8<8><<<nblocks, dim3(WARP_SIZE, 8)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 16) {
        matmultadd_seqsell_tiled_kernel8<16><<<nblocks, dim3(WARP_SIZE, 16)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
      } else {
        matmultadd_seqsell_tiled_kernel8<2><<<nblocks, dim3(WARP_SIZE, 2)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
      }
      break;
    case 7:
      nblocks = 1 + (nrows - 1) / (blocky * sliceheight);
      if (blocky == 2) {
        matmultadd_seqsell_tiled_kernel7<<<nblocks, dim3(WARP_SIZE, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 4) {
        matmultadd_seqsell_tiled_kernel7<<<nblocks, dim3(WARP_SIZE, 4)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 8) {
        matmultadd_seqsell_tiled_kernel7<<<nblocks, dim3(WARP_SIZE, 8)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      } else if (blocky == 16) {
        matmultadd_seqsell_tiled_kernel7<<<nblocks, dim3(WARP_SIZE, 16)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      } else {
        nblocks = 1 + (nrows - 1) / (2 * sliceheight);
        matmultadd_seqsell_tiled_kernel7<<<nblocks, dim3(WARP_SIZE, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      }
      break;
#endif
    case 6:
      nblocks = 1 + (nrows - 1) / (blocksize / 32);
      matmultadd_seqsell_tiled_kernel6<<<nblocks, block32>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      break;
    case 5:
      nblocks = 1 + (nrows - 1) / (blocksize / 16);
      matmultadd_seqsell_tiled_kernel5<<<nblocks, block16>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      break;
    case 4:
      nblocks = 1 + (nrows - 1) / (blocksize / 8);
      matmultadd_seqsell_tiled_kernel4<<<nblocks, block8>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      break;
    case 3:
      nblocks = 1 + (nrows - 1) / (blocksize / 4);
      matmultadd_seqsell_tiled_kernel3<<<nblocks, block4>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
      break;
    case 2:
      nblocks = 1 + (nrows - 1) / (blocksize / 2);
      matmultadd_seqsell_tiled_kernel2<<<nblocks, block2>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
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
        nchunks         = hipstruct->totalchunks;
        blocky          = a->chunksize / 32;
        chunksperblock  = hipstruct->chunksperblock ? hipstruct->chunksperblock : 1 + (hipstruct->totalentries / hipstruct->totalslices - 1) / a->chunksize;
        nblocks         = 1 + (nchunks - 1) / chunksperblock;
        chunk_slice_map = hipstruct->chunk_slice_map;
        if (blocky == 2) {
          matmultadd_seqsell_tiled_kernel8<2><<<nblocks, dim3(WARP_SIZE, 2)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
        } else if (blocky == 4) {
          matmultadd_seqsell_tiled_kernel8<4><<<nblocks, dim3(WARP_SIZE, 4)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
        } else if (blocky == 8) {
          matmultadd_seqsell_tiled_kernel8<8><<<nblocks, dim3(WARP_SIZE, 8)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
        } else if (blocky == 16) {
          matmultadd_seqsell_tiled_kernel8<16><<<nblocks, dim3(WARP_SIZE, 16)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
        } else {
          matmultadd_seqsell_tiled_kernel8<2><<<nblocks, dim3(WARP_SIZE, 2)>>>(nrows, sliceheight, chunksperblock, nchunks, chunk_slice_map, acolidx, aval, sliidx, x, y, z);
        }
      } else {
        PetscInt avgslicesize = sliceheight * a->avgslicewidth;
        if (avgslicesize <= 432) {
          if (sliceheight * a->maxslicewidth < 2048 && nrows > 100000) {
            nblocks = 1 + (nrows - 1) / (2 * sliceheight); /* two slices per block */
            matmultadd_seqsell_tiled_kernel7<<<nblocks, dim3(WARP_SIZE, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
          } else {
            nblocks = 1 + (nrows - 1) / sliceheight;
            matmultadd_seqsell_tiled_kernel9<2><<<nblocks, dim3(WARP_SIZE, 2)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
          }
        } else if (avgslicesize <= 2400) {
          nblocks = 1 + (nrows - 1) / sliceheight;
          matmultadd_seqsell_tiled_kernel9<8><<<nblocks, dim3(WARP_SIZE, 8)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
        } else {
          nblocks = 1 + (nrows - 1) / sliceheight;
          matmultadd_seqsell_tiled_kernel9<16><<<nblocks, dim3(WARP_SIZE, 16)>>>(nrows, sliceheight, acolidx, aval, sliidx, x, y, z);
        }
      }
      break;
#endif
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "unsupported kernel choice %" PetscInt_FMT " for MatMult_SeqSELLHIP.", hipstruct->kernelchoice);
    }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(VecHIPRestoreArrayRead(xx, &x));
    PetscCall(VecHIPRestoreArrayRead(yy, &y));
    PetscCall(VecHIPRestoreArrayWrite(zz, &z));
    PetscCall(PetscLogGpuFlops(2.0 * a->nz));
  } else {
    PetscCall(VecCopy(yy, zz));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_SeqSELLHIP(Mat A, PetscOptionItems PetscOptionsObject)
{
  Mat_SeqSELLHIP *hipstruct = (Mat_SeqSELLHIP *)A->spptr;
  PetscInt        kernel, blocky;
  PetscBool       flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "SeqSELLHIP options");
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mat_sell_spmv_hip_blocky", &blocky, &flg));
  if (flg) {
    PetscCheck(blocky == 2 || blocky == 4 || blocky == 8 || blocky == 16 || blocky == 32, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported blocky: %" PetscInt_FMT " it should be in {2,4,8,16,32}", blocky);
    hipstruct->blocky = blocky;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mat_sell_spmv_hip_kernel", &kernel, &flg));
  if (flg) {
    PetscCheck(kernel >= 0 && kernel <= 9, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Wrong kernel choice: %" PetscInt_FMT " it should be in [0,9]", kernel);
    hipstruct->kernelchoice = kernel;
    if (kernel == 8) PetscCall(PetscOptionsGetInt(NULL, NULL, "-mat_sell_spmv_hip_chunksperblock", &hipstruct->chunksperblock, &flg));
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

static PetscErrorCode MatAssemblyEnd_SeqSELLHIP(Mat A, MatAssemblyType mode)
{
  PetscFunctionBegin;
  PetscCall(MatAssemblyEnd_SeqSELL(A, mode));
  PetscCall(MatAssemblyEnd_SpMV_Preprocessing_Private(A));
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(PETSC_SUCCESS);
  if (A->factortype == MAT_FACTOR_NONE) PetscCall(MatSeqSELLHIPCopyToGPU(A));
  A->ops->mult    = MatMult_SeqSELLHIP;
  A->ops->multadd = MatMultAdd_SeqSELLHIP;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroEntries_SeqSELLHIP(Mat A)
{
  PetscBool    both = PETSC_FALSE;
  Mat_SeqSELL *a    = (Mat_SeqSELL *)A->data;

  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE) {
    Mat_SeqSELLHIP *hipstruct = (Mat_SeqSELLHIP *)A->spptr;
    if (hipstruct->val) {
      both = PETSC_TRUE;
      PetscCallHIP(hipMemset(hipstruct->val, 0, a->sliidx[a->totalslices] * sizeof(*hipstruct->val)));
    }
  }
  PetscCall(PetscArrayzero(a->val, a->sliidx[a->totalslices]));
  PetscCall(MatSeqSELLInvalidateDiagonal(A));
  if (both) A->offloadmask = PETSC_OFFLOAD_BOTH;
  else A->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_SeqSELLHIP(Mat A)
{
  PetscFunctionBegin;
  if (A->factortype == MAT_FACTOR_NONE && A->offloadmask != PETSC_OFFLOAD_UNALLOCATED) PetscCall(MatSeqSELLHIP_Destroy((Mat_SeqSELLHIP **)&A->spptr));
  PetscCall(MatDestroy_SeqSELL(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqSELL_SeqSELLHIP(Mat);
static PetscErrorCode       MatDuplicate_SeqSELLHIP(Mat A, MatDuplicateOption cpvalues, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatDuplicate_SeqSELL(A, cpvalues, B));
  PetscCall(MatConvert_SeqSELL_SeqSELLHIP(*B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqSELL_SeqSELLHIP(Mat B)
{
  Mat_SeqSELLHIP *hipstruct;

  PetscFunctionBegin;
  PetscCall(PetscFree(B->defaultvectype));
  PetscCall(PetscStrallocpy(VECHIP, &B->defaultvectype));

  if (!B->spptr) {
    if (B->factortype == MAT_FACTOR_NONE) {
      PetscCall(PetscNew(&hipstruct));
      B->spptr = hipstruct;
    }
  }

  B->ops->assemblyend    = MatAssemblyEnd_SeqSELLHIP;
  B->ops->destroy        = MatDestroy_SeqSELLHIP;
  B->ops->setfromoptions = MatSetFromOptions_SeqSELLHIP;
  B->ops->mult           = MatMult_SeqSELLHIP;
  B->ops->multadd        = MatMultAdd_SeqSELLHIP;
  B->ops->duplicate      = MatDuplicate_SeqSELLHIP;
  B->ops->zeroentries    = MatZeroEntries_SeqSELLHIP;

  /* No need to assemble SeqSELL, but need to do the preprocessing for SpMV */
  PetscCall(MatAssemblyEnd_SpMV_Preprocessing_Private(B));

  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQSELLHIP));
  B->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  MATSEQSELLHIP - MATSELLHIP = "(seq)sellhip" - A matrix type to be used for sparse matrices on AMD GPUs.

  Options Database Keys:
+  -mat_type seqsellhip - sets the matrix type to "seqsellhip" during a call to `MatSetFromOptions()`
.  -mat_sell_spmv_hip_kernel - selects a spmv kernel for MatSELLHIP
-  -mat_sell_spmv_hip_blocky - sets the y dimension of the block size of the spmv kernels. These kernels use a 2D block with the x dimension equal to the wrap size (normally 64 for AMD GPUs)

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MATSELLHIP`
M*/

PETSC_EXTERN PetscErrorCode MatCreate_SeqSELLHIP(Mat B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate_SeqSELL(B));
  PetscCall(MatConvert_SeqSELL_SeqSELLHIP(B));
  PetscCall(MatSetFromOptions(B));
  PetscFunctionReturn(PETSC_SUCCESS);
}
