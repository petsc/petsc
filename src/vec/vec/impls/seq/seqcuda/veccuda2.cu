/*
   Implements the sequential cuda vectors.
*/

#define PETSC_SKIP_SPINLOCK

#include <petscconf.h>
#include <petsc/private/vecimpl.h>
#include <../src/vec/vec/impls/dvecimpl.h>
#include <petsc/private/cudavecimpl.h>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#if defined(PETSC_USE_COMPLEX)
#include <thrust/transform_reduce.h>
#endif

#if THRUST_VERSION >= 101600 && !PetscDefined(USE_DEBUG)
static thrust::cuda_cub::par_nosync_t::stream_attachment_type VecCUDAThrustPolicy(Vec x) {
    return thrust::cuda::par_nosync.on(((Vec_CUDA*)x->spptr)->stream);
}
#else
static thrust::cuda_cub::par_t::stream_attachment_type VecCUDAThrustPolicy(Vec x) {
    return thrust::cuda::par.on(((Vec_CUDA*)x->spptr)->stream);
}
#endif

/*
    Allocates space for the vector array on the GPU if it does not exist.
    Does NOT change the PetscCUDAFlag for the vector
    Does NOT zero the CUDA array

 */
PetscErrorCode VecCUDAAllocateCheck(Vec v)
{
  Vec_CUDA       *veccuda;
  PetscBool      option_set;

  PetscFunctionBegin;
  if (!v->spptr) {
    PetscReal pinned_memory_min;
    PetscCall(PetscCalloc(sizeof(Vec_CUDA),&v->spptr));
    veccuda = (Vec_CUDA*)v->spptr;
    PetscCallCUDA(cudaMalloc((void**)&veccuda->GPUarray_allocated,sizeof(PetscScalar)*((PetscBLASInt)v->map->n)));
    veccuda->GPUarray = veccuda->GPUarray_allocated;
    if (v->offloadmask == PETSC_OFFLOAD_UNALLOCATED) {
      if (v->data && ((Vec_Seq*)v->data)->array) {
        v->offloadmask = PETSC_OFFLOAD_CPU;
      } else {
        v->offloadmask = PETSC_OFFLOAD_GPU;
      }
    }
    pinned_memory_min = 0;

    /* Need to parse command line for minimum size to use for pinned memory allocations on host here.
       Note: This same code duplicated in VecCreate_SeqCUDA_Private() and VecCreate_MPICUDA_Private(). Is there a good way to avoid this? */
    PetscOptionsBegin(PetscObjectComm((PetscObject)v),((PetscObject)v)->prefix,"VECCUDA Options","Vec");
    PetscCall(PetscOptionsReal("-vec_pinned_memory_min","Minimum size (in bytes) for an allocation to use pinned memory on host","VecSetPinnedMemoryMin",pinned_memory_min,&pinned_memory_min,&option_set));
    if (option_set) v->minimum_bytes_pinned_memory = pinned_memory_min;
    PetscOptionsEnd();
  }
  PetscFunctionReturn(0);
}

/* Copies a vector from the CPU to the GPU unless we already have an up-to-date copy on the GPU */
PetscErrorCode VecCUDACopyToGPU(Vec v)
{
  Vec_CUDA       *veccuda;
  PetscScalar    *varray;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  PetscCall(VecCUDAAllocateCheck(v));
  if (v->offloadmask == PETSC_OFFLOAD_CPU) {
    PetscCall(PetscLogEventBegin(VEC_CUDACopyToGPU,v,0,0,0));
    veccuda        = (Vec_CUDA*)v->spptr;
    varray         = veccuda->GPUarray;
    PetscCallCUDA(cudaMemcpy(varray,((Vec_Seq*)v->data)->array,v->map->n*sizeof(PetscScalar),cudaMemcpyHostToDevice));
    PetscCall(PetscLogCpuToGpu((v->map->n)*sizeof(PetscScalar)));
    PetscCall(PetscLogEventEnd(VEC_CUDACopyToGPU,v,0,0,0));
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

/*
     VecCUDACopyFromGPU - Copies a vector from the GPU to the CPU unless we already have an up-to-date copy on the CPU
*/
PetscErrorCode VecCUDACopyFromGPU(Vec v)
{
  Vec_CUDA       *veccuda;
  PetscScalar    *varray;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  PetscCall(VecCUDAAllocateCheckHost(v));
  if (v->offloadmask == PETSC_OFFLOAD_GPU) {
    PetscCall(PetscLogEventBegin(VEC_CUDACopyFromGPU,v,0,0,0));
    veccuda        = (Vec_CUDA*)v->spptr;
    varray         = veccuda->GPUarray;
    PetscCallCUDA(cudaMemcpy(((Vec_Seq*)v->data)->array,varray,v->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
    PetscCall(PetscLogGpuToCpu((v->map->n)*sizeof(PetscScalar)));
    PetscCall(PetscLogEventEnd(VEC_CUDACopyFromGPU,v,0,0,0));
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

/*MC
   VECSEQCUDA - VECSEQCUDA = "seqcuda" - The basic sequential vector, modified to use CUDA

   Options Database Keys:
. -vec_type seqcuda - sets the vector type to VECSEQCUDA during a call to VecSetFromOptions()

  Level: beginner

.seealso: `VecCreate()`, `VecSetType()`, `VecSetFromOptions()`, `VecCreateSeqWithArray()`, `VECMPI`, `VecType`, `VecCreateMPI()`, `VecCreateSeq()`, `VecSetPinnedMemoryMin()`
M*/

PetscErrorCode VecAYPX_SeqCUDA(Vec yin,PetscScalar alpha,Vec xin)
{
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscBLASInt      one = 1,bn = 0;
  PetscScalar       sone = 1.0;
  cublasHandle_t    cublasv2handle;

  PetscFunctionBegin;
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(PetscBLASIntCast(yin->map->n,&bn));
  PetscCall(VecCUDAGetArrayRead(xin,&xarray));
  PetscCall(VecCUDAGetArray(yin,&yarray));
  PetscCall(PetscLogGpuTimeBegin());
  if (alpha == (PetscScalar)0.0) {
    PetscCallCUDA(cudaMemcpy(yarray,xarray,bn*sizeof(PetscScalar),cudaMemcpyDeviceToDevice));
  } else if (alpha == (PetscScalar)1.0) {
    PetscCallCUBLAS(cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,yarray,one));
    PetscCall(PetscLogGpuFlops(1.0*yin->map->n));
  } else {
    PetscCallCUBLAS(cublasXscal(cublasv2handle,bn,&alpha,yarray,one));
    PetscCallCUBLAS(cublasXaxpy(cublasv2handle,bn,&sone,xarray,one,yarray,one));
    PetscCall(PetscLogGpuFlops(2.0*yin->map->n));
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(VecCUDARestoreArrayRead(xin,&xarray));
  PetscCall(VecCUDARestoreArray(yin,&yarray));
  PetscCall(PetscLogCpuToGpuScalar(sizeof(PetscScalar)));
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPY_SeqCUDA(Vec yin,PetscScalar alpha,Vec xin)
{
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;
  PetscBool         xiscuda;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) PetscFunctionReturn(0);
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)xin,&xiscuda,VECSEQCUDA,VECMPICUDA,""));
  if (xiscuda) {
    PetscCall(PetscBLASIntCast(yin->map->n,&bn));
    PetscCall(VecCUDAGetArrayRead(xin,&xarray));
    PetscCall(VecCUDAGetArray(yin,&yarray));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUBLAS(cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,yarray,one));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(VecCUDARestoreArrayRead(xin,&xarray));
    PetscCall(VecCUDARestoreArray(yin,&yarray));
    PetscCall(PetscLogGpuFlops(2.0*yin->map->n));
    PetscCall(PetscLogCpuToGpuScalar(sizeof(PetscScalar)));
  } else {
    PetscCall(VecAXPY_Seq(yin,alpha,xin));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseDivide_SeqCUDA(Vec win, Vec xin, Vec yin)
{
  PetscInt                              n = xin->map->n;
  const PetscScalar                     *xarray=NULL,*yarray=NULL;
  PetscScalar                           *warray=NULL;
  thrust::device_ptr<const PetscScalar> xptr,yptr;
  thrust::device_ptr<PetscScalar>       wptr;

  PetscFunctionBegin;
  if (xin->boundtocpu || yin->boundtocpu) {
    PetscCall(VecPointwiseDivide_Seq(win,xin,yin));
    PetscFunctionReturn(0);
  }
  PetscCall(VecCUDAGetArrayWrite(win,&warray));
  PetscCall(VecCUDAGetArrayRead(xin,&xarray));
  PetscCall(VecCUDAGetArrayRead(yin,&yarray));
  PetscCall(PetscLogGpuTimeBegin());
  try {
    wptr = thrust::device_pointer_cast(warray);
    xptr = thrust::device_pointer_cast(xarray);
    yptr = thrust::device_pointer_cast(yarray);
    thrust::transform(VecCUDAThrustPolicy(win),xptr,xptr+n,yptr,wptr,thrust::divides<PetscScalar>());
  } catch (char *ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(PetscLogGpuFlops(n));
  PetscCall(VecCUDARestoreArrayRead(xin,&xarray));
  PetscCall(VecCUDARestoreArrayRead(yin,&yarray));
  PetscCall(VecCUDARestoreArrayWrite(win,&warray));
  PetscFunctionReturn(0);
}

PetscErrorCode VecWAXPY_SeqCUDA(Vec win,PetscScalar alpha,Vec xin, Vec yin)
{
  const PetscScalar *xarray=NULL,*yarray=NULL;
  PetscScalar       *warray=NULL;
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;
  cudaStream_t      stream;

  PetscFunctionBegin;
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(PetscBLASIntCast(win->map->n,&bn));
  if (alpha == (PetscScalar)0.0) {
    PetscCall(VecCopy_SeqCUDA(yin,win));
  } else {
    PetscCall(VecCUDAGetArrayRead(xin,&xarray));
    PetscCall(VecCUDAGetArrayRead(yin,&yarray));
    PetscCall(VecCUDAGetArrayWrite(win,&warray));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUBLAS(cublasGetStream(cublasv2handle,&stream));
    PetscCallCUDA(cudaMemcpyAsync(warray,yarray,win->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice,stream));
    PetscCallCUBLAS(cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,warray,one));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(2*win->map->n));
    PetscCall(VecCUDARestoreArrayRead(xin,&xarray));
    PetscCall(VecCUDARestoreArrayRead(yin,&yarray));
    PetscCall(VecCUDARestoreArrayWrite(win,&warray));
    PetscCall(PetscLogCpuToGpuScalar(sizeof(PetscScalar)));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecMAXPY_SeqCUDA(Vec xin, PetscInt nv,const PetscScalar *alpha,Vec *y)
{
  PetscInt          n = xin->map->n,j;
  PetscScalar       *xarray;
  const PetscScalar *yarray;
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;

  PetscFunctionBegin;
  PetscCall(PetscLogGpuFlops(nv*2.0*n));
  PetscCall(PetscLogCpuToGpuScalar(nv*sizeof(PetscScalar)));
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(PetscBLASIntCast(n,&bn));
  PetscCall(VecCUDAGetArray(xin,&xarray));
  PetscCall(PetscLogGpuTimeBegin());
  for (j=0; j<nv; j++) {
    PetscCall(VecCUDAGetArrayRead(y[j],&yarray));
    PetscCallCUBLAS(cublasXaxpy(cublasv2handle,bn,alpha+j,yarray,one,xarray,one));
    PetscCall(VecCUDARestoreArrayRead(y[j],&yarray));
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(VecCUDARestoreArray(xin,&xarray));
  PetscFunctionReturn(0);
}

PetscErrorCode VecDot_SeqCUDA(Vec xin,Vec yin,PetscScalar *z)
{
  const PetscScalar *xarray,*yarray;
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;

  PetscFunctionBegin;
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(PetscBLASIntCast(yin->map->n,&bn));
  PetscCall(VecCUDAGetArrayRead(xin,&xarray));
  PetscCall(VecCUDAGetArrayRead(yin,&yarray));
  /* arguments y, x are reversed because BLAS complex conjugates the first argument, PETSc the second */
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUBLAS(cublasXdot(cublasv2handle,bn,yarray,one,xarray,one,z));
  PetscCall(PetscLogGpuTimeEnd());
  if (xin->map->n >0) {
    PetscCall(PetscLogGpuFlops(2.0*xin->map->n-1));
  }
  PetscCall(PetscLogGpuToCpuScalar(sizeof(PetscScalar)));
  PetscCall(VecCUDARestoreArrayRead(xin,&xarray));
  PetscCall(VecCUDARestoreArrayRead(yin,&yarray));
  PetscFunctionReturn(0);
}

//
// CUDA kernels for MDot to follow
//

// set work group size to be a power of 2 (128 is usually a good compromise between portability and speed)
#define MDOT_WORKGROUP_SIZE 128
#define MDOT_WORKGROUP_NUM  128

#if !defined(PETSC_USE_COMPLEX)
// M = 2:
__global__ void VecMDot_SeqCUDA_kernel2(const PetscScalar *x,const PetscScalar *y0,const PetscScalar *y1,
                                        PetscInt size, PetscScalar *group_results)
{
  __shared__ PetscScalar tmp_buffer[2*MDOT_WORKGROUP_SIZE];
  PetscInt entries_per_group = (size - 1) / gridDim.x + 1;
  entries_per_group = (entries_per_group == 0) ? 1 : entries_per_group;  // for very small vectors, a group should still do some work
  PetscInt vec_start_index = blockIdx.x * entries_per_group;
  PetscInt vec_stop_index  = PetscMin((blockIdx.x + 1) * entries_per_group, size); // don't go beyond vec size

  PetscScalar entry_x    = 0;
  PetscScalar group_sum0 = 0;
  PetscScalar group_sum1 = 0;
  for (PetscInt i = vec_start_index + threadIdx.x; i < vec_stop_index; i += blockDim.x) {
    entry_x     = x[i];   // load only once from global memory!
    group_sum0 += entry_x * y0[i];
    group_sum1 += entry_x * y1[i];
  }
  tmp_buffer[threadIdx.x]                       = group_sum0;
  tmp_buffer[threadIdx.x + MDOT_WORKGROUP_SIZE] = group_sum1;

  // parallel reduction
  for (PetscInt stride = blockDim.x/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      tmp_buffer[threadIdx.x                      ] += tmp_buffer[threadIdx.x+stride                      ];
      tmp_buffer[threadIdx.x + MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + MDOT_WORKGROUP_SIZE];
    }
  }

  // write result of group to group_results
  if (threadIdx.x == 0) {
    group_results[blockIdx.x]             = tmp_buffer[0];
    group_results[blockIdx.x + gridDim.x] = tmp_buffer[MDOT_WORKGROUP_SIZE];
  }
}

// M = 3:
__global__ void VecMDot_SeqCUDA_kernel3(const PetscScalar *x,const PetscScalar *y0,const PetscScalar *y1,const PetscScalar *y2,
                                        PetscInt size, PetscScalar *group_results)
{
  __shared__ PetscScalar tmp_buffer[3*MDOT_WORKGROUP_SIZE];
  PetscInt entries_per_group = (size - 1) / gridDim.x + 1;
  entries_per_group = (entries_per_group == 0) ? 1 : entries_per_group;  // for very small vectors, a group should still do some work
  PetscInt vec_start_index = blockIdx.x * entries_per_group;
  PetscInt vec_stop_index  = PetscMin((blockIdx.x + 1) * entries_per_group, size); // don't go beyond vec size

  PetscScalar entry_x    = 0;
  PetscScalar group_sum0 = 0;
  PetscScalar group_sum1 = 0;
  PetscScalar group_sum2 = 0;
  for (PetscInt i = vec_start_index + threadIdx.x; i < vec_stop_index; i += blockDim.x) {
    entry_x     = x[i];   // load only once from global memory!
    group_sum0 += entry_x * y0[i];
    group_sum1 += entry_x * y1[i];
    group_sum2 += entry_x * y2[i];
  }
  tmp_buffer[threadIdx.x]                           = group_sum0;
  tmp_buffer[threadIdx.x +     MDOT_WORKGROUP_SIZE] = group_sum1;
  tmp_buffer[threadIdx.x + 2 * MDOT_WORKGROUP_SIZE] = group_sum2;

  // parallel reduction
  for (PetscInt stride = blockDim.x/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      tmp_buffer[threadIdx.x                          ] += tmp_buffer[threadIdx.x+stride                          ];
      tmp_buffer[threadIdx.x +     MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride +     MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 2 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 2 * MDOT_WORKGROUP_SIZE];
    }
  }

  // write result of group to group_results
  if (threadIdx.x == 0) {
    group_results[blockIdx.x                ] = tmp_buffer[0];
    group_results[blockIdx.x +     gridDim.x] = tmp_buffer[    MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 2 * gridDim.x] = tmp_buffer[2 * MDOT_WORKGROUP_SIZE];
  }
}

// M = 4:
__global__ void VecMDot_SeqCUDA_kernel4(const PetscScalar *x,const PetscScalar *y0,const PetscScalar *y1,const PetscScalar *y2,const PetscScalar *y3,
                                        PetscInt size, PetscScalar *group_results)
{
  __shared__ PetscScalar tmp_buffer[4*MDOT_WORKGROUP_SIZE];
  PetscInt entries_per_group = (size - 1) / gridDim.x + 1;
  entries_per_group = (entries_per_group == 0) ? 1 : entries_per_group;  // for very small vectors, a group should still do some work
  PetscInt vec_start_index = blockIdx.x * entries_per_group;
  PetscInt vec_stop_index  = PetscMin((blockIdx.x + 1) * entries_per_group, size); // don't go beyond vec size

  PetscScalar entry_x    = 0;
  PetscScalar group_sum0 = 0;
  PetscScalar group_sum1 = 0;
  PetscScalar group_sum2 = 0;
  PetscScalar group_sum3 = 0;
  for (PetscInt i = vec_start_index + threadIdx.x; i < vec_stop_index; i += blockDim.x) {
    entry_x     = x[i];   // load only once from global memory!
    group_sum0 += entry_x * y0[i];
    group_sum1 += entry_x * y1[i];
    group_sum2 += entry_x * y2[i];
    group_sum3 += entry_x * y3[i];
  }
  tmp_buffer[threadIdx.x]                           = group_sum0;
  tmp_buffer[threadIdx.x +     MDOT_WORKGROUP_SIZE] = group_sum1;
  tmp_buffer[threadIdx.x + 2 * MDOT_WORKGROUP_SIZE] = group_sum2;
  tmp_buffer[threadIdx.x + 3 * MDOT_WORKGROUP_SIZE] = group_sum3;

  // parallel reduction
  for (PetscInt stride = blockDim.x/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      tmp_buffer[threadIdx.x                          ] += tmp_buffer[threadIdx.x+stride                          ];
      tmp_buffer[threadIdx.x +     MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride +     MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 2 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 2 * MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 3 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 3 * MDOT_WORKGROUP_SIZE];
    }
  }

  // write result of group to group_results
  if (threadIdx.x == 0) {
    group_results[blockIdx.x                ] = tmp_buffer[0];
    group_results[blockIdx.x +     gridDim.x] = tmp_buffer[    MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 2 * gridDim.x] = tmp_buffer[2 * MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 3 * gridDim.x] = tmp_buffer[3 * MDOT_WORKGROUP_SIZE];
  }
}

// M = 8:
__global__ void VecMDot_SeqCUDA_kernel8(const PetscScalar *x,const PetscScalar *y0,const PetscScalar *y1,const PetscScalar *y2,const PetscScalar *y3,
                                          const PetscScalar *y4,const PetscScalar *y5,const PetscScalar *y6,const PetscScalar *y7,
                                          PetscInt size, PetscScalar *group_results)
{
  __shared__ PetscScalar tmp_buffer[8*MDOT_WORKGROUP_SIZE];
  PetscInt entries_per_group = (size - 1) / gridDim.x + 1;
  entries_per_group = (entries_per_group == 0) ? 1 : entries_per_group;  // for very small vectors, a group should still do some work
  PetscInt vec_start_index = blockIdx.x * entries_per_group;
  PetscInt vec_stop_index  = PetscMin((blockIdx.x + 1) * entries_per_group, size); // don't go beyond vec size

  PetscScalar entry_x    = 0;
  PetscScalar group_sum0 = 0;
  PetscScalar group_sum1 = 0;
  PetscScalar group_sum2 = 0;
  PetscScalar group_sum3 = 0;
  PetscScalar group_sum4 = 0;
  PetscScalar group_sum5 = 0;
  PetscScalar group_sum6 = 0;
  PetscScalar group_sum7 = 0;
  for (PetscInt i = vec_start_index + threadIdx.x; i < vec_stop_index; i += blockDim.x) {
    entry_x     = x[i];   // load only once from global memory!
    group_sum0 += entry_x * y0[i];
    group_sum1 += entry_x * y1[i];
    group_sum2 += entry_x * y2[i];
    group_sum3 += entry_x * y3[i];
    group_sum4 += entry_x * y4[i];
    group_sum5 += entry_x * y5[i];
    group_sum6 += entry_x * y6[i];
    group_sum7 += entry_x * y7[i];
  }
  tmp_buffer[threadIdx.x]                           = group_sum0;
  tmp_buffer[threadIdx.x +     MDOT_WORKGROUP_SIZE] = group_sum1;
  tmp_buffer[threadIdx.x + 2 * MDOT_WORKGROUP_SIZE] = group_sum2;
  tmp_buffer[threadIdx.x + 3 * MDOT_WORKGROUP_SIZE] = group_sum3;
  tmp_buffer[threadIdx.x + 4 * MDOT_WORKGROUP_SIZE] = group_sum4;
  tmp_buffer[threadIdx.x + 5 * MDOT_WORKGROUP_SIZE] = group_sum5;
  tmp_buffer[threadIdx.x + 6 * MDOT_WORKGROUP_SIZE] = group_sum6;
  tmp_buffer[threadIdx.x + 7 * MDOT_WORKGROUP_SIZE] = group_sum7;

  // parallel reduction
  for (PetscInt stride = blockDim.x/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      tmp_buffer[threadIdx.x                          ] += tmp_buffer[threadIdx.x+stride                          ];
      tmp_buffer[threadIdx.x +     MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride +     MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 2 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 2 * MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 3 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 3 * MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 4 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 4 * MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 5 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 5 * MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 6 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 6 * MDOT_WORKGROUP_SIZE];
      tmp_buffer[threadIdx.x + 7 * MDOT_WORKGROUP_SIZE] += tmp_buffer[threadIdx.x+stride + 7 * MDOT_WORKGROUP_SIZE];
    }
  }

  // write result of group to group_results
  if (threadIdx.x == 0) {
    group_results[blockIdx.x                ] = tmp_buffer[0];
    group_results[blockIdx.x +     gridDim.x] = tmp_buffer[    MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 2 * gridDim.x] = tmp_buffer[2 * MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 3 * gridDim.x] = tmp_buffer[3 * MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 4 * gridDim.x] = tmp_buffer[4 * MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 5 * gridDim.x] = tmp_buffer[5 * MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 6 * gridDim.x] = tmp_buffer[6 * MDOT_WORKGROUP_SIZE];
    group_results[blockIdx.x + 7 * gridDim.x] = tmp_buffer[7 * MDOT_WORKGROUP_SIZE];
  }
}
#endif /* !defined(PETSC_USE_COMPLEX) */

PetscErrorCode VecMDot_SeqCUDA(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscInt          i,n = xin->map->n,current_y_index = 0;
  const PetscScalar *xptr,*y0ptr,*y1ptr,*y2ptr,*y3ptr,*y4ptr,*y5ptr,*y6ptr,*y7ptr;
#if !defined(PETSC_USE_COMPLEX)
  PetscInt          nv1 = ((nv % 4) == 1) ? nv-1: nv,j;
  PetscScalar       *group_results_gpu,*group_results_cpu;
#endif
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;

  PetscFunctionBegin;
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(PetscBLASIntCast(xin->map->n,&bn));
  PetscCheck(nv > 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Number of vectors provided to VecMDot_SeqCUDA not positive.");
  /* Handle the case of local size zero first */
  if (!xin->map->n) {
    for (i=0; i<nv; ++i) z[i] = 0;
    PetscFunctionReturn(0);
  }

#if !defined(PETSC_USE_COMPLEX)
  // allocate scratchpad memory for the results of individual work groups:
  PetscCallCUDA(cudaMalloc((void**)&group_results_gpu, nv1*sizeof(PetscScalar)*MDOT_WORKGROUP_NUM));
  PetscCall(PetscMalloc1(nv1*MDOT_WORKGROUP_NUM,&group_results_cpu));
#endif
  PetscCall(VecCUDAGetArrayRead(xin,&xptr));
  PetscCall(PetscLogGpuTimeBegin());

  while (current_y_index < nv)
  {
    switch (nv - current_y_index) {

      case 7:
      case 6:
      case 5:
      case 4:
        PetscCall(VecCUDAGetArrayRead(yin[current_y_index  ],&y0ptr));
        PetscCall(VecCUDAGetArrayRead(yin[current_y_index+1],&y1ptr));
        PetscCall(VecCUDAGetArrayRead(yin[current_y_index+2],&y2ptr));
        PetscCall(VecCUDAGetArrayRead(yin[current_y_index+3],&y3ptr));
#if defined(PETSC_USE_COMPLEX)
        PetscCallCUBLAS(cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]));
        PetscCallCUBLAS(cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,&z[current_y_index+1]));
        PetscCallCUBLAS(cublasXdot(cublasv2handle,bn,y2ptr,one,xptr,one,&z[current_y_index+2]));
        PetscCallCUBLAS(cublasXdot(cublasv2handle,bn,y3ptr,one,xptr,one,&z[current_y_index+3]));
#else
        VecMDot_SeqCUDA_kernel4<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,y2ptr,y3ptr,n,group_results_gpu+current_y_index*MDOT_WORKGROUP_NUM);
#endif
        PetscCall(VecCUDARestoreArrayRead(yin[current_y_index  ],&y0ptr));
        PetscCall(VecCUDARestoreArrayRead(yin[current_y_index+1],&y1ptr));
        PetscCall(VecCUDARestoreArrayRead(yin[current_y_index+2],&y2ptr));
        PetscCall(VecCUDARestoreArrayRead(yin[current_y_index+3],&y3ptr));
        current_y_index += 4;
        break;

      case 3:
        PetscCall(VecCUDAGetArrayRead(yin[current_y_index  ],&y0ptr));
        PetscCall(VecCUDAGetArrayRead(yin[current_y_index+1],&y1ptr));
        PetscCall(VecCUDAGetArrayRead(yin[current_y_index+2],&y2ptr));

#if defined(PETSC_USE_COMPLEX)
        PetscCallCUBLAS(cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]));
        PetscCallCUBLAS(cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,&z[current_y_index+1]));
        PetscCallCUBLAS(cublasXdot(cublasv2handle,bn,y2ptr,one,xptr,one,&z[current_y_index+2]));
#else
        VecMDot_SeqCUDA_kernel3<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,y2ptr,n,group_results_gpu+current_y_index*MDOT_WORKGROUP_NUM);
#endif
        PetscCall(VecCUDARestoreArrayRead(yin[current_y_index  ],&y0ptr));
        PetscCall(VecCUDARestoreArrayRead(yin[current_y_index+1],&y1ptr));
        PetscCall(VecCUDARestoreArrayRead(yin[current_y_index+2],&y2ptr));
        current_y_index += 3;
        break;

      case 2:
        PetscCall(VecCUDAGetArrayRead(yin[current_y_index],&y0ptr));
        PetscCall(VecCUDAGetArrayRead(yin[current_y_index+1],&y1ptr));
#if defined(PETSC_USE_COMPLEX)
        PetscCallCUBLAS(cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]));
        PetscCallCUBLAS(cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,&z[current_y_index+1]));
#else
        VecMDot_SeqCUDA_kernel2<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,n,group_results_gpu+current_y_index*MDOT_WORKGROUP_NUM);
#endif
        PetscCall(VecCUDARestoreArrayRead(yin[current_y_index],&y0ptr));
        PetscCall(VecCUDARestoreArrayRead(yin[current_y_index+1],&y1ptr));
        current_y_index += 2;
        break;

      case 1:
        PetscCall(VecCUDAGetArrayRead(yin[current_y_index],&y0ptr));
        PetscCallCUBLAS(cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]));
        PetscCall(VecCUDARestoreArrayRead(yin[current_y_index],&y0ptr));
        current_y_index += 1;
        break;

      default: // 8 or more vectors left
        PetscCall(VecCUDAGetArrayRead(yin[current_y_index  ],&y0ptr));
        PetscCall(VecCUDAGetArrayRead(yin[current_y_index+1],&y1ptr));
        PetscCall(VecCUDAGetArrayRead(yin[current_y_index+2],&y2ptr));
        PetscCall(VecCUDAGetArrayRead(yin[current_y_index+3],&y3ptr));
        PetscCall(VecCUDAGetArrayRead(yin[current_y_index+4],&y4ptr));
        PetscCall(VecCUDAGetArrayRead(yin[current_y_index+5],&y5ptr));
        PetscCall(VecCUDAGetArrayRead(yin[current_y_index+6],&y6ptr));
        PetscCall(VecCUDAGetArrayRead(yin[current_y_index+7],&y7ptr));
#if defined(PETSC_USE_COMPLEX)
        PetscCallCUBLAS(cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]));
        PetscCallCUBLAS(cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,&z[current_y_index+1]));
        PetscCallCUBLAS(cublasXdot(cublasv2handle,bn,y2ptr,one,xptr,one,&z[current_y_index+2]));
        PetscCallCUBLAS(cublasXdot(cublasv2handle,bn,y3ptr,one,xptr,one,&z[current_y_index+3]));
        PetscCallCUBLAS(cublasXdot(cublasv2handle,bn,y4ptr,one,xptr,one,&z[current_y_index+4]));
        PetscCallCUBLAS(cublasXdot(cublasv2handle,bn,y5ptr,one,xptr,one,&z[current_y_index+5]));
        PetscCallCUBLAS(cublasXdot(cublasv2handle,bn,y6ptr,one,xptr,one,&z[current_y_index+6]));
        PetscCallCUBLAS(cublasXdot(cublasv2handle,bn,y7ptr,one,xptr,one,&z[current_y_index+7]));
#else
        VecMDot_SeqCUDA_kernel8<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,y2ptr,y3ptr,y4ptr,y5ptr,y6ptr,y7ptr,n,group_results_gpu+current_y_index*MDOT_WORKGROUP_NUM);
#endif
        PetscCall(VecCUDARestoreArrayRead(yin[current_y_index  ],&y0ptr));
        PetscCall(VecCUDARestoreArrayRead(yin[current_y_index+1],&y1ptr));
        PetscCall(VecCUDARestoreArrayRead(yin[current_y_index+2],&y2ptr));
        PetscCall(VecCUDARestoreArrayRead(yin[current_y_index+3],&y3ptr));
        PetscCall(VecCUDARestoreArrayRead(yin[current_y_index+4],&y4ptr));
        PetscCall(VecCUDARestoreArrayRead(yin[current_y_index+5],&y5ptr));
        PetscCall(VecCUDARestoreArrayRead(yin[current_y_index+6],&y6ptr));
        PetscCall(VecCUDARestoreArrayRead(yin[current_y_index+7],&y7ptr));
        current_y_index += 8;
        break;
    }
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(VecCUDARestoreArrayRead(xin,&xptr));

#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscLogGpuToCpuScalar(nv*sizeof(PetscScalar)));
#else
  // copy results to CPU
  PetscCallCUDA(cudaMemcpy(group_results_cpu,group_results_gpu,nv1*sizeof(PetscScalar)*MDOT_WORKGROUP_NUM,cudaMemcpyDeviceToHost));

  // sum group results into z
  for (j=0; j<nv1; ++j) {
    z[j] = 0;
    for (i=j*MDOT_WORKGROUP_NUM; i<(j+1)*MDOT_WORKGROUP_NUM; ++i) z[j] += group_results_cpu[i];
  }
  PetscCall(PetscLogFlops(nv1*MDOT_WORKGROUP_NUM));
  PetscCallCUDA(cudaFree(group_results_gpu));
  PetscCall(PetscFree(group_results_cpu));
  PetscCall(PetscLogGpuToCpuScalar(nv1*sizeof(PetscScalar)*MDOT_WORKGROUP_NUM));
#endif
  PetscCall(PetscLogGpuFlops(PetscMax(nv*(2.0*n-1),0.0)));
  PetscFunctionReturn(0);
}

#undef MDOT_WORKGROUP_SIZE
#undef MDOT_WORKGROUP_NUM

PetscErrorCode VecSet_SeqCUDA(Vec xin,PetscScalar alpha)
{
  PetscInt                        n = xin->map->n;
  PetscScalar                     *xarray = NULL;
  thrust::device_ptr<PetscScalar> xptr;

  PetscFunctionBegin;
  PetscCall(VecCUDAGetArrayWrite(xin,&xarray));
  PetscCall(PetscLogGpuTimeBegin());
  if (alpha == (PetscScalar)0.0) {
    PetscCallCUDA(cudaMemset(xarray,0,n*sizeof(PetscScalar)));
  } else {
    try {
      xptr = thrust::device_pointer_cast(xarray);
      thrust::fill(xptr,xptr+n,alpha);
    } catch (char *ex) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
    }
    PetscCall(PetscLogCpuToGpuScalar(sizeof(PetscScalar)));
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(VecCUDARestoreArrayWrite(xin,&xarray));
  PetscFunctionReturn(0);
}

struct PetscScalarReciprocal
{
  __host__ __device__
  PetscScalar operator()(const PetscScalar& s)
  {
    return (s != (PetscScalar)0.0) ? (PetscScalar)1.0/s : 0.0;
  }
};

PetscErrorCode VecReciprocal_SeqCUDA(Vec v)
{
  PetscInt       n;
  PetscScalar    *x;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(VecCUDAGetArray(v,&x));
  PetscCall(PetscLogGpuTimeBegin());
  try {
    auto xptr = thrust::device_pointer_cast(x);
    thrust::transform(VecCUDAThrustPolicy(v),xptr,xptr+n,xptr,PetscScalarReciprocal());
  } catch (char *ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(VecCUDARestoreArray(v,&x));
  PetscFunctionReturn(0);
}

PetscErrorCode VecScale_SeqCUDA(Vec xin,PetscScalar alpha)
{
  PetscScalar    *xarray;
  PetscBLASInt   one = 1,bn = 0;
  cublasHandle_t cublasv2handle;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    PetscCall(VecSet_SeqCUDA(xin,alpha));
  } else if (alpha != (PetscScalar)1.0) {
    PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
    PetscCall(PetscBLASIntCast(xin->map->n,&bn));
    PetscCall(VecCUDAGetArray(xin,&xarray));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUBLAS(cublasXscal(cublasv2handle,bn,&alpha,xarray,one));
    PetscCall(VecCUDARestoreArray(xin,&xarray));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogCpuToGpuScalar(sizeof(PetscScalar)));
    PetscCall(PetscLogGpuFlops(xin->map->n));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_SeqCUDA(Vec xin,Vec yin,PetscScalar *z)
{
  const PetscScalar *xarray,*yarray;
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;

  PetscFunctionBegin;
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(PetscBLASIntCast(xin->map->n,&bn));
  PetscCall(VecCUDAGetArrayRead(xin,&xarray));
  PetscCall(VecCUDAGetArrayRead(yin,&yarray));
  PetscCall(PetscLogGpuTimeBegin());
  PetscCallCUBLAS(cublasXdotu(cublasv2handle,bn,xarray,one,yarray,one,z));
  PetscCall(PetscLogGpuTimeEnd());
  if (xin->map->n > 0) {
    PetscCall(PetscLogGpuFlops(2.0*xin->map->n-1));
  }
  PetscCall(PetscLogGpuToCpuScalar(sizeof(PetscScalar)));
  PetscCall(VecCUDARestoreArrayRead(yin,&yarray));
  PetscCall(VecCUDARestoreArrayRead(xin,&xarray));
  PetscFunctionReturn(0);
}

PetscErrorCode VecCopy_SeqCUDA(Vec xin,Vec yin)
{
  const PetscScalar *xarray;
  PetscScalar       *yarray;

  PetscFunctionBegin;
  if (xin != yin) {
    if (xin->offloadmask == PETSC_OFFLOAD_GPU) {
      PetscBool yiscuda;

      PetscCall(PetscObjectTypeCompareAny((PetscObject)yin,&yiscuda,VECSEQCUDA,VECMPICUDA,""));
      PetscCall(VecCUDAGetArrayRead(xin,&xarray));
      if (yiscuda) {
        PetscCall(VecCUDAGetArrayWrite(yin,&yarray));
      } else {
        PetscCall(VecGetArrayWrite(yin,&yarray));
      }
      PetscCall(PetscLogGpuTimeBegin());
      if (yiscuda) {
        PetscCallCUDA(cudaMemcpyAsync(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice,PetscDefaultCudaStream));
      } else {
        PetscCallCUDA(cudaMemcpy(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
        PetscCall(PetscLogGpuToCpu((yin->map->n)*sizeof(PetscScalar)));
      }
      PetscCall(PetscLogGpuTimeEnd());
      PetscCall(VecCUDARestoreArrayRead(xin,&xarray));
      if (yiscuda) {
        PetscCall(VecCUDARestoreArrayWrite(yin,&yarray));
      } else {
        PetscCall(VecRestoreArrayWrite(yin,&yarray));
      }
    } else if (xin->offloadmask == PETSC_OFFLOAD_CPU) {
      /* copy in CPU if we are on the CPU */
      PetscCall(VecCopy_SeqCUDA_Private(xin,yin));
    } else if (xin->offloadmask == PETSC_OFFLOAD_BOTH) {
      /* if xin is valid in both places, see where yin is and copy there (because it's probably where we'll want to next use it) */
      if (yin->offloadmask == PETSC_OFFLOAD_CPU) {
        /* copy in CPU */
        PetscCall(VecCopy_SeqCUDA_Private(xin,yin));
      } else if (yin->offloadmask == PETSC_OFFLOAD_GPU) {
        /* copy in GPU */
        PetscCall(VecCUDAGetArrayRead(xin,&xarray));
        PetscCall(VecCUDAGetArrayWrite(yin,&yarray));
        PetscCall(PetscLogGpuTimeBegin());
        PetscCallCUDA(cudaMemcpyAsync(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice,PetscDefaultCudaStream));
        PetscCall(PetscLogGpuTimeEnd());
        PetscCall(VecCUDARestoreArrayRead(xin,&xarray));
        PetscCall(VecCUDARestoreArrayWrite(yin,&yarray));
      } else if (yin->offloadmask == PETSC_OFFLOAD_BOTH) {
        /* xin and yin are both valid in both places (or yin was unallocated before the earlier call to allocatecheck
           default to copy in GPU (this is an arbitrary choice) */
        PetscCall(VecCUDAGetArrayRead(xin,&xarray));
        PetscCall(VecCUDAGetArrayWrite(yin,&yarray));
        PetscCall(PetscLogGpuTimeBegin());
        PetscCallCUDA(cudaMemcpyAsync(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice,PetscDefaultCudaStream));
        PetscCall(PetscLogGpuTimeEnd());
        PetscCall(VecCUDARestoreArrayRead(xin,&xarray));
        PetscCall(VecCUDARestoreArrayWrite(yin,&yarray));
      } else {
        PetscCall(VecCopy_SeqCUDA_Private(xin,yin));
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecSwap_SeqCUDA(Vec xin,Vec yin)
{
  PetscBLASInt   one = 1,bn = 0;
  PetscScalar    *xarray,*yarray;
  cublasHandle_t cublasv2handle;

  PetscFunctionBegin;
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(PetscBLASIntCast(xin->map->n,&bn));
  if (xin != yin) {
    PetscCall(VecCUDAGetArray(xin,&xarray));
    PetscCall(VecCUDAGetArray(yin,&yarray));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUBLAS(cublasXswap(cublasv2handle,bn,xarray,one,yarray,one));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(VecCUDARestoreArray(xin,&xarray));
    PetscCall(VecCUDARestoreArray(yin,&yarray));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBY_SeqCUDA(Vec yin,PetscScalar alpha,PetscScalar beta,Vec xin)
{
  PetscScalar       a = alpha,b = beta;
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscBLASInt      one = 1, bn = 0;
  cublasHandle_t    cublasv2handle;

  PetscFunctionBegin;
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(PetscBLASIntCast(yin->map->n,&bn));
  if (a == (PetscScalar)0.0) {
    PetscCall(VecScale_SeqCUDA(yin,beta));
  } else if (b == (PetscScalar)1.0) {
    PetscCall(VecAXPY_SeqCUDA(yin,alpha,xin));
  } else if (a == (PetscScalar)1.0) {
    PetscCall(VecAYPX_SeqCUDA(yin,beta,xin));
  } else if (b == (PetscScalar)0.0) {
    PetscCall(VecCUDAGetArrayRead(xin,&xarray));
    PetscCall(VecCUDAGetArray(yin,&yarray));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUDA(cudaMemcpy(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice));
    PetscCallCUBLAS(cublasXscal(cublasv2handle,bn,&alpha,yarray,one));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(xin->map->n));
    PetscCall(PetscLogCpuToGpuScalar(sizeof(PetscScalar)));
    PetscCall(VecCUDARestoreArrayRead(xin,&xarray));
    PetscCall(VecCUDARestoreArray(yin,&yarray));
  } else {
    PetscCall(VecCUDAGetArrayRead(xin,&xarray));
    PetscCall(VecCUDAGetArray(yin,&yarray));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUBLAS(cublasXscal(cublasv2handle,bn,&beta,yarray,one));
    PetscCallCUBLAS(cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,yarray,one));
    PetscCall(VecCUDARestoreArrayRead(xin,&xarray));
    PetscCall(VecCUDARestoreArray(yin,&yarray));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(3.0*xin->map->n));
    PetscCall(PetscLogCpuToGpuScalar(2*sizeof(PetscScalar)));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBYPCZ_SeqCUDA(Vec zin,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec xin,Vec yin)
{
  PetscInt       n = zin->map->n;

  PetscFunctionBegin;
  if (gamma == (PetscScalar)1.0) {
    /* z = ax + b*y + z */
    PetscCall(VecAXPY_SeqCUDA(zin,alpha,xin));
    PetscCall(VecAXPY_SeqCUDA(zin,beta,yin));
    PetscCall(PetscLogGpuFlops(4.0*n));
  } else {
    /* z = a*x + b*y + c*z */
    PetscCall(VecScale_SeqCUDA(zin,gamma));
    PetscCall(VecAXPY_SeqCUDA(zin,alpha,xin));
    PetscCall(VecAXPY_SeqCUDA(zin,beta,yin));
    PetscCall(PetscLogGpuFlops(5.0*n));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseMult_SeqCUDA(Vec win,Vec xin,Vec yin)
{
  PetscInt                              n = win->map->n;
  const PetscScalar                     *xarray,*yarray;
  PetscScalar                           *warray;
  thrust::device_ptr<const PetscScalar> xptr,yptr;
  thrust::device_ptr<PetscScalar>       wptr;

  PetscFunctionBegin;
  if (xin->boundtocpu || yin->boundtocpu) {
    PetscCall(VecPointwiseMult_Seq(win,xin,yin));
    PetscFunctionReturn(0);
  }
  PetscCall(VecCUDAGetArrayRead(xin,&xarray));
  PetscCall(VecCUDAGetArrayRead(yin,&yarray));
  PetscCall(VecCUDAGetArrayWrite(win,&warray));
  PetscCall(PetscLogGpuTimeBegin());
  try {
    wptr = thrust::device_pointer_cast(warray);
    xptr = thrust::device_pointer_cast(xarray);
    yptr = thrust::device_pointer_cast(yarray);
    thrust::transform(VecCUDAThrustPolicy(win),xptr,xptr+n,yptr,wptr,thrust::multiplies<PetscScalar>());
  } catch (char *ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(VecCUDARestoreArrayRead(xin,&xarray));
  PetscCall(VecCUDARestoreArrayRead(yin,&yarray));
  PetscCall(VecCUDARestoreArrayWrite(win,&warray));
  PetscCall(PetscLogGpuFlops(n));
  PetscFunctionReturn(0);
}

/* should do infinity norm in cuda */

PetscErrorCode VecNorm_SeqCUDA(Vec xin,NormType type,PetscReal *z)
{
  PetscInt          n = xin->map->n;
  PetscBLASInt      one = 1, bn = 0;
  const PetscScalar *xarray;
  cublasHandle_t    cublasv2handle;

  PetscFunctionBegin;
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  PetscCall(PetscBLASIntCast(n,&bn));
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    PetscCall(VecCUDAGetArrayRead(xin,&xarray));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUBLAS(cublasXnrm2(cublasv2handle,bn,xarray,one,z));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(VecCUDARestoreArrayRead(xin,&xarray));
    PetscCall(PetscLogGpuFlops(PetscMax(2.0*n-1,0.0)));
  } else if (type == NORM_INFINITY) {
    int  i;
    PetscCall(VecCUDAGetArrayRead(xin,&xarray));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUBLAS(cublasIXamax(cublasv2handle,bn,xarray,one,&i));
    PetscCall(PetscLogGpuTimeEnd());
    if (bn) {
      PetscScalar zs;
      PetscCallCUDA(cudaMemcpy(&zs,xarray+i-1,sizeof(PetscScalar),cudaMemcpyDeviceToHost));
      *z = PetscAbsScalar(zs);
    } else *z = 0.0;
    PetscCall(VecCUDARestoreArrayRead(xin,&xarray));
  } else if (type == NORM_1) {
    PetscCall(VecCUDAGetArrayRead(xin,&xarray));
    PetscCall(PetscLogGpuTimeBegin());
    PetscCallCUBLAS(cublasXasum(cublasv2handle,bn,xarray,one,z));
    PetscCall(VecCUDARestoreArrayRead(xin,&xarray));
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(PetscLogGpuFlops(PetscMax(n-1.0,0.0)));
  } else if (type == NORM_1_AND_2) {
    PetscCall(VecNorm_SeqCUDA(xin,NORM_1,z));
    PetscCall(VecNorm_SeqCUDA(xin,NORM_2,z+1));
  }
  PetscCall(PetscLogGpuToCpuScalar(sizeof(PetscReal)));
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotNorm2_SeqCUDA(Vec s, Vec t, PetscScalar *dp, PetscScalar *nm)
{
  PetscFunctionBegin;
  PetscCall(VecDot_SeqCUDA(s,t,dp));
  PetscCall(VecDot_SeqCUDA(t,t,nm));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecResetPreallocationCOO_SeqCUDA(Vec x)
{
  Vec_CUDA     *veccuda = static_cast<Vec_CUDA*>(x->spptr);

  PetscFunctionBegin;
  if (veccuda) {
    PetscCallCUDA(cudaFree(veccuda->jmap1_d));
    PetscCallCUDA(cudaFree(veccuda->perm1_d));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetPreallocationCOO_SeqCUDA(Vec x, PetscCount ncoo, const PetscInt coo_i[])
{
  Vec_Seq      *vecseq = static_cast<Vec_Seq*>(x->data);
  Vec_CUDA     *veccuda = static_cast<Vec_CUDA*>(x->spptr);
  PetscInt     m;

  PetscFunctionBegin;
  PetscCall(VecResetPreallocationCOO_SeqCUDA(x));
  PetscCall(VecSetPreallocationCOO_Seq(x,ncoo,coo_i));
  PetscCall(VecGetLocalSize(x,&m));
  PetscCallCUDA(cudaMalloc((void**)&veccuda->jmap1_d,sizeof(PetscCount)*(m+1)));
  PetscCallCUDA(cudaMalloc((void**)&veccuda->perm1_d,sizeof(PetscCount)*vecseq->tot1));
  PetscCallCUDA(cudaMemcpy(veccuda->jmap1_d,vecseq->jmap1,sizeof(PetscCount)*(m+1),cudaMemcpyHostToDevice));
  PetscCallCUDA(cudaMemcpy(veccuda->perm1_d,vecseq->perm1,sizeof(PetscCount)*vecseq->tot1,cudaMemcpyHostToDevice));
  PetscFunctionReturn(0);
}

__global__ static void VecAddCOOValues(const PetscScalar vv[],PetscCount m,const PetscCount jmap1[],const PetscCount perm1[],InsertMode imode,PetscScalar xv[])
{
  PetscCount        i = blockIdx.x*blockDim.x + threadIdx.x;
  const PetscCount  grid_size = gridDim.x * blockDim.x;
  for (; i<m; i+= grid_size) {
    PetscScalar sum = 0.0;
    for (PetscCount k=jmap1[i]; k<jmap1[i+1]; k++) sum += vv[perm1[k]];
    xv[i] = (imode == INSERT_VALUES? 0.0 : xv[i]) + sum;
  }
}

PetscErrorCode VecSetValuesCOO_SeqCUDA(Vec x,const PetscScalar v[],InsertMode imode)
{
  Vec_Seq                     *vecseq = static_cast<Vec_Seq*>(x->data);
  Vec_CUDA                    *veccuda = static_cast<Vec_CUDA*>(x->spptr);
  const PetscCount            *jmap1 = veccuda->jmap1_d;
  const PetscCount            *perm1 = veccuda->perm1_d;
  PetscScalar                 *xv;
  const PetscScalar           *vv = v;
  PetscInt                    m;
  PetscMemType                memtype;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(x,&m));
  PetscCall(PetscGetMemType(v,&memtype));

  if (PetscMemTypeHost(memtype)) { /* If user gave v[] in host, we might need to copy it to device if any */
    PetscCallCUDA(cudaMalloc((void**)&vv,vecseq->coo_n*sizeof(PetscScalar)));
    PetscCallCUDA(cudaMemcpy((void*)vv,v,vecseq->coo_n*sizeof(PetscScalar),cudaMemcpyHostToDevice));
  }

  if (imode == INSERT_VALUES) PetscCall(VecCUDAGetArrayWrite(x,&xv)); /* write vector */
  else PetscCall(VecCUDAGetArray(x,&xv)); /* read & write vector */

  if (m) {
    VecAddCOOValues<<<(m+255)/256,256>>>(vv,m,jmap1,perm1,imode,xv);
    PetscCallCUDA(cudaPeekAtLastError());
  }
  if (imode == INSERT_VALUES) PetscCall(VecCUDARestoreArrayWrite(x,&xv));
  else PetscCall(VecCUDARestoreArray(x,&xv));

  if (PetscMemTypeHost(memtype)) PetscCallCUDA(cudaFree((void*)vv));
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroy_SeqCUDA(Vec v)
{
  Vec_CUDA       *veccuda = (Vec_CUDA*)v->spptr;

  PetscFunctionBegin;
  if (v->spptr) {
    if (veccuda->GPUarray_allocated) {
     #if defined(PETSC_HAVE_NVSHMEM)
      if (veccuda->nvshmem) {
        PetscCall(PetscNvshmemFree(veccuda->GPUarray_allocated));
        veccuda->nvshmem = PETSC_FALSE;
      }
      else
     #endif
      PetscCallCUDA(cudaFree(veccuda->GPUarray_allocated));
      veccuda->GPUarray_allocated = NULL;
    }
    if (veccuda->stream) {
      PetscCallCUDA(cudaStreamDestroy(veccuda->stream));
    }
    PetscCall(VecResetPreallocationCOO_SeqCUDA(v));
  }
  PetscCall(VecDestroy_SeqCUDA_Private(v));
  PetscCall(PetscFree(v->spptr));
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_COMPLEX)
struct conjugate
{
  __host__ __device__
    PetscScalar operator()(const PetscScalar& x)
    {
      return PetscConj(x);
    }
};
#endif

PetscErrorCode VecConjugate_SeqCUDA(Vec xin)
{
#if defined(PETSC_USE_COMPLEX)
  PetscScalar                     *xarray;
  PetscInt                        n = xin->map->n;
  thrust::device_ptr<PetscScalar> xptr;

  PetscFunctionBegin;
  PetscCall(VecCUDAGetArray(xin,&xarray));
  PetscCall(PetscLogGpuTimeBegin());
  try {
    xptr = thrust::device_pointer_cast(xarray);
    thrust::transform(VecCUDAThrustPolicy(xin),xptr,xptr+n,xptr,conjugate());
  } catch (char *ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(VecCUDARestoreArray(xin,&xarray));
#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(0);
}

static inline PetscErrorCode VecGetLocalVectorK_SeqCUDA(Vec v,Vec w,PetscBool read)
{
  PetscBool      wisseqcuda;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(w,VEC_CLASSID,2);
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  PetscCall(PetscObjectTypeCompare((PetscObject)w,VECSEQCUDA,&wisseqcuda));
  if (w->data && wisseqcuda) {
    if (((Vec_Seq*)w->data)->array_allocated) {
      if (w->pinned_memory) {
        PetscCall(PetscMallocSetCUDAHost());
      }
      PetscCall(PetscFree(((Vec_Seq*)w->data)->array_allocated));
      if (w->pinned_memory) {
        PetscCall(PetscMallocResetCUDAHost());
        w->pinned_memory = PETSC_FALSE;
      }
    }
    ((Vec_Seq*)w->data)->array = NULL;
    ((Vec_Seq*)w->data)->unplacedarray = NULL;
  }
  if (w->spptr && wisseqcuda) {
    if (((Vec_CUDA*)w->spptr)->GPUarray) {
      PetscCallCUDA(cudaFree(((Vec_CUDA*)w->spptr)->GPUarray));
      ((Vec_CUDA*)w->spptr)->GPUarray = NULL;
    }
    if (((Vec_CUDA*)w->spptr)->stream) {
      PetscCallCUDA(cudaStreamDestroy(((Vec_CUDA*)w->spptr)->stream));
    }
    PetscCall(PetscFree(w->spptr));
  }

  if (v->petscnative && wisseqcuda) {
    PetscCall(PetscFree(w->data));
    w->data = v->data;
    w->offloadmask = v->offloadmask;
    w->pinned_memory = v->pinned_memory;
    w->spptr = v->spptr;
    PetscCall(PetscObjectStateIncrease((PetscObject)w));
  } else {
    if (read) {
      PetscCall(VecGetArrayRead(v,(const PetscScalar**)&((Vec_Seq*)w->data)->array));
    } else {
      PetscCall(VecGetArray(v,&((Vec_Seq*)w->data)->array));
    }
    w->offloadmask = PETSC_OFFLOAD_CPU;
    if (wisseqcuda) {
      PetscCall(VecCUDAAllocateCheck(w));
    }
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode VecRestoreLocalVectorK_SeqCUDA(Vec v,Vec w,PetscBool read)
{
  PetscBool      wisseqcuda;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(w,VEC_CLASSID,2);
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  PetscCall(PetscObjectTypeCompare((PetscObject)w,VECSEQCUDA,&wisseqcuda));
  if (v->petscnative && wisseqcuda) {
    v->data = w->data;
    v->offloadmask = w->offloadmask;
    v->pinned_memory = w->pinned_memory;
    v->spptr = w->spptr;
    w->data = 0;
    w->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
    w->spptr = 0;
  } else {
    if (read) {
      PetscCall(VecRestoreArrayRead(v,(const PetscScalar**)&((Vec_Seq*)w->data)->array));
    } else {
      PetscCall(VecRestoreArray(v,&((Vec_Seq*)w->data)->array));
    }
    if ((Vec_CUDA*)w->spptr && wisseqcuda) {
      PetscCallCUDA(cudaFree(((Vec_CUDA*)w->spptr)->GPUarray));
      ((Vec_CUDA*)w->spptr)->GPUarray = NULL;
      if (((Vec_CUDA*)v->spptr)->stream) {
        PetscCallCUDA(cudaStreamDestroy(((Vec_CUDA*)w->spptr)->stream));
      }
      PetscCall(PetscFree(w->spptr));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetLocalVector_SeqCUDA(Vec v,Vec w)
{
  PetscFunctionBegin;
  PetscCall(VecGetLocalVectorK_SeqCUDA(v,w,PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetLocalVectorRead_SeqCUDA(Vec v,Vec w)
{
  PetscFunctionBegin;
  PetscCall(VecGetLocalVectorK_SeqCUDA(v,w,PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreLocalVector_SeqCUDA(Vec v,Vec w)
{
  PetscFunctionBegin;
  PetscCall(VecRestoreLocalVectorK_SeqCUDA(v,w,PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreLocalVectorRead_SeqCUDA(Vec v,Vec w)
{
  PetscFunctionBegin;
  PetscCall(VecRestoreLocalVectorK_SeqCUDA(v,w,PETSC_TRUE));
  PetscFunctionReturn(0);
}

struct petscrealpart : public thrust::unary_function<PetscScalar,PetscReal>
{
  __host__ __device__
  PetscReal operator()(const PetscScalar& x) {
    return PetscRealPart(x);
  }
};

struct petscrealparti : public thrust::unary_function<thrust::tuple<PetscScalar, PetscInt>,thrust::tuple<PetscReal, PetscInt>>
{
  __host__ __device__
  thrust::tuple<PetscReal, PetscInt> operator()(const thrust::tuple<PetscScalar, PetscInt>& x) {
    return thrust::make_tuple(PetscRealPart(x.get<0>()), x.get<1>());
  }
};

struct petscmax : public thrust::binary_function<PetscReal,PetscReal,PetscReal>
{
  __host__ __device__
  PetscReal operator()(const PetscReal& x, const PetscReal& y) {
    return x < y ? y : x;
  }
};

struct petscmaxi : public thrust::binary_function<thrust::tuple<PetscReal, PetscInt>,thrust::tuple<PetscReal, PetscInt>,thrust::tuple<PetscReal, PetscInt>>
{
  __host__ __device__
  thrust::tuple<PetscReal, PetscInt> operator()(const thrust::tuple<PetscReal, PetscInt>& x, const thrust::tuple<PetscReal, PetscInt>& y) {
    return x.get<0>() < y.get<0>() ? thrust::make_tuple(y.get<0>(), y.get<1>()) :
           (x.get<0>() != y.get<0>() ? thrust::make_tuple(x.get<0>(), x.get<1>()) :
           (x.get<1>() < y.get<1>() ? thrust::make_tuple(x.get<0>(), x.get<1>()) : thrust::make_tuple(y.get<0>(), y.get<1>())));
  }
};

struct petscmin : public thrust::binary_function<PetscReal,PetscReal,PetscReal>
{
  __host__ __device__
  PetscReal operator()(const PetscReal& x, const PetscReal& y) {
    return x < y ? x : y;
  }
};

struct petscmini : public thrust::binary_function<thrust::tuple<PetscReal, PetscInt>,thrust::tuple<PetscReal, PetscInt>,thrust::tuple<PetscReal, PetscInt>>
{
  __host__ __device__
  thrust::tuple<PetscReal, PetscInt> operator()(const thrust::tuple<PetscReal, PetscInt>& x, const thrust::tuple<PetscReal, PetscInt>& y) {
    return x.get<0>() > y.get<0>() ? thrust::make_tuple(y.get<0>(), y.get<1>()) :
           (x.get<0>() != y.get<0>() ? thrust::make_tuple(x.get<0>(), x.get<1>()) :
           (x.get<1>() < y.get<1>() ? thrust::make_tuple(x.get<0>(), x.get<1>()) : thrust::make_tuple(y.get<0>(), y.get<1>())));
  }
};

PetscErrorCode VecMax_SeqCUDA(Vec v, PetscInt *p, PetscReal *m)
{
  PetscInt                              n = v->map->n;
  const PetscScalar                     *av;
  thrust::device_ptr<const PetscScalar> avpt;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  if (!n) {
    *m = PETSC_MIN_REAL;
    if (p) *p = -1;
    PetscFunctionReturn(0);
  }
  PetscCall(VecCUDAGetArrayRead(v,&av));
  avpt = thrust::device_pointer_cast(av);
  PetscCall(PetscLogGpuTimeBegin());
  if (p) {
    thrust::tuple<PetscReal,PetscInt> res(PETSC_MIN_REAL,-1);
    auto zibit = thrust::make_zip_iterator(thrust::make_tuple(avpt,thrust::counting_iterator<PetscInt>(0)));
    try {
#if defined(PETSC_USE_COMPLEX)
      res = thrust::transform_reduce(VecCUDAThrustPolicy(v),zibit,zibit+n,petscrealparti(),res,petscmaxi());
#else
      res = thrust::reduce(VecCUDAThrustPolicy(v),zibit,zibit+n,res,petscmaxi());
#endif
    } catch (char *ex) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
    }
    *m = res.get<0>();
    *p = res.get<1>();
  } else {
    try {
#if defined(PETSC_USE_COMPLEX)
      *m = thrust::transform_reduce(VecCUDAThrustPolicy(v),avpt,avpt+n,petscrealpart(),PETSC_MIN_REAL,petscmax());
#else
      *m = thrust::reduce(VecCUDAThrustPolicy(v),avpt,avpt+n,PETSC_MIN_REAL,petscmax());
#endif
    } catch (char *ex) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
    }
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(VecCUDARestoreArrayRead(v,&av));
  PetscFunctionReturn(0);
}

PetscErrorCode VecMin_SeqCUDA(Vec v, PetscInt *p, PetscReal *m)
{
  PetscInt                              n = v->map->n;
  const PetscScalar                     *av;
  thrust::device_ptr<const PetscScalar> avpt;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  if (!n) {
    *m = PETSC_MAX_REAL;
    if (p) *p = -1;
    PetscFunctionReturn(0);
  }
  PetscCall(VecCUDAGetArrayRead(v,&av));
  avpt = thrust::device_pointer_cast(av);
  PetscCall(PetscLogGpuTimeBegin());
  if (p) {
    thrust::tuple<PetscReal,PetscInt> res(PETSC_MAX_REAL,-1);
    auto zibit = thrust::make_zip_iterator(thrust::make_tuple(avpt,thrust::counting_iterator<PetscInt>(0)));
    try {
#if defined(PETSC_USE_COMPLEX)
      res = thrust::transform_reduce(VecCUDAThrustPolicy(v),zibit,zibit+n,petscrealparti(),res,petscmini());
#else
      res = thrust::reduce(VecCUDAThrustPolicy(v),zibit,zibit+n,res,petscmini());
#endif
    } catch (char *ex) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
    }
    *m = res.get<0>();
    *p = res.get<1>();
  } else {
    try {
#if defined(PETSC_USE_COMPLEX)
      *m = thrust::transform_reduce(VecCUDAThrustPolicy(v),avpt,avpt+n,petscrealpart(),PETSC_MAX_REAL,petscmin());
#else
      *m = thrust::reduce(VecCUDAThrustPolicy(v),avpt,avpt+n,PETSC_MAX_REAL,petscmin());
#endif
    } catch (char *ex) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
    }
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(VecCUDARestoreArrayRead(v,&av));
  PetscFunctionReturn(0);
}

PetscErrorCode VecSum_SeqCUDA(Vec v,PetscScalar *sum)
{
  PetscInt                              n = v->map->n;
  const PetscScalar                     *a;
  thrust::device_ptr<const PetscScalar> dptr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  PetscCall(VecCUDAGetArrayRead(v,&a));
  dptr = thrust::device_pointer_cast(a);
  PetscCall(PetscLogGpuTimeBegin());
  try {
    *sum = thrust::reduce(VecCUDAThrustPolicy(v),dptr,dptr+n,PetscScalar(0.0));
  } catch (char *ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(VecCUDARestoreArrayRead(v,&a));
  PetscFunctionReturn(0);
}

struct petscshift : public thrust::unary_function<PetscScalar,PetscScalar>
{
  const PetscScalar shift_;
  petscshift(PetscScalar shift) : shift_(shift){}
  __host__ __device__
  PetscScalar operator()(PetscScalar x) {return x + shift_;}
};

PetscErrorCode VecShift_SeqCUDA(Vec v,PetscScalar shift)
{
  PetscInt                              n = v->map->n;
  PetscScalar                           *a;
  thrust::device_ptr<PetscScalar>       dptr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  PetscCall(VecCUDAGetArray(v,&a));
  dptr = thrust::device_pointer_cast(a);
  PetscCall(PetscLogGpuTimeBegin());
  try {
    thrust::transform(VecCUDAThrustPolicy(v),dptr,dptr+n,dptr,petscshift(shift)); /* in-place transform */
  } catch (char *ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  PetscCall(PetscLogGpuTimeEnd());
  PetscCall(VecCUDARestoreArray(v,&a));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_NVSHMEM)
/* Free old CUDA array and re-allocate a new one from nvshmem symmetric heap.
   New array does not retain values in the old array. The offload mask is not changed.

   Note: the function is only meant to be used in MatAssemblyEnd_MPIAIJCUSPARSE.
 */
PetscErrorCode  VecAllocateNVSHMEM_SeqCUDA(Vec v)
{
  cudaError_t    cerr;
  Vec_CUDA       *veccuda = (Vec_CUDA*)v->spptr;
  PetscInt       n;

  PetscFunctionBegin;
  PetscCallCUDA(cudaFree(veccuda->GPUarray_allocated));
  PetscCall(VecGetLocalSize(v,&n));
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE,&n,1,MPIU_INT,MPI_MAX,PETSC_COMM_WORLD));
  PetscCall(PetscNvshmemMalloc(n*sizeof(PetscScalar),(void**)&veccuda->GPUarray_allocated));
  veccuda->GPUarray = veccuda->GPUarray_allocated;
  veccuda->nvshmem  = PETSC_TRUE;
  PetscFunctionReturn(0);
}
#endif
