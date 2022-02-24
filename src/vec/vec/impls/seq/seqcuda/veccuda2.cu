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
  PetscErrorCode ierr;
  Vec_CUDA       *veccuda;
  PetscBool      option_set;

  PetscFunctionBegin;
  if (!v->spptr) {
    PetscReal pinned_memory_min;
    CHKERRQ(PetscCalloc(sizeof(Vec_CUDA),&v->spptr));
    veccuda = (Vec_CUDA*)v->spptr;
    CHKERRCUDA(cudaMalloc((void**)&veccuda->GPUarray_allocated,sizeof(PetscScalar)*((PetscBLASInt)v->map->n)));
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
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)v),((PetscObject)v)->prefix,"VECCUDA Options","Vec");CHKERRQ(ierr);
    CHKERRQ(PetscOptionsReal("-vec_pinned_memory_min","Minimum size (in bytes) for an allocation to use pinned memory on host","VecSetPinnedMemoryMin",pinned_memory_min,&pinned_memory_min,&option_set));
    if (option_set) v->minimum_bytes_pinned_memory = pinned_memory_min;
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
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
  CHKERRQ(VecCUDAAllocateCheck(v));
  if (v->offloadmask == PETSC_OFFLOAD_CPU) {
    CHKERRQ(PetscLogEventBegin(VEC_CUDACopyToGPU,v,0,0,0));
    veccuda        = (Vec_CUDA*)v->spptr;
    varray         = veccuda->GPUarray;
    CHKERRCUDA(cudaMemcpy(varray,((Vec_Seq*)v->data)->array,v->map->n*sizeof(PetscScalar),cudaMemcpyHostToDevice));
    CHKERRQ(PetscLogCpuToGpu((v->map->n)*sizeof(PetscScalar)));
    CHKERRQ(PetscLogEventEnd(VEC_CUDACopyToGPU,v,0,0,0));
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
  CHKERRQ(VecCUDAAllocateCheckHost(v));
  if (v->offloadmask == PETSC_OFFLOAD_GPU) {
    CHKERRQ(PetscLogEventBegin(VEC_CUDACopyFromGPU,v,0,0,0));
    veccuda        = (Vec_CUDA*)v->spptr;
    varray         = veccuda->GPUarray;
    CHKERRCUDA(cudaMemcpy(((Vec_Seq*)v->data)->array,varray,v->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
    CHKERRQ(PetscLogGpuToCpu((v->map->n)*sizeof(PetscScalar)));
    CHKERRQ(PetscLogEventEnd(VEC_CUDACopyFromGPU,v,0,0,0));
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

/*MC
   VECSEQCUDA - VECSEQCUDA = "seqcuda" - The basic sequential vector, modified to use CUDA

   Options Database Keys:
. -vec_type seqcuda - sets the vector type to VECSEQCUDA during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateSeqWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateSeq(), VecSetPinnedMemoryMin()
M*/

PetscErrorCode VecAYPX_SeqCUDA(Vec yin,PetscScalar alpha,Vec xin)
{
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscBLASInt      one = 1,bn = 0;
  PetscScalar       sone = 1.0;
  cublasHandle_t    cublasv2handle;

  PetscFunctionBegin;
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  CHKERRQ(PetscBLASIntCast(yin->map->n,&bn));
  CHKERRQ(VecCUDAGetArrayRead(xin,&xarray));
  CHKERRQ(VecCUDAGetArray(yin,&yarray));
  CHKERRQ(PetscLogGpuTimeBegin());
  if (alpha == (PetscScalar)0.0) {
    CHKERRCUDA(cudaMemcpy(yarray,xarray,bn*sizeof(PetscScalar),cudaMemcpyDeviceToDevice));
  } else if (alpha == (PetscScalar)1.0) {
    CHKERRCUBLAS(cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,yarray,one));
    CHKERRQ(PetscLogGpuFlops(1.0*yin->map->n));
  } else {
    CHKERRCUBLAS(cublasXscal(cublasv2handle,bn,&alpha,yarray,one));
    CHKERRCUBLAS(cublasXaxpy(cublasv2handle,bn,&sone,xarray,one,yarray,one));
    CHKERRQ(PetscLogGpuFlops(2.0*yin->map->n));
  }
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(VecCUDARestoreArrayRead(xin,&xarray));
  CHKERRQ(VecCUDARestoreArray(yin,&yarray));
  CHKERRQ(PetscLogCpuToGpuScalar(sizeof(PetscScalar)));
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
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  CHKERRQ(PetscObjectTypeCompareAny((PetscObject)xin,&xiscuda,VECSEQCUDA,VECMPICUDA,""));
  if (xiscuda) {
    CHKERRQ(PetscBLASIntCast(yin->map->n,&bn));
    CHKERRQ(VecCUDAGetArrayRead(xin,&xarray));
    CHKERRQ(VecCUDAGetArray(yin,&yarray));
    CHKERRQ(PetscLogGpuTimeBegin());
    CHKERRCUBLAS(cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,yarray,one));
    CHKERRQ(PetscLogGpuTimeEnd());
    CHKERRQ(VecCUDARestoreArrayRead(xin,&xarray));
    CHKERRQ(VecCUDARestoreArray(yin,&yarray));
    CHKERRQ(PetscLogGpuFlops(2.0*yin->map->n));
    CHKERRQ(PetscLogCpuToGpuScalar(sizeof(PetscScalar)));
  } else {
    CHKERRQ(VecAXPY_Seq(yin,alpha,xin));
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
    CHKERRQ(VecPointwiseDivide_Seq(win,xin,yin));
    PetscFunctionReturn(0);
  }
  CHKERRQ(VecCUDAGetArrayWrite(win,&warray));
  CHKERRQ(VecCUDAGetArrayRead(xin,&xarray));
  CHKERRQ(VecCUDAGetArrayRead(yin,&yarray));
  CHKERRQ(PetscLogGpuTimeBegin());
  try {
    wptr = thrust::device_pointer_cast(warray);
    xptr = thrust::device_pointer_cast(xarray);
    yptr = thrust::device_pointer_cast(yarray);
    thrust::transform(VecCUDAThrustPolicy(win),xptr,xptr+n,yptr,wptr,thrust::divides<PetscScalar>());
  } catch (char *ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(PetscLogGpuFlops(n));
  CHKERRQ(VecCUDARestoreArrayRead(xin,&xarray));
  CHKERRQ(VecCUDARestoreArrayRead(yin,&yarray));
  CHKERRQ(VecCUDARestoreArrayWrite(win,&warray));
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
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  CHKERRQ(PetscBLASIntCast(win->map->n,&bn));
  if (alpha == (PetscScalar)0.0) {
    CHKERRQ(VecCopy_SeqCUDA(yin,win));
  } else {
    CHKERRQ(VecCUDAGetArrayRead(xin,&xarray));
    CHKERRQ(VecCUDAGetArrayRead(yin,&yarray));
    CHKERRQ(VecCUDAGetArrayWrite(win,&warray));
    CHKERRQ(PetscLogGpuTimeBegin());
    CHKERRCUBLAS(cublasGetStream(cublasv2handle,&stream));
    CHKERRCUDA(cudaMemcpyAsync(warray,yarray,win->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice,stream));
    CHKERRCUBLAS(cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,warray,one));
    CHKERRQ(PetscLogGpuTimeEnd());
    CHKERRQ(PetscLogGpuFlops(2*win->map->n));
    CHKERRQ(VecCUDARestoreArrayRead(xin,&xarray));
    CHKERRQ(VecCUDARestoreArrayRead(yin,&yarray));
    CHKERRQ(VecCUDARestoreArrayWrite(win,&warray));
    CHKERRQ(PetscLogCpuToGpuScalar(sizeof(PetscScalar)));
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
  CHKERRQ(PetscLogGpuFlops(nv*2.0*n));
  CHKERRQ(PetscLogCpuToGpuScalar(nv*sizeof(PetscScalar)));
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  CHKERRQ(PetscBLASIntCast(n,&bn));
  CHKERRQ(VecCUDAGetArray(xin,&xarray));
  CHKERRQ(PetscLogGpuTimeBegin());
  for (j=0; j<nv; j++) {
    CHKERRQ(VecCUDAGetArrayRead(y[j],&yarray));
    CHKERRCUBLAS(cublasXaxpy(cublasv2handle,bn,alpha+j,yarray,one,xarray,one));
    CHKERRQ(VecCUDARestoreArrayRead(y[j],&yarray));
  }
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(VecCUDARestoreArray(xin,&xarray));
  PetscFunctionReturn(0);
}

PetscErrorCode VecDot_SeqCUDA(Vec xin,Vec yin,PetscScalar *z)
{
  const PetscScalar *xarray,*yarray;
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;

  PetscFunctionBegin;
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  CHKERRQ(PetscBLASIntCast(yin->map->n,&bn));
  CHKERRQ(VecCUDAGetArrayRead(xin,&xarray));
  CHKERRQ(VecCUDAGetArrayRead(yin,&yarray));
  /* arguments y, x are reversed because BLAS complex conjugates the first argument, PETSc the second */
  CHKERRQ(PetscLogGpuTimeBegin());
  CHKERRCUBLAS(cublasXdot(cublasv2handle,bn,yarray,one,xarray,one,z));
  CHKERRQ(PetscLogGpuTimeEnd());
  if (xin->map->n >0) {
    CHKERRQ(PetscLogGpuFlops(2.0*xin->map->n-1));
  }
  CHKERRQ(PetscLogGpuToCpuScalar(sizeof(PetscScalar)));
  CHKERRQ(VecCUDARestoreArrayRead(xin,&xarray));
  CHKERRQ(VecCUDARestoreArrayRead(yin,&yarray));
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
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  CHKERRQ(PetscBLASIntCast(xin->map->n,&bn));
  PetscCheckFalse(nv <= 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Number of vectors provided to VecMDot_SeqCUDA not positive.");
  /* Handle the case of local size zero first */
  if (!xin->map->n) {
    for (i=0; i<nv; ++i) z[i] = 0;
    PetscFunctionReturn(0);
  }

#if !defined(PETSC_USE_COMPLEX)
  // allocate scratchpad memory for the results of individual work groups:
  CHKERRCUDA(cudaMalloc((void**)&group_results_gpu, nv1*sizeof(PetscScalar)*MDOT_WORKGROUP_NUM));
  CHKERRQ(PetscMalloc1(nv1*MDOT_WORKGROUP_NUM,&group_results_cpu));
#endif
  CHKERRQ(VecCUDAGetArrayRead(xin,&xptr));
  CHKERRQ(PetscLogGpuTimeBegin());

  while (current_y_index < nv)
  {
    switch (nv - current_y_index) {

      case 7:
      case 6:
      case 5:
      case 4:
        CHKERRQ(VecCUDAGetArrayRead(yin[current_y_index  ],&y0ptr));
        CHKERRQ(VecCUDAGetArrayRead(yin[current_y_index+1],&y1ptr));
        CHKERRQ(VecCUDAGetArrayRead(yin[current_y_index+2],&y2ptr));
        CHKERRQ(VecCUDAGetArrayRead(yin[current_y_index+3],&y3ptr));
#if defined(PETSC_USE_COMPLEX)
        CHKERRCUBLAS(cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]));
        CHKERRCUBLAS(cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,&z[current_y_index+1]));
        CHKERRCUBLAS(cublasXdot(cublasv2handle,bn,y2ptr,one,xptr,one,&z[current_y_index+2]));
        CHKERRCUBLAS(cublasXdot(cublasv2handle,bn,y3ptr,one,xptr,one,&z[current_y_index+3]));
#else
        VecMDot_SeqCUDA_kernel4<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,y2ptr,y3ptr,n,group_results_gpu+current_y_index*MDOT_WORKGROUP_NUM);
#endif
        CHKERRQ(VecCUDARestoreArrayRead(yin[current_y_index  ],&y0ptr));
        CHKERRQ(VecCUDARestoreArrayRead(yin[current_y_index+1],&y1ptr));
        CHKERRQ(VecCUDARestoreArrayRead(yin[current_y_index+2],&y2ptr));
        CHKERRQ(VecCUDARestoreArrayRead(yin[current_y_index+3],&y3ptr));
        current_y_index += 4;
        break;

      case 3:
        CHKERRQ(VecCUDAGetArrayRead(yin[current_y_index  ],&y0ptr));
        CHKERRQ(VecCUDAGetArrayRead(yin[current_y_index+1],&y1ptr));
        CHKERRQ(VecCUDAGetArrayRead(yin[current_y_index+2],&y2ptr));

#if defined(PETSC_USE_COMPLEX)
        CHKERRCUBLAS(cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]));
        CHKERRCUBLAS(cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,&z[current_y_index+1]));
        CHKERRCUBLAS(cublasXdot(cublasv2handle,bn,y2ptr,one,xptr,one,&z[current_y_index+2]));
#else
        VecMDot_SeqCUDA_kernel3<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,y2ptr,n,group_results_gpu+current_y_index*MDOT_WORKGROUP_NUM);
#endif
        CHKERRQ(VecCUDARestoreArrayRead(yin[current_y_index  ],&y0ptr));
        CHKERRQ(VecCUDARestoreArrayRead(yin[current_y_index+1],&y1ptr));
        CHKERRQ(VecCUDARestoreArrayRead(yin[current_y_index+2],&y2ptr));
        current_y_index += 3;
        break;

      case 2:
        CHKERRQ(VecCUDAGetArrayRead(yin[current_y_index],&y0ptr));
        CHKERRQ(VecCUDAGetArrayRead(yin[current_y_index+1],&y1ptr));
#if defined(PETSC_USE_COMPLEX)
        CHKERRCUBLAS(cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]));
        CHKERRCUBLAS(cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,&z[current_y_index+1]));
#else
        VecMDot_SeqCUDA_kernel2<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,n,group_results_gpu+current_y_index*MDOT_WORKGROUP_NUM);
#endif
        CHKERRQ(VecCUDARestoreArrayRead(yin[current_y_index],&y0ptr));
        CHKERRQ(VecCUDARestoreArrayRead(yin[current_y_index+1],&y1ptr));
        current_y_index += 2;
        break;

      case 1:
        CHKERRQ(VecCUDAGetArrayRead(yin[current_y_index],&y0ptr));
        CHKERRCUBLAS(cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]));
        CHKERRQ(VecCUDARestoreArrayRead(yin[current_y_index],&y0ptr));
        current_y_index += 1;
        break;

      default: // 8 or more vectors left
        CHKERRQ(VecCUDAGetArrayRead(yin[current_y_index  ],&y0ptr));
        CHKERRQ(VecCUDAGetArrayRead(yin[current_y_index+1],&y1ptr));
        CHKERRQ(VecCUDAGetArrayRead(yin[current_y_index+2],&y2ptr));
        CHKERRQ(VecCUDAGetArrayRead(yin[current_y_index+3],&y3ptr));
        CHKERRQ(VecCUDAGetArrayRead(yin[current_y_index+4],&y4ptr));
        CHKERRQ(VecCUDAGetArrayRead(yin[current_y_index+5],&y5ptr));
        CHKERRQ(VecCUDAGetArrayRead(yin[current_y_index+6],&y6ptr));
        CHKERRQ(VecCUDAGetArrayRead(yin[current_y_index+7],&y7ptr));
#if defined(PETSC_USE_COMPLEX)
        CHKERRCUBLAS(cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]));
        CHKERRCUBLAS(cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,&z[current_y_index+1]));
        CHKERRCUBLAS(cublasXdot(cublasv2handle,bn,y2ptr,one,xptr,one,&z[current_y_index+2]));
        CHKERRCUBLAS(cublasXdot(cublasv2handle,bn,y3ptr,one,xptr,one,&z[current_y_index+3]));
        CHKERRCUBLAS(cublasXdot(cublasv2handle,bn,y4ptr,one,xptr,one,&z[current_y_index+4]));
        CHKERRCUBLAS(cublasXdot(cublasv2handle,bn,y5ptr,one,xptr,one,&z[current_y_index+5]));
        CHKERRCUBLAS(cublasXdot(cublasv2handle,bn,y6ptr,one,xptr,one,&z[current_y_index+6]));
        CHKERRCUBLAS(cublasXdot(cublasv2handle,bn,y7ptr,one,xptr,one,&z[current_y_index+7]));
#else
        VecMDot_SeqCUDA_kernel8<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,y2ptr,y3ptr,y4ptr,y5ptr,y6ptr,y7ptr,n,group_results_gpu+current_y_index*MDOT_WORKGROUP_NUM);
#endif
        CHKERRQ(VecCUDARestoreArrayRead(yin[current_y_index  ],&y0ptr));
        CHKERRQ(VecCUDARestoreArrayRead(yin[current_y_index+1],&y1ptr));
        CHKERRQ(VecCUDARestoreArrayRead(yin[current_y_index+2],&y2ptr));
        CHKERRQ(VecCUDARestoreArrayRead(yin[current_y_index+3],&y3ptr));
        CHKERRQ(VecCUDARestoreArrayRead(yin[current_y_index+4],&y4ptr));
        CHKERRQ(VecCUDARestoreArrayRead(yin[current_y_index+5],&y5ptr));
        CHKERRQ(VecCUDARestoreArrayRead(yin[current_y_index+6],&y6ptr));
        CHKERRQ(VecCUDARestoreArrayRead(yin[current_y_index+7],&y7ptr));
        current_y_index += 8;
        break;
    }
  }
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(VecCUDARestoreArrayRead(xin,&xptr));

#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscLogGpuToCpuScalar(nv*sizeof(PetscScalar)));
#else
  // copy results to CPU
  CHKERRCUDA(cudaMemcpy(group_results_cpu,group_results_gpu,nv1*sizeof(PetscScalar)*MDOT_WORKGROUP_NUM,cudaMemcpyDeviceToHost));

  // sum group results into z
  for (j=0; j<nv1; ++j) {
    z[j] = 0;
    for (i=j*MDOT_WORKGROUP_NUM; i<(j+1)*MDOT_WORKGROUP_NUM; ++i) z[j] += group_results_cpu[i];
  }
  CHKERRQ(PetscLogFlops(nv1*MDOT_WORKGROUP_NUM));
  CHKERRCUDA(cudaFree(group_results_gpu));
  CHKERRQ(PetscFree(group_results_cpu));
  CHKERRQ(PetscLogGpuToCpuScalar(nv1*sizeof(PetscScalar)*MDOT_WORKGROUP_NUM));
#endif
  CHKERRQ(PetscLogGpuFlops(PetscMax(nv*(2.0*n-1),0.0)));
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
  CHKERRQ(VecCUDAGetArrayWrite(xin,&xarray));
  CHKERRQ(PetscLogGpuTimeBegin());
  if (alpha == (PetscScalar)0.0) {
    CHKERRCUDA(cudaMemset(xarray,0,n*sizeof(PetscScalar)));
  } else {
    try {
      xptr = thrust::device_pointer_cast(xarray);
      thrust::fill(xptr,xptr+n,alpha);
    } catch (char *ex) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
    }
    CHKERRQ(PetscLogCpuToGpuScalar(sizeof(PetscScalar)));
  }
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(VecCUDARestoreArrayWrite(xin,&xarray));
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
  CHKERRQ(VecGetLocalSize(v,&n));
  CHKERRQ(VecCUDAGetArray(v,&x));
  CHKERRQ(PetscLogGpuTimeBegin());
  try {
    auto xptr = thrust::device_pointer_cast(x);
    thrust::transform(VecCUDAThrustPolicy(v),xptr,xptr+n,xptr,PetscScalarReciprocal());
  } catch (char *ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(VecCUDARestoreArray(v,&x));
  PetscFunctionReturn(0);
}

PetscErrorCode VecScale_SeqCUDA(Vec xin,PetscScalar alpha)
{
  PetscScalar    *xarray;
  PetscBLASInt   one = 1,bn = 0;
  cublasHandle_t cublasv2handle;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    CHKERRQ(VecSet_SeqCUDA(xin,alpha));
  } else if (alpha != (PetscScalar)1.0) {
    CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
    CHKERRQ(PetscBLASIntCast(xin->map->n,&bn));
    CHKERRQ(VecCUDAGetArray(xin,&xarray));
    CHKERRQ(PetscLogGpuTimeBegin());
    CHKERRCUBLAS(cublasXscal(cublasv2handle,bn,&alpha,xarray,one));
    CHKERRQ(VecCUDARestoreArray(xin,&xarray));
    CHKERRQ(PetscLogGpuTimeEnd());
    CHKERRQ(PetscLogCpuToGpuScalar(sizeof(PetscScalar)));
    CHKERRQ(PetscLogGpuFlops(xin->map->n));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_SeqCUDA(Vec xin,Vec yin,PetscScalar *z)
{
  const PetscScalar *xarray,*yarray;
  PetscBLASInt      one = 1,bn = 0;
  cublasHandle_t    cublasv2handle;

  PetscFunctionBegin;
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  CHKERRQ(PetscBLASIntCast(xin->map->n,&bn));
  CHKERRQ(VecCUDAGetArrayRead(xin,&xarray));
  CHKERRQ(VecCUDAGetArrayRead(yin,&yarray));
  CHKERRQ(PetscLogGpuTimeBegin());
  CHKERRCUBLAS(cublasXdotu(cublasv2handle,bn,xarray,one,yarray,one,z));
  CHKERRQ(PetscLogGpuTimeEnd());
  if (xin->map->n > 0) {
    CHKERRQ(PetscLogGpuFlops(2.0*xin->map->n-1));
  }
  CHKERRQ(PetscLogGpuToCpuScalar(sizeof(PetscScalar)));
  CHKERRQ(VecCUDARestoreArrayRead(yin,&yarray));
  CHKERRQ(VecCUDARestoreArrayRead(xin,&xarray));
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

      CHKERRQ(PetscObjectTypeCompareAny((PetscObject)yin,&yiscuda,VECSEQCUDA,VECMPICUDA,""));
      CHKERRQ(VecCUDAGetArrayRead(xin,&xarray));
      if (yiscuda) {
        CHKERRQ(VecCUDAGetArrayWrite(yin,&yarray));
      } else {
        CHKERRQ(VecGetArrayWrite(yin,&yarray));
      }
      CHKERRQ(PetscLogGpuTimeBegin());
      if (yiscuda) {
        CHKERRCUDA(cudaMemcpyAsync(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice,PetscDefaultCudaStream));
      } else {
        CHKERRCUDA(cudaMemcpy(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToHost));
        CHKERRQ(PetscLogGpuToCpu((yin->map->n)*sizeof(PetscScalar)));
      }
      CHKERRQ(PetscLogGpuTimeEnd());
      CHKERRQ(VecCUDARestoreArrayRead(xin,&xarray));
      if (yiscuda) {
        CHKERRQ(VecCUDARestoreArrayWrite(yin,&yarray));
      } else {
        CHKERRQ(VecRestoreArrayWrite(yin,&yarray));
      }
    } else if (xin->offloadmask == PETSC_OFFLOAD_CPU) {
      /* copy in CPU if we are on the CPU */
      CHKERRQ(VecCopy_SeqCUDA_Private(xin,yin));
    } else if (xin->offloadmask == PETSC_OFFLOAD_BOTH) {
      /* if xin is valid in both places, see where yin is and copy there (because it's probably where we'll want to next use it) */
      if (yin->offloadmask == PETSC_OFFLOAD_CPU) {
        /* copy in CPU */
        CHKERRQ(VecCopy_SeqCUDA_Private(xin,yin));
      } else if (yin->offloadmask == PETSC_OFFLOAD_GPU) {
        /* copy in GPU */
        CHKERRQ(VecCUDAGetArrayRead(xin,&xarray));
        CHKERRQ(VecCUDAGetArrayWrite(yin,&yarray));
        CHKERRQ(PetscLogGpuTimeBegin());
        CHKERRCUDA(cudaMemcpyAsync(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice,PetscDefaultCudaStream));
        CHKERRQ(PetscLogGpuTimeEnd());
        CHKERRQ(VecCUDARestoreArrayRead(xin,&xarray));
        CHKERRQ(VecCUDARestoreArrayWrite(yin,&yarray));
      } else if (yin->offloadmask == PETSC_OFFLOAD_BOTH) {
        /* xin and yin are both valid in both places (or yin was unallocated before the earlier call to allocatecheck
           default to copy in GPU (this is an arbitrary choice) */
        CHKERRQ(VecCUDAGetArrayRead(xin,&xarray));
        CHKERRQ(VecCUDAGetArrayWrite(yin,&yarray));
        CHKERRQ(PetscLogGpuTimeBegin());
        CHKERRCUDA(cudaMemcpyAsync(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice,PetscDefaultCudaStream));
        CHKERRQ(PetscLogGpuTimeEnd());
        CHKERRQ(VecCUDARestoreArrayRead(xin,&xarray));
        CHKERRQ(VecCUDARestoreArrayWrite(yin,&yarray));
      } else {
        CHKERRQ(VecCopy_SeqCUDA_Private(xin,yin));
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
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  CHKERRQ(PetscBLASIntCast(xin->map->n,&bn));
  if (xin != yin) {
    CHKERRQ(VecCUDAGetArray(xin,&xarray));
    CHKERRQ(VecCUDAGetArray(yin,&yarray));
    CHKERRQ(PetscLogGpuTimeBegin());
    CHKERRCUBLAS(cublasXswap(cublasv2handle,bn,xarray,one,yarray,one));
    CHKERRQ(PetscLogGpuTimeEnd());
    CHKERRQ(VecCUDARestoreArray(xin,&xarray));
    CHKERRQ(VecCUDARestoreArray(yin,&yarray));
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
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  CHKERRQ(PetscBLASIntCast(yin->map->n,&bn));
  if (a == (PetscScalar)0.0) {
    CHKERRQ(VecScale_SeqCUDA(yin,beta));
  } else if (b == (PetscScalar)1.0) {
    CHKERRQ(VecAXPY_SeqCUDA(yin,alpha,xin));
  } else if (a == (PetscScalar)1.0) {
    CHKERRQ(VecAYPX_SeqCUDA(yin,beta,xin));
  } else if (b == (PetscScalar)0.0) {
    CHKERRQ(VecCUDAGetArrayRead(xin,&xarray));
    CHKERRQ(VecCUDAGetArray(yin,&yarray));
    CHKERRQ(PetscLogGpuTimeBegin());
    CHKERRCUDA(cudaMemcpy(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice));
    CHKERRCUBLAS(cublasXscal(cublasv2handle,bn,&alpha,yarray,one));
    CHKERRQ(PetscLogGpuTimeEnd());
    CHKERRQ(PetscLogGpuFlops(xin->map->n));
    CHKERRQ(PetscLogCpuToGpuScalar(sizeof(PetscScalar)));
    CHKERRQ(VecCUDARestoreArrayRead(xin,&xarray));
    CHKERRQ(VecCUDARestoreArray(yin,&yarray));
  } else {
    CHKERRQ(VecCUDAGetArrayRead(xin,&xarray));
    CHKERRQ(VecCUDAGetArray(yin,&yarray));
    CHKERRQ(PetscLogGpuTimeBegin());
    CHKERRCUBLAS(cublasXscal(cublasv2handle,bn,&beta,yarray,one));
    CHKERRCUBLAS(cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,yarray,one));
    CHKERRQ(VecCUDARestoreArrayRead(xin,&xarray));
    CHKERRQ(VecCUDARestoreArray(yin,&yarray));
    CHKERRQ(PetscLogGpuTimeEnd());
    CHKERRQ(PetscLogGpuFlops(3.0*xin->map->n));
    CHKERRQ(PetscLogCpuToGpuScalar(2*sizeof(PetscScalar)));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBYPCZ_SeqCUDA(Vec zin,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec xin,Vec yin)
{
  PetscInt       n = zin->map->n;

  PetscFunctionBegin;
  if (gamma == (PetscScalar)1.0) {
    /* z = ax + b*y + z */
    CHKERRQ(VecAXPY_SeqCUDA(zin,alpha,xin));
    CHKERRQ(VecAXPY_SeqCUDA(zin,beta,yin));
    CHKERRQ(PetscLogGpuFlops(4.0*n));
  } else {
    /* z = a*x + b*y + c*z */
    CHKERRQ(VecScale_SeqCUDA(zin,gamma));
    CHKERRQ(VecAXPY_SeqCUDA(zin,alpha,xin));
    CHKERRQ(VecAXPY_SeqCUDA(zin,beta,yin));
    CHKERRQ(PetscLogGpuFlops(5.0*n));
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
    CHKERRQ(VecPointwiseMult_Seq(win,xin,yin));
    PetscFunctionReturn(0);
  }
  CHKERRQ(VecCUDAGetArrayRead(xin,&xarray));
  CHKERRQ(VecCUDAGetArrayRead(yin,&yarray));
  CHKERRQ(VecCUDAGetArrayWrite(win,&warray));
  CHKERRQ(PetscLogGpuTimeBegin());
  try {
    wptr = thrust::device_pointer_cast(warray);
    xptr = thrust::device_pointer_cast(xarray);
    yptr = thrust::device_pointer_cast(yarray);
    thrust::transform(VecCUDAThrustPolicy(win),xptr,xptr+n,yptr,wptr,thrust::multiplies<PetscScalar>());
  } catch (char *ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(VecCUDARestoreArrayRead(xin,&xarray));
  CHKERRQ(VecCUDARestoreArrayRead(yin,&yarray));
  CHKERRQ(VecCUDARestoreArrayWrite(win,&warray));
  CHKERRQ(PetscLogGpuFlops(n));
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
  CHKERRQ(PetscCUBLASGetHandle(&cublasv2handle));
  CHKERRQ(PetscBLASIntCast(n,&bn));
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    CHKERRQ(VecCUDAGetArrayRead(xin,&xarray));
    CHKERRQ(PetscLogGpuTimeBegin());
    CHKERRCUBLAS(cublasXnrm2(cublasv2handle,bn,xarray,one,z));
    CHKERRQ(PetscLogGpuTimeEnd());
    CHKERRQ(VecCUDARestoreArrayRead(xin,&xarray));
    CHKERRQ(PetscLogGpuFlops(PetscMax(2.0*n-1,0.0)));
  } else if (type == NORM_INFINITY) {
    int  i;
    CHKERRQ(VecCUDAGetArrayRead(xin,&xarray));
    CHKERRQ(PetscLogGpuTimeBegin());
    CHKERRCUBLAS(cublasIXamax(cublasv2handle,bn,xarray,one,&i));
    CHKERRQ(PetscLogGpuTimeEnd());
    if (bn) {
      PetscScalar zs;
      CHKERRCUDA(cudaMemcpy(&zs,xarray+i-1,sizeof(PetscScalar),cudaMemcpyDeviceToHost));
      *z = PetscAbsScalar(zs);
    } else *z = 0.0;
    CHKERRQ(VecCUDARestoreArrayRead(xin,&xarray));
  } else if (type == NORM_1) {
    CHKERRQ(VecCUDAGetArrayRead(xin,&xarray));
    CHKERRQ(PetscLogGpuTimeBegin());
    CHKERRCUBLAS(cublasXasum(cublasv2handle,bn,xarray,one,z));
    CHKERRQ(VecCUDARestoreArrayRead(xin,&xarray));
    CHKERRQ(PetscLogGpuTimeEnd());
    CHKERRQ(PetscLogGpuFlops(PetscMax(n-1.0,0.0)));
  } else if (type == NORM_1_AND_2) {
    CHKERRQ(VecNorm_SeqCUDA(xin,NORM_1,z));
    CHKERRQ(VecNorm_SeqCUDA(xin,NORM_2,z+1));
  }
  CHKERRQ(PetscLogGpuToCpuScalar(sizeof(PetscReal)));
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotNorm2_SeqCUDA(Vec s, Vec t, PetscScalar *dp, PetscScalar *nm)
{
  PetscFunctionBegin;
  CHKERRQ(VecDot_SeqCUDA(s,t,dp));
  CHKERRQ(VecDot_SeqCUDA(t,t,nm));
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
        CHKERRQ(PetscNvshmemFree(veccuda->GPUarray_allocated));
        veccuda->nvshmem = PETSC_FALSE;
      }
      else
     #endif
      CHKERRCUDA(cudaFree(veccuda->GPUarray_allocated));
      veccuda->GPUarray_allocated = NULL;
    }
    if (veccuda->stream) {
      CHKERRCUDA(cudaStreamDestroy(veccuda->stream));
    }
  }
  CHKERRQ(VecDestroy_SeqCUDA_Private(v));
  CHKERRQ(PetscFree(v->spptr));
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
  CHKERRQ(VecCUDAGetArray(xin,&xarray));
  CHKERRQ(PetscLogGpuTimeBegin());
  try {
    xptr = thrust::device_pointer_cast(xarray);
    thrust::transform(VecCUDAThrustPolicy(xin),xptr,xptr+n,xptr,conjugate());
  } catch (char *ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(VecCUDARestoreArray(xin,&xarray));
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
  CHKERRQ(PetscObjectTypeCompare((PetscObject)w,VECSEQCUDA,&wisseqcuda));
  if (w->data && wisseqcuda) {
    if (((Vec_Seq*)w->data)->array_allocated) {
      if (w->pinned_memory) {
        CHKERRQ(PetscMallocSetCUDAHost());
      }
      CHKERRQ(PetscFree(((Vec_Seq*)w->data)->array_allocated));
      if (w->pinned_memory) {
        CHKERRQ(PetscMallocResetCUDAHost());
        w->pinned_memory = PETSC_FALSE;
      }
    }
    ((Vec_Seq*)w->data)->array = NULL;
    ((Vec_Seq*)w->data)->unplacedarray = NULL;
  }
  if (w->spptr && wisseqcuda) {
    if (((Vec_CUDA*)w->spptr)->GPUarray) {
      CHKERRCUDA(cudaFree(((Vec_CUDA*)w->spptr)->GPUarray));
      ((Vec_CUDA*)w->spptr)->GPUarray = NULL;
    }
    if (((Vec_CUDA*)w->spptr)->stream) {
      CHKERRCUDA(cudaStreamDestroy(((Vec_CUDA*)w->spptr)->stream));
    }
    CHKERRQ(PetscFree(w->spptr));
  }

  if (v->petscnative && wisseqcuda) {
    CHKERRQ(PetscFree(w->data));
    w->data = v->data;
    w->offloadmask = v->offloadmask;
    w->pinned_memory = v->pinned_memory;
    w->spptr = v->spptr;
    CHKERRQ(PetscObjectStateIncrease((PetscObject)w));
  } else {
    if (read) {
      CHKERRQ(VecGetArrayRead(v,(const PetscScalar**)&((Vec_Seq*)w->data)->array));
    } else {
      CHKERRQ(VecGetArray(v,&((Vec_Seq*)w->data)->array));
    }
    w->offloadmask = PETSC_OFFLOAD_CPU;
    if (wisseqcuda) {
      CHKERRQ(VecCUDAAllocateCheck(w));
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
  CHKERRQ(PetscObjectTypeCompare((PetscObject)w,VECSEQCUDA,&wisseqcuda));
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
      CHKERRQ(VecRestoreArrayRead(v,(const PetscScalar**)&((Vec_Seq*)w->data)->array));
    } else {
      CHKERRQ(VecRestoreArray(v,&((Vec_Seq*)w->data)->array));
    }
    if ((Vec_CUDA*)w->spptr && wisseqcuda) {
      CHKERRCUDA(cudaFree(((Vec_CUDA*)w->spptr)->GPUarray));
      ((Vec_CUDA*)w->spptr)->GPUarray = NULL;
      if (((Vec_CUDA*)v->spptr)->stream) {
        CHKERRCUDA(cudaStreamDestroy(((Vec_CUDA*)w->spptr)->stream));
      }
      CHKERRQ(PetscFree(w->spptr));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetLocalVector_SeqCUDA(Vec v,Vec w)
{
  PetscFunctionBegin;
  CHKERRQ(VecGetLocalVectorK_SeqCUDA(v,w,PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetLocalVectorRead_SeqCUDA(Vec v,Vec w)
{
  PetscFunctionBegin;
  CHKERRQ(VecGetLocalVectorK_SeqCUDA(v,w,PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreLocalVector_SeqCUDA(Vec v,Vec w)
{
  PetscFunctionBegin;
  CHKERRQ(VecRestoreLocalVectorK_SeqCUDA(v,w,PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreLocalVectorRead_SeqCUDA(Vec v,Vec w)
{
  PetscFunctionBegin;
  CHKERRQ(VecRestoreLocalVectorK_SeqCUDA(v,w,PETSC_TRUE));
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
  CHKERRQ(VecCUDAGetArrayRead(v,&av));
  avpt = thrust::device_pointer_cast(av);
  CHKERRQ(PetscLogGpuTimeBegin());
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
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(VecCUDARestoreArrayRead(v,&av));
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
  CHKERRQ(VecCUDAGetArrayRead(v,&av));
  avpt = thrust::device_pointer_cast(av);
  CHKERRQ(PetscLogGpuTimeBegin());
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
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(VecCUDARestoreArrayRead(v,&av));
  PetscFunctionReturn(0);
}

PetscErrorCode VecSum_SeqCUDA(Vec v,PetscScalar *sum)
{
  PetscInt                              n = v->map->n;
  const PetscScalar                     *a;
  thrust::device_ptr<const PetscScalar> dptr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  CHKERRQ(VecCUDAGetArrayRead(v,&a));
  dptr = thrust::device_pointer_cast(a);
  CHKERRQ(PetscLogGpuTimeBegin());
  try {
    *sum = thrust::reduce(VecCUDAThrustPolicy(v),dptr,dptr+n,PetscScalar(0.0));
  } catch (char *ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(VecCUDARestoreArrayRead(v,&a));
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
  CHKERRQ(VecCUDAGetArray(v,&a));
  dptr = thrust::device_pointer_cast(a);
  CHKERRQ(PetscLogGpuTimeBegin());
  try {
    thrust::transform(VecCUDAThrustPolicy(v),dptr,dptr+n,dptr,petscshift(shift)); /* in-place transform */
  } catch (char *ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  CHKERRQ(PetscLogGpuTimeEnd());
  CHKERRQ(VecCUDARestoreArray(v,&a));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_NVSHMEM)
/* Free old CUDA array and re-allocate a new one from nvshmem symmetric heap.
   New array does not retain values in the old array. The offload mask is not changed.

   Note: the function is only meant to be used in MatAssemblyEnd_MPIAIJCUSPARSE.
 */
PetscErrorCode  VecAllocateNVSHMEM_SeqCUDA(Vec v)
{
  PetscErrorCode ierr;
  cudaError_t    cerr;
  Vec_CUDA       *veccuda = (Vec_CUDA*)v->spptr;
  PetscInt       n;

  PetscFunctionBegin;
  CHKERRCUDA(cudaFree(veccuda->GPUarray_allocated));
  CHKERRQ(VecGetLocalSize(v,&n));
  CHKERRMPI(MPIU_Allreduce(MPI_IN_PLACE,&n,1,MPIU_INT,MPI_MAX,PETSC_COMM_WORLD));
  CHKERRQ(PetscNvshmemMalloc(n*sizeof(PetscScalar),(void**)&veccuda->GPUarray_allocated));
  veccuda->GPUarray = veccuda->GPUarray_allocated;
  veccuda->nvshmem  = PETSC_TRUE;
  PetscFunctionReturn(0);
}
#endif
