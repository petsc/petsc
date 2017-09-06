/*
   Implements the sequential cuda vectors.
*/

#define PETSC_SKIP_SPINLOCK

#include <petscconf.h>
#include <petsc/private/vecimpl.h>
#include <../src/vec/vec/impls/dvecimpl.h>
#include <../src/vec/vec/impls/seq/seqcuda/cudavecimpl.h>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

/*
    Allocates space for the vector array on the GPU if it does not exist.
    Does NOT change the PetscCUDAFlag for the vector
    Does NOT zero the CUDA array

 */
PetscErrorCode VecCUDAAllocateCheck(Vec v)
{
  PetscErrorCode ierr;
  cudaError_t    err;
  cudaStream_t   stream;
  Vec_CUDA       *veccuda;

  PetscFunctionBegin;
  if (!v->spptr) {
    ierr = PetscMalloc(sizeof(Vec_CUDA),&v->spptr);CHKERRQ(ierr);
    veccuda = (Vec_CUDA*)v->spptr;
    err = cudaMalloc((void**)&veccuda->GPUarray_allocated,sizeof(PetscScalar)*((PetscBLASInt)v->map->n));CHKERRCUDA(err);
    veccuda->GPUarray = veccuda->GPUarray_allocated;
    err = cudaStreamCreate(&stream);CHKERRCUDA(err);
    veccuda->stream = stream;
    veccuda->hostDataRegisteredAsPageLocked = PETSC_FALSE;
    if (v->valid_GPU_array == PETSC_CUDA_UNALLOCATED) {
      if (v->data && ((Vec_Seq*)v->data)->array) {
        v->valid_GPU_array = PETSC_CUDA_CPU;
      } else {
        v->valid_GPU_array = PETSC_CUDA_GPU;
      }
    }
  }
  PetscFunctionReturn(0);
}

/* Copies a vector from the CPU to the GPU unless we already have an up-to-date copy on the GPU */
PetscErrorCode VecCUDACopyToGPU(Vec v)
{
  PetscErrorCode ierr;
  cudaError_t    err;
  Vec_CUDA       *veccuda;
  PetscScalar    *varray;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  ierr = VecCUDAAllocateCheck(v);CHKERRQ(ierr);
  if (v->valid_GPU_array == PETSC_CUDA_CPU) {
    ierr = PetscLogEventBegin(VEC_CUDACopyToGPU,v,0,0,0);CHKERRQ(ierr);
    veccuda=(Vec_CUDA*)v->spptr;
    varray=veccuda->GPUarray;
    err = cudaMemcpy(varray,((Vec_Seq*)v->data)->array,v->map->n*sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(err);
    ierr = PetscLogEventEnd(VEC_CUDACopyToGPU,v,0,0,0);CHKERRQ(ierr);
    v->valid_GPU_array = PETSC_CUDA_BOTH;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecCUDACopyToGPUSome(Vec v, PetscCUDAIndices ci)
{
  PetscScalar    *varray;
  PetscErrorCode ierr;
  cudaError_t    err;
  PetscScalar    *cpuPtr, *gpuPtr;
  Vec_Seq        *s;
  VecScatterCUDAIndices_PtoP ptop_scatter = (VecScatterCUDAIndices_PtoP)ci->scatter;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  ierr = VecCUDAAllocateCheck(v);CHKERRQ(ierr);
  if (v->valid_GPU_array == PETSC_CUDA_CPU) {
    s = (Vec_Seq*)v->data;

    ierr   = PetscLogEventBegin(VEC_CUDACopyToGPUSome,v,0,0,0);CHKERRQ(ierr);
    varray = ((Vec_CUDA*)v->spptr)->GPUarray;
    gpuPtr = varray + ptop_scatter->recvLowestIndex;
    cpuPtr = s->array + ptop_scatter->recvLowestIndex;

    /* Note : this code copies the smallest contiguous chunk of data
       containing ALL of the indices */
    err = cudaMemcpy(gpuPtr,cpuPtr,ptop_scatter->nr*sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(err);

    // Set the buffer states
    v->valid_GPU_array = PETSC_CUDA_BOTH;
    ierr = PetscLogEventEnd(VEC_CUDACopyToGPUSome,v,0,0,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*
     VecCUDACopyFromGPU - Copies a vector from the GPU to the CPU unless we already have an up-to-date copy on the CPU
*/
PetscErrorCode VecCUDACopyFromGPU(Vec v)
{
  PetscErrorCode ierr;
  cudaError_t    err;
  Vec_CUDA       *veccuda;
  PetscScalar    *varray;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  ierr = VecCUDAAllocateCheckHost(v);CHKERRQ(ierr);
  if (v->valid_GPU_array == PETSC_CUDA_GPU) {
    ierr = PetscLogEventBegin(VEC_CUDACopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    veccuda=(Vec_CUDA*)v->spptr;
    varray=veccuda->GPUarray;
    err = cudaMemcpy(((Vec_Seq*)v->data)->array,varray,v->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUDA(err);
    ierr = PetscLogEventEnd(VEC_CUDACopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    v->valid_GPU_array = PETSC_CUDA_BOTH;
  }
  PetscFunctionReturn(0);
}

/* Note that this function only copies *some* of the values up from the GPU to CPU,
   which means that we need recombine the data at some point before using any of the standard functions.
   We could add another few flag-types to keep track of this, or treat things like VecGetArray VecRestoreArray
   where you have to always call in pairs
*/
PetscErrorCode VecCUDACopyFromGPUSome(Vec v, PetscCUDAIndices ci)
{
  const PetscScalar *varray, *gpuPtr;
  PetscErrorCode    ierr;
  cudaError_t       err;
  PetscScalar       *cpuPtr;
  Vec_Seq           *s;
  VecScatterCUDAIndices_PtoP ptop_scatter = (VecScatterCUDAIndices_PtoP)ci->scatter;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  ierr = VecCUDAAllocateCheckHost(v);CHKERRQ(ierr);
  if (v->valid_GPU_array == PETSC_CUDA_GPU) {
    ierr   = PetscLogEventBegin(VEC_CUDACopyFromGPUSome,v,0,0,0);CHKERRQ(ierr);

    varray=((Vec_CUDA*)v->spptr)->GPUarray;
    s = (Vec_Seq*)v->data;
    gpuPtr = varray + ptop_scatter->sendLowestIndex;
    cpuPtr = s->array + ptop_scatter->sendLowestIndex;

    /* Note : this code copies the smallest contiguous chunk of data
       containing ALL of the indices */
    err = cudaMemcpy(cpuPtr,gpuPtr,ptop_scatter->ns*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUDA(err);

    ierr = VecCUDARestoreArrayRead(v,&varray);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(VEC_CUDACopyFromGPUSome,v,0,0,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   VECSEQCUDA - VECSEQCUDA = "seqcuda" - The basic sequential vector, modified to use CUDA

   Options Database Keys:
. -vec_type seqcuda - sets the vector type to VECSEQCUDA during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateSeqWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateSeq()
M*/

PetscErrorCode VecAYPX_SeqCUDA(Vec yin,PetscScalar alpha,Vec xin)
{
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscErrorCode    ierr;
  PetscBLASInt      one=1,bn;
  PetscScalar       sone=1.0;
  cublasStatus_t    cberr;
  cudaError_t       err;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
  if (alpha == (PetscScalar)0.0) {
    err = cudaMemcpy(yarray,xarray,bn*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRCUDA(err);
  } else if (alpha == (PetscScalar)1.0) {
    cberr = cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogFlops(2.0*yin->map->n);CHKERRQ(ierr);
  } else {
    cberr = cublasXscal(cublasv2handle,bn,&alpha,yarray,one);CHKERRCUBLAS(cberr);
    cberr = cublasXaxpy(cublasv2handle,bn,&sone,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogFlops(2.0*yin->map->n);CHKERRQ(ierr);
  }
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPY_SeqCUDA(Vec yin,PetscScalar alpha,Vec xin)
{
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscErrorCode    ierr;
  PetscBLASInt      one=1,bn;
  cublasStatus_t    cberr;

  PetscFunctionBegin;
  if (alpha != (PetscScalar)0.0) {
    ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    cberr = cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = WaitForGPU();CHKERRCUDA(ierr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogFlops(2.0*yin->map->n);CHKERRQ(ierr);
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
  PetscErrorCode                        ierr;

  PetscFunctionBegin;
  ierr = VecCUDAGetArrayWrite(win,&warray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  try {
    wptr = thrust::device_pointer_cast(warray);
    xptr = thrust::device_pointer_cast(xarray);
    yptr = thrust::device_pointer_cast(yarray);
    thrust::transform(xptr,xptr+n,yptr,wptr,thrust::divides<PetscScalar>());
    ierr = WaitForGPU();CHKERRCUDA(ierr);
  } catch (char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(win,&warray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecWAXPY_SeqCUDA(Vec win,PetscScalar alpha,Vec xin, Vec yin)
{
  const PetscScalar *xarray=NULL,*yarray=NULL;
  PetscScalar       *warray=NULL;
  PetscErrorCode    ierr;
  PetscBLASInt      one=1,bn;
  cublasStatus_t    cberr;
  cudaError_t       err;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(win->map->n,&bn);CHKERRQ(ierr);
  if (alpha == (PetscScalar)0.0) {
    ierr = VecCopy_SeqCUDA(yin,win);CHKERRQ(ierr);
  } else {
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayWrite(win,&warray);CHKERRQ(ierr);
    err = cudaMemcpy(warray,yarray,win->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRCUDA(err);
    cberr = cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,warray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogFlops(2*win->map->n);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRCUDA(ierr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayWrite(win,&warray);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecMAXPY_SeqCUDA(Vec xin, PetscInt nv,const PetscScalar *alpha,Vec *y)
{
  PetscErrorCode ierr;
  PetscInt       n = xin->map->n,j,j_rem;
  PetscScalar    alpha0,alpha1,alpha2,alpha3;

  PetscFunctionBegin;
  ierr = PetscLogFlops(nv*2.0*n);CHKERRQ(ierr);
  switch (j_rem=nv&0x3) {
    case 3:
      alpha0 = alpha[0];
      alpha1 = alpha[1];
      alpha2 = alpha[2];
      alpha += 3;
      ierr   = VecAXPY_SeqCUDA(xin,alpha0,y[0]);CHKERRQ(ierr);
      ierr   = VecAXPY_SeqCUDA(xin,alpha1,y[1]);CHKERRQ(ierr);
      ierr   = VecAXPY_SeqCUDA(xin,alpha2,y[2]);CHKERRQ(ierr);
      y   += 3;
      break;
    case 2:
      alpha0 = alpha[0];
      alpha1 = alpha[1];
      alpha +=2;
      ierr   = VecAXPY_SeqCUDA(xin,alpha0,y[0]);CHKERRQ(ierr);
      ierr   = VecAXPY_SeqCUDA(xin,alpha1,y[1]);CHKERRQ(ierr);
      y +=2;
      break;
    case 1:
      alpha0 = *alpha++;
      ierr   = VecAXPY_SeqCUDA(xin,alpha0,y[0]);CHKERRQ(ierr);
      y     +=1;
      break;
  }
  for (j=j_rem; j<nv; j+=4) {
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha2 = alpha[2];
    alpha3 = alpha[3];
    alpha += 4;
    ierr   = VecAXPY_SeqCUDA(xin,alpha0,y[0]);CHKERRQ(ierr);
    ierr   = VecAXPY_SeqCUDA(xin,alpha1,y[1]);CHKERRQ(ierr);
    ierr   = VecAXPY_SeqCUDA(xin,alpha2,y[2]);CHKERRQ(ierr);
    ierr   = VecAXPY_SeqCUDA(xin,alpha3,y[3]);CHKERRQ(ierr);
    y   += 4;
  }
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecDot_SeqCUDA(Vec xin,Vec yin,PetscScalar *z)
{
  const PetscScalar *xarray,*yarray;
  PetscErrorCode    ierr;
  PetscBLASInt      one=1,bn;
  cublasStatus_t    cberr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  /* arguments y, x are reversed because BLAS complex conjugates the first argument, PETSc the second */
  cberr = cublasXdot(cublasv2handle,bn,yarray,one,xarray,one,z);CHKERRCUBLAS(cberr);
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  if (xin->map->n >0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  PetscInt          i,n = xin->map->n,current_y_index = 0;
  const PetscScalar *xptr,*y0ptr,*y1ptr,*y2ptr,*y3ptr,*y4ptr,*y5ptr,*y6ptr,*y7ptr;
  PetscScalar       *group_results_gpu;
#if !defined(PETSC_USE_COMPLEX)
  PetscInt          j;
  PetscScalar       group_results_cpu[MDOT_WORKGROUP_NUM * 8]; // we process at most eight vectors in one kernel
#endif
  cudaError_t    cuda_ierr;
  PetscBLASInt   one=1,bn;
  cublasStatus_t cberr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
  if (nv <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Number of vectors provided to VecMDot_SeqCUDA not positive.");
  /* Handle the case of local size zero first */
  if (!xin->map->n) {
    for (i=0; i<nv; ++i) z[i] = 0;
    PetscFunctionReturn(0);
  }

  // allocate scratchpad memory for the results of individual work groups:
  cuda_ierr = cudaMalloc((void**)&group_results_gpu, sizeof(PetscScalar) * MDOT_WORKGROUP_NUM * 8);CHKERRCUDA(cuda_ierr);

  ierr = VecCUDAGetArrayRead(xin,&xptr);CHKERRQ(ierr);

  while (current_y_index < nv)
  {
    switch (nv - current_y_index) {

      case 7:
      case 6:
      case 5:
      case 4:
        ierr = VecCUDAGetArrayRead(yin[current_y_index  ],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+2],&y2ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+3],&y3ptr);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
        cberr = cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,&z[current_y_index+1]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y2ptr,one,xptr,one,&z[current_y_index+2]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y3ptr,one,xptr,one,&z[current_y_index+3]);CHKERRCUBLAS(cberr);
#else
        // run kernel:
        VecMDot_SeqCUDA_kernel4<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,y2ptr,y3ptr,n,group_results_gpu);

        // copy results back to
        cuda_ierr = cudaMemcpy(group_results_cpu,group_results_gpu,sizeof(PetscScalar) * MDOT_WORKGROUP_NUM * 4,cudaMemcpyDeviceToHost);CHKERRCUDA(cuda_ierr);

        // sum group results into z:
        for (j=0; j<4; ++j) {
          z[current_y_index + j] = 0;
          for (i=j*MDOT_WORKGROUP_NUM; i<(j+1)*MDOT_WORKGROUP_NUM; ++i) z[current_y_index + j] += group_results_cpu[i];
        }
#endif
        ierr = VecCUDARestoreArrayRead(yin[current_y_index  ],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+2],&y2ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+3],&y3ptr);CHKERRQ(ierr);
        current_y_index += 4;
        break;

      case 3:
        ierr = VecCUDAGetArrayRead(yin[current_y_index  ],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+2],&y2ptr);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
        cberr = cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,&z[current_y_index+1]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y2ptr,one,xptr,one,&z[current_y_index+2]);CHKERRCUBLAS(cberr);
#else
        // run kernel:
        VecMDot_SeqCUDA_kernel3<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,y2ptr,n,group_results_gpu);

        // copy results back to
        cuda_ierr = cudaMemcpy(group_results_cpu,group_results_gpu,sizeof(PetscScalar) * MDOT_WORKGROUP_NUM * 3,cudaMemcpyDeviceToHost);CHKERRCUDA(cuda_ierr);

        // sum group results into z:
        for (j=0; j<3; ++j) {
          z[current_y_index + j] = 0;
          for (i=j*MDOT_WORKGROUP_NUM; i<(j+1)*MDOT_WORKGROUP_NUM; ++i) z[current_y_index + j] += group_results_cpu[i];
        }
#endif

        ierr = VecCUDARestoreArrayRead(yin[current_y_index  ],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+2],&y2ptr);CHKERRQ(ierr);
        current_y_index += 3;
        break;

      case 2:
        ierr = VecCUDAGetArrayRead(yin[current_y_index],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
        cberr = cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,&z[current_y_index+1]);CHKERRCUBLAS(cberr);
#else
        // run kernel:
        VecMDot_SeqCUDA_kernel2<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,n,group_results_gpu);

        // copy results back to
        cuda_ierr = cudaMemcpy(group_results_cpu,group_results_gpu,sizeof(PetscScalar) * MDOT_WORKGROUP_NUM * 2,cudaMemcpyDeviceToHost);CHKERRCUDA(cuda_ierr);

        // sum group results into z:
        for (j=0; j<2; ++j) {
          z[current_y_index + j] = 0;
          for (i=j*MDOT_WORKGROUP_NUM; i<(j+1)*MDOT_WORKGROUP_NUM; ++i) z[current_y_index + j] += group_results_cpu[i];
        }
#endif
        ierr = VecCUDARestoreArrayRead(yin[current_y_index],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        current_y_index += 2;
        break;

      case 1:
        ierr = VecCUDAGetArrayRead(yin[current_y_index],&y0ptr);CHKERRQ(ierr);
        cberr = cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]);CHKERRCUBLAS(cberr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index],&y0ptr);CHKERRQ(ierr);
        current_y_index += 1;
        break;

      default: // 8 or more vectors left
        ierr = VecCUDAGetArrayRead(yin[current_y_index  ],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+2],&y2ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+3],&y3ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+4],&y4ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+5],&y5ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+6],&y6ptr);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(yin[current_y_index+7],&y7ptr);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
        cberr = cublasXdot(cublasv2handle,bn,y0ptr,one,xptr,one,&z[current_y_index]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y1ptr,one,xptr,one,&z[current_y_index+1]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y2ptr,one,xptr,one,&z[current_y_index+2]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y3ptr,one,xptr,one,&z[current_y_index+3]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y4ptr,one,xptr,one,&z[current_y_index+4]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y5ptr,one,xptr,one,&z[current_y_index+5]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y6ptr,one,xptr,one,&z[current_y_index+6]);CHKERRCUBLAS(cberr);
        cberr = cublasXdot(cublasv2handle,bn,y7ptr,one,xptr,one,&z[current_y_index+7]);CHKERRCUBLAS(cberr);
#else
        // run kernel:
        VecMDot_SeqCUDA_kernel8<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,y2ptr,y3ptr,y4ptr,y5ptr,y6ptr,y7ptr,n,group_results_gpu);

        // copy results back to
        cuda_ierr = cudaMemcpy(group_results_cpu,group_results_gpu,sizeof(PetscScalar) * MDOT_WORKGROUP_NUM * 8,cudaMemcpyDeviceToHost);CHKERRCUDA(cuda_ierr);

        // sum group results into z:
        for (j=0; j<8; ++j) {
          z[current_y_index + j] = 0;
          for (i=j*MDOT_WORKGROUP_NUM; i<(j+1)*MDOT_WORKGROUP_NUM; ++i) z[current_y_index + j] += group_results_cpu[i];
        }
#endif
        ierr = VecCUDARestoreArrayRead(yin[current_y_index  ],&y0ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+1],&y1ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+2],&y2ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+3],&y3ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+4],&y4ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+5],&y5ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+6],&y6ptr);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(yin[current_y_index+7],&y7ptr);CHKERRQ(ierr);
        current_y_index += 8;
        break;
    }
  }
  ierr = VecCUDARestoreArrayRead(xin,&xptr);CHKERRQ(ierr);

  cuda_ierr = cudaFree(group_results_gpu);CHKERRCUDA(cuda_ierr);
  ierr = PetscLogFlops(PetscMax(nv*(2.0*n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef MDOT_WORKGROUP_SIZE
#undef MDOT_WORKGROUP_NUM

PetscErrorCode VecSet_SeqCUDA(Vec xin,PetscScalar alpha)
{
  PetscInt                        n = xin->map->n;
  PetscScalar                     *xarray=NULL;
  thrust::device_ptr<PetscScalar> xptr;
  PetscErrorCode                  ierr;
  cudaError_t                     err;

  PetscFunctionBegin;
  ierr = VecCUDAGetArrayWrite(xin,&xarray);CHKERRQ(ierr);
  if (alpha == (PetscScalar)0.0) {
    err = cudaMemset(xarray,0,n*sizeof(PetscScalar));CHKERRCUDA(err);
  } else {
    try {
      xptr = thrust::device_pointer_cast(xarray);
      thrust::fill(xptr,xptr+n,alpha);
    } catch (char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
    }
  }
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  ierr = VecCUDARestoreArrayWrite(xin,&xarray);
  PetscFunctionReturn(0);
}

PetscErrorCode VecScale_SeqCUDA(Vec xin,PetscScalar alpha)
{
  PetscScalar    *xarray;
  PetscErrorCode ierr;
  PetscBLASInt   one=1,bn;
  cublasStatus_t cberr;

  PetscFunctionBegin;
  if (alpha == (PetscScalar)0.0) {
    ierr = VecSet_SeqCUDA(xin,alpha);CHKERRQ(ierr);
  } else if (alpha != (PetscScalar)1.0) {
    ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
    cberr = cublasXscal(cublasv2handle,bn,&alpha,xarray,one);CHKERRCUBLAS(cberr);
    ierr = VecCUDARestoreArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
  }
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_SeqCUDA(Vec xin,Vec yin,PetscScalar *z)
{
  const PetscScalar *xarray,*yarray;
  PetscErrorCode    ierr;
  PetscBLASInt      one=1,bn;
  cublasStatus_t    cberr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  cberr = cublasXdotu(cublasv2handle,bn,xarray,one,yarray,one,z);CHKERRCUBLAS(cberr);
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  if (xin->map->n > 0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecCopy_SeqCUDA(Vec xin,Vec yin)
{
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscErrorCode    ierr;
  cudaError_t       err;

  PetscFunctionBegin;
  if (xin != yin) {
    if (xin->valid_GPU_array == PETSC_CUDA_GPU) {
      ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
      ierr = VecCUDAGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
      err = cudaMemcpy(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRCUDA(err);
      ierr = WaitForGPU();CHKERRCUDA(ierr);
      ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
      ierr = VecCUDARestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);

    } else if (xin->valid_GPU_array == PETSC_CUDA_CPU) {
      /* copy in CPU if we are on the CPU*/
      ierr = VecCopy_SeqCUDA_Private(xin,yin);CHKERRQ(ierr);
    } else if (xin->valid_GPU_array == PETSC_CUDA_BOTH) {
      /* if xin is valid in both places, see where yin is and copy there (because it's probably where we'll want to next use it) */
      if (yin->valid_GPU_array == PETSC_CUDA_CPU) {
        /* copy in CPU */
        ierr = VecCopy_SeqCUDA_Private(xin,yin);CHKERRQ(ierr);

      } else if (yin->valid_GPU_array == PETSC_CUDA_GPU) {
        /* copy in GPU */
        ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
        err = cudaMemcpy(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRCUDA(err);
        ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
      } else if (yin->valid_GPU_array == PETSC_CUDA_BOTH) {
        /* xin and yin are both valid in both places (or yin was unallocated before the earlier call to allocatecheck
           default to copy in GPU (this is an arbitrary choice) */
        ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
        err = cudaMemcpy(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRCUDA(err);
        ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
      } else {
        ierr = VecCopy_SeqCUDA_Private(xin,yin);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecSwap_SeqCUDA(Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscBLASInt   one = 1,bn;
  PetscScalar    *xarray,*yarray;
  cublasStatus_t cberr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
  if (xin != yin) {
    ierr = VecCUDAGetArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    cberr = cublasXswap(cublasv2handle,bn,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = WaitForGPU();CHKERRCUDA(ierr);
    ierr = VecCUDARestoreArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBY_SeqCUDA(Vec yin,PetscScalar alpha,PetscScalar beta,Vec xin)
{
  PetscErrorCode    ierr;
  PetscScalar       a = alpha,b = beta;
  const PetscScalar *xarray;
  PetscScalar       *yarray;
  PetscBLASInt      one = 1, bn;
  cublasStatus_t    cberr;
  cudaError_t       err;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(yin->map->n,&bn);CHKERRQ(ierr);
  if (a == (PetscScalar)0.0) {
    ierr = VecScale_SeqCUDA(yin,beta);CHKERRQ(ierr);
  } else if (b == (PetscScalar)1.0) {
    ierr = VecAXPY_SeqCUDA(yin,alpha,xin);CHKERRQ(ierr);
  } else if (a == (PetscScalar)1.0) {
    ierr = VecAYPX_SeqCUDA(yin,beta,xin);CHKERRQ(ierr);
  } else if (b == (PetscScalar)0.0) {
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    err = cudaMemcpy(yarray,xarray,yin->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToDevice);CHKERRCUDA(err);
    cberr = cublasXscal(cublasv2handle,bn,&alpha,yarray,one);CHKERRCUBLAS(cberr);
    ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRCUDA(ierr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
  } else {
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    cberr = cublasXscal(cublasv2handle,bn,&beta,yarray,one);CHKERRCUBLAS(cberr);
    cberr = cublasXaxpy(cublasv2handle,bn,&alpha,xarray,one,yarray,one);CHKERRCUBLAS(cberr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRCUDA(ierr);
    ierr = PetscLogFlops(3.0*xin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPBYPCZ_SeqCUDA(Vec zin,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = zin->map->n;

  PetscFunctionBegin;
  if (gamma == (PetscScalar)1.0) {
    /* z = ax + b*y + z */
    ierr = VecAXPY_SeqCUDA(zin,alpha,xin);CHKERRQ(ierr);
    ierr = VecAXPY_SeqCUDA(zin,beta,yin);CHKERRQ(ierr);
    ierr = PetscLogFlops(4.0*n);CHKERRQ(ierr);
  } else {
    /* z = a*x + b*y + c*z */
    ierr = VecScale_SeqCUDA(zin,gamma);CHKERRQ(ierr);
    ierr = VecAXPY_SeqCUDA(zin,alpha,xin);CHKERRQ(ierr);
    ierr = VecAXPY_SeqCUDA(zin,beta,yin);CHKERRQ(ierr);
    ierr = PetscLogFlops(5.0*n);CHKERRQ(ierr);
  }
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseMult_SeqCUDA(Vec win,Vec xin,Vec yin)
{
  PetscInt                              n = win->map->n;
  const PetscScalar                     *xarray,*yarray;
  PetscScalar                           *warray;
  thrust::device_ptr<const PetscScalar> xptr,yptr;
  thrust::device_ptr<PetscScalar>       wptr;
  PetscErrorCode                        ierr;

  PetscFunctionBegin;
  ierr = VecCUDAGetArrayReadWrite(win,&warray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  try {
    wptr = thrust::device_pointer_cast(warray);
    xptr = thrust::device_pointer_cast(xarray);
    yptr = thrust::device_pointer_cast(yarray);
    thrust::transform(xptr,xptr+n,yptr,wptr,thrust::multiplies<PetscScalar>());
    ierr = WaitForGPU();CHKERRCUDA(ierr);
  } catch (char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
  ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayReadWrite(win,&warray);CHKERRQ(ierr);
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* should do infinity norm in cuda */

PetscErrorCode VecNorm_SeqCUDA(Vec xin,NormType type,PetscReal *z)
{
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n;
  PetscBLASInt      one = 1, bn;
  const PetscScalar *xarray;
  cublasStatus_t    cberr;
  cudaError_t       err;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    cberr = cublasXnrm2(cublasv2handle,bn,xarray,one,z);CHKERRCUBLAS(cberr);
    ierr = WaitForGPU();CHKERRCUDA(ierr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = PetscLogFlops(PetscMax(2.0*n-1,0.0));CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    int  i;
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    cberr = cublasIXamax(cublasv2handle,bn,xarray,one,&i);CHKERRCUBLAS(cberr);
    err = cudaMemcpy(z,xarray+i,sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUDA(err);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  } else if (type == NORM_1) {
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    cberr = cublasXasum(cublasv2handle,bn,xarray,one,z);CHKERRCUBLAS(cberr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRCUDA(ierr);
    ierr = PetscLogFlops(PetscMax(n-1.0,0.0));CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    ierr = VecNorm_SeqCUDA(xin,NORM_1,z);CHKERRQ(ierr);
    ierr = VecNorm_SeqCUDA(xin,NORM_2,z+1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecDotNorm2_SeqCUDA(Vec s, Vec t, PetscScalar *dp, PetscScalar *nm)
{
  PetscErrorCode    ierr;
  PetscReal         n=s->map->n;
  const PetscScalar *sarray,*tarray;

  PetscFunctionBegin;
  ierr = VecCUDAGetArrayRead(s,&sarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(t,&tarray);CHKERRQ(ierr);
  ierr = VecDot_SeqCUDA(s,t,dp);CHKERRQ(ierr);
  ierr = VecDot_SeqCUDA(t,t,nm);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(s,&sarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(t,&tarray);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  ierr = PetscLogFlops(4.0*n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroy_SeqCUDA(Vec v)
{
  PetscErrorCode ierr;
  cudaError_t    err;

  PetscFunctionBegin;
  if (v->spptr) {
    if (((Vec_CUDA*)v->spptr)->GPUarray_allocated) {
      err = cudaFree(((Vec_CUDA*)v->spptr)->GPUarray_allocated);CHKERRCUDA(err);
    }
    if (((Vec_CUDA*)v->spptr)->stream) {
      err = cudaStreamDestroy(((Vec_CUDA*)v->spptr)->stream);CHKERRCUDA(err);
    }
    ierr = PetscFree(v->spptr);CHKERRQ(ierr);
  }
  ierr = VecDestroy_SeqCUDA_Private(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_USE_COMPLEX)
struct conjugate
{
  __host__ __device__
    PetscScalar operator()(PetscScalar x)
    {
      return PetscConj(x);
    }
};
#endif

PetscErrorCode VecConjugate_SeqCUDA(Vec xin)
{
  PetscScalar                     *xarray;
  PetscErrorCode                  ierr;
#if defined(PETSC_USE_COMPLEX)
  PetscInt                        n = xin->map->n;
  thrust::device_ptr<PetscScalar> xptr;
#endif

  PetscFunctionBegin;
  ierr = VecCUDAGetArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  try {
    xptr = thrust::device_pointer_cast(xarray);
    thrust::transform(xptr,xptr+n,xptr,conjugate());
    ierr = WaitForGPU();CHKERRCUDA(ierr);
  } catch (char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Thrust error: %s", ex);
  }
#endif
  ierr = VecCUDARestoreArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetLocalVector_SeqCUDA(Vec v,Vec w)
{
  PetscErrorCode ierr;
  cudaError_t    err;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(w,VEC_CLASSID,2);
  PetscCheckTypeName(w,VECSEQCUDA);

  if (w->data) {
    if (((Vec_Seq*)w->data)->array_allocated) {
      ierr = PetscFree(((Vec_Seq*)w->data)->array_allocated);CHKERRQ(ierr);
    }
    ((Vec_Seq*)w->data)->array = NULL;
    ((Vec_Seq*)w->data)->unplacedarray = NULL;
  }
  if (w->spptr) {
    if (((Vec_CUDA*)w->spptr)->GPUarray) {
      err = cudaFree(((Vec_CUDA*)w->spptr)->GPUarray);CHKERRCUDA(err);
      ((Vec_CUDA*)w->spptr)->GPUarray = NULL;
    }
    err = cudaStreamDestroy(((Vec_CUDA*)w->spptr)->stream);CHKERRCUDA(err);
    ierr = PetscFree(w->spptr);CHKERRQ(ierr);
  }

  if (v->petscnative) {
    ierr = PetscFree(w->data);CHKERRQ(ierr);
    w->data = v->data;
    w->valid_GPU_array = v->valid_GPU_array;
    w->spptr = v->spptr;
    ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
  } else {
    ierr = VecGetArray(v,&((Vec_Seq*)w->data)->array);CHKERRQ(ierr);
    w->valid_GPU_array = PETSC_CUDA_CPU;
    ierr = VecCUDAAllocateCheck(w);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreLocalVector_SeqCUDA(Vec v,Vec w)
{
  PetscErrorCode ierr;
  cudaError_t    err;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(w,VEC_CLASSID,2);
  PetscCheckTypeName(w,VECSEQCUDA);

  if (v->petscnative) {
    v->data = w->data;
    v->valid_GPU_array = w->valid_GPU_array;
    v->spptr = w->spptr;
    ierr = VecCUDACopyFromGPU(v);CHKERRQ(ierr);
    ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
    w->data = 0;
    w->valid_GPU_array = PETSC_CUDA_UNALLOCATED;
    w->spptr = 0;
  } else {
    ierr = VecRestoreArray(v,&((Vec_Seq*)w->data)->array);CHKERRQ(ierr);
    if ((Vec_CUDA*)w->spptr) {
      err = cudaFree(((Vec_CUDA*)w->spptr)->GPUarray);CHKERRCUDA(err);
      ((Vec_CUDA*)w->spptr)->GPUarray = NULL;
      err = cudaStreamDestroy(((Vec_CUDA*)w->spptr)->stream);CHKERRCUDA(err);
      ierr = PetscFree(w->spptr);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   VecCUDAGetArrayReadWrite - Provides access to the CUDA buffer inside a vector.

   This function has semantics similar to VecGetArray():  the pointer
   returned by this function points to a consistent view of the vector
   data.  This may involve a copy operation of data from the host to the
   device if the data on the device is out of date.  If the device
   memory hasn't been allocated previously it will be allocated as part
   of this function call.  VecCUDAGetArrayReadWrite() assumes that
   the user will modify the vector data.  This is similar to
   intent(inout) in fortran.

   The CUDA device pointer has to be released by calling
   VecCUDARestoreArrayReadWrite().  Upon restoring the vector data
   the data on the host will be marked as out of date.  A subsequent
   access of the host data will thus incur a data transfer from the
   device to the host.


   Input Parameter:
.  v - the vector

   Output Parameter:
.  a - the CUDA device pointer

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecCUDARestoreArrayReadWrite(), VecCUDAGetArrayRead(), VecCUDAGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUDAGetArrayReadWrite(Vec v, PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  *a   = 0;
  ierr = VecCUDACopyToGPU(v);CHKERRQ(ierr);
  *a   = ((Vec_CUDA*)v->spptr)->GPUarray;
  PetscFunctionReturn(0);
}

/*@C
   VecCUDARestoreArrayReadWrite - Restore a CUDA device pointer previously acquired with VecCUDAGetArrayReadWrite().

   This marks the host data as out of date.  Subsequent access to the
   vector data on the host side with for instance VecGetArray() incurs a
   data transfer.

   Input Parameter:
+  v - the vector
-  a - the CUDA device pointer.  This pointer is invalid after
       VecCUDARestoreArrayReadWrite() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecCUDAGetArrayReadWrite(), VecCUDAGetArrayRead(), VecCUDAGetArrayWrite(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUDARestoreArrayReadWrite(Vec v, PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  v->valid_GPU_array = PETSC_CUDA_GPU;

  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCUDAGetArrayRead - Provides read access to the CUDA buffer inside a vector.

   This function is analogous to VecGetArrayRead():  The pointer
   returned by this function points to a consistent view of the vector
   data.  This may involve a copy operation of data from the host to the
   device if the data on the device is out of date.  If the device
   memory hasn't been allocated previously it will be allocated as part
   of this function call.  VecCUDAGetArrayRead() assumes that the
   user will not modify the vector data.  This is analgogous to
   intent(in) in Fortran.

   The CUDA device pointer has to be released by calling
   VecCUDARestoreArrayRead().  If the data on the host side was
   previously up to date it will remain so, i.e. data on both the device
   and the host is up to date.  Accessing data on the host side does not
   incur a device to host data transfer.

   Input Parameter:
.  v - the vector

   Output Parameter:
.  a - the CUDA pointer.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecCUDARestoreArrayRead(), VecCUDAGetArrayReadWrite(), VecCUDAGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUDAGetArrayRead(Vec v, const PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  *a   = 0;
  ierr = VecCUDACopyToGPU(v);CHKERRQ(ierr);
  *a   = ((Vec_CUDA*)v->spptr)->GPUarray;
  PetscFunctionReturn(0);
}

/*@C
   VecCUDARestoreArrayRead - Restore a CUDA device pointer previously acquired with VecCUDAGetArrayRead().

   If the data on the host side was previously up to date it will remain
   so, i.e. data on both the device and the host is up to date.
   Accessing data on the host side e.g. with VecGetArray() does not
   incur a device to host data transfer.

   Input Parameter:
+  v - the vector
-  a - the CUDA device pointer.  This pointer is invalid after
       VecCUDARestoreArrayRead() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecCUDAGetArrayRead(), VecCUDAGetArrayWrite(), VecCUDAGetArrayReadWrite(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUDARestoreArrayRead(Vec v, const PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  PetscFunctionReturn(0);
}

/*@C
   VecCUDAGetArrayWrite - Provides write access to the CUDA buffer inside a vector.

   The data pointed to by the device pointer is uninitialized.  The user
   may not read from this data.  Furthermore, the entire array needs to
   be filled by the user to obtain well-defined behaviour.  The device
   memory will be allocated by this function if it hasn't been allocated
   previously.  This is analogous to intent(out) in Fortran.

   The device pointer needs to be released with
   VecCUDARestoreArrayWrite().  When the pointer is released the
   host data of the vector is marked as out of data.  Subsequent access
   of the host data with e.g. VecGetArray() incurs a device to host data
   transfer.


   Input Parameter:
.  v - the vector

   Output Parameter:
.  a - the CUDA pointer

   Fortran note:
   This function is not currently available from Fortran.

   Level: advanced

.seealso: VecCUDARestoreArrayWrite(), VecCUDAGetArrayReadWrite(), VecCUDAGetArrayRead(), VecCUDAGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUDAGetArrayWrite(Vec v, PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  *a   = 0;
  ierr = VecCUDAAllocateCheck(v);CHKERRQ(ierr);
  *a   = ((Vec_CUDA*)v->spptr)->GPUarray;
  PetscFunctionReturn(0);
}

/*@C
   VecCUDARestoreArrayWrite - Restore a CUDA device pointer previously acquired with VecCUDAGetArrayWrite().

   Data on the host will be marked as out of date.  Subsequent access of
   the data on the host side e.g. with VecGetArray() will incur a device
   to host data transfer.

   Input Parameter:
+  v - the vector
-  a - the CUDA device pointer.  This pointer is invalid after
       VecCUDARestoreArrayWrite() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecCUDAGetArrayWrite(), VecCUDAGetArrayReadWrite(), VecCUDAGetArrayRead(), VecCUDAGetArrayWrite(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUDARestoreArrayWrite(Vec v, PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUDA,VECMPICUDA);
  v->valid_GPU_array = PETSC_CUDA_GPU;

  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCUDAPlaceArray - Allows one to replace the GPU array in a vector with a
   GPU array provided by the user. This is useful to avoid copying an
   array into a vector.

   Not Collective

   Input Parameters:
+  vec - the vector
-  array - the GPU array

   Notes:
   You can return to the original GPU array with a call to VecCUDAResetArray()
   It is not possible to use VecCUDAPlaceArray() and VecPlaceArray() at the
   same time on the same vector.

   Level: developer

.seealso: VecPlaceArray(), VecGetArray(), VecRestoreArray(), VecReplaceArray(), VecResetArray(), VecCUDAResetArray(), VecCUDAReplaceArray()

@*/
PetscErrorCode VecCUDAPlaceArray(Vec vin,PetscScalar *a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(vin,VECSEQCUDA,VECMPICUDA);
  ierr = VecCUDACopyToGPU(vin);CHKERRQ(ierr);
  if (((Vec_Seq*)vin->data)->unplacedarray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"VecCUDAPlaceArray()/VecPlaceArray() was already called on this vector, without a call to VecCUDAResetArray()/VecResetArray()");
  ((Vec_Seq*)vin->data)->unplacedarray  = (PetscScalar *) ((Vec_CUDA*)vin->spptr)->GPUarray; /* save previous GPU array so reset can bring it back */
  ((Vec_CUDA*)vin->spptr)->GPUarray = a;
  vin->valid_GPU_array = PETSC_CUDA_GPU;
  ierr = PetscObjectStateIncrease((PetscObject)vin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCUDAReplaceArray - Allows one to replace the GPU array in a vector
   with a GPU array provided by the user. This is useful to avoid copying
   a GPU array into a vector.

   Not Collective

   Input Parameters:
+  vec - the vector
-  array - the GPU array

   Notes:
   This permanently replaces the GPU array and frees the memory associated
   with the old GPU array.

   The memory passed in CANNOT be freed by the user. It will be freed
   when the vector is destroyed.

   Not supported from Fortran

   Level: developer

.seealso: VecGetArray(), VecRestoreArray(), VecPlaceArray(), VecResetArray(), VecCUDAResetArray(), VecCUDAPlaceArray(), VecReplaceArray()

@*/
PetscErrorCode VecCUDAReplaceArray(Vec vin,PetscScalar *a)
{
  cudaError_t err;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(vin,VECSEQCUDA,VECMPICUDA);
  err = cudaFree(((Vec_CUDA*)vin->spptr)->GPUarray);CHKERRCUDA(err);
  ((Vec_CUDA*)vin->spptr)->GPUarray = a;
  vin->valid_GPU_array = PETSC_CUDA_GPU;
  ierr = PetscObjectStateIncrease((PetscObject)vin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCUDAResetArray - Resets a vector to use its default memory. Call this
   after the use of VecCUDAPlaceArray().

   Not Collective

   Input Parameters:
.  vec - the vector

   Level: developer

.seealso: VecGetArray(), VecRestoreArray(), VecReplaceArray(), VecPlaceArray(), VecResetArray(), VecCUDAPlaceArray(), VecCUDAReplaceArray()

@*/
PetscErrorCode VecCUDAResetArray(Vec vin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(vin,VECSEQCUDA,VECMPICUDA);
  ierr = VecCUDACopyToGPU(vin);CHKERRQ(ierr);
  ((Vec_CUDA*)vin->spptr)->GPUarray = (PetscScalar *) ((Vec_Seq*)vin->data)->unplacedarray;
  ((Vec_Seq*)vin->data)->unplacedarray = 0;
  vin->valid_GPU_array = PETSC_CUDA_GPU;
  ierr = PetscObjectStateIncrease((PetscObject)vin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
