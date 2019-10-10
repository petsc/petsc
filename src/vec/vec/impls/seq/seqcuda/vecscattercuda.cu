/*
   Implements the various scatter operations on cuda vectors
*/

#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_CXX_COMPLEX_FIX

#include <petscconf.h>
#include <petsc/private/vecimpl.h>          /*I "petscvec.h" I*/
#include <../src/vec/vec/impls/dvecimpl.h>
#include <../src/vec/vec/impls/seq/seqcuda/cudavecimpl.h>

#include <cuda_runtime.h>

PetscErrorCode VecScatterCUDAIndicesCreate_StoS(PetscInt n,PetscInt toFirst,PetscInt fromFirst,PetscInt toStep, PetscInt fromStep,PetscInt *tslots, PetscInt *fslots,PetscCUDAIndices *ci) {

  PetscCUDAIndices           cci;
  VecScatterCUDAIndices_StoS stos_scatter;
  cudaError_t                err;
  PetscInt                   *intVecGPU;
  int                        device;
  cudaDeviceProp             props;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  cci = new struct _p_PetscCUDAIndices;
  stos_scatter = new struct _p_VecScatterCUDAIndices_StoS;

  /* create the "from" indices */
  stos_scatter->fslots = 0;
  stos_scatter->fromFirst = 0;
  stos_scatter->fromStep = 0;
  if (n) {
    if (fslots) {
      /* allocate GPU memory for the to-slots */
      err = cudaMalloc((void **)&intVecGPU,n*sizeof(PetscInt));CHKERRCUDA(err);
      err = cudaMemcpy(intVecGPU,fslots,n*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUDA(err);
      ierr = PetscLogCpuToGpu(n*sizeof(PetscInt));CHKERRQ(ierr);

      /* assign the pointer to the struct */
      stos_scatter->fslots = intVecGPU;
      stos_scatter->fromMode = VEC_SCATTER_CUDA_GENERAL;
    } else if (fromStep) {
      stos_scatter->fromFirst = fromFirst;
      stos_scatter->fromStep = fromStep;
      stos_scatter->fromMode = VEC_SCATTER_CUDA_STRIDED;
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide fslots or fromStep.");
  }

  /* create the "to" indices */
  stos_scatter->tslots = 0;
  stos_scatter->toFirst = 0;
  stos_scatter->toStep = 0;
  if (n) {
    if (tslots) {
      /* allocate GPU memory for the to-slots */
      err = cudaMalloc((void **)&intVecGPU,n*sizeof(PetscInt));CHKERRCUDA(err);
      err = cudaMemcpy(intVecGPU,tslots,n*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUDA(err);
      ierr = PetscLogCpuToGpu(n*sizeof(PetscInt));CHKERRQ(ierr);

      /* assign the pointer to the struct */
      stos_scatter->tslots = intVecGPU;
      stos_scatter->toMode = VEC_SCATTER_CUDA_GENERAL;
    } else if (toStep) {
      stos_scatter->toFirst = toFirst;
      stos_scatter->toStep = toStep;
      stos_scatter->toMode = VEC_SCATTER_CUDA_STRIDED;
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide tslots or toStep.");
  }

  /* Scatter uses default stream as well.
     Note: Here we have used a separate stream earlier.
           However, without proper synchronization with the default stream, one will inevitably run into data races.
           Please keep that in mind when trying to reintroduce streams for scatters.
           Ultimately, it's better to have correct code/results at 90 percent of peak performance than to have incorrect code/results at 99 percent of peak performance. */
  stos_scatter->stream = 0;

  /* the number of indices */
  stos_scatter->n = n;

  /* get the maximum number of coresident thread blocks */
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);
  stos_scatter->MAX_CORESIDENT_THREADS = props.maxThreadsPerMultiProcessor;
  if (props.major>=3) {
    stos_scatter->MAX_BLOCKS = 16*props.multiProcessorCount;
  } else {
    stos_scatter->MAX_BLOCKS = 8*props.multiProcessorCount;
  }

  /* assign the indices */
  cci->scatter = (VecScatterCUDAIndices_StoS)stos_scatter;
  cci->scatterType = VEC_SCATTER_CUDA_STOS;
  *ci = cci;
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterCUDAIndicesCreate_PtoP(PetscInt ns,PetscInt *sendIndices,PetscInt nr,PetscInt *recvIndices,PetscCUDAIndices *ci)
{
  PetscCUDAIndices           cci;
  VecScatterCUDAIndices_PtoP ptop_scatter;

  PetscFunctionBegin;
  cci = new struct _p_PetscCUDAIndices;
  ptop_scatter = new struct _p_VecScatterCUDAIndices_PtoP;

  /* this calculation assumes that the input indices are sorted */
  if (sendIndices) {
    ptop_scatter->ns = sendIndices[ns-1]-sendIndices[0]+1;
    ptop_scatter->sendLowestIndex = sendIndices[0];
  } else {
    ptop_scatter->ns = 0;
    ptop_scatter->sendLowestIndex = 0;
  }
  if (recvIndices) {
    ptop_scatter->nr = recvIndices[nr-1]-recvIndices[0]+1;
    ptop_scatter->recvLowestIndex = recvIndices[0];
  } else {
    ptop_scatter->nr = 0;
    ptop_scatter->recvLowestIndex = 0;
  }

  /* assign indices */
  cci->scatter = (VecScatterCUDAIndices_PtoP)ptop_scatter;
  cci->scatterType = VEC_SCATTER_CUDA_PTOP;

  *ci = cci;
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterCUDAIndicesDestroy(PetscCUDAIndices *ci)
{
  PetscFunctionBegin;
  if (*ci) {
    if ((*ci)->scatterType == VEC_SCATTER_CUDA_PTOP) {
      delete (VecScatterCUDAIndices_PtoP)(*ci)->scatter;
      (*ci)->scatter = 0;
    } else {
      cudaError_t err;
      VecScatterCUDAIndices_StoS stos_scatter = (VecScatterCUDAIndices_StoS)(*ci)->scatter;
      if (stos_scatter->fslots) {
        err = cudaFree(stos_scatter->fslots);CHKERRCUDA(err);
        stos_scatter->fslots = 0;
      }

      /* free the GPU memory for the to-slots */
      if (stos_scatter->tslots) {
        err = cudaFree(stos_scatter->tslots);CHKERRCUDA(err);
        stos_scatter->tslots = 0;
      }

      /* free the stream variable */
      if (stos_scatter->stream) {
        err = cudaStreamDestroy(stos_scatter->stream);CHKERRCUDA(err);
        stos_scatter->stream = 0;
      }
      delete stos_scatter;
      (*ci)->scatter = 0;
    }
    delete *ci;
    *ci = 0;
  }
  PetscFunctionReturn(0);
}

/* Insert operator */
class Insert {
  public:
    __device__ PetscScalar operator() (PetscScalar a,PetscScalar b) const {
      return a;
    }
};

/* Add operator */
class Add {
  public:
    __device__ PetscScalar operator() (PetscScalar a,PetscScalar b) const {
      return a+b;
    }
};

/* Add operator */
class Max {
  public:
    __device__ PetscScalar operator() (PetscScalar a,PetscScalar b) const {
      return PetscMax(PetscRealPart(a),PetscRealPart(b));
    }
};

/* Sequential general to sequential general GPU kernel */
template<class OPERATOR>
__global__ void VecScatterCUDA_SGtoSG_kernel(PetscInt n,PetscInt *xind,const PetscScalar *x,PetscInt *yind,PetscScalar *y,OPERATOR OP) {
  const int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  const int grid_size = gridDim.x * blockDim.x;
  for (int i = tidx; i < n; i += grid_size) {
    y[yind[i]] = OP(x[xind[i]],y[yind[i]]);
  }
}

/* Sequential general to sequential strided GPU kernel */
template<class OPERATOR>
__global__ void VecScatterCUDA_SGtoSS_kernel(PetscInt n,PetscInt *xind,const PetscScalar *x,PetscInt toFirst,PetscInt toStep,PetscScalar *y,OPERATOR OP) {
  const int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  const int grid_size = gridDim.x * blockDim.x;
  for (int i = tidx; i < n; i += grid_size) {
    y[toFirst+i*toStep] = OP(x[xind[i]],y[toFirst+i*toStep]);
  }
}

/* Sequential strided to sequential strided GPU kernel */
template<class OPERATOR>
__global__ void VecScatterCUDA_SStoSS_kernel(PetscInt n,PetscInt fromFirst,PetscInt fromStep,const PetscScalar *x,PetscInt toFirst,PetscInt toStep,PetscScalar *y,OPERATOR OP) {
  const int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  const int grid_size = gridDim.x * blockDim.x;
  for (int i = tidx; i < n; i += grid_size) {
    y[toFirst+i*toStep] = OP(x[fromFirst+i*fromStep],y[toFirst+i*toStep]);
  }
}

/* Sequential strided to sequential general GPU kernel */
template<class OPERATOR>
__global__ void VecScatterCUDA_SStoSG_kernel(PetscInt n,PetscInt fromFirst,PetscInt fromStep,const PetscScalar *x,PetscInt *yind,PetscScalar *y,OPERATOR OP) {
  const int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  const int grid_size = gridDim.x * blockDim.x;
  for (int i = tidx; i < n; i += grid_size) {
    y[yind[i]] = OP(x[fromFirst+i*fromStep],y[yind[i]]);
  }
}

template<class OPERATOR>
void VecScatterCUDA_StoS_Dispatcher(const PetscScalar *xarray,PetscScalar *yarray,PetscCUDAIndices ci,ScatterMode mode,OPERATOR OP) {

  PetscInt                   nBlocks=0,nThreads=128;
  VecScatterCUDAIndices_StoS stos_scatter = (VecScatterCUDAIndices_StoS)ci->scatter;

  nBlocks=(int)ceil(((float) stos_scatter->n)/((float) nThreads))+1;
  if (nBlocks>stos_scatter->MAX_CORESIDENT_THREADS/nThreads) {
    nBlocks = stos_scatter->MAX_CORESIDENT_THREADS/nThreads;
  }
  dim3 block(nThreads,1,1);
  dim3 grid(nBlocks,1,1);

  if (mode == SCATTER_FORWARD) {
    if (stos_scatter->fromMode == VEC_SCATTER_CUDA_GENERAL && stos_scatter->toMode == VEC_SCATTER_CUDA_GENERAL) {
      VecScatterCUDA_SGtoSG_kernel<<<grid,block,0,stos_scatter->stream>>>(stos_scatter->n,stos_scatter->fslots,xarray,stos_scatter->tslots,yarray,OP);
    } else if (stos_scatter->fromMode == VEC_SCATTER_CUDA_GENERAL && stos_scatter->toMode == VEC_SCATTER_CUDA_STRIDED) {
      VecScatterCUDA_SGtoSS_kernel<<<grid,block,0,stos_scatter->stream>>>(stos_scatter->n,stos_scatter->fslots,xarray,stos_scatter->toFirst,stos_scatter->toStep,yarray,OP);
    } else if (stos_scatter->fromMode == VEC_SCATTER_CUDA_STRIDED && stos_scatter->toMode == VEC_SCATTER_CUDA_STRIDED) {
      VecScatterCUDA_SStoSS_kernel<<<grid,block,0,stos_scatter->stream>>>(stos_scatter->n,stos_scatter->fromFirst,stos_scatter->fromStep,xarray,stos_scatter->toFirst,stos_scatter->toStep,yarray,OP);
    } else if (stos_scatter->fromMode == VEC_SCATTER_CUDA_STRIDED && stos_scatter->toMode == VEC_SCATTER_CUDA_GENERAL) {
      VecScatterCUDA_SStoSG_kernel<<<grid,block,0,stos_scatter->stream>>>(stos_scatter->n,stos_scatter->fromFirst,stos_scatter->fromStep,xarray,stos_scatter->tslots,yarray,OP);
    }
  } else {
    if (stos_scatter->toMode == VEC_SCATTER_CUDA_GENERAL && stos_scatter->fromMode == VEC_SCATTER_CUDA_GENERAL) {
      VecScatterCUDA_SGtoSG_kernel<<<grid,block,0,stos_scatter->stream>>>(stos_scatter->n,stos_scatter->tslots,xarray,stos_scatter->fslots,yarray,OP);
    } else if (stos_scatter->toMode == VEC_SCATTER_CUDA_GENERAL && stos_scatter->fromMode == VEC_SCATTER_CUDA_STRIDED) {
      VecScatterCUDA_SGtoSS_kernel<<<grid,block,0,stos_scatter->stream>>>(stos_scatter->n,stos_scatter->tslots,xarray,stos_scatter->fromFirst,stos_scatter->fromStep,yarray,OP);
    } else if (stos_scatter->toMode == VEC_SCATTER_CUDA_STRIDED && stos_scatter->fromMode == VEC_SCATTER_CUDA_STRIDED) {
      VecScatterCUDA_SStoSS_kernel<<<grid,block,0,stos_scatter->stream>>>(stos_scatter->n,stos_scatter->toFirst,stos_scatter->toStep,xarray,stos_scatter->fromFirst,stos_scatter->fromStep,yarray,OP);
    } else if (stos_scatter->toMode == VEC_SCATTER_CUDA_STRIDED && stos_scatter->fromMode == VEC_SCATTER_CUDA_GENERAL) {
      VecScatterCUDA_SStoSG_kernel<<<grid,block,0,stos_scatter->stream>>>(stos_scatter->n,stos_scatter->toFirst,stos_scatter->toStep,xarray,stos_scatter->fslots,yarray,OP);
    }
  }
}

PetscErrorCode VecScatterCUDA_StoS(Vec x,Vec y,PetscCUDAIndices ci,InsertMode addv,ScatterMode mode)
{
  PetscErrorCode             ierr;
  const PetscScalar          *xarray;
  PetscScalar                *yarray;
  VecScatterCUDAIndices_StoS stos_scatter = (VecScatterCUDAIndices_StoS)ci->scatter;
  cudaError_t                err;

  PetscFunctionBegin;
  ierr = VecCUDAAllocateCheck(x);CHKERRQ(ierr);
  ierr = VecCUDAAllocateCheck(y);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(x,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArray(y,&yarray);CHKERRQ(ierr);
  if (stos_scatter->n) {
    if (addv == INSERT_VALUES)
      VecScatterCUDA_StoS_Dispatcher(xarray,yarray,ci,mode,Insert());
    else if (addv == ADD_VALUES)
      VecScatterCUDA_StoS_Dispatcher(xarray,yarray,ci,mode,Add());
    else if (addv == MAX_VALUES)
      VecScatterCUDA_StoS_Dispatcher(xarray,yarray,ci,mode,Max());
    else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
    err = cudaGetLastError();CHKERRCUDA(err);
    err = cudaStreamSynchronize(stos_scatter->stream);CHKERRCUDA(err);
  }
  ierr = VecCUDARestoreArrayRead(x,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArray(y,&yarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
