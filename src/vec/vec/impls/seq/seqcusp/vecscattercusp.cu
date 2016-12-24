/*
   Implements the various scatter operations on cusp vectors
*/

#define PETSC_SKIP_COMPLEX
#define PETSC_SKIP_SPINLOCK

#include <petscconf.h>
#include <petsc/private/vecimpl.h>          /*I "petscvec.h" I*/
#include <../src/vec/vec/impls/dvecimpl.h>
#include <../src/vec/vec/impls/seq/seqcusp/cuspvecimpl.h>

#include <cuda_runtime.h>

PetscErrorCode VecScatterCUSPIndicesCreate_StoS(PetscInt n,PetscInt toFirst,PetscInt fromFirst,PetscInt toStep, PetscInt fromStep,PetscInt *tslots, PetscInt *fslots,PetscCUSPIndices *ci) {

  PetscCUSPIndices           cci;
  VecScatterCUSPIndices_StoS stos_scatter;
  cudaError_t                err = cudaSuccess;
  cudaStream_t               stream;
  PetscInt                   *intVecGPU;
  int                        device;
  cudaDeviceProp             props;

  PetscFunctionBegin;
  cci = new struct _p_PetscCUSPIndices;
  stos_scatter = new struct _p_VecScatterCUSPIndices_StoS;

  /* create the "from" indices */
  stos_scatter->fslots = 0;
  stos_scatter->fromFirst = 0;
  stos_scatter->fromStep = 0;
  if (n) {
    if (fslots) {
      /* allocate GPU memory for the to-slots */
      err = cudaMalloc((void **)&intVecGPU,n*sizeof(PetscInt));CHKERRCUSP((int)err);
      err = cudaMemcpy(intVecGPU,fslots,n*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUSP((int)err);

      /* assign the pointer to the struct */
      stos_scatter->fslots = intVecGPU;
      stos_scatter->fromMode = VEC_SCATTER_CUSP_GENERAL;
    } else if (fromStep) {
      stos_scatter->fromFirst = fromFirst;
      stos_scatter->fromStep = fromStep;
      stos_scatter->fromMode = VEC_SCATTER_CUSP_STRIDED;
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide fslots or fromStep.");
  }

  /* create the "to" indices */
  stos_scatter->tslots = 0;
  stos_scatter->toFirst = 0;
  stos_scatter->toStep = 0;
  if (n) {
    if (tslots) {
      /* allocate GPU memory for the to-slots */
      err = cudaMalloc((void **)&intVecGPU,n*sizeof(PetscInt));CHKERRCUSP((int)err);
      err = cudaMemcpy(intVecGPU,tslots,n*sizeof(PetscInt),cudaMemcpyHostToDevice);CHKERRCUSP((int)err);

      /* assign the pointer to the struct */
      stos_scatter->tslots = intVecGPU;
      stos_scatter->toMode = VEC_SCATTER_CUSP_GENERAL;
    } else if (toStep) {
      stos_scatter->toFirst = toFirst;
      stos_scatter->toStep = toStep;
      stos_scatter->toMode = VEC_SCATTER_CUSP_STRIDED;
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Must provide tslots or toStep.");
  }

  /* allocate the stream variable */
  err = cudaStreamCreate(&stream);CHKERRCUSP((int)err);
  stos_scatter->stream = stream;

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
  cci->scatter = (VecScatterCUSPIndices_StoS)stos_scatter;
  cci->scatterType = VEC_SCATTER_CUSP_STOS;
  *ci = cci;
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterCUSPIndicesCreate_PtoP(PetscInt ns,PetscInt *sendIndices,PetscInt nr,PetscInt *recvIndices,PetscCUSPIndices *ci)
{
  PetscCUSPIndices           cci;
  VecScatterCUSPIndices_PtoP ptop_scatter;

  PetscFunctionBegin;
  cci = new struct _p_PetscCUSPIndices;
  ptop_scatter = new struct _p_VecScatterCUSPIndices_PtoP;

  /* this calculation assumes that the input indices are sorted */
  ptop_scatter->ns = sendIndices[ns-1]-sendIndices[0]+1;
  ptop_scatter->sendLowestIndex = sendIndices[0];
  ptop_scatter->nr = recvIndices[nr-1]-recvIndices[0]+1;
  ptop_scatter->recvLowestIndex = recvIndices[0];

  /* assign indices */
  cci->scatter = (VecScatterCUSPIndices_PtoP)ptop_scatter;
  cci->scatterType = VEC_SCATTER_CUSP_PTOP;

  *ci = cci;
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterCUSPIndicesDestroy(PetscCUSPIndices *ci)
{
  PetscFunctionBegin;
  if (!(*ci)) PetscFunctionReturn(0);
  try {
    if (ci) {
      if ((*ci)->scatterType == VEC_SCATTER_CUSP_PTOP) {
	delete (VecScatterCUSPIndices_PtoP)(*ci)->scatter;
	(*ci)->scatter = 0;
      } else {
	cudaError_t err = cudaSuccess;
	VecScatterCUSPIndices_StoS stos_scatter = (VecScatterCUSPIndices_StoS)(*ci)->scatter;
	if (stos_scatter->fslots) {
	  err = cudaFree(stos_scatter->fslots);CHKERRCUSP((int)err);
	  stos_scatter->fslots = 0;
	}

	/* free the GPU memory for the to-slots */
	if (stos_scatter->tslots) {
	  err = cudaFree(stos_scatter->tslots);CHKERRCUSP((int)err);
	  stos_scatter->tslots = 0;
	}

	/* free the stream variable */
	if (stos_scatter->stream) {
	  err = cudaStreamDestroy(stos_scatter->stream);CHKERRCUSP((int)err);
	  stos_scatter->stream = 0;
	}
	delete stos_scatter;
	(*ci)->scatter = 0;
      }
      delete *ci;
      *ci = 0;
    }
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
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
#if !defined(PETSC_USE_COMPLEX)
    return PetscMax(a,b);
#endif
  }
};

/* Sequential general to sequential general GPU kernel */
template<class OPERATOR>
__global__ void VecScatterCUSP_SGtoSG_kernel(PetscInt n,PetscInt *xind,PetscScalar *x,PetscInt *yind,PetscScalar *y,OPERATOR OP) {
  const int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  const int grid_size = gridDim.x * blockDim.x;
  for (int i = tidx; i < n; i += grid_size) {
    y[yind[i]] = OP(x[xind[i]],y[yind[i]]);
  }
}

/* Sequential general to sequential strided GPU kernel */
template<class OPERATOR>
__global__ void VecScatterCUSP_SGtoSS_kernel(PetscInt n,PetscInt *xind,PetscScalar *x,PetscInt toFirst,PetscInt toStep,PetscScalar *y,OPERATOR OP) {
  const int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  const int grid_size = gridDim.x * blockDim.x;
  for (int i = tidx; i < n; i += grid_size) {
    y[toFirst+i*toStep] = OP(x[xind[i]],y[toFirst+i*toStep]);
  }
}

/* Sequential strided to sequential strided GPU kernel */
template<class OPERATOR>
__global__ void VecScatterCUSP_SStoSS_kernel(PetscInt n,PetscInt fromFirst,PetscInt fromStep,PetscScalar *x,PetscInt toFirst,PetscInt toStep,PetscScalar *y,OPERATOR OP) {
  const int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  const int grid_size = gridDim.x * blockDim.x;
  for (int i = tidx; i < n; i += grid_size) {
    y[toFirst+i*toStep] = OP(x[fromFirst+i*fromStep],y[toFirst+i*toStep]);
  }
}

/* Sequential strided to sequential general GPU kernel */
template<class OPERATOR>
__global__ void VecScatterCUSP_SStoSG_kernel(PetscInt n,PetscInt fromFirst,PetscInt fromStep,PetscScalar *x,PetscInt *yind,PetscScalar *y,OPERATOR OP) {
  const int tidx = blockIdx.x*blockDim.x + threadIdx.x;
  const int grid_size = gridDim.x * blockDim.x;
  for (int i = tidx; i < n; i += grid_size) {
    y[yind[i]] = OP(x[fromFirst+i*fromStep],y[yind[i]]);
  }
}

template<class OPERATOR>
void VecScatterCUSP_StoS_Dispatcher(CUSPARRAY *xarray,CUSPARRAY *yarray,PetscCUSPIndices ci,ScatterMode mode,OPERATOR OP) {

  PetscInt                   nBlocks=0,nThreads=128;
  VecScatterCUSPIndices_StoS stos_scatter = (VecScatterCUSPIndices_StoS)ci->scatter;

  nBlocks=(int)ceil(((float) stos_scatter->n)/((float) nThreads))+1;
  if (nBlocks>stos_scatter->MAX_CORESIDENT_THREADS/nThreads) {
    nBlocks = stos_scatter->MAX_CORESIDENT_THREADS/nThreads;
  }
  dim3 block(nThreads,1,1);
  dim3 grid(nBlocks,1,1);

  if (mode == SCATTER_FORWARD) {
    if (stos_scatter->fromMode == VEC_SCATTER_CUSP_GENERAL && stos_scatter->toMode == VEC_SCATTER_CUSP_GENERAL) {
      VecScatterCUSP_SGtoSG_kernel<<<grid,block,0,stos_scatter->stream>>>(stos_scatter->n,stos_scatter->fslots,xarray->data().get(),stos_scatter->tslots,yarray->data().get(),OP);
    } else if (stos_scatter->fromMode == VEC_SCATTER_CUSP_GENERAL && stos_scatter->toMode == VEC_SCATTER_CUSP_STRIDED) {
      VecScatterCUSP_SGtoSS_kernel<<<grid,block,0,stos_scatter->stream>>>(stos_scatter->n,stos_scatter->fslots,xarray->data().get(),stos_scatter->toFirst,stos_scatter->toStep,yarray->data().get(),OP);
    } else if (stos_scatter->fromMode == VEC_SCATTER_CUSP_STRIDED && stos_scatter->toMode == VEC_SCATTER_CUSP_STRIDED) {
      VecScatterCUSP_SStoSS_kernel<<<grid,block,0,stos_scatter->stream>>>(stos_scatter->n,stos_scatter->fromFirst,stos_scatter->fromStep,xarray->data().get(),stos_scatter->toFirst,stos_scatter->toStep,yarray->data().get(),OP);
    } else if (stos_scatter->fromMode == VEC_SCATTER_CUSP_STRIDED && stos_scatter->toMode == VEC_SCATTER_CUSP_GENERAL) {
      VecScatterCUSP_SStoSG_kernel<<<grid,block,0,stos_scatter->stream>>>(stos_scatter->n,stos_scatter->fromFirst,stos_scatter->fromStep,xarray->data().get(),stos_scatter->tslots,yarray->data().get(),OP);
    }
  } else {
    if (stos_scatter->toMode == VEC_SCATTER_CUSP_GENERAL && stos_scatter->fromMode == VEC_SCATTER_CUSP_GENERAL) {
      VecScatterCUSP_SGtoSG_kernel<<<grid,block,0,stos_scatter->stream>>>(stos_scatter->n,stos_scatter->tslots,xarray->data().get(),stos_scatter->fslots,yarray->data().get(),OP);
    } else if (stos_scatter->toMode == VEC_SCATTER_CUSP_GENERAL && stos_scatter->fromMode == VEC_SCATTER_CUSP_STRIDED) {
      VecScatterCUSP_SGtoSS_kernel<<<grid,block,0,stos_scatter->stream>>>(stos_scatter->n,stos_scatter->tslots,xarray->data().get(),stos_scatter->fromFirst,stos_scatter->fromStep,yarray->data().get(),OP);
    } else if (stos_scatter->toMode == VEC_SCATTER_CUSP_STRIDED && stos_scatter->fromMode == VEC_SCATTER_CUSP_STRIDED) {
      VecScatterCUSP_SStoSS_kernel<<<grid,block,0,stos_scatter->stream>>>(stos_scatter->n,stos_scatter->toFirst,stos_scatter->toStep,xarray->data().get(),stos_scatter->fromFirst,stos_scatter->fromStep,yarray->data().get(),OP);
    } else if (stos_scatter->toMode == VEC_SCATTER_CUSP_STRIDED && stos_scatter->fromMode == VEC_SCATTER_CUSP_GENERAL) {
      VecScatterCUSP_SStoSG_kernel<<<grid,block,0,stos_scatter->stream>>>(stos_scatter->n,stos_scatter->toFirst,stos_scatter->toStep,xarray->data().get(),stos_scatter->fslots,yarray->data().get(),OP);
    }
  }
}

PetscErrorCode VecScatterCUSP_StoS(Vec x,Vec y,PetscCUSPIndices ci,InsertMode addv,ScatterMode mode)
{
  PetscErrorCode             ierr;
  CUSPARRAY                  *xarray,*yarray;
  VecScatterCUSPIndices_StoS stos_scatter = (VecScatterCUSPIndices_StoS)ci->scatter;
  cudaError_t                err = cudaSuccess;

  PetscFunctionBegin;
  ierr = VecCUSPAllocateCheck(x);CHKERRQ(ierr);
  ierr = VecCUSPAllocateCheck(y);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(x,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayReadWrite(y,&yarray);CHKERRQ(ierr);
  if (stos_scatter->n) {
    if (addv == INSERT_VALUES)
      VecScatterCUSP_StoS_Dispatcher(xarray,yarray,ci,mode,Insert());
    else if (addv == ADD_VALUES)
      VecScatterCUSP_StoS_Dispatcher(xarray,yarray,ci,mode,Add());
#if !defined(PETSC_USE_COMPLEX)
    else if (addv == MAX_VALUES)
      VecScatterCUSP_StoS_Dispatcher(xarray,yarray,ci,mode,Max());
#endif
    else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Wrong insert option");
    err = cudaGetLastError();CHKERRCUSP((int)err);
    err = cudaStreamSynchronize(stos_scatter->stream);CHKERRCUSP((int)err);
  }
  ierr = VecCUSPRestoreArrayRead(x,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayReadWrite(y,&yarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
