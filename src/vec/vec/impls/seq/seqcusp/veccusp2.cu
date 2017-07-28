/*
   Implements the sequential cusp vectors.
*/

#define PETSC_SKIP_COMPLEX
#define PETSC_SKIP_SPINLOCK

#include <petscconf.h>
#include <petsc/private/vecimpl.h>
#include <../src/vec/vec/impls/dvecimpl.h>
#include <../src/vec/vec/impls/seq/seqcusp/cuspvecimpl.h>

#include <cuda_runtime.h>

/*
    Allocates space for the vector array on the GPU if it does not exist.
    Does NOT change the PetscCUSPFlag for the vector
    Does NOT zero the CUSP array

 */
PetscErrorCode VecCUSPAllocateCheck(Vec v)
{
  cudaError_t    err;
  cudaStream_t   stream;
  Vec_CUSP       *veccusp;

  PetscFunctionBegin;
  if (!v->spptr) {
    try {
      v->spptr = new Vec_CUSP;
      veccusp = (Vec_CUSP*)v->spptr;
      veccusp->GPUarray = new CUSPARRAY;
      veccusp->GPUarray->resize((PetscBLASInt)v->map->n);
      err = cudaStreamCreate(&stream);CHKERRCUSP(err);
      veccusp->stream = stream;
      veccusp->hostDataRegisteredAsPageLocked = PETSC_FALSE;
      v->ops->destroy = VecDestroy_SeqCUSP;
      if (v->valid_GPU_array == PETSC_CUSP_UNALLOCATED) {
        if (v->data && ((Vec_Seq*)v->data)->array) {
          v->valid_GPU_array = PETSC_CUSP_CPU;
        } else {
          v->valid_GPU_array = PETSC_CUSP_GPU;
        }
      }
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
  }
  PetscFunctionReturn(0);
}

/* Copies a vector from the CPU to the GPU unless we already have an up-to-date copy on the GPU */
PetscErrorCode VecCUSPCopyToGPU(Vec v)
{
  PetscErrorCode ierr;
  cudaError_t    err;
  Vec_CUSP       *veccusp;
  CUSPARRAY      *varray;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUSP,VECMPICUSP);
  ierr = VecCUSPAllocateCheck(v);CHKERRQ(ierr);
  if (v->valid_GPU_array == PETSC_CUSP_CPU) {
    ierr = PetscLogEventBegin(VEC_CUSPCopyToGPU,v,0,0,0);CHKERRQ(ierr);
    try {
      veccusp=(Vec_CUSP*)v->spptr;
      varray=veccusp->GPUarray;
      err = cudaMemcpy(varray->data().get(),((Vec_Seq*)v->data)->array,v->map->n*sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUSP(err);
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = PetscLogEventEnd(VEC_CUSPCopyToGPU,v,0,0,0);CHKERRQ(ierr);
    v->valid_GPU_array = PETSC_CUSP_BOTH;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecCUSPCopyToGPUSome(Vec v, PetscCUSPIndices ci)
{
  CUSPARRAY      *varray;
  PetscErrorCode ierr;
  cudaError_t    err;
  PetscScalar    *cpuPtr, *gpuPtr;
  Vec_Seq        *s;
  VecScatterCUSPIndices_PtoP ptop_scatter = (VecScatterCUSPIndices_PtoP)ci->scatter;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUSP,VECMPICUSP);
  ierr = VecCUSPAllocateCheck(v);CHKERRQ(ierr);
  if (v->valid_GPU_array == PETSC_CUSP_CPU) {
    s = (Vec_Seq*)v->data;

    ierr   = PetscLogEventBegin(VEC_CUSPCopyToGPUSome,v,0,0,0);CHKERRQ(ierr);
    varray = ((Vec_CUSP*)v->spptr)->GPUarray;
    gpuPtr = varray->data().get() + ptop_scatter->recvLowestIndex;
    cpuPtr = s->array + ptop_scatter->recvLowestIndex;

    /* Note : this code copies the smallest contiguous chunk of data
       containing ALL of the indices */
    err = cudaMemcpy(gpuPtr,cpuPtr,ptop_scatter->nr*sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUSP(err);

    // Set the buffer states
    v->valid_GPU_array = PETSC_CUSP_BOTH;
    ierr = PetscLogEventEnd(VEC_CUSPCopyToGPUSome,v,0,0,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*
     VecCUSPCopyFromGPU - Copies a vector from the GPU to the CPU unless we already have an up-to-date copy on the CPU
*/
PetscErrorCode VecCUSPCopyFromGPU(Vec v)
{
  PetscErrorCode ierr;
  cudaError_t    err;
  Vec_CUSP       *veccusp;
  CUSPARRAY      *varray;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUSP,VECMPICUSP);
  ierr = VecCUSPAllocateCheckHost(v);CHKERRQ(ierr);
  if (v->valid_GPU_array == PETSC_CUSP_GPU) {
    ierr = PetscLogEventBegin(VEC_CUSPCopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    try {
      veccusp=(Vec_CUSP*)v->spptr;
      varray=veccusp->GPUarray;
      err = cudaMemcpy(((Vec_Seq*)v->data)->array,varray->data().get(),v->map->n*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUSP(err);
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = PetscLogEventEnd(VEC_CUSPCopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    v->valid_GPU_array = PETSC_CUSP_BOTH;
  }
  PetscFunctionReturn(0);
}

/* Note that this function only copies *some* of the values up from the GPU to CPU,
   which means that we need recombine the data at some point before using any of the standard functions.
   We could add another few flag-types to keep track of this, or treat things like VecGetArray VecRestoreArray
   where you have to always call in pairs
*/
PetscErrorCode VecCUSPCopyFromGPUSome(Vec v, PetscCUSPIndices ci)
{
  CUSPARRAY      *varray;
  PetscErrorCode ierr;
  cudaError_t    err;
  PetscScalar    *cpuPtr, *gpuPtr;
  Vec_Seq        *s;
  VecScatterCUSPIndices_PtoP ptop_scatter = (VecScatterCUSPIndices_PtoP)ci->scatter;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUSP,VECMPICUSP);
  ierr = VecCUSPAllocateCheckHost(v);CHKERRQ(ierr);
  if (v->valid_GPU_array == PETSC_CUSP_GPU) {
    ierr   = PetscLogEventBegin(VEC_CUSPCopyFromGPUSome,v,0,0,0);CHKERRQ(ierr);

    varray=((Vec_CUSP*)v->spptr)->GPUarray;
    s = (Vec_Seq*)v->data;
    gpuPtr = varray->data().get() + ptop_scatter->sendLowestIndex;
    cpuPtr = s->array + ptop_scatter->sendLowestIndex;

    /* Note : this code copies the smallest contiguous chunk of data
       containing ALL of the indices */
    err = cudaMemcpy(cpuPtr,gpuPtr,ptop_scatter->ns*sizeof(PetscScalar),cudaMemcpyDeviceToHost);CHKERRCUSP(err);

    ierr = VecCUSPRestoreArrayRead(v,&varray);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(VEC_CUSPCopyFromGPUSome,v,0,0,0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   VECSEQCUSP - VECSEQCUSP = "seqcusp" - The basic sequential vector, modified to use CUSP

   Options Database Keys:
. -vec_type seqcusp - sets the vector type to VECSEQCUSP during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateSeqWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateSeq()
M*/

/* for VecAYPX_SeqCUSP*/
namespace cusp
{
namespace blas
{
namespace detail
{
  template <typename T>
    struct AYPX : public thrust::binary_function<T,T,T>
    {
      T alpha;

      AYPX(T _alpha) : alpha(_alpha) {}

      __host__ __device__
      T operator()(T x, T y)
      {
        return alpha * y + x;
      }
    };
}

 template <typename ForwardIterator1,
           typename ForwardIterator2,
           typename ScalarType>
void aypx(ForwardIterator1 first1,ForwardIterator1 last1,ForwardIterator2 first2,ScalarType alpha)
           {
             thrust::transform(first1,last1,first2,first2,detail::AYPX<ScalarType>(alpha));
           }
 template <typename Array1, typename Array2, typename ScalarType>
   void aypx(const Array1& x, Array2& y, ScalarType alpha)
 {
#if defined(CUSP_VERSION) && CUSP_VERSION >= 500
   cusp::assert_same_dimensions(x,y);
#else
   detail::assert_same_dimensions(x,y);
#endif
   aypx(x.begin(),x.end(),y.begin(),alpha);
 }
}
}

PetscErrorCode VecAYPX_SeqCUSP(Vec yin, PetscScalar alpha, Vec xin)
{
  CUSPARRAY      *xarray,*yarray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
  try {
    if (alpha != 0.0) {
      cusp::blas::aypx(*xarray,*yarray,alpha);
      ierr = PetscLogFlops(2.0*yin->map->n);CHKERRQ(ierr);
    } else {
      cusp::blas::copy(*xarray,*yarray);
    }
    ierr = WaitForGPU();CHKERRCUSP(ierr);
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode VecAXPY_SeqCUSP(Vec yin,PetscScalar alpha,Vec xin)
{
  CUSPARRAY      *xarray,*yarray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha != 0.0) {
    ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    try {
      cusp::blas::axpy(*xarray,*yarray,alpha);
      ierr = WaitForGPU();CHKERRCUSP(ierr);
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogFlops(2.0*yin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

struct VecCUSPPointwiseDivide
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = thrust::get<1>(t) / thrust::get<2>(t);
  }
};

PetscErrorCode VecPointwiseDivide_SeqCUSP(Vec win, Vec xin, Vec yin)
{
  CUSPARRAY      *warray=NULL,*xarray=NULL,*yarray=NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayWrite(win,&warray);CHKERRQ(ierr);
  try {
    thrust::for_each(
      thrust::make_zip_iterator(
        thrust::make_tuple(
          warray->begin(),
          xarray->begin(),
          yarray->begin())),
      thrust::make_zip_iterator(
        thrust::make_tuple(
          warray->end(),
          xarray->end(),
          yarray->end())),
      VecCUSPPointwiseDivide());
    ierr = WaitForGPU();CHKERRCUSP(ierr);
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  ierr = PetscLogFlops(win->map->n);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(win,&warray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


struct VecCUSPWAXPY
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = thrust::get<1>(t) + thrust::get<2>(t)*thrust::get<3>(t);
  }
};

struct VecCUSPSum
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = thrust::get<1>(t) + thrust::get<2>(t);
  }
};

struct VecCUSPDiff
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = thrust::get<1>(t) - thrust::get<2>(t);
  }
};

PetscErrorCode VecWAXPY_SeqCUSP(Vec win,PetscScalar alpha,Vec xin, Vec yin)
{
  CUSPARRAY      *xarray=NULL,*yarray=NULL,*warray=NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == 0.0) {
    ierr = VecCopy_SeqCUSP(yin,win);CHKERRQ(ierr);
  } else {
    ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(yin,&yarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayWrite(win,&warray);CHKERRQ(ierr);
    if (alpha == 1.0) {
      try {
        thrust::for_each(
          thrust::make_zip_iterator(
            thrust::make_tuple(
              warray->begin(),
              yarray->begin(),
              xarray->begin())),
          thrust::make_zip_iterator(
            thrust::make_tuple(
              warray->end(),
              yarray->end(),
              xarray->end())),
          VecCUSPSum());
      } catch(char *ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
      }
      ierr = PetscLogFlops(win->map->n);CHKERRQ(ierr);
    } else if (alpha == -1.0) {
      try {
        thrust::for_each(
          thrust::make_zip_iterator(
            thrust::make_tuple(
              warray->begin(),
              yarray->begin(),
              xarray->begin())),
          thrust::make_zip_iterator(
            thrust::make_tuple(
              warray->end(),
              yarray->end(),
              xarray->end())),
          VecCUSPDiff());
      } catch(char *ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
      }
      ierr = PetscLogFlops(win->map->n);CHKERRQ(ierr);
    } else {
      try {
        thrust::for_each(
          thrust::make_zip_iterator(
            thrust::make_tuple(
              warray->begin(),
              yarray->begin(),
              thrust::make_constant_iterator(alpha),
              xarray->begin())),
          thrust::make_zip_iterator(
            thrust::make_tuple(
              warray->end(),
              yarray->end(),
              thrust::make_constant_iterator(alpha),
              xarray->end())),
          VecCUSPWAXPY());
      } catch(char *ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
      }
      ierr = PetscLogFlops(2*win->map->n);CHKERRQ(ierr);
    }
    ierr = WaitForGPU();CHKERRCUSP(ierr);
    ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayWrite(win,&warray);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* These functions are for the CUSP implementation of MAXPY with the loop unrolled on the CPU */
struct VecCUSPMAXPY4
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    /*y += a1*x1 +a2*x2 + 13*x3 +a4*x4 */
    thrust::get<0>(t) += thrust::get<1>(t)*thrust::get<2>(t)+thrust::get<3>(t)*thrust::get<4>(t)+thrust::get<5>(t)*thrust::get<6>(t)+thrust::get<7>(t)*thrust::get<8>(t);
  }
};


struct VecCUSPMAXPY3
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    /*y += a1*x1 +a2*x2 + a3*x3 */
    thrust::get<0>(t) += thrust::get<1>(t)*thrust::get<2>(t)+thrust::get<3>(t)*thrust::get<4>(t)+thrust::get<5>(t)*thrust::get<6>(t);
  }
};

struct VecCUSPMAXPY2
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    /*y += a1*x1 +a2*x2*/
    thrust::get<0>(t) += thrust::get<1>(t)*thrust::get<2>(t)+thrust::get<3>(t)*thrust::get<4>(t);
  }
};
PetscErrorCode VecMAXPY_SeqCUSP(Vec xin, PetscInt nv,const PetscScalar *alpha,Vec *y)
{
  PetscErrorCode ierr;
  CUSPARRAY      *xarray,*yy0,*yy1,*yy2,*yy3;
  PetscInt       n = xin->map->n,j,j_rem;
  PetscScalar    alpha0,alpha1,alpha2,alpha3;

  PetscFunctionBegin;
  ierr = PetscLogFlops(nv*2.0*n);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
  switch (j_rem=nv&0x3) {
  case 3:
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha2 = alpha[2];
    alpha += 3;
    ierr   = VecCUSPGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr   = VecCUSPGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr   = VecCUSPGetArrayRead(y[2],&yy2);CHKERRQ(ierr);
    try {
      thrust::for_each(
        thrust::make_zip_iterator(
          thrust::make_tuple(
            xarray->begin(),
            thrust::make_constant_iterator(alpha0),
            yy0->begin(),
            thrust::make_constant_iterator(alpha1),
            yy1->begin(),
            thrust::make_constant_iterator(alpha2),
            yy2->begin())),
        thrust::make_zip_iterator(
          thrust::make_tuple(
            xarray->end(),
            thrust::make_constant_iterator(alpha0),
            yy0->end(),
            thrust::make_constant_iterator(alpha1),
            yy1->end(),
            thrust::make_constant_iterator(alpha2),
            yy2->end())),
        VecCUSPMAXPY3());
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = VecCUSPRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(y[2],&yy2);CHKERRQ(ierr);
    y   += 3;
    break;
  case 2:
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha +=2;
    ierr   = VecCUSPGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr   = VecCUSPGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
    try {
      thrust::for_each(
        thrust::make_zip_iterator(
          thrust::make_tuple(
            xarray->begin(),
            thrust::make_constant_iterator(alpha0),
            yy0->begin(),
            thrust::make_constant_iterator(alpha1),
            yy1->begin())),
        thrust::make_zip_iterator(
          thrust::make_tuple(
            xarray->end(),
            thrust::make_constant_iterator(alpha0),
            yy0->end(),
            thrust::make_constant_iterator(alpha1),
            yy1->end())),
        VecCUSPMAXPY2());
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    y +=2;
    break;
  case 1:
    alpha0 = *alpha++;
    ierr   = VecAXPY_SeqCUSP(xin,alpha0,y[0]);
    y     +=1;
    break;
  }
  for (j=j_rem; j<nv; j+=4) {
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha2 = alpha[2];
    alpha3 = alpha[3];
    alpha += 4;
    ierr   = VecCUSPGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr   = VecCUSPGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr   = VecCUSPGetArrayRead(y[2],&yy2);CHKERRQ(ierr);
    ierr   = VecCUSPGetArrayRead(y[3],&yy3);CHKERRQ(ierr);
    try {
      thrust::for_each(
        thrust::make_zip_iterator(
          thrust::make_tuple(
            xarray->begin(),
            thrust::make_constant_iterator(alpha0),
            yy0->begin(),
            thrust::make_constant_iterator(alpha1),
            yy1->begin(),
            thrust::make_constant_iterator(alpha2),
            yy2->begin(),
            thrust::make_constant_iterator(alpha3),
            yy3->begin())),
        thrust::make_zip_iterator(
          thrust::make_tuple(
            xarray->end(),
            thrust::make_constant_iterator(alpha0),
            yy0->end(),
            thrust::make_constant_iterator(alpha1),
            yy1->end(),
            thrust::make_constant_iterator(alpha2),
            yy2->end(),
            thrust::make_constant_iterator(alpha3),
            yy3->end())),
        VecCUSPMAXPY4());
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = VecCUSPRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(y[2],&yy2);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(y[3],&yy3);CHKERRQ(ierr);
    y   += 4;
  }
  ierr = VecCUSPRestoreArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode VecDot_SeqCUSP(Vec xin,Vec yin,PetscScalar *z)
{
  CUSPARRAY      *xarray,*yarray;
  PetscErrorCode ierr;
  //  PetscScalar    *xptr,*yptr,*zgpu;
  //PetscReal tmp;

  PetscFunctionBegin;
  //VecNorm_SeqCUSP(xin, NORM_2, &tmp);
  //VecNorm_SeqCUSP(yin, NORM_2, &tmp);
  ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  try {
#if defined(PETSC_USE_COMPLEX)
    *z = cusp::blas::dotc(*yarray,*xarray);
#else
    *z = cusp::blas::dot(*yarray,*xarray);
#endif
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  if (xin->map->n >0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

//
// CUDA kernels for MDot to follow
//

// set work group size to be a power of 2 (128 is usually a good compromise between portability and speed)
#define MDOT_WORKGROUP_SIZE 128
#define MDOT_WORKGROUP_NUM  128

// M = 2:
__global__ void VecMDot_SeqCUSP_kernel2(const PetscScalar *x,const PetscScalar *y0,const PetscScalar *y1,
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
__global__ void VecMDot_SeqCUSP_kernel3(const PetscScalar *x,const PetscScalar *y0,const PetscScalar *y1,const PetscScalar *y2,
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
__global__ void VecMDot_SeqCUSP_kernel4(const PetscScalar *x,const PetscScalar *y0,const PetscScalar *y1,const PetscScalar *y2,const PetscScalar *y3,
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
__global__ void VecMDot_SeqCUSP_kernel8(const PetscScalar *x,const PetscScalar *y0,const PetscScalar *y1,const PetscScalar *y2,const PetscScalar *y3,
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


PetscErrorCode VecMDot_SeqCUSP(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode ierr;
  PetscInt       i,j,n = xin->map->n,current_y_index = 0;
  CUSPARRAY      *xarray,*y0array,*y1array,*y2array,*y3array,*y4array,*y5array,*y6array,*y7array;
  PetscScalar    *group_results_gpu,*xptr,*y0ptr,*y1ptr,*y2ptr,*y3ptr,*y4ptr,*y5ptr,*y6ptr,*y7ptr;
  PetscScalar    group_results_cpu[MDOT_WORKGROUP_NUM * 8]; // we process at most eight vectors in one kernel
  cudaError_t    cuda_ierr;

  PetscFunctionBegin;
  if (nv <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Number of vectors provided to VecMDot_SeqCUSP not positive.");
  /* Handle the case of local size zero first */
  if (!xin->map->n) {
    for (i=0; i<nv; ++i) z[i] = 0;
    PetscFunctionReturn(0);
  }

  // allocate scratchpad memory for the results of individual work groups:
  cuda_ierr = cudaMalloc((void**)&group_results_gpu, sizeof(PetscScalar) * MDOT_WORKGROUP_NUM * 8);
  if (cuda_ierr != cudaSuccess) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Could not allocate CUDA work memory. Error code: %d", (int)cuda_ierr);

  ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  xptr = thrust::raw_pointer_cast(xarray->data());

  while (current_y_index < nv)
  {
    switch (nv - current_y_index) {

    case 7:
    case 6:
    case 5:
    case 4:
      ierr = VecCUSPGetArrayRead(yin[current_y_index  ],&y0array);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(yin[current_y_index+1],&y1array);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(yin[current_y_index+2],&y2array);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(yin[current_y_index+3],&y3array);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
      z[current_y_index]   = cusp::blas::dot(*y0array,*xarray);
      z[current_y_index+1] = cusp::blas::dot(*y1array,*xarray);
      z[current_y_index+2] = cusp::blas::dot(*y2array,*xarray);
      z[current_y_index+3] = cusp::blas::dot(*y3array,*xarray);
#else
      // extract raw device pointers:
      y0ptr = thrust::raw_pointer_cast(y0array->data());
      y1ptr = thrust::raw_pointer_cast(y1array->data());
      y2ptr = thrust::raw_pointer_cast(y2array->data());
      y3ptr = thrust::raw_pointer_cast(y3array->data());

      // run kernel:
      VecMDot_SeqCUSP_kernel4<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,y2ptr,y3ptr,n,group_results_gpu);

      // copy results back to
      cuda_ierr = cudaMemcpy(group_results_cpu,group_results_gpu,sizeof(PetscScalar) * MDOT_WORKGROUP_NUM * 4,cudaMemcpyDeviceToHost);
      if (cuda_ierr != cudaSuccess) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Could not copy CUDA buffer to host. Error code: %d", (int)cuda_ierr);

      // sum group results into z:
      for (j=0; j<4; ++j) {
        z[current_y_index + j] = 0;
        for (i=j*MDOT_WORKGROUP_NUM; i<(j+1)*MDOT_WORKGROUP_NUM; ++i) z[current_y_index + j] += group_results_cpu[i];
      }
#endif
      ierr = VecCUSPRestoreArrayRead(yin[current_y_index  ],&y0array);CHKERRQ(ierr);
      ierr = VecCUSPRestoreArrayRead(yin[current_y_index+1],&y1array);CHKERRQ(ierr);
      ierr = VecCUSPRestoreArrayRead(yin[current_y_index+2],&y2array);CHKERRQ(ierr);
      ierr = VecCUSPRestoreArrayRead(yin[current_y_index+3],&y3array);CHKERRQ(ierr);
      current_y_index += 4;
      break;

    case 3:
      ierr = VecCUSPGetArrayRead(yin[current_y_index  ],&y0array);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(yin[current_y_index+1],&y1array);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(yin[current_y_index+2],&y2array);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
      z[current_y_index]   = cusp::blas::dot(*y0array,*xarray);
      z[current_y_index+1] = cusp::blas::dot(*y1array,*xarray);
      z[current_y_index+2] = cusp::blas::dot(*y2array,*xarray);
#else
      // extract raw device pointers:
      y0ptr = thrust::raw_pointer_cast(y0array->data());
      y1ptr = thrust::raw_pointer_cast(y1array->data());
      y2ptr = thrust::raw_pointer_cast(y2array->data());

      // run kernel:
      VecMDot_SeqCUSP_kernel3<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,y2ptr,n,group_results_gpu);

      // copy results back to
      cuda_ierr = cudaMemcpy(group_results_cpu,group_results_gpu,sizeof(PetscScalar) * MDOT_WORKGROUP_NUM * 3,cudaMemcpyDeviceToHost);
      if (cuda_ierr != cudaSuccess) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Could not copy CUDA buffer to host. Error code: %d", (int)cuda_ierr);

      // sum group results into z:
      for (j=0; j<3; ++j) {
        z[current_y_index + j] = 0;
        for (i=j*MDOT_WORKGROUP_NUM; i<(j+1)*MDOT_WORKGROUP_NUM; ++i) z[current_y_index + j] += group_results_cpu[i];
      }
#endif

      ierr = VecCUSPRestoreArrayRead(yin[current_y_index  ],&y0array);CHKERRQ(ierr);
      ierr = VecCUSPRestoreArrayRead(yin[current_y_index+1],&y1array);CHKERRQ(ierr);
      ierr = VecCUSPRestoreArrayRead(yin[current_y_index+2],&y2array);CHKERRQ(ierr);
      current_y_index += 3;
      break;

    case 2:
      ierr = VecCUSPGetArrayRead(yin[current_y_index],&y0array);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(yin[current_y_index+1],&y1array);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
      z[current_y_index]   = cusp::blas::dot(*y0array,*xarray);
      z[current_y_index+1] = cusp::blas::dot(*y1array,*xarray);
#else
      // extract raw device pointers:
      y0ptr = thrust::raw_pointer_cast(y0array->data());
      y1ptr = thrust::raw_pointer_cast(y1array->data());

      // run kernel:
      VecMDot_SeqCUSP_kernel2<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,n,group_results_gpu);

      // copy results back to
      cuda_ierr = cudaMemcpy(group_results_cpu,group_results_gpu,sizeof(PetscScalar) * MDOT_WORKGROUP_NUM * 2,cudaMemcpyDeviceToHost);
      if (cuda_ierr != cudaSuccess) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Could not copy CUDA buffer to host. Error code: %d", (int)cuda_ierr);

      // sum group results into z:
      for (j=0; j<2; ++j) {
        z[current_y_index + j] = 0;
        for (i=j*MDOT_WORKGROUP_NUM; i<(j+1)*MDOT_WORKGROUP_NUM; ++i) z[current_y_index + j] += group_results_cpu[i];
      }
#endif
      ierr = VecCUSPRestoreArrayRead(yin[current_y_index],&y0array);CHKERRQ(ierr);
      ierr = VecCUSPRestoreArrayRead(yin[current_y_index+1],&y1array);CHKERRQ(ierr);
      current_y_index += 2;
      break;

    case 1:
      ierr = VecCUSPGetArrayRead(yin[current_y_index],&y0array);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
      z[current_y_index] = cusp::blas::dotc(*y0array, *xarray);
#else
      z[current_y_index] = cusp::blas::dot(*xarray, *y0array);
#endif
      ierr = VecCUSPRestoreArrayRead(yin[current_y_index],&y0array);CHKERRQ(ierr);
      current_y_index += 1;
      break;

    default: // 8 or more vectors left
      ierr = VecCUSPGetArrayRead(yin[current_y_index  ],&y0array);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(yin[current_y_index+1],&y1array);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(yin[current_y_index+2],&y2array);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(yin[current_y_index+3],&y3array);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(yin[current_y_index+4],&y4array);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(yin[current_y_index+5],&y5array);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(yin[current_y_index+6],&y6array);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayRead(yin[current_y_index+7],&y7array);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
      z[current_y_index]   = cusp::blas::dot(*y0array,*xarray);
      z[current_y_index+1] = cusp::blas::dot(*y1array,*xarray);
      z[current_y_index+2] = cusp::blas::dot(*y2array,*xarray);
      z[current_y_index+3] = cusp::blas::dot(*y3array,*xarray);
      z[current_y_index+4] = cusp::blas::dot(*y4array,*xarray);
      z[current_y_index+5] = cusp::blas::dot(*y5array,*xarray);
      z[current_y_index+6] = cusp::blas::dot(*y6array,*xarray);
      z[current_y_index+7] = cusp::blas::dot(*y7array,*xarray);
#else
      // extract raw device pointers:
      y0ptr = thrust::raw_pointer_cast(y0array->data());
      y1ptr = thrust::raw_pointer_cast(y1array->data());
      y2ptr = thrust::raw_pointer_cast(y2array->data());
      y3ptr = thrust::raw_pointer_cast(y3array->data());
      y4ptr = thrust::raw_pointer_cast(y4array->data());
      y5ptr = thrust::raw_pointer_cast(y5array->data());
      y6ptr = thrust::raw_pointer_cast(y6array->data());
      y7ptr = thrust::raw_pointer_cast(y7array->data());

      // run kernel:
      VecMDot_SeqCUSP_kernel8<<<MDOT_WORKGROUP_NUM,MDOT_WORKGROUP_SIZE>>>(xptr,y0ptr,y1ptr,y2ptr,y3ptr,y4ptr,y5ptr,y6ptr,y7ptr,n,group_results_gpu);

      // copy results back to
      cuda_ierr = cudaMemcpy(group_results_cpu,group_results_gpu,sizeof(PetscScalar) * MDOT_WORKGROUP_NUM * 8,cudaMemcpyDeviceToHost);
      if (cuda_ierr != cudaSuccess) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Could not copy CUDA buffer to host. Error code: %d", (int)cuda_ierr);

      // sum group results into z:
      for (j=0; j<8; ++j) {
        z[current_y_index + j] = 0;
        for (i=j*MDOT_WORKGROUP_NUM; i<(j+1)*MDOT_WORKGROUP_NUM; ++i) z[current_y_index + j] += group_results_cpu[i];
      }
#endif
      ierr = VecCUSPRestoreArrayRead(yin[current_y_index  ],&y0array);CHKERRQ(ierr);
      ierr = VecCUSPRestoreArrayRead(yin[current_y_index+1],&y1array);CHKERRQ(ierr);
      ierr = VecCUSPRestoreArrayRead(yin[current_y_index+2],&y2array);CHKERRQ(ierr);
      ierr = VecCUSPRestoreArrayRead(yin[current_y_index+3],&y3array);CHKERRQ(ierr);
      ierr = VecCUSPRestoreArrayRead(yin[current_y_index+4],&y4array);CHKERRQ(ierr);
      ierr = VecCUSPRestoreArrayRead(yin[current_y_index+5],&y5array);CHKERRQ(ierr);
      ierr = VecCUSPRestoreArrayRead(yin[current_y_index+6],&y6array);CHKERRQ(ierr);
      ierr = VecCUSPRestoreArrayRead(yin[current_y_index+7],&y7array);CHKERRQ(ierr);
      current_y_index += 8;
      break;
    }
  }
  ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);

  cuda_ierr = cudaFree(group_results_gpu);
  if (cuda_ierr != cudaSuccess) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Could not copy CUDA buffer to host: %d", (int)cuda_ierr);
  ierr = PetscLogFlops(PetscMax(nv*(2.0*n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef MDOT_WORKGROUP_SIZE
#undef MDOT_WORKGROUP_NUM



PetscErrorCode VecSet_SeqCUSP(Vec xin,PetscScalar alpha)
{
  CUSPARRAY      *xarray=NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* if there's a faster way to do the case alpha=0.0 on the GPU we should do that*/
  ierr = VecCUSPGetArrayWrite(xin,&xarray);CHKERRQ(ierr);
  try {
    cusp::blas::fill(*xarray,alpha);
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = VecCUSPRestoreArrayWrite(xin,&xarray);
  PetscFunctionReturn(0);
}

PetscErrorCode VecScale_SeqCUSP(Vec xin, PetscScalar alpha)
{
  CUSPARRAY      *xarray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == 0.0) {
    ierr = VecSet_SeqCUSP(xin,alpha);CHKERRQ(ierr);
  } else if (alpha != 1.0) {
    ierr = VecCUSPGetArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
    try {
      cusp::blas::scal(*xarray,alpha);
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = VecCUSPRestoreArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
  }
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_SeqCUSP(Vec xin,Vec yin,PetscScalar *z)
{
  CUSPARRAY      *xarray,*yarray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  //#if defined(PETSC_USE_COMPLEX)
  /*Not working for complex*/
  //#else
  ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  try {
    *z = cusp::blas::dot(*xarray,*yarray);
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  //#endif
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  if (xin->map->n > 0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  ierr = VecCUSPRestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecCopy_SeqCUSP(Vec xin,Vec yin)
{
  CUSPARRAY      *xarray,*yarray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (xin != yin) {
    if (xin->valid_GPU_array == PETSC_CUSP_GPU) {
      ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
      ierr = VecCUSPGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
      try {
        cusp::blas::copy(*xarray,*yarray);
      } catch(char *ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
      }
      ierr = WaitForGPU();CHKERRCUSP(ierr);
      ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
      ierr = VecCUSPRestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);

    } else if (xin->valid_GPU_array == PETSC_CUSP_CPU) {
      /* copy in CPU if we are on the CPU*/
      ierr = VecCopy_SeqCUSP_Private(xin,yin);CHKERRQ(ierr);
    } else if (xin->valid_GPU_array == PETSC_CUSP_BOTH) {
      /* if xin is valid in both places, see where yin is and copy there (because it's probably where we'll want to next use it) */
      if (yin->valid_GPU_array == PETSC_CUSP_CPU) {
        /* copy in CPU */
        ierr = VecCopy_SeqCUSP_Private(xin,yin);CHKERRQ(ierr);

      } else if (yin->valid_GPU_array == PETSC_CUSP_GPU) {
        /* copy in GPU */
        ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUSPGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
        try {
          cusp::blas::copy(*xarray,*yarray);
          ierr = WaitForGPU();CHKERRCUSP(ierr);
        } catch(char *ex) {
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
        }
        ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUSPRestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
      } else if (yin->valid_GPU_array == PETSC_CUSP_BOTH) {
        /* xin and yin are both valid in both places (or yin was unallocated before the earlier call to allocatecheck
           default to copy in GPU (this is an arbitrary choice) */
        ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUSPGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
        try {
          cusp::blas::copy(*xarray,*yarray);
          ierr = WaitForGPU();CHKERRCUSP(ierr);
        } catch(char *ex) {
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
        }
        ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUSPRestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
      } else {
        ierr = VecCopy_SeqCUSP_Private(xin,yin);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}


PetscErrorCode VecSwap_SeqCUSP(Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscBLASInt   one = 1,bn;
  CUSPARRAY      *xarray,*yarray;
  cublasStatus_t cberr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(xin->map->n,&bn);CHKERRQ(ierr);
  if (xin != yin) {
    ierr = VecCUSPGetArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayReadWrite(yin,&yarray);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
    cberr = cublasCswap(cublasv2handle,bn,(cuFloatComplex*)VecCUSPCastToRawPtr(*xarray),one,(cuFloatComplex*)VecCUSPCastToRawPtr(*yarray),one);CHKERRCUBLAS(cberr);
#else
    cberr = cublasZswap(cublasv2handle,bn,(cuDoubleComplex*)VecCUSPCastToRawPtr(*xarray),one,(cuDoubleComplex*)VecCUSPCastToRawPtr(*yarray),one);CHKERRCUBLAS(cberr);
#endif
#else
#if defined(PETSC_USE_REAL_SINGLE)
    cberr = cublasSswap(cublasv2handle,bn,VecCUSPCastToRawPtr(*xarray),one,VecCUSPCastToRawPtr(*yarray),one);CHKERRCUBLAS(cberr);
#else
    cberr = cublasDswap(cublasv2handle,bn,VecCUSPCastToRawPtr(*xarray),one,VecCUSPCastToRawPtr(*yarray),one);CHKERRCUBLAS(cberr);
#endif
#endif
    ierr = WaitForGPU();CHKERRCUSP(ierr);
    ierr = VecCUSPRestoreArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

struct VecCUSPAX
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = thrust::get<1>(t)*thrust::get<2>(t);
  }
};

PetscErrorCode VecAXPBY_SeqCUSP(Vec yin,PetscScalar alpha,PetscScalar beta,Vec xin)
{
  PetscErrorCode ierr;
  PetscScalar    a = alpha,b = beta;
  CUSPARRAY      *xarray,*yarray;

  PetscFunctionBegin;
  if (a == 0.0) {
    ierr = VecScale_SeqCUSP(yin,beta);CHKERRQ(ierr);
  } else if (b == 1.0) {
    ierr = VecAXPY_SeqCUSP(yin,alpha,xin);CHKERRQ(ierr);
  } else if (a == 1.0) {
    ierr = VecAYPX_SeqCUSP(yin,beta,xin);CHKERRQ(ierr);
  } else if (b == 0.0) {
    ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    try {
      thrust::for_each(
        thrust::make_zip_iterator(
          thrust::make_tuple(
            yarray->begin(),
            thrust::make_constant_iterator(a),
            xarray->begin())),
        thrust::make_zip_iterator(
          thrust::make_tuple(
            yarray->end(),
            thrust::make_constant_iterator(a),
            xarray->end())),
        VecCUSPAX());
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRCUSP(ierr);
    ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
  } else {
    ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    try {
      cusp::blas::axpby(*xarray,*yarray,*yarray,a,b);
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRCUSP(ierr);
    ierr = PetscLogFlops(3.0*xin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* structs below are for special cases of VecAXPBYPCZ_SeqCUSP */
struct VecCUSPXPBYPCZ
{
  /* z = x + b*y + c*z */
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = thrust::get<1>(t)*thrust::get<0>(t)+thrust::get<2>(t)+thrust::get<4>(t)*thrust::get<3>(t);
  }
};

struct VecCUSPAXPBYPZ
{
  /* z = ax + b*y + z */
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) += thrust::get<2>(t)*thrust::get<1>(t)+thrust::get<4>(t)*thrust::get<3>(t);
  }
};

PetscErrorCode VecAXPBYPCZ_SeqCUSP(Vec zin,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = zin->map->n;
  CUSPARRAY      *xarray,*yarray,*zarray;

  PetscFunctionBegin;
  ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayReadWrite(zin,&zarray);CHKERRQ(ierr);
  if (alpha == 1.0) {
    try {
      thrust::for_each(
        thrust::make_zip_iterator(
          thrust::make_tuple(
            zarray->begin(),
            thrust::make_constant_iterator(gamma),
            xarray->begin(),
            yarray->begin(),
            thrust::make_constant_iterator(beta))),
        thrust::make_zip_iterator(
          thrust::make_tuple(
            zarray->end(),
            thrust::make_constant_iterator(gamma),
            xarray->end(),
            yarray->end(),
            thrust::make_constant_iterator(beta))),
        VecCUSPXPBYPCZ());
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = PetscLogFlops(4.0*n);CHKERRQ(ierr);
  } else if (gamma == 1.0) {
    try {
      thrust::for_each(
        thrust::make_zip_iterator(
          thrust::make_tuple(
            zarray->begin(),
            xarray->begin(),
            thrust::make_constant_iterator(alpha),
            yarray->begin(),
            thrust::make_constant_iterator(beta))),
        thrust::make_zip_iterator(
          thrust::make_tuple(
            zarray->end(),
            xarray->end(),
            thrust::make_constant_iterator(alpha),
            yarray->end(),
            thrust::make_constant_iterator(beta))),
        VecCUSPAXPBYPZ());
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = PetscLogFlops(4.0*n);CHKERRQ(ierr);
  } else {
    try {
      cusp::blas::axpbypcz(*xarray,*yarray,*zarray,*zarray,alpha,beta,gamma);
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = VecCUSPRestoreArrayReadWrite(zin,&zarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogFlops(5.0*n);CHKERRQ(ierr);
  }
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecPointwiseMult_SeqCUSP(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = win->map->n;
  CUSPARRAY      *xarray,*yarray,*warray;

  PetscFunctionBegin;
  ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayReadWrite(win,&warray);CHKERRQ(ierr);
  try {
    cusp::blas::xmy(*xarray,*yarray,*warray);
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayReadWrite(win,&warray);CHKERRQ(ierr);
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  PetscFunctionReturn(0);
}


/* should do infinity norm in cusp */

PetscErrorCode VecNorm_SeqCUSP(Vec xin,NormType type,PetscReal *z)
{
  const PetscScalar *xx;
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n;
  PetscBLASInt      one = 1, bn;
  CUSPARRAY         *xarray;
  cublasStatus_t    cberr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    try {
      *z = cusp::blas::nrm2(*xarray);
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = WaitForGPU();CHKERRCUSP(ierr);
    ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = PetscLogFlops(PetscMax(2.0*n-1,0.0));CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    PetscInt  i;
    PetscReal max = 0.0,tmp;

    ierr = VecGetArrayRead(xin,&xx);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      if ((tmp = PetscAbsScalar(*xx)) > max) max = tmp;
      /* check special case of tmp == NaN */
      if (tmp != tmp) {max = tmp; break;}
      xx++;
    }
    ierr = VecRestoreArrayRead(xin,&xx);CHKERRQ(ierr);
    *z   = max;
  } else if (type == NORM_1) {
    ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
    cberr = cublasSasum(cublasv2handle,bn,(cuFloatComplex*)VecCUSPCastToRawPtr(*xarray),one,z);CHKERRCUBLAS(cberr);
#else
    cberr = cublasDasum(cublasv2handle,bn,(cuDoubleComplex*)VecCUSPCastToRawPtr(*xarray),one,z);CHKERRCUBLAS(cberr);
#endif
#else
#if defined(PETSC_USE_REAL_SINGLE)
    cberr = cublasSasum(cublasv2handle,bn,VecCUSPCastToRawPtr(*xarray),one,z);CHKERRCUBLAS(cberr);
#else
    cberr = cublasDasum(cublasv2handle,bn,VecCUSPCastToRawPtr(*xarray),one,z);CHKERRCUBLAS(cberr);
#endif
#endif
    ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRCUSP(ierr);
    ierr = PetscLogFlops(PetscMax(n-1.0,0.0));CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    ierr = VecNorm_SeqCUSP(xin,NORM_1,z);CHKERRQ(ierr);
    ierr = VecNorm_SeqCUSP(xin,NORM_2,z+1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*The following template functions are for VecDotNorm2_SeqCUSP.  Note that there is no complex support as currently written*/
template <typename T>
struct cuspdotnormcalculate : thrust::unary_function<T,T>
{
  __host__ __device__
  T operator()(T x)
  {
#if defined(PETSC_USE_COMPLEX)
    //return thrust::make_tuple(thrust::get<0>(x)*thrust::get<1>(x), thrust::get<1>(x)*thrust::get<1>(x));
#else
    return thrust::make_tuple(thrust::get<0>(x)*thrust::get<1>(x), thrust::get<1>(x)*thrust::get<1>(x));
#endif
  }
};

template <typename T>
struct cuspdotnormreduce : thrust::binary_function<T,T,T>
{
  __host__ __device__
  T operator()(T x,T y)
  {
    return thrust::make_tuple(thrust::get<0>(x)+thrust::get<0>(y), thrust::get<1>(x)+thrust::get<1>(y));
  }
};

PetscErrorCode VecDotNorm2_SeqCUSP(Vec s, Vec t, PetscScalar *dp, PetscScalar *nm)
{
  PetscErrorCode                         ierr;
  PetscScalar                            zero = 0.0;
  PetscReal                              n=s->map->n;
  thrust::tuple<PetscScalar,PetscScalar> result;
  CUSPARRAY                              *sarray,*tarray;

  PetscFunctionBegin;
  /*ierr = VecCUSPCopyToGPU(s);CHKERRQ(ierr);
   ierr = VecCUSPCopyToGPU(t);CHKERRQ(ierr);*/
  ierr = VecCUSPGetArrayRead(s,&sarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(t,&tarray);CHKERRQ(ierr);
  try {
#if defined(PETSC_USE_COMPLEX)
    ierr = VecDot_SeqCUSP(s,t,dp);CHKERRQ(ierr);
    ierr = VecDot_SeqCUSP(t,t,nm);CHKERRQ(ierr);
    //printf("VecDotNorm2_SeqCUSP=%1.5g,%1.5g\n",PetscRealPart(*dp),PetscImaginaryPart(*dp));
    //printf("VecDotNorm2_SeqCUSP=%1.5g,%1.5g\n",PetscRealPart(*nm),PetscImaginaryPart(*nm));
#else
    result = thrust::transform_reduce(
              thrust::make_zip_iterator(
                thrust::make_tuple(
                  sarray->begin(),
                  tarray->begin())),
              thrust::make_zip_iterator(
                thrust::make_tuple(
                  sarray->end(),
                  tarray->end())),
              cuspdotnormcalculate<thrust::tuple<PetscScalar,PetscScalar> >(),
              thrust::make_tuple(zero,zero),                                   /*init */
              cuspdotnormreduce<thrust::tuple<PetscScalar, PetscScalar> >());  /* binary function */
    *dp = thrust::get<0>(result);
    *nm = thrust::get<1>(result);
#endif
  } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  ierr = VecCUSPRestoreArrayRead(s,&sarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayRead(t,&tarray);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = PetscLogFlops(4.0*n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecDestroy_SeqCUSP(Vec v)
{
  PetscErrorCode ierr;
  cudaError_t    err;

  PetscFunctionBegin;
  try {
    if (v->spptr) {
      delete ((Vec_CUSP*)v->spptr)->GPUarray;
      err = cudaStreamDestroy(((Vec_CUSP*)v->spptr)->stream);CHKERRCUSP(err);
      delete (Vec_CUSP*)v->spptr;
    }
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  ierr = VecDestroy_SeqCUSP_Private(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#if defined(PETSC_USE_COMPLEX)
struct conjugate
{
  __host__ __device__
  PetscScalar operator()(PetscScalar x)
  {
    return cusp::conj(x);
  }
};
#endif


PetscErrorCode VecConjugate_SeqCUSP(Vec xin)
{
  PetscErrorCode ierr;
  CUSPARRAY      *xarray;

  PetscFunctionBegin;
  ierr = VecCUSPGetArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  thrust::transform(xarray->begin(), xarray->end(), xarray->begin(), conjugate());
#endif
  ierr = VecCUSPRestoreArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetLocalVector_SeqCUSP(Vec v,Vec w)
{
  PetscErrorCode ierr;
  cudaError_t    err;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(w,VEC_CLASSID,2);
  PetscCheckTypeName(w,VECSEQCUSP);

  if (w->data) {
    if (((Vec_Seq*)w->data)->array_allocated) PetscFree(((Vec_Seq*)w->data)->array_allocated);
    ((Vec_Seq*)w->data)->array = 0;
    ((Vec_Seq*)w->data)->array_allocated = 0;
    ((Vec_Seq*)w->data)->unplacedarray = 0;
  }
  if (w->spptr) {
    if (((Vec_CUSP*)w->spptr)->GPUarray) delete ((Vec_CUSP*)w->spptr)->GPUarray;
    err = cudaStreamDestroy(((Vec_CUSP*)w->spptr)->stream);CHKERRCUSP(err);
    delete (Vec_CUSP*)w->spptr;
    w->spptr = 0;
  }

  if (v->petscnative) {
    ierr = PetscFree(w->data);CHKERRQ(ierr);
    w->data = v->data;
    w->valid_GPU_array = v->valid_GPU_array;
    w->spptr = v->spptr;
    ierr = PetscObjectStateIncrease((PetscObject)w);CHKERRQ(ierr);
  } else {
    ierr = VecGetArray(v,&((Vec_Seq*)w->data)->array);CHKERRQ(ierr);
    w->valid_GPU_array = PETSC_CUSP_CPU;
    ierr = VecCUSPAllocateCheck(w);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreLocalVector_SeqCUSP(Vec v,Vec w)
{
  PetscErrorCode ierr;
  cudaError_t    err;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(w,VEC_CLASSID,2);
  PetscCheckTypeName(w,VECSEQCUSP);

  if (v->petscnative) {
    v->data = w->data;
    v->valid_GPU_array = w->valid_GPU_array;
    v->spptr = w->spptr;
    ierr = VecCUSPCopyFromGPU(v);CHKERRQ(ierr);
    ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
    w->data = 0;
    w->valid_GPU_array = PETSC_CUSP_UNALLOCATED;
    w->spptr = 0;
  } else {
    ierr = VecRestoreArray(v,&((Vec_Seq*)w->data)->array);CHKERRQ(ierr);
    if ((Vec_CUSP*)w->spptr) {
      delete ((Vec_CUSP*)w->spptr)->GPUarray;
      err = cudaStreamDestroy(((Vec_CUSP*)w->spptr)->stream);CHKERRCUSP(err);
      delete (Vec_CUSP*)w->spptr;
    }
  }
  PetscFunctionReturn(0);
}

/*@C
   VecCUSPGetArrayReadWrite - Provides access to the CUSP vector inside a vector.

   This function has semantics similar to VecGetArray():  the CUSP
   vector returned by this function points to a consistent view of the
   vector data.  This may involve a copy operation of data from the host
   to the device if the data on the device is out of date.  If the
   device memory hasn't been allocated previously it will be allocated
   as part of this function call.  VecCUSPGetArrayReadWrite() assumes
   that the user will modify the vector data.  This is similar to
   intent(inout) in fortran.

   The CUSP device vector has to be released by calling
   VecCUSPRestoreArrayReadWrite().  Upon restoring the vector data the
   data on the host will be marked as out of date.  A subsequent access
   of the host data will thus incur a data transfer from the device to
   the host.


   Input Parameter:
.  v - the vector

   Output Parameter:
.  a - the CUSP device vector
   
   Fortran note: This function is not currently available from Fortran.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecCUSPRestoreArrayReadWrite(), VecCUSPGetArrayRead(), VecCUSPGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUSPGetArrayReadWrite(Vec v, CUSPARRAY **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUSP,VECMPICUSP);
  *a   = 0;
  ierr = VecCUSPCopyToGPU(v);CHKERRQ(ierr);
  *a   = ((Vec_CUSP*)v->spptr)->GPUarray;
  PetscFunctionReturn(0);
}

/*@C
   VecCUSPRestoreArrayReadWrite - Restore a CUSP device vector previously acquired with VecCUSPGetArrayReadWrite().

   This marks the host data as out of date.  Subsequent access to the
   vector data on the host side with for instance VecGetArray() incurs a
   data transfer.

   Input Parameter:
+  v - the vector
-  a - the CUSP device vector.  This pointer is invalid after
       VecCUSPRestoreArrayReadWrite() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecCUSPGetCUDAArrayRead(), VecCUSPGetCUDAArrayWrite(), VecCUSPGetArrayReadWrite(), VecCUSPGetArrayRead(), VecCUSPGetArrayWrite(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUSPRestoreArrayReadWrite(Vec v, CUSPARRAY **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUSP,VECMPICUSP);
  v->valid_GPU_array = PETSC_CUSP_GPU;

  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCUSPGetArrayRead - Provides read access to the CUSP device vector inside a vector.

   This function is analogous to VecGetArrayRead():  The CUSP vector
   returned by this function points to a consistent view of the vector
   data.  This may involve a copy operation of data from the host to the
   device if the data on the device is out of date.  If the device
   memory hasn't been allocated previously it will be allocated as part
   of this function call.  VecCUSPGetArrayRead() assumes that the user
   will not modify the vector data.  This is analogous to intent(in) in
   Fortran.

   The CUSP device vector has to be released by calling
   VecCUSPRestoreArrayRead().  If the data on the host side was
   previously up to date it will remain so, i.e. data on both the device
   and the host is up to date.  Accessing data on the host side does not
   incur a device to host data transfer.

   Input Parameter:
.  v - the vector

   Output Parameter:
.  a - the CUSP device vector

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecCUSPRestoreArrayRead(), VecCUSPGetArrayReadWrite(), VecCUSPGetArrayWrite(), VecCUSPGetArrayReadWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUSPGetArrayRead(Vec v, CUSPARRAY **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUSP,VECMPICUSP);
  *a   = 0;
  ierr = VecCUSPCopyToGPU(v);CHKERRQ(ierr);
  *a   = ((Vec_CUSP*)v->spptr)->GPUarray;
  PetscFunctionReturn(0);
}

/*@C
   VecCUSPRestoreArrayRead - Restore a CUSP device vector previously acquired with VecCUSPGetArrayRead().

   If the data on the host side was previously up to date it will remain
   so, i.e. data on both the device and the host is up to date.
   Accessing data on the host side e.g. with VecGetArray() does not
   incur a device to host data transfer.

   Input Parameter:
+  v - the vector
-  a - the CUSP device vector.  This pointer is invalid after
       VecCUSPRestoreArrayRead() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecCUSPGetArrayRead(), VecCUSPGetArrayWrite(), VecCUSPGetArrayReadWrite(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUSPRestoreArrayRead(Vec v, CUSPARRAY **a)
{
  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUSP,VECMPICUSP);
  PetscFunctionReturn(0);
}

/*@C
   VecCUSPGetArrayWrite - Provides write access to the CUSP device vector inside a vector.

   The data pointed to by the device vector is uninitialized.  The user
   must not read this data.  Furthermore, the entire array needs to be
   filled by the user to obtain well-defined behaviour.  The device
   memory will be allocated by this function if it hasn't been allocated
   previously.  This is analogous to intent(out) in Fortran.

   The CUSP device vector needs to be released with
   VecCUSPRestoreArrayWrite().  When the pointer is released the host
   data of the vector is marked as out of data.  Subsequent access of
   the host data with e.g. VecGetArray() incurs a device to host data
   transfer.


   Input Parameter:
.  v - the vector

   Output Parameter:
.  a - the CUDA pointer

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecCUSPRestoreArrayWrite(), VecCUSPGetArrayReadWrite(), VecCUSPGetArrayRead(), VecCUSPGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUSPGetArrayWrite(Vec v, CUSPARRAY **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUSP,VECMPICUSP);
  *a   = 0;
  ierr = VecCUSPAllocateCheck(v);CHKERRQ(ierr);
  *a   = ((Vec_CUSP*)v->spptr)->GPUarray;
  PetscFunctionReturn(0);
}

/*@C
   VecCUSPRestoreArrayWrite - Restore a CUSP device vector previously acquired with VecCUSPGetArrayWrite().

   Data on the host will be marked as out of date.  Subsequent access of
   the data on the host side e.g. with VecGetArray() will incur a device
   to host data transfer.

   Input Parameter:
+  v - the vector
-  a - the CUDA device pointer.  This pointer is invalid after
       VecCUSPRestoreArrayWrite() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: intermediate

.seealso: VecCUSPGetArrayWrite(), VecCUSPGetArrayReadWrite(), VecCUSPGetArrayRead(), VecCUSPGetArrayWrite(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUSPRestoreArrayWrite(Vec v, CUSPARRAY **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUSP,VECMPICUSP);
  v->valid_GPU_array = PETSC_CUSP_GPU;

  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCUSPGetCUDAArrayReadWrite - Provides access to the CUDA buffer inside a vector.

   This function has semantics similar to VecGetArray():  the pointer
   returned by this function points to a consistent view of the vector
   data.  This may involve a copy operation of data from the host to the
   device if the data on the device is out of date.  If the device
   memory hasn't been allocated previously it will be allocated as part
   of this function call.  VecCUSPGetCUDAArrayReadWrite() assumes that
   the user will modify the vector data.  This is similar to
   intent(inout) in fortran.

   The CUDA device pointer has to be released by calling
   VecCUSPRestoreCUDAArrayReadWrite().  Upon restoring the vector data
   the data on the host will be marked as out of date.  A subsequent
   access of the host data will thus incur a data transfer from the
   device to the host.


   Input Parameter:
.  v - the vector

   Output Parameter:
.  a - the CUDA device pointer

   Fortran note:
   This function is not currently available from Fortran.

   Level: advanced

.seealso: VecCUSPRestoreCUDAArrayReadWrite(), VecCUSPGetCUDAArrayRead(), VecCUSPGetCUDAArrayWrite(), VecCUSPGetArrayReadWrite(), VecCUSPGetArrayRead(), VecCUSPGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUSPGetCUDAArrayReadWrite(Vec v, PetscScalar **a)
{
  PetscErrorCode ierr;
  CUSPARRAY      *cusparray;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUSP,VECMPICUSP);
  PetscValidPointer(a,1);
  ierr = VecCUSPGetArrayReadWrite(v, &cusparray);CHKERRQ(ierr);
  *a   = thrust::raw_pointer_cast(cusparray->data());
  PetscFunctionReturn(0);
}

/*@C
   VecCUSPRestoreCUDAArrayReadWrite - Restore a device vector previously acquired with VecCUSPGetCUDAArrayReadWrite().

   This marks the host data as out of date.  Subsequent access to the
   vector data on the host side with for instance VecGetArray() incurs a
   data transfer.

   Input Parameter:
+  v - the vector
-  a - the CUDA device pointer.  This pointer is invalid after
       VecCUSPRestoreCUDAArrayReadWrite() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: advanced

.seealso: VecCUSPGetCUDAArrayRead(), VecCUSPGetCUDAArrayWrite(), VecCUSPGetArrayReadWrite(), VecCUSPGetArrayRead(), VecCUSPGetArrayWrite(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUSPRestoreCUDAArrayReadWrite(Vec v, PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUSP,VECMPICUSP);
  v->valid_GPU_array = PETSC_CUSP_GPU;
  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCUSPGetCUDAArrayRead - Provides read access to the CUDA buffer inside a vector.

   This function is analogous to VecGetArrayRead():  The pointer
   returned by this function points to a consistent view of the vector
   data.  This may involve a copy operation of data from the host to the
   device if the data on the device is out of date.  If the device
   memory hasn't been allocated previously it will be allocated as part
   of this function call.  VecCUSPGetCUDAArrayRead() assumes that the
   user will not modify the vector data.  This is analgogous to
   intent(in) in Fortran.

   The CUDA device pointer has to be released by calling
   VecCUSPRestoreCUDAArrayRead().  If the data on the host side was
   previously up to date it will remain so, i.e. data on both the device
   and the host is up to date.  Accessing data on the host side does not
   incur a device to host data transfer.

   Input Parameter:
.  v - the vector

   Output Parameter:
.  a - the CUDA pointer.

   Fortran note:
   This function is not currently available from Fortran.

   Level: advanced

.seealso: VecCUSPRestoreCUDAArrayRead(), VecCUSPGetCUDAArrayReadWrite(), VecCUSPGetCUDAArrayWrite(), VecCUSPGetArrayReadWrite(), VecCUSPGetArrayRead(), VecCUSPGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUSPGetCUDAArrayRead(Vec v, PetscScalar **a)
{
  PetscErrorCode ierr;
  CUSPARRAY      *cusparray;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUSP,VECMPICUSP);
  PetscValidPointer(a,1);
  ierr = VecCUSPGetArrayRead(v, &cusparray);CHKERRQ(ierr);
  *a   = thrust::raw_pointer_cast(cusparray->data());
  PetscFunctionReturn(0);
}

/*@C
   VecCUSPRestoreCUDAArrayRead - Restore a device vector previously acquired with VecCUSPGetCUDAArrayRead()

   If the data on the host side was previously up to date it will remain
   so, i.e. data on both the device and the host is up to date.
   Accessing data on the host side e.g. with VecGetArray() does not
   incur a device to host data transfer.

   Input Parameter:
+  v - the vector

-  a - the CUDA device pointer.  This pointer is invalid after
       VecCUSPRestoreCUDAArrayRead() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: advanced

   Fortran note: This function is not currently available from Fortran.

.seealso: VecCUSPGetCUDAArrayRead(), VecCUSPGetCUDAArrayWrite(), VecCUSPGetArrayReadWrite(), VecCUSPGetArrayRead(), VecCUSPGetArrayWrite(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUSPRestoreCUDAArrayRead(Vec v, PetscScalar **a)
{
  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUSP,VECMPICUSP);
  v->valid_GPU_array = PETSC_CUSP_BOTH;
  PetscFunctionReturn(0);
}

/*@C
   VecCUSPGetCUDAArrayWrite - Provides write access to the CUDA buffer inside a vector.

   The data pointed to by the device pointer is uninitialized.  The user
   may not read from this data.  Furthermore, the entire array needs to
   be filled by the user to obtain well-defined behaviour.  The device
   memory will be allocated by this function if it hasn't been allocated
   previously.  This is analogous to intent(out) in Fortran.

   The device pointer needs to be released with
   VecCUSPRestoreCUDAArrayWrite().  When the pointer is released the
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

.seealso: VecCUSPRestoreCUDAArrayWrite(), VecCUSPGetCUDAArrayReadWrite(), VecCUSPGetCUDAArrayWrite(), VecCUSPGetArrayReadWrite(), VecCUSPGetArrayRead(), VecCUSPGetArrayWrite(), VecGetArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUSPGetCUDAArrayWrite(Vec v, PetscScalar **a)
{
  PetscErrorCode ierr;
  CUSPARRAY      *cusparray;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUSP,VECMPICUSP);
  PetscValidPointer(a,1);
  ierr = VecCUSPGetArrayWrite(v, &cusparray);CHKERRQ(ierr);
  *a   = thrust::raw_pointer_cast(cusparray->data());
  PetscFunctionReturn(0);
}

/*@C
   VecCUSPRestoreCUDAArrayWrite - Restore a device vector previously acquired with VecCUSPGetCUDAArrayWrite().

   Data on the host will be marked as out of date.  Subsequent access of
   the data on the host side e.g. with VecGetArray() will incur a device
   to host data transfer.

   Input Parameter:
+  v - the vector

-  a - the CUDA device pointer.  This pointer is invalid after
       VecCUSPRestoreCUDAArrayWrite() returns.

   Fortran note:
   This function is not currently available from Fortran.

   Level: advanced

.seealso: VecCUSPGetCUDAArrayWrite(), VecCUSPGetCUDAArrayWrite(), VecCUSPGetArrayReadWrite(), VecCUSPGetArrayRead(), VecCUSPGetArrayWrite(), VecGetArray(), VecRestoreArray(), VecGetArrayRead()
@*/
PETSC_EXTERN PetscErrorCode VecCUSPRestoreCUDAArrayWrite(Vec v, PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQCUSP,VECMPICUSP);
  v->valid_GPU_array = PETSC_CUSP_GPU;
  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*@C
   VecCUSPPlaceArray - Allows one to replace the array in a vector with a
   CUSPARRAY provided by the user. This is useful to avoid copying a
   CUSPARRAY into a vector.

   Not Collective

   Input Parameters:
+  vec - the vector
-  array - the CUSPARRAY

   Notes:
   You can return to the original CUSPARRAY with a call to VecCUSPResetArray()
   It is not possible to use VecCUSPPlaceArray() and VecPlaceArray() at the
   same time on the same vector.

   Level: developer

.seealso: VecPlaceArray(), VecGetArray(), VecRestoreArray(), VecReplaceArray(), VecResetArray(), VecCUSPResetArray(), VecCUSPReplaceArray()

@*/
PetscErrorCode VecCUSPPlaceArray(Vec vin,CUSPARRAY *a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(vin,VECSEQCUSP,VECMPICUSP);
  ierr = VecCUSPCopyToGPU(vin);CHKERRQ(ierr);
  if (((Vec_Seq*)vin->data)->unplacedarray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"VecCUSPPlaceArray()/VecPlaceArray() was already called on this vector, without a call to VecCUSPResetArray()/VecResetArray()");
  ((Vec_Seq*)vin->data)->unplacedarray  = (PetscScalar *) ((Vec_CUSP*)vin->spptr)->GPUarray; /* save previous CUDAARRAY so reset can bring it back */
  ((Vec_CUSP*)vin->spptr)->GPUarray = a;
  vin->valid_GPU_array = PETSC_CUSP_GPU;
  ierr = PetscObjectStateIncrease((PetscObject)vin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCUSPReplaceArray - Allows one to replace the CUSPARRAY in a vector
   with a CUSPARRAY provided by the user. This is useful to avoid copying
   a CUSPARRAY into a vector.

   Not Collective

   Input Parameters:
+  vec - the vector
-  array - the CUSPARRAY

   Notes:
   This permanently replaces the CUSPARRAY and frees the memory associated
   with the old CUSPARRAY.

   The memory passed in CANNOT be freed by the user. It will be freed
   when the vector is destroy.

   Not supported from Fortran

   Level: developer

.seealso: VecGetArray(), VecRestoreArray(), VecPlaceArray(), VecResetArray(), VecCUSPResetArray(), VecCUSPPlaceArray(), VecReplaceArray()

@*/
PetscErrorCode VecCUSPReplaceArray(Vec vin,CUSPARRAY *a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(vin,VECSEQCUSP,VECMPICUSP);
  delete ((Vec_CUSP*)vin->spptr)->GPUarray;
  ((Vec_CUSP*)vin->spptr)->GPUarray = a;
  vin->valid_GPU_array = PETSC_CUSP_GPU;
  ierr = PetscObjectStateIncrease((PetscObject)vin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCUSPResetArray - Resets a vector to use its default memory. Call this
   after the use of VecCUSPPlaceArray().

   Not Collective

   Input Parameters:
.  vec - the vector

   Level: developer

.seealso: VecGetArray(), VecRestoreArray(), VecReplaceArray(), VecPlaceArray(), VecResetArray(), VecCUSPPlaceArray(), VecCUSPReplaceArray()

@*/
PetscErrorCode VecCUSPResetArray(Vec vin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(vin,VECSEQCUSP,VECMPICUSP);
  ierr = VecCUSPCopyToGPU(vin);CHKERRQ(ierr);
  ((Vec_CUSP*)vin->spptr)->GPUarray = (CUSPARRAY *) ((Vec_Seq*)vin->data)->unplacedarray;
  ((Vec_Seq*)vin->data)->unplacedarray = 0;
  vin->valid_GPU_array = PETSC_CUSP_GPU;
  ierr = PetscObjectStateIncrease((PetscObject)vin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
