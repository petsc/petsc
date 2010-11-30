#define PETSCVEC_DLL
/*
   Implements the sequential cuda vectors.
*/

#include "petscconf.h"
PETSC_CUDA_EXTERN_C_BEGIN
#include "private/vecimpl.h"          /*I "petscvec.h" I*/
#include "../src/vec/vec/impls/dvecimpl.h"
PETSC_CUDA_EXTERN_C_END
#include "../src/vec/vec/impls/seq/seqcuda/cudavecimpl.h"

#undef __FUNCT__
#define __FUNCT__ "VecCopy_Seq"
static PetscErrorCode VecCopy_Seq(Vec xin,Vec yin)
{
  PetscScalar       *ya;
  const PetscScalar *xa;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (xin != yin) {
    ierr = VecGetArrayRead(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
    ierr = PetscMemcpy(ya,xa,xin->map->n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecSetRandom_Seq"
static PetscErrorCode VecSetRandom_Seq(Vec xin,PetscRandom r)
{
  PetscErrorCode ierr;
  PetscInt       n = xin->map->n,i;
  PetscScalar    *xx;

  PetscFunctionBegin;
  ierr = VecGetArray(xin,&xx);CHKERRQ(ierr);
  for (i=0; i<n; i++) {ierr = PetscRandomGetValue(r,&xx[i]);CHKERRQ(ierr);}
  ierr = VecRestoreArray(xin,&xx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDestroy_Seq"
static PetscErrorCode VecDestroy_Seq(Vec v)
{
  Vec_Seq        *vs = (Vec_Seq*)v->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectDepublish(v);CHKERRQ(ierr);

#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%D",v->map->n);
#endif
  ierr = PetscFree(vs->array_allocated);CHKERRQ(ierr);
  ierr = PetscFree(vs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecResetArray_Seq"
static PetscErrorCode VecResetArray_Seq(Vec vin)
{
  Vec_Seq *v = (Vec_Seq *)vin->data;

  PetscFunctionBegin;
  v->array         = v->unplacedarray;
  v->unplacedarray = 0;
  PetscFunctionReturn(0);
}

/* these following 3 public versions are necessary because we use CUSP in the regular PETSc code and these need to be called from plain C code. */
#undef __FUNCT__
#define __FUNCT__ "VecCUDAAllocateCheck_Public"
PetscErrorCode VecCUDAAllocateCheck_Public(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDAAllocateCheck(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCUDACopyToGPU_Public"
PetscErrorCode VecCUDACopyToGPU_Public(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDACopyToGPU(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

struct  _p_PetscCUSPIndices {
  CUSPINTARRAYCPU indicesCPU;
  CUSPINTARRAYGPU indicesGPU;
};


#undef __FUNCT__
#define __FUNCT__ "PetscCUSPIndicesCreate"
/*
    PetscCUSPIndicesCreate - creates the data structure needed by VecCUDACopyToGPUSome_Public()

   Input Parameters:
+    n - the number of indices
-    indices - integer list of indices

   Output Parameter:
.    ci - the CUSPIndices object suitable to pass to VecCUDACopyToGPUSome_Public()

.seealso: PetscCUSPIndicesDestroy(), VecCUDACopyToGPUSome_Public()
*/
PetscErrorCode PetscCUSPIndicesCreate(PetscInt n,const PetscInt *indices,PetscCUSPIndices *ci)
{
  PetscCUSPIndices  cci;

  PetscFunctionBegin;
  cci = new struct _p_PetscCUSPIndices;
  cci->indicesCPU.assign(indices,indices+n);
  cci->indicesGPU.assign(indices,indices+n);
  *ci = cci;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCUSPIndicesDestroy"
/*
    PetscCUSPIndicesDestroy - destroys the data structure needed by VecCUDACopyToGPUSome_Public()

   Input Parameters:
.    ci - the CUSPIndices object suitable to pass to VecCUDACopyToGPUSome_Public()

.seealso: PetscCUSPIndicesCreate(), VecCUDACopyToGPUSome_Public()
*/
PetscErrorCode PetscCUSPIndicesDestroy(PetscCUSPIndices ci)
{
  PetscFunctionBegin;
  if (!ci) PetscFunctionReturn(0);
  try {
    delete ci;
  } catch(char* ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCUDACopyToGPUSome_Public"
/*
    VecCUDACopyToGPUSome_Public - Copies certain entries down to the GPU from the CPU of a vector

   Input Parameters:
+    v - the vector
-    indices - the requested indices, this should be created with CUSPIndicesCreate()

*/
PetscErrorCode VecCUDACopyToGPUSome_Public(Vec v, PetscCUSPIndices ci)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = VecCUDACopyToGPUSome(v,&ci->indicesCPU,&ci->indicesGPU);CHKERRCUDA(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "VecCUDACopyFromGPU"
/*@C
     VecCUDACopyFromGPU - Copies a vector from the GPU to the CPU unless we already have an up-to-date copy on the CPU
@*/
PetscErrorCode VecCUDACopyFromGPU(Vec v)
{
  PetscErrorCode ierr;
  CUSPARRAY      *GPUvector;
  PetscScalar    *array;
  Vec_Seq        *s;
  PetscInt       n = v->map->n;

  PetscFunctionBegin;
  s = (Vec_Seq*)v->data;
  if (s->array == 0){
    ierr               = PetscMalloc(n*sizeof(PetscScalar),&array);CHKERRQ(ierr);
    ierr               = PetscLogObjectMemory(v,n*sizeof(PetscScalar));CHKERRQ(ierr);
    s->array           = array;
    s->array_allocated = array;
  }
  if (v->valid_GPU_array == PETSC_CUDA_GPU){
    GPUvector  = ((Vec_CUDA*)v->spptr)->GPUarray;
    ierr       = PetscLogEventBegin(VEC_CUDACopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    try{
      thrust::copy(GPUvector->begin(),GPUvector->end(),*(PetscScalar**)v->data);
      ierr = WaitForGPU();CHKERRCUDA(ierr);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
    ierr = PetscLogEventEnd(VEC_CUDACopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    v->valid_GPU_array = PETSC_CUDA_BOTH;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCUDACopyFromGPUSome"
/* Note that this function only copies *some* of the values up from the GPU to CPU,
   which means that we need recombine the data at some point before using any of the standard functions.
   We could add another few flag-types to keep track of this, or treat things like VecGetArray VecRestoreArray
   where you have to always call in pairs
*/
PetscErrorCode VecCUDACopyFromGPUSome(Vec v,CUSPINTARRAYCPU *indicesCPU,CUSPINTARRAYGPU *indicesGPU)
{
  Vec_Seq        *s;
  PetscInt       n = v->map->n;
  PetscScalar    *array;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDAAllocateCheck(v);CHKERRCUDA(ierr);
  s = (Vec_Seq*)v->data;
  if (s->array == 0){
    ierr               = PetscMalloc(n*sizeof(PetscScalar),&array);CHKERRQ(ierr);
    ierr               = PetscLogObjectMemory(v,n*sizeof(PetscScalar));CHKERRQ(ierr);
    s->array           = array;
    s->array_allocated = array;
  }
  if (v->valid_GPU_array == PETSC_CUDA_GPU) {
    ierr = PetscLogEventBegin(VEC_CUDACopyFromGPUSome,v,0,0,0);CHKERRQ(ierr);
    thrust::copy(
		 thrust::make_permutation_iterator(((Vec_CUDA *)v->spptr)->GPUarray->begin(),indicesGPU->begin()),
		 thrust::make_permutation_iterator(((Vec_CUDA *)v->spptr)->GPUarray->begin(),indicesGPU->end()),
		 thrust::make_permutation_iterator(s->array,indicesCPU->begin()));
    ierr = PetscLogEventEnd(VEC_CUDACopyFromGPUSome,v,0,0,0);CHKERRQ(ierr);
  }
  /*v->valid_GPU_array = PETSC_CUDA_CPU; */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCUDACopyFromGPUSome_Public"
/*
  VecCUDACopyFromGPUSome_Public - Copies certain entries up to the CPU from the GPU of a vector

  Input Parameters:
 +    v - the vector
 -    indices - the requested indices, this should be created with CUSPIndicesCreate()
*/
PetscErrorCode VecCUDACopyFromGPUSome_Public(Vec v, PetscCUSPIndices ci)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUDACopyFromGPUSome(v,&ci->indicesCPU,&ci->indicesGPU);CHKERRCUDA(ierr);
  PetscFunctionReturn(0);
}


/*MC
   VECSEQCUDA - VECSEQCUDA = "seqcuda" - The basic sequential vector, modified to use CUDA

   Options Database Keys:
. -vec_type seqcuda - sets the vector type to VECSEQCUDA during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateSeqWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateSeq()
M*/

/* for VecAYPX_SeqCUDA*/
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
   detail::assert_same_dimensions(x,y);
   aypx(x.begin(),x.end(),y.begin(),alpha);
 }
}
}

#undef __FUNCT__
#define __FUNCT__ "VecAYPX_SeqCUDA"
PetscErrorCode VecAYPX_SeqCUDA(Vec yin, PetscScalar alpha, Vec xin)
{
  CUSPARRAY      *xarray,*yarray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha != 0.0) {/*
    ierr = VecCUDACopyToGPU(xin);CHKERRQ(ierr);
                      ierr = VecCUDACopyToGPU(yin);CHKERRQ(ierr);*/
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    try{
      /*cusp::blas::aypx(*((Vec_CUDA*)xin->spptr)->GPUarray,*((Vec_CUDA*)yin->spptr)->GPUarray,alpha);
       yin->valid_GPU_array = PETSC_CUDA_GPU;*/
      cusp::blas::aypx(*xarray,*yarray,alpha);
      ierr = WaitForGPU();CHKERRCUDA(ierr);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogFlops(2.0*yin->map->n);CHKERRQ(ierr);
   }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecAXPY_SeqCUDA"
PetscErrorCode VecAXPY_SeqCUDA(Vec yin,PetscScalar alpha,Vec xin)
{
  CUSPARRAY      *xarray,*yarray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha != 0.0) {
    /*ierr = VecCUDACopyToGPU(xin);CHKERRQ(ierr);
     ierr = VecCUDACopyToGPU(yin);CHKERRQ(ierr);*/
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    try {
      /*cusp::blas::axpy(*((Vec_CUDA*)xin->spptr)->GPUarray,*((Vec_CUDA*)yin->spptr)->GPUarray,alpha);
       yin->valid_GPU_array = PETSC_CUDA_GPU;*/
      cusp::blas::axpy(*xarray,*yarray,alpha);
      ierr = WaitForGPU();CHKERRCUDA(ierr);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogFlops(2.0*yin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

struct VecCUDAPointwiseDivide
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = thrust::get<1>(t) / thrust::get<2>(t);
  }
};

#undef __FUNCT__
#define __FUNCT__ "VecPointwiseDivide_SeqCUDA"
PetscErrorCode VecPointwiseDivide_SeqCUDA(Vec win, Vec xin, Vec yin)
{
  CUSPARRAY      *warray,*xarray,*yarray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*ierr = VecCUDAAllocateCheck(win);CHKERRQ(ierr);
  ierr = VecCUDACopyToGPU(xin);CHKERRQ(ierr);
   ierr = VecCUDACopyToGPU(yin);CHKERRQ(ierr);*/
  ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayWrite(win,&warray);CHKERRQ(ierr);
  try{
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
	VecCUDAPointwiseDivide());
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
  ierr = PetscLogFlops(win->map->n);CHKERRQ(ierr);
  /*win->valid_GPU_array = PETSC_CUDA_GPU;*/
  ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(win,&warray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


struct VecCUDAWAXPY
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = thrust::get<1>(t) + thrust::get<2>(t)*thrust::get<3>(t);
  }
};

struct VecCUDASum
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = thrust::get<1>(t) + thrust::get<2>(t);
  }
};

struct VecCUDADiff
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = thrust::get<1>(t) - thrust::get<2>(t);
  }
};

#undef __FUNCT__
#define __FUNCT__ "VecWAXPY_SeqCUDA"
PetscErrorCode VecWAXPY_SeqCUDA(Vec win,PetscScalar alpha,Vec xin, Vec yin)
{
  CUSPARRAY      *xarray,*yarray,*warray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*ierr = VecCUDACopyToGPU(xin);CHKERRQ(ierr);
  ierr = VecCUDACopyToGPU(yin);CHKERRQ(ierr);
   ierr = VecCUDAAllocateCheck(win);CHKERRQ(ierr);*/
    if (alpha == 0.0) {
    ierr = VecCopy_SeqCUDA(yin,win);CHKERRQ(ierr);
  } else {
      ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
      ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
      ierr = VecCUDAGetArrayWrite(win,&warray);CHKERRQ(ierr);
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
            VecCUDASum());
        } catch(char* ex) {
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
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
            VecCUDADiff());
        } catch(char* ex) {
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
        }
        ierr = PetscLogFlops(win->map->n);CHKERRQ(ierr);
      } else {
        try {
          thrust::for_each(
            thrust::make_zip_iterator(
              thrust::make_tuple(
		warray->begin(),
		yarray->begin(),
		thrust::make_constant_iterator(alpha,0),
		xarray->begin())),
            thrust::make_zip_iterator(
              thrust::make_tuple(
		warray->end(),
		yarray->end(),
		thrust::make_constant_iterator(alpha,win->map->n),
		xarray->end())),
            VecCUDAWAXPY());
        } catch(char* ex) {
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
        }
        ierr = PetscLogFlops(2*win->map->n);CHKERRQ(ierr);
      }
      ierr = WaitForGPU();CHKERRCUDA(ierr);
      /*win->valid_GPU_array = PETSC_CUDA_GPU;*/
      ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
      ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
      ierr = VecCUDARestoreArrayWrite(win,&warray);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

/* These functions are for the CUDA implementation of MAXPY with the loop unrolled on the CPU */
struct VecCUDAMAXPY4
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    /*y += a1*x1 +a2*x2 + 13*x3 +a4*x4 */
    thrust::get<0>(t) += thrust::get<1>(t)*thrust::get<2>(t)+thrust::get<3>(t)*thrust::get<4>(t)+thrust::get<5>(t)*thrust::get<6>(t)+thrust::get<7>(t)*thrust::get<8>(t);
  }
};


struct VecCUDAMAXPY3
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    /*y += a1*x1 +a2*x2 + 13*x3 */
    thrust::get<0>(t) += thrust::get<1>(t)*thrust::get<2>(t)+thrust::get<3>(t)*thrust::get<4>(t)+thrust::get<5>(t)*thrust::get<6>(t);
  }
};

struct VecCUDAMAXPY2
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    /*y += a1*x1 +a2*x2*/
    thrust::get<0>(t) += thrust::get<1>(t)*thrust::get<2>(t)+thrust::get<3>(t)*thrust::get<4>(t);
  }
};
#undef __FUNCT__
#define __FUNCT__ "VecMAXPY_SeqCUDA"
PetscErrorCode VecMAXPY_SeqCUDA(Vec xin, PetscInt nv,const PetscScalar *alpha,Vec *y)
{
  PetscErrorCode    ierr;
  CUSPARRAY         *xarray,*yy0,*yy1,*yy2,*yy3;
  PetscInt          n = xin->map->n,j,j_rem;
  PetscScalar       alpha0,alpha1,alpha2,alpha3;

  PetscFunctionBegin;
  ierr = PetscLogFlops(nv*2.0*n);CHKERRQ(ierr);
  /*ierr = VecCUDACopyToGPU(xin);CHKERRQ(ierr);*/
  ierr = VecCUDAGetArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
  switch (j_rem=nv&0x3) {
  case 3:
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha2 = alpha[2];
    alpha += 3;
    /*yy0    = y[0];
    yy1    = y[1];
    yy2    = y[2];
    ierr   = VecCUDACopyToGPU(yy0);CHKERRQ(ierr);
    ierr   = VecCUDACopyToGPU(yy1);CHKERRQ(ierr);
     ierr   = VecCUDACopyToGPU(yy2);CHKERRQ(ierr);*/
    ierr = VecCUDAGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(y[2],&yy2);CHKERRQ(ierr);
    try {
      thrust::for_each(
	thrust::make_zip_iterator(
	    thrust::make_tuple(
		xarray->begin(),
		thrust::make_constant_iterator(alpha0,0),
		yy0->begin(),
		thrust::make_constant_iterator(alpha1,0),
		yy1->begin(),
		thrust::make_constant_iterator(alpha2,0),
		yy2->begin())),
	thrust::make_zip_iterator(
	    thrust::make_tuple(
		xarray->end(),
		thrust::make_constant_iterator(alpha0,n),
		yy0->end(),
		thrust::make_constant_iterator(alpha1,n),
		yy1->end(),
		thrust::make_constant_iterator(alpha2,n),
		yy2->end())),
	VecCUDAMAXPY3());
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
    ierr = VecCUDARestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(y[2],&yy2);CHKERRQ(ierr);
    y     += 3;
    break;
  case 2:
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha +=2;
    /*yy0    = y[0];
    yy1    = y[1];
    ierr   = VecCUDACopyToGPU(yy0);CHKERRQ(ierr);
     ierr   = VecCUDACopyToGPU(yy1);CHKERRQ(ierr);*/
    ierr = VecCUDAGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
    try {
      thrust::for_each(
	thrust::make_zip_iterator(
	    thrust::make_tuple(
		xarray->begin(),
		thrust::make_constant_iterator(alpha0,0),
		yy0->begin(),
		thrust::make_constant_iterator(alpha1,0),
		yy1->begin())),
	thrust::make_zip_iterator(
	    thrust::make_tuple(
		xarray->end(),
		thrust::make_constant_iterator(alpha0,n),
		yy0->end(),
		thrust::make_constant_iterator(alpha1,n),
		yy1->end())),
	VecCUDAMAXPY2());
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
    y     +=2;
    break;
  case 1:
    alpha0 = *alpha++;
    ierr = VecAXPY_SeqCUDA(xin,alpha0,y[0]);
    y     +=1;
    break;
  }
  for (j=j_rem; j<nv; j+=4) {
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha2 = alpha[2];
    alpha3 = alpha[3];
    alpha  += 4;
    /*yy0    = y[0];
    yy1    = y[1];
    yy2    = y[2];
    yy3    = y[3];
    ierr   = VecCUDACopyToGPU(yy0);CHKERRQ(ierr);
    ierr   = VecCUDACopyToGPU(yy1);CHKERRQ(ierr);
    ierr   = VecCUDACopyToGPU(yy2);CHKERRQ(ierr);
     ierr   = VecCUDACopyToGPU(yy3);CHKERRQ(ierr);*/
    ierr = VecCUDAGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(y[2],&yy2);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(y[3],&yy3);CHKERRQ(ierr);
    try {
      thrust::for_each(
	thrust::make_zip_iterator(
	    thrust::make_tuple(
		xarray->begin(),
		thrust::make_constant_iterator(alpha0,0),
		yy0->begin(),
		thrust::make_constant_iterator(alpha1,0),
		yy1->begin(),
		thrust::make_constant_iterator(alpha2,0),
		yy2->begin(),
		thrust::make_constant_iterator(alpha3,0),
		yy3->begin())),
	thrust::make_zip_iterator(
	    thrust::make_tuple(
		xarray->end(),
		thrust::make_constant_iterator(alpha0,n),
		yy0->end(),
		thrust::make_constant_iterator(alpha1,n),
		yy1->end(),
		thrust::make_constant_iterator(alpha2,n),
		yy2->end(),
		thrust::make_constant_iterator(alpha3,n),
		yy3->end())),
	VecCUDAMAXPY4());
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
    ierr = VecCUDARestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(y[2],&yy2);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(y[3],&yy3);CHKERRQ(ierr);
    y      += 4;
  }
  /*xin->valid_GPU_array = PETSC_CUDA_GPU;*/
  ierr = VecCUDARestoreArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecDot_SeqCUDA"
PetscErrorCode VecDot_SeqCUDA(Vec xin,Vec yin,PetscScalar *z)
{
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    *ya,*xa;
#endif
  CUSPARRAY      *xarray,*yarray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  /*Not working for complex*/
#else
  {
    /* ierr = VecCUDACopyToGPU(xin);CHKERRQ(ierr);
     ierr = VecCUDACopyToGPU(yin);CHKERRQ(ierr);*/
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
    try {
      *z = cusp::blas::dot(*xarray,*yarray);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
  }
#endif
 ierr = WaitForGPU();CHKERRCUDA(ierr);
 if (xin->map->n >0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
 ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
 ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
 PetscFunctionReturn(0);
}

/*The following few template functions are for VecMDot_SeqCUDA*/

template <typename T1,typename T2>
struct cudamult2 : thrust::unary_function<T1,T2>
{
	__host__ __device__
	T2 operator()(T1 x)
	{
		return thrust::make_tuple(thrust::get<0>(x)*thrust::get<1>(x),thrust::get<0>(x)*thrust::get<2>(x));
	}
};

template <typename T>
struct cudaadd2 : thrust::binary_function<T,T,T>
{
	__host__ __device__
	T operator()(T x,T y)
	{
		return thrust::make_tuple(thrust::get<0>(x)+thrust::get<0>(y),thrust::get<1>(x)+thrust::get<1>(y));
	}
};

template <typename T1,typename T2>
struct cudamult3 : thrust::unary_function<T1,T2>
{
	__host__ __device__
	T2 operator()(T1 x)
	{
	  return thrust::make_tuple(thrust::get<0>(x)*thrust::get<1>(x),thrust::get<0>(x)*thrust::get<2>(x),thrust::get<0>(x)*thrust::get<3>(x));
	}
};

template <typename T>
struct cudaadd3 : thrust::binary_function<T,T,T>
{
	__host__ __device__
	T operator()(T x,T y)
	{
	  return thrust::make_tuple(thrust::get<0>(x)+thrust::get<0>(y),thrust::get<1>(x)+thrust::get<1>(y),thrust::get<2>(x)+thrust::get<2>(y));
	}
};
	template <typename T1,typename T2>
struct cudamult4 : thrust::unary_function<T1,T2>
{
	__host__ __device__
	T2 operator()(T1 x)
	{
	  return thrust::make_tuple(thrust::get<0>(x)*thrust::get<1>(x),thrust::get<0>(x)*thrust::get<2>(x),thrust::get<0>(x)*thrust::get<3>(x),thrust::get<0>(x)*thrust::get<4>(x));
	}
};

template <typename T>
struct cudaadd4 : thrust::binary_function<T,T,T>
{
	__host__ __device__
	T operator()(T x,T y)
	{
	  return thrust::make_tuple(thrust::get<0>(x)+thrust::get<0>(y),thrust::get<1>(x)+thrust::get<1>(y),thrust::get<2>(x)+thrust::get<2>(y),thrust::get<3>(x)+thrust::get<3>(y));
	}
};


#undef __FUNCT__
#define __FUNCT__ "VecMDot_SeqCUDA"
PetscErrorCode VecMDot_SeqCUDA(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n,j,j_rem;
  /*Vec               yy0,yy1,yy2,yy3;*/
  CUSPARRAY         *xarray,*yy0,*yy1,*yy2,*yy3;
  PetscScalar       zero=0.0;
  thrust::tuple<PetscScalar,PetscScalar> result2;
  thrust::tuple<PetscScalar,PetscScalar,PetscScalar> result3;
  thrust::tuple<PetscScalar,PetscScalar,PetscScalar,PetscScalar>result4;

  PetscFunctionBegin;
  ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  switch(j_rem=nv&0x3) {
  case 3:
    /*yy0  =  yin[0];
    yy1  =  yin[1];
    yy2  =  yin[2];
    ierr =  VecCUDACopyToGPU(yy0);CHKERRQ(ierr);
    ierr =  VecCUDACopyToGPU(yy1);CHKERRQ(ierr);
     ierr =  VecCUDACopyToGPU(yy2);CHKERRQ(ierr);*/
    ierr = VecCUDAGetArrayRead(yin[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(yin[1],&yy1);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(yin[2],&yy2);CHKERRQ(ierr);
    try {
      result3 = thrust::transform_reduce(
		     thrust::make_zip_iterator(
			  thrust::make_tuple(
				   xarray->begin(),
				   yy0->begin(),
				   yy1->begin(),
				   yy2->begin())),
		     thrust::make_zip_iterator(
			  thrust::make_tuple(
				   xarray->end(),
				   yy0->end(),
				   yy1->end(),
				   yy2->end())),
		     cudamult3<thrust::tuple<PetscScalar,PetscScalar,PetscScalar,PetscScalar>, thrust::tuple<PetscScalar,PetscScalar,PetscScalar> >(),
		     thrust::make_tuple(zero,zero,zero), /*init */
		     cudaadd3<thrust::tuple<PetscScalar,PetscScalar,PetscScalar> >()); /* binary function */
      z[0] = thrust::get<0>(result3);
      z[1] = thrust::get<1>(result3);
      z[2] = thrust::get<2>(result3);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
    z    += 3;
    ierr = VecCUDARestoreArrayRead(yin[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(yin[1],&yy1);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(yin[2],&yy2);CHKERRQ(ierr);
    yin  += 3;
    break;
  case 2:
    /*yy0  =  yin[0];
    yy1  =  yin[1];
    ierr =  VecCUDACopyToGPU(yy0);CHKERRQ(ierr);
     ierr =  VecCUDACopyToGPU(yy1);CHKERRQ(ierr);*/
    ierr = VecCUDAGetArrayRead(yin[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(yin[1],&yy1);CHKERRQ(ierr);
    try {
      result2 = thrust::transform_reduce(
		    thrust::make_zip_iterator(
			thrust::make_tuple(
				  xarray->begin(),
				  yy0->begin(),
				  yy1->begin())),
		    thrust::make_zip_iterator(
			thrust::make_tuple(
				  xarray->end(),
				  yy0->end(),
				  yy1->end())),
		    cudamult2<thrust::tuple<PetscScalar,PetscScalar,PetscScalar>, thrust::tuple<PetscScalar,PetscScalar> >(),
		    thrust::make_tuple(zero,zero), /*init */
		    cudaadd2<thrust::tuple<PetscScalar, PetscScalar> >()); /* binary function */
      z[0] = thrust::get<0>(result2);
      z[1] = thrust::get<1>(result2);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
    z    += 2;
    ierr = VecCUDARestoreArrayRead(yin[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(yin[1],&yy1);CHKERRQ(ierr);
    yin  += 2;
    break;
  case 1:
    /*yy0  =  yin[0];
     ierr =  VecCUDACopyToGPU(yy0);CHKERRQ(ierr);*/
    ierr =  VecDot_SeqCUDA(xin,yin[0],&z[0]);CHKERRQ(ierr);
    z    += 1;
    yin  += 1;
    break;
  }
  for (j=j_rem; j<nv; j+=4) {
    /*yy0  =  yin[0];
    yy1  =  yin[1];
    yy2  =  yin[2];
    yy3  =  yin[3];
    ierr =  VecCUDACopyToGPU(yy0);CHKERRQ(ierr);
    ierr =  VecCUDACopyToGPU(yy1);CHKERRQ(ierr);
    ierr =  VecCUDACopyToGPU(yy2);CHKERRQ(ierr);
     ierr =  VecCUDACopyToGPU(yy3);CHKERRQ(ierr);*/
    ierr = VecCUDAGetArrayRead(yin[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(yin[1],&yy1);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(yin[2],&yy2);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayRead(yin[3],&yy3);CHKERRQ(ierr);
    try {
      result4 = thrust::transform_reduce(
		    thrust::make_zip_iterator(
			thrust::make_tuple(
				  xarray->begin(),
				  yy0->begin(),
				  yy1->begin(),
				  yy2->begin(),
				  yy3->begin())),
		    thrust::make_zip_iterator(
			thrust::make_tuple(
				  xarray->end(),
				  yy0->end(),
				  yy1->end(),
				  yy2->end(),
				  yy3->end())),
		     cudamult4<thrust::tuple<PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar>, thrust::tuple<PetscScalar,PetscScalar,PetscScalar,PetscScalar> >(),
		     thrust::make_tuple(zero,zero,zero,zero), /*init */
		     cudaadd4<thrust::tuple<PetscScalar,PetscScalar,PetscScalar,PetscScalar> >()); /* binary function */
      z[0] = thrust::get<0>(result4);
      z[1] = thrust::get<1>(result4);
      z[2] = thrust::get<2>(result4);
      z[3] = thrust::get<3>(result4);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
    z    += 4;
    ierr = VecCUDARestoreArrayRead(yin[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(yin[1],&yy1);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(yin[2],&yy2);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(yin[3],&yy3);CHKERRQ(ierr);
    yin  += 4;
  }
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  ierr = PetscLogFlops(PetscMax(nv*(2.0*n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecSet_SeqCUDA"
PetscErrorCode VecSet_SeqCUDA(Vec xin,PetscScalar alpha)
{
  CUSPARRAY      *xarray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* if there's a faster way to do the case alpha=0.0 on the GPU we should do that*/
  /*ierr = VecCUDAAllocateCheck(xin);CHKERRQ(ierr);*/
  ierr = VecCUDAGetArrayWrite(xin,&xarray);CHKERRQ(ierr);
  try {
    /*cusp::blas::fill(*((Vec_CUDA*)xin->spptr)->GPUarray,alpha);*/
    cusp::blas::fill(*xarray,alpha);
  } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
  }
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  /*xin->valid_GPU_array = PETSC_CUDA_GPU;*/
  ierr = VecCUDARestoreArrayWrite(xin,&xarray);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecScale_SeqCUDA"
PetscErrorCode VecScale_SeqCUDA(Vec xin, PetscScalar alpha)
{
  CUSPARRAY      *xarray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == 0.0) {
    ierr = VecSet_SeqCUDA(xin,alpha);CHKERRQ(ierr);
  } else if (alpha != 1.0) {
    /*ierr = VecCUDACopyToGPU(xin);CHKERRQ(ierr);*/
    ierr = VecCUDAGetArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
    try {
      cusp::blas::scal(*xarray,alpha);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
    /*xin->valid_GPU_array = PETSC_CUDA_GPU;*/
    ierr = VecCUDARestoreArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
  }
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecTDot_SeqCUDA"
PetscErrorCode VecTDot_SeqCUDA(Vec xin,Vec yin,PetscScalar *z)
{
#if defined(PETSC_USE_COMPLEX)
  PetscScalar    *ya,*xa;
#endif
  CUSPARRAY      *xarray,*yarray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  /*Not working for complex*/
#else
  /*ierr = VecCUDACopyToGPU(xin);CHKERRQ(ierr);
   ierr = VecCUDACopyToGPU(yin);CHKERRQ(ierr);*/
 ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
 ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
 try {
   *z = cusp::blas::dot(*xarray,*yarray);
 } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
 }
#endif
 ierr = WaitForGPU();CHKERRCUDA(ierr);
  if (xin->map->n > 0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "VecCopy_SeqCUDA"
PetscErrorCode VecCopy_SeqCUDA(Vec xin,Vec yin)
{
  CUSPARRAY      *xarray,*yarray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (xin != yin) {
    if (xin->valid_GPU_array == PETSC_CUDA_GPU) {
      /* copy in GPU */
      /*ierr = VecCUDAAllocateCheck(yin);CHKERRQ(ierr);*/
      ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
      ierr = VecCUDAGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
       try {
	 cusp::blas::copy(*xarray,*yarray);
       } catch(char* ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
      }
      ierr = WaitForGPU();CHKERRCUDA(ierr);
      /*yin->valid_GPU_array = PETSC_CUDA_GPU;*/
      ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
      ierr = VecCUDARestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);

    } else if (xin->valid_GPU_array == PETSC_CUDA_CPU || xin->valid_GPU_array == PETSC_CUDA_UNALLOCATED) {
      /* copy in CPU if we are on the CPU*/
      ierr = VecCopy_Seq(xin,yin);CHKERRQ(ierr);
    } else if (xin->valid_GPU_array == PETSC_CUDA_BOTH) {
      /* if xin is valid in both places, see where yin is and copy there (because it's probably where we'll want to next use it) */
      if (yin->valid_GPU_array == PETSC_CUDA_CPU) {
	/* copy in CPU */
	ierr = VecCopy_Seq(xin,yin);CHKERRQ(ierr);

      } else if (yin->valid_GPU_array == PETSC_CUDA_GPU) {
	/* copy in GPU */
	/*ierr = VecCUDACopyToGPU(xin);CHKERRQ(ierr);*/
        ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
	try {
	  cusp::blas::copy(*xarray,*yarray);
	  ierr = WaitForGPU();CHKERRCUDA(ierr);
	} catch(char* ex) {
	  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
	}
	/*yin->valid_GPU_array = PETSC_CUDA_GPU;*/
        ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
      } else if (yin->valid_GPU_array == PETSC_CUDA_BOTH) {
	/* xin and yin are both valid in both places (or yin was unallocated before the earlier call to allocatecheck
	   default to copy in GPU (this is an arbitrary choice) */
        ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUDAGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
	try {
	  cusp::blas::copy(*xarray,*yarray);
	  ierr = WaitForGPU();CHKERRCUDA(ierr);
	} catch(char* ex) {
	  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
	}
	/*yin->valid_GPU_array = PETSC_CUDA_GPU;*/
        ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
      } else {
	ierr = VecCopy_Seq(xin,yin);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecSwap_SeqCUDA"
PetscErrorCode VecSwap_SeqCUDA(Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscBLASInt   one = 1,bn = PetscBLASIntCast(xin->map->n);
  CUSPARRAY      *xarray,*yarray;

  PetscFunctionBegin;
  if (xin != yin) {
    ierr = VecCUDAGetArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    /*ierr = VecCUDACopyToGPU(xin);CHKERRQ(ierr);
     ierr = VecCUDACopyToGPU(yin);CHKERRQ(ierr);*/
#if defined(PETSC_USE_SCALAR_SINGLE)
    cublasSswap(bn,VecCUDACastToRawPtr(*xarray),one,VecCUDACastToRawPtr(*yarray),one);
#else
    cublasDswap(bn,VecCUDACastToRawPtr(*xarray),one,VecCUDACastToRawPtr(*yarray),one);
#endif
    ierr = cublasGetError();CHKERRCUDA(ierr);
    ierr = WaitForGPU();CHKERRCUDA(ierr);
    /*xin->valid_GPU_array = PETSC_CUDA_GPU;
     yin->valid_GPU_array = PETSC_CUDA_GPU;*/
    ierr = VecCUDARestoreArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

struct VecCUDAAX
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = thrust::get<1>(t)*thrust::get<2>(t);
  }
};
#undef __FUNCT__
#define __FUNCT__ "VecAXPBY_SeqCUDA"
PetscErrorCode VecAXPBY_SeqCUDA(Vec yin,PetscScalar alpha,PetscScalar beta,Vec xin)
{
  PetscErrorCode    ierr;
  PetscInt          n = yin->map->n;
  PetscScalar       a = alpha,b = beta;
  CUSPARRAY         *xarray,*yarray;

  PetscFunctionBegin;
  if (a == 0.0) {
    ierr = VecScale_SeqCUDA(yin,beta);CHKERRQ(ierr);
  } else if (b == 1.0) {
    ierr = VecAXPY_SeqCUDA(yin,alpha,xin);CHKERRQ(ierr);
  } else if (a == 1.0) {
    ierr = VecAYPX_SeqCUDA(yin,beta,xin);CHKERRQ(ierr);
  } else if (b == 0.0) {
    /*
    ierr = VecCUDACopyToGPU(xin);CHKERRQ(ierr);
     ierr = VecCUDACopyToGPU(yin);CHKERRQ(ierr);*/
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    try {
      thrust::for_each(
	thrust::make_zip_iterator(
	    thrust::make_tuple(
                yarray->begin(),
		thrust::make_constant_iterator(a,0),
		xarray->begin())),
	thrust::make_zip_iterator(
	    thrust::make_tuple(
		yarray->end(),
		thrust::make_constant_iterator(a,n),
		xarray->end())),
	VecCUDAAX());
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
    ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRCUDA(ierr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
  } else {
    /*ierr = VecCUDACopyToGPU(xin);CHKERRQ(ierr);
     ierr = VecCUDACopyToGPU(yin);CHKERRQ(ierr);*/
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDAGetArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    try {
      cusp::blas::axpby(*xarray,*yarray,*yarray,a,b);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
    /*yin->valid_GPU_array = PETSC_CUDA_GPU;*/
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRCUDA(ierr);
    ierr = PetscLogFlops(3.0*xin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* structs below are for special cases of VecAXPBYPCZ_SeqCUDA */
struct VecCUDAXPBYPCZ
{
  /* z = x + b*y + c*z */
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) = thrust::get<1>(t)*thrust::get<0>(t)+thrust::get<2>(t)+thrust::get<4>(t)*thrust::get<3>(t);
  }
};
struct VecCUDAAXPBYPZ
{
  /* z = ax + b*y + z */
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<0>(t) += thrust::get<2>(t)*thrust::get<1>(t)+thrust::get<4>(t)*thrust::get<3>(t);
  }
};

#undef __FUNCT__
#define __FUNCT__ "VecAXPBYPCZ_SeqCUDA"
PetscErrorCode VecAXPBYPCZ_SeqCUDA(Vec zin,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec xin,Vec yin)
{
  PetscErrorCode     ierr;
  PetscInt           n = zin->map->n;
  CUSPARRAY          *xarray,*yarray,*zarray;

  PetscFunctionBegin;
  /*
    ierr = VecCUDACopyToGPU(xin);CHKERRQ(ierr);
    ierr = VecCUDACopyToGPU(yin);CHKERRQ(ierr);
    ierr = VecCUDACopyToGPU(zin);CHKERRQ(ierr);
   */
  ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayReadWrite(zin,&zarray);CHKERRQ(ierr);
  if (alpha == 1.0) {
    try {
      thrust::for_each(
	thrust::make_zip_iterator(
	    thrust::make_tuple(
		zarray->begin(),
		thrust::make_constant_iterator(gamma,0),
		xarray->begin(),
		yarray->begin(),
		thrust::make_constant_iterator(beta,0))),
	thrust::make_zip_iterator(
	    thrust::make_tuple(
		zarray->end(),
		thrust::make_constant_iterator(gamma,n),
		xarray->end(),
		yarray->end(),
		thrust::make_constant_iterator(beta,n))),
	VecCUDAXPBYPCZ());
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
    ierr = PetscLogFlops(4.0*n);CHKERRQ(ierr);
  } else if (gamma == 1.0) {
    try {
      thrust::for_each(
	thrust::make_zip_iterator(
	    thrust::make_tuple(
		zarray->begin(),
		xarray->begin(),
		thrust::make_constant_iterator(alpha,0),
		yarray->begin(),
		thrust::make_constant_iterator(beta,0))),
	thrust::make_zip_iterator(
	    thrust::make_tuple(
		zarray->end(),
		xarray->end(),
		thrust::make_constant_iterator(alpha,n),
		yarray->end(),
		thrust::make_constant_iterator(beta,n))),
	VecCUDAAXPBYPZ());
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
    ierr = PetscLogFlops(4.0*n);CHKERRQ(ierr);
  } else {
    try {
      cusp::blas::axpbypcz(*xarray,*yarray,*zarray,*zarray,alpha,beta,gamma);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
    /*zin->valid_GPU_array = PETSC_CUDA_GPU;*/
    ierr = VecCUDARestoreArrayReadWrite(zin,&zarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogFlops(5.0*n);CHKERRQ(ierr);
  }
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPointwiseMult_SeqCUDA"
PetscErrorCode VecPointwiseMult_SeqCUDA(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscInt       n = win->map->n;
  CUSPARRAY      *xarray,*yarray,*warray;

  PetscFunctionBegin;
  /*ierr = VecCUDACopyToGPU(xin);CHKERRQ(ierr);
  ierr = VecCUDACopyToGPU(yin);CHKERRQ(ierr);
   ierr = VecCUDAAllocateCheck(win);CHKERRQ(ierr);*/
  ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayReadWrite(win,&warray);CHKERRQ(ierr);
  try {
    cusp::blas::xmy(*xarray,*yarray,*warray);
  } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
  }
  /*win->valid_GPU_array = PETSC_CUDA_GPU;*/
  ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayReadWrite(win,&warray);CHKERRQ(ierr);
  ierr = PetscLogFlops(n);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  PetscFunctionReturn(0);
}


/* should do infinity norm in cuda */

#undef __FUNCT__
#define __FUNCT__ "VecNorm_SeqCUDA"
PetscErrorCode VecNorm_SeqCUDA(Vec xin,NormType type,PetscReal* z)
{
  const PetscScalar *xx;
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n;
  PetscBLASInt      one = 1, bn = PetscBLASIntCast(n);
  CUSPARRAY         *xarray;

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    /*ierr = VecCUDACopyToGPU(xin);CHKERRQ(ierr);*/
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    try {
      *z = cusp::blas::nrm2(*xarray);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    }
    ierr = WaitForGPU();CHKERRCUDA(ierr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = PetscLogFlops(PetscMax(2.0*n-1,0.0));CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    PetscInt     i;
    PetscReal    max = 0.0,tmp;

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
    /*ierr = VecCUDACopyToGPU(xin);CHKERRQ(ierr);*/
    ierr = VecCUDAGetArrayRead(xin,&xarray);CHKERRQ(ierr);
#if defined(PETSC_USE_SCALAR_SINGLE)
    *z = cublasSasum(bn,VecCUDACastToRawPtr(*xarray),one);
#else
    *z = cublasDasum(bn,VecCUDACastToRawPtr(*xarray),one);
#endif
    ierr = cublasGetError();CHKERRCUDA(ierr);
    ierr = VecCUDARestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRCUDA(ierr);
    ierr = PetscLogFlops(PetscMax(n-1.0,0.0));CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    ierr = VecNorm_SeqCUDA(xin,NORM_1,z);CHKERRQ(ierr);
    ierr = VecNorm_SeqCUDA(xin,NORM_2,z+1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*the following few functions should be modified to actually work with the GPU so they don't force unneccesary allocation of CPU memory */

#undef __FUNCT__
#define __FUNCT__ "VecSetRandom_SeqCUDA"
PetscErrorCode VecSetRandom_SeqCUDA(Vec xin,PetscRandom r)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecSetRandom_Seq(xin,r);CHKERRQ(ierr);
  if (xin->valid_GPU_array != PETSC_CUDA_UNALLOCATED){
    xin->valid_GPU_array = PETSC_CUDA_CPU;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecResetArray_SeqCUDA"
PetscErrorCode VecResetArray_SeqCUDA(Vec vin)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecResetArray_Seq(vin);CHKERRQ(ierr);
  if (vin->valid_GPU_array != PETSC_CUDA_UNALLOCATED){
    vin->valid_GPU_array = PETSC_CUDA_CPU;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPlaceArray_SeqCUDA"
PetscErrorCode VecPlaceArray_SeqCUDA(Vec vin,const PetscScalar *a)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecPlaceArray_Seq(vin,a);CHKERRQ(ierr);
  if (vin->valid_GPU_array != PETSC_CUDA_UNALLOCATED){
    vin->valid_GPU_array = PETSC_CUDA_CPU;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecReplaceArray_SeqCUDA"
PetscErrorCode VecReplaceArray_SeqCUDA(Vec vin,const PetscScalar *a)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecReplaceArray_Seq(vin,a);CHKERRQ(ierr);
  if (vin->valid_GPU_array != PETSC_CUDA_UNALLOCATED){
    vin->valid_GPU_array = PETSC_CUDA_CPU;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecCreateSeqCUDA"
/*@
   VecCreateSeqCUDA - Creates a standard, sequential array-style vector.

   Collective on MPI_Comm

   Input Parameter:
+  comm - the communicator, should be PETSC_COMM_SELF
-  n - the vector length

   Output Parameter:
.  V - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   Level: intermediate

   Concepts: vectors^creating sequential

.seealso: VecCreateMPI(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecCreateSeqCUDA(MPI_Comm comm,PetscInt n,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,n);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECSEQCUDA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*The following template functions are for VecDotNorm2_SeqCUDA.  Note that there is no complex support as currently written*/
template <typename T>
struct cudadotnormcalculate : thrust::unary_function<T,T>
{
	__host__ __device__
	T operator()(T x)
	{
		return thrust::make_tuple(thrust::get<0>(x)*thrust::get<1>(x),thrust::get<1>(x)*thrust::get<1>(x));
	}
};

template <typename T>
struct cudadotnormreduce : thrust::binary_function<T,T,T>
{
	__host__ __device__
	T operator()(T x,T y)
	{
		return thrust::make_tuple(thrust::get<0>(x)+thrust::get<0>(y),thrust::get<1>(x)+thrust::get<1>(y));
	}
};

#undef __FUNCT__
#define __FUNCT__ "VecDotNorm2_SeqCUDA"
PetscErrorCode VecDotNorm2_SeqCUDA(Vec s, Vec t, PetscScalar *dp, PetscScalar *nm)
{
  PetscErrorCode                         ierr;
  PetscScalar                            zero = 0.0,n=s->map->n;
  thrust::tuple<PetscScalar,PetscScalar> result;
  CUSPARRAY                              *sarray,*tarray;

  PetscFunctionBegin;
  /*ierr = VecCUDACopyToGPU(s);CHKERRQ(ierr);
   ierr = VecCUDACopyToGPU(t);CHKERRQ(ierr);*/
  ierr = VecCUDAGetArrayRead(s,&sarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(t,&tarray);CHKERRQ(ierr);
  try {
    result = thrust::transform_reduce(
		 thrust::make_zip_iterator(
		     thrust::make_tuple(
                         sarray->begin(),
			 tarray->begin())),
		 thrust::make_zip_iterator(
                     thrust::make_tuple(
			 sarray->end(),
			 tarray->end())),
		  cudadotnormcalculate<thrust::tuple<PetscScalar,PetscScalar> >(),
		  thrust::make_tuple(zero,zero), /*init */
		  cudadotnormreduce<thrust::tuple<PetscScalar, PetscScalar> >()); /* binary function */
    *dp = thrust::get<0>(result);
    *nm = thrust::get<1>(result);
  } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
  }
  ierr = VecCUDARestoreArrayRead(s,&sarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayRead(t,&tarray);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUDA(ierr);
  ierr = PetscLogFlops(4.0*n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDuplicate_SeqCUDA"
PetscErrorCode VecDuplicate_SeqCUDA(Vec win,Vec *V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateSeqCUDA(((PetscObject)win)->comm,win->map->n,V);CHKERRQ(ierr);
  if (win->mapping) {
    ierr = PetscObjectReference((PetscObject)win->mapping);CHKERRQ(ierr);
    (*V)->mapping = win->mapping;
  }
  if (win->bmapping) {
    ierr = PetscObjectReference((PetscObject)win->bmapping);CHKERRQ(ierr);
    (*V)->bmapping = win->bmapping;
  }
  (*V)->map->bs = win->map->bs;
  ierr = PetscOListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*V))->olist);CHKERRQ(ierr);
  ierr = PetscFListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*V))->qlist);CHKERRQ(ierr);

  (*V)->stash.ignorenegidx = win->stash.ignorenegidx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDestroy_SeqCUDA"
PetscErrorCode VecDestroy_SeqCUDA(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  try {
    if (v->spptr) {
      delete ((Vec_CUDA *)v->spptr)->GPUarray;
      delete (Vec_CUDA *)v->spptr;
    }
  } catch(char* ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
  }
  ierr = VecDestroy_Seq(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "VecCreate_SeqCUDA"
PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_SeqCUDA(Vec V)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)V)->comm,&size);CHKERRQ(ierr);
  if  (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQCUDA on more than one process");
  ierr = VecCreate_Seq_Private(V,0);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)V,VECSEQCUDA);CHKERRQ(ierr);
  V->ops->dot             = VecDot_SeqCUDA;
  V->ops->norm            = VecNorm_SeqCUDA;
  V->ops->tdot            = VecTDot_SeqCUDA;
  V->ops->scale           = VecScale_SeqCUDA;
  V->ops->copy            = VecCopy_SeqCUDA;
  V->ops->set             = VecSet_SeqCUDA;
  V->ops->swap            = VecSwap_SeqCUDA;
  V->ops->axpy            = VecAXPY_SeqCUDA;
  V->ops->axpby           = VecAXPBY_SeqCUDA;
  V->ops->axpbypcz        = VecAXPBYPCZ_SeqCUDA;
  V->ops->pointwisemult   = VecPointwiseMult_SeqCUDA;
  V->ops->pointwisedivide = VecPointwiseDivide_SeqCUDA;
  V->ops->setrandom       = VecSetRandom_SeqCUDA;
  V->ops->dot_local       = VecDot_SeqCUDA;
  V->ops->tdot_local      = VecTDot_SeqCUDA;
  V->ops->norm_local      = VecNorm_SeqCUDA;
  V->ops->maxpy           = VecMAXPY_SeqCUDA;
  V->ops->mdot            = VecMDot_SeqCUDA;
  V->ops->aypx            = VecAYPX_SeqCUDA;
  V->ops->waxpy           = VecWAXPY_SeqCUDA;
  V->ops->dotnorm2        = VecDotNorm2_SeqCUDA;
  V->ops->placearray      = VecPlaceArray_SeqCUDA;
  V->ops->replacearray    = VecReplaceArray_SeqCUDA;
  V->ops->resetarray      = VecResetArray_SeqCUDA;
  V->ops->destroy         = VecDestroy_SeqCUDA;
  V->ops->duplicate       = VecDuplicate_SeqCUDA;
  V->valid_GPU_array      = PETSC_CUDA_UNALLOCATED;
  PetscFunctionReturn(0);
}
EXTERN_C_END
