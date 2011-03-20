
/*
   Implements the sequential cusp vectors.
*/

#include <petscconf.h>
PETSC_CUDA_EXTERN_C_BEGIN
#include <private/vecimpl.h>          /*I "petscvec.h" I*/
#include <../src/vec/vec/impls/dvecimpl.h>
PETSC_CUDA_EXTERN_C_END
#include <../src/vec/vec/impls/seq/seqcusp/cuspvecimpl.h>

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
#define __FUNCT__ "VecCUSPAllocateCheck_Public"
PetscErrorCode VecCUSPAllocateCheck_Public(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUSPAllocateCheck(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCUSPCopyToGPU_Public"
PetscErrorCode VecCUSPCopyToGPU_Public(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUSPCopyToGPU(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

struct  _p_PetscCUSPIndices {
  CUSPINTARRAYCPU indicesCPU;
  CUSPINTARRAYGPU indicesGPU;
};


#undef __FUNCT__
#define __FUNCT__ "PetscCUSPIndicesCreate"
/*
    PetscCUSPIndicesCreate - creates the data structure needed by VecCUSPCopyToGPUSome_Public()

   Input Parameters:
+    n - the number of indices
-    indices - integer list of indices

   Output Parameter:
.    ci - the CUSPIndices object suitable to pass to VecCUSPCopyToGPUSome_Public()

.seealso: PetscCUSPIndicesDestroy(), VecCUSPCopyToGPUSome_Public()
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
    PetscCUSPIndicesDestroy - destroys the data structure needed by VecCUSPCopyToGPUSome_Public()

   Input Parameters:
.    ci - the CUSPIndices object suitable to pass to VecCUSPCopyToGPUSome_Public()

.seealso: PetscCUSPIndicesCreate(), VecCUSPCopyToGPUSome_Public()
*/
PetscErrorCode PetscCUSPIndicesDestroy(PetscCUSPIndices ci)
{
  PetscFunctionBegin;
  if (!ci) PetscFunctionReturn(0);
  try {
    delete ci;
  } catch(char* ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCUSPCopyToGPUSome_Public"
/*
    VecCUSPCopyToGPUSome_Public - Copies certain entries down to the GPU from the CPU of a vector

   Input Parameters:
+    v - the vector
-    indices - the requested indices, this should be created with CUSPIndicesCreate()

*/
PetscErrorCode VecCUSPCopyToGPUSome_Public(Vec v, PetscCUSPIndices ci)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = VecCUSPCopyToGPUSome(v,&ci->indicesCPU,&ci->indicesGPU);CHKERRCUSP(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "VecCUSPCopyFromGPU"
/*@C
     VecCUSPCopyFromGPU - Copies a vector from the GPU to the CPU unless we already have an up-to-date copy on the CPU
@*/
PetscErrorCode VecCUSPCopyFromGPU(Vec v)
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
  if (v->valid_GPU_array == PETSC_CUSP_GPU){
    GPUvector  = ((Vec_CUSP*)v->spptr)->GPUarray;
    ierr       = PetscLogEventBegin(VEC_CUSPCopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    try{
      thrust::copy(GPUvector->begin(),GPUvector->end(),*(PetscScalar**)v->data);
      ierr = WaitForGPU();CHKERRCUSP(ierr);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = PetscLogEventEnd(VEC_CUSPCopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    v->valid_GPU_array = PETSC_CUSP_BOTH;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCUSPCopyFromGPUSome"
/* Note that this function only copies *some* of the values up from the GPU to CPU,
   which means that we need recombine the data at some point before using any of the standard functions.
   We could add another few flag-types to keep track of this, or treat things like VecGetArray VecRestoreArray
   where you have to always call in pairs
*/
PetscErrorCode VecCUSPCopyFromGPUSome(Vec v,CUSPINTARRAYCPU *indicesCPU,CUSPINTARRAYGPU *indicesGPU)
{
  Vec_Seq        *s;
  PetscInt       n = v->map->n;
  PetscScalar    *array;
  CUSPARRAY      *varray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUSPAllocateCheck(v);CHKERRCUSP(ierr);
  s = (Vec_Seq*)v->data;
  if (s->array == 0){
    ierr               = PetscMalloc(n*sizeof(PetscScalar),&array);CHKERRQ(ierr);
    ierr               = PetscLogObjectMemory(v,n*sizeof(PetscScalar));CHKERRQ(ierr);
    s->array           = array;
    s->array_allocated = array;
  }
  if (v->valid_GPU_array == PETSC_CUSP_GPU) {
    ierr = PetscLogEventBegin(VEC_CUSPCopyFromGPUSome,v,0,0,0);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(v,&varray);CHKERRQ(ierr);
    thrust::copy(
		 thrust::make_permutation_iterator(varray->begin(),indicesGPU->begin()),
		 thrust::make_permutation_iterator(varray->begin(),indicesGPU->end()),
		 thrust::make_permutation_iterator(s->array,indicesCPU->begin()));
    ierr = VecCUSPRestoreArrayRead(v,&varray);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(VEC_CUSPCopyFromGPUSome,v,0,0,0);CHKERRQ(ierr);
  }
  /*v->valid_GPU_array = PETSC_CUSP_CPU; */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecCUSPCopyFromGPUSome_Public"
/*
  VecCUSPCopyFromGPUSome_Public - Copies certain entries up to the CPU from the GPU of a vector

  Input Parameters:
 +    v - the vector
 -    indices - the requested indices, this should be created with CUSPIndicesCreate()
*/
PetscErrorCode VecCUSPCopyFromGPUSome_Public(Vec v, PetscCUSPIndices ci)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUSPCopyFromGPUSome(v,&ci->indicesCPU,&ci->indicesGPU);CHKERRCUSP(ierr);
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
   detail::assert_same_dimensions(x,y);
   aypx(x.begin(),x.end(),y.begin(),alpha);
 }
}
}

#undef __FUNCT__
#define __FUNCT__ "VecAYPX_SeqCUSP"
PetscErrorCode VecAYPX_SeqCUSP(Vec yin, PetscScalar alpha, Vec xin)
{
  CUSPARRAY      *xarray,*yarray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha != 0.0) {
    ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    try{
      cusp::blas::aypx(*xarray,*yarray,alpha);
      ierr = WaitForGPU();CHKERRCUSP(ierr);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
    ierr = PetscLogFlops(2.0*yin->map->n);CHKERRQ(ierr);
   }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecAXPY_SeqCUSP"
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
    } catch(char* ex) {
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

#undef __FUNCT__
#define __FUNCT__ "VecPointwiseDivide_SeqCUSP"
PetscErrorCode VecPointwiseDivide_SeqCUSP(Vec win, Vec xin, Vec yin)
{
  CUSPARRAY      *warray,*xarray,*yarray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayWrite(win,&warray);CHKERRQ(ierr);
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
	VecCUSPPointwiseDivide());
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  } catch(char* ex) {
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

#undef __FUNCT__
#define __FUNCT__ "VecWAXPY_SeqCUSP"
PetscErrorCode VecWAXPY_SeqCUSP(Vec win,PetscScalar alpha,Vec xin, Vec yin)
{
  CUSPARRAY      *xarray,*yarray,*warray;
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
        } catch(char* ex) {
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
        } catch(char* ex) {
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
		thrust::make_constant_iterator(alpha,0),
		xarray->begin())),
            thrust::make_zip_iterator(
              thrust::make_tuple(
		warray->end(),
		yarray->end(),
		thrust::make_constant_iterator(alpha,win->map->n),
		xarray->end())),
            VecCUSPWAXPY());
        } catch(char* ex) {
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
    /*y += a1*x1 +a2*x2 + 13*x3 */
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
#undef __FUNCT__
#define __FUNCT__ "VecMAXPY_SeqCUSP"
PetscErrorCode VecMAXPY_SeqCUSP(Vec xin, PetscInt nv,const PetscScalar *alpha,Vec *y)
{
  PetscErrorCode    ierr;
  CUSPARRAY         *xarray,*yy0,*yy1,*yy2,*yy3;
  PetscInt          n = xin->map->n,j,j_rem;
  PetscScalar       alpha0,alpha1,alpha2,alpha3;

  PetscFunctionBegin;
  ierr = PetscLogFlops(nv*2.0*n);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
  switch (j_rem=nv&0x3) {
  case 3:
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha2 = alpha[2];
    alpha += 3;
    ierr = VecCUSPGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(y[2],&yy2);CHKERRQ(ierr);
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
	VecCUSPMAXPY3());
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = VecCUSPRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(y[2],&yy2);CHKERRQ(ierr);
    y     += 3;
    break;
  case 2:
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha +=2;
    ierr = VecCUSPGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
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
	VecCUSPMAXPY2());
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    y     +=2;
    break;
  case 1:
    alpha0 = *alpha++;
    ierr = VecAXPY_SeqCUSP(xin,alpha0,y[0]);
    y     +=1;
    break;
  }
  for (j=j_rem; j<nv; j+=4) {
    alpha0 = alpha[0];
    alpha1 = alpha[1];
    alpha2 = alpha[2];
    alpha3 = alpha[3];
    alpha  += 4;
    ierr = VecCUSPGetArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(y[2],&yy2);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(y[3],&yy3);CHKERRQ(ierr);
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
	VecCUSPMAXPY4());
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = VecCUSPRestoreArrayRead(y[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(y[1],&yy1);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(y[2],&yy2);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(y[3],&yy3);CHKERRQ(ierr);
    y      += 4;
  }
  ierr = VecCUSPRestoreArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecDot_SeqCUSP"
PetscErrorCode VecDot_SeqCUSP(Vec xin,Vec yin,PetscScalar *z)
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
    ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(yin,&yarray);CHKERRQ(ierr);
    try {
      *z = cusp::blas::dot(*xarray,*yarray);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
  }
#endif
 ierr = WaitForGPU();CHKERRCUSP(ierr);
 if (xin->map->n >0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
 ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
 ierr = VecCUSPRestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
 PetscFunctionReturn(0);
}

/*The following few template functions are for VecMDot_SeqCUSP*/

template <typename T1,typename T2>
struct cuspmult2 : thrust::unary_function<T1,T2>
{
	__host__ __device__
	T2 operator()(T1 x)
	{
		return thrust::make_tuple(thrust::get<0>(x)*thrust::get<1>(x),thrust::get<0>(x)*thrust::get<2>(x));
	}
};

template <typename T>
struct cuspadd2 : thrust::binary_function<T,T,T>
{
	__host__ __device__
	T operator()(T x,T y)
	{
		return thrust::make_tuple(thrust::get<0>(x)+thrust::get<0>(y),thrust::get<1>(x)+thrust::get<1>(y));
	}
};

template <typename T1,typename T2>
struct cuspmult3 : thrust::unary_function<T1,T2>
{
	__host__ __device__
	T2 operator()(T1 x)
	{
	  return thrust::make_tuple(thrust::get<0>(x)*thrust::get<1>(x),thrust::get<0>(x)*thrust::get<2>(x),thrust::get<0>(x)*thrust::get<3>(x));
	}
};

template <typename T>
struct cuspadd3 : thrust::binary_function<T,T,T>
{
	__host__ __device__
	T operator()(T x,T y)
	{
	  return thrust::make_tuple(thrust::get<0>(x)+thrust::get<0>(y),thrust::get<1>(x)+thrust::get<1>(y),thrust::get<2>(x)+thrust::get<2>(y));
	}
};
	template <typename T1,typename T2>
struct cuspmult4 : thrust::unary_function<T1,T2>
{
	__host__ __device__
	T2 operator()(T1 x)
	{
	  return thrust::make_tuple(thrust::get<0>(x)*thrust::get<1>(x),thrust::get<0>(x)*thrust::get<2>(x),thrust::get<0>(x)*thrust::get<3>(x),thrust::get<0>(x)*thrust::get<4>(x));
	}
};

template <typename T>
struct cuspadd4 : thrust::binary_function<T,T,T>
{
	__host__ __device__
	T operator()(T x,T y)
	{
	  return thrust::make_tuple(thrust::get<0>(x)+thrust::get<0>(y),thrust::get<1>(x)+thrust::get<1>(y),thrust::get<2>(x)+thrust::get<2>(y),thrust::get<3>(x)+thrust::get<3>(y));
	}
};


#undef __FUNCT__
#define __FUNCT__ "VecMDot_SeqCUSP"
PetscErrorCode VecMDot_SeqCUSP(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
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
  ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
  switch(j_rem=nv&0x3) {
  case 3:
    ierr = VecCUSPGetArrayRead(yin[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(yin[1],&yy1);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(yin[2],&yy2);CHKERRQ(ierr);
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
		     cuspmult3<thrust::tuple<PetscScalar,PetscScalar,PetscScalar,PetscScalar>, thrust::tuple<PetscScalar,PetscScalar,PetscScalar> >(),
		     thrust::make_tuple(zero,zero,zero), /*init */
		     cuspadd3<thrust::tuple<PetscScalar,PetscScalar,PetscScalar> >()); /* binary function */
      z[0] = thrust::get<0>(result3);
      z[1] = thrust::get<1>(result3);
      z[2] = thrust::get<2>(result3);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    z    += 3;
    ierr = VecCUSPRestoreArrayRead(yin[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(yin[1],&yy1);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(yin[2],&yy2);CHKERRQ(ierr);
    yin  += 3;
    break;
  case 2:
    ierr = VecCUSPGetArrayRead(yin[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(yin[1],&yy1);CHKERRQ(ierr);
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
		    cuspmult2<thrust::tuple<PetscScalar,PetscScalar,PetscScalar>, thrust::tuple<PetscScalar,PetscScalar> >(),
		    thrust::make_tuple(zero,zero), /*init */
		    cuspadd2<thrust::tuple<PetscScalar, PetscScalar> >()); /* binary function */
      z[0] = thrust::get<0>(result2);
      z[1] = thrust::get<1>(result2);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    z    += 2;
    ierr = VecCUSPRestoreArrayRead(yin[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(yin[1],&yy1);CHKERRQ(ierr);
    yin  += 2;
    break;
  case 1:
    ierr =  VecDot_SeqCUSP(xin,yin[0],&z[0]);CHKERRQ(ierr);
    z    += 1;
    yin  += 1;
    break;
  }
  for (j=j_rem; j<nv; j+=4) {
    ierr = VecCUSPGetArrayRead(yin[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(yin[1],&yy1);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(yin[2],&yy2);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayRead(yin[3],&yy3);CHKERRQ(ierr);
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
		     cuspmult4<thrust::tuple<PetscScalar,PetscScalar,PetscScalar,PetscScalar,PetscScalar>, thrust::tuple<PetscScalar,PetscScalar,PetscScalar,PetscScalar> >(),
		     thrust::make_tuple(zero,zero,zero,zero), /*init */
		     cuspadd4<thrust::tuple<PetscScalar,PetscScalar,PetscScalar,PetscScalar> >()); /* binary function */
      z[0] = thrust::get<0>(result4);
      z[1] = thrust::get<1>(result4);
      z[2] = thrust::get<2>(result4);
      z[3] = thrust::get<3>(result4);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    z    += 4;
    ierr = VecCUSPRestoreArrayRead(yin[0],&yy0);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(yin[1],&yy1);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(yin[2],&yy2);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayRead(yin[3],&yy3);CHKERRQ(ierr);
    yin  += 4;
  }
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = PetscLogFlops(PetscMax(nv*(2.0*n-1),0.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecSet_SeqCUSP"
PetscErrorCode VecSet_SeqCUSP(Vec xin,PetscScalar alpha)
{
  CUSPARRAY      *xarray;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* if there's a faster way to do the case alpha=0.0 on the GPU we should do that*/
  ierr = VecCUSPGetArrayWrite(xin,&xarray);CHKERRQ(ierr);
  try {
    cusp::blas::fill(*xarray,alpha);
  } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = VecCUSPRestoreArrayWrite(xin,&xarray);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecScale_SeqCUSP"
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
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = VecCUSPRestoreArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
  }
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecTDot_SeqCUSP"
PetscErrorCode VecTDot_SeqCUSP(Vec xin,Vec yin,PetscScalar *z)
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
 ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
 ierr = VecCUSPGetArrayRead(yin,&yarray);CHKERRQ(ierr);
 try {
   *z = cusp::blas::dot(*xarray,*yarray);
 } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
 }
#endif
 ierr = WaitForGPU();CHKERRCUSP(ierr);
  if (xin->map->n > 0) {
    ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
  }
  ierr = VecCUSPRestoreArrayRead(yin,&yarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "VecCopy_SeqCUSP"
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
       } catch(char* ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
      }
      ierr = WaitForGPU();CHKERRCUSP(ierr);
      ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
      ierr = VecCUSPRestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);

    } else if (xin->valid_GPU_array == PETSC_CUSP_CPU || xin->valid_GPU_array == PETSC_CUSP_UNALLOCATED) {
      /* copy in CPU if we are on the CPU*/
      ierr = VecCopy_Seq(xin,yin);CHKERRQ(ierr);
    } else if (xin->valid_GPU_array == PETSC_CUSP_BOTH) {
      /* if xin is valid in both places, see where yin is and copy there (because it's probably where we'll want to next use it) */
      if (yin->valid_GPU_array == PETSC_CUSP_CPU) {
	/* copy in CPU */
	ierr = VecCopy_Seq(xin,yin);CHKERRQ(ierr);

      } else if (yin->valid_GPU_array == PETSC_CUSP_GPU) {
	/* copy in GPU */
        ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUSPGetArrayWrite(yin,&yarray);CHKERRQ(ierr);
	try {
	  cusp::blas::copy(*xarray,*yarray);
	  ierr = WaitForGPU();CHKERRCUSP(ierr);
	} catch(char* ex) {
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
	} catch(char* ex) {
	  SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
	}
        ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
        ierr = VecCUSPRestoreArrayWrite(yin,&yarray);CHKERRQ(ierr);
      } else {
	ierr = VecCopy_Seq(xin,yin);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecSwap_SeqCUSP"
PetscErrorCode VecSwap_SeqCUSP(Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  PetscBLASInt   one = 1,bn = PetscBLASIntCast(xin->map->n);
  CUSPARRAY      *xarray,*yarray;

  PetscFunctionBegin;
  if (xin != yin) {
    ierr = VecCUSPGetArrayReadWrite(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPGetArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
#if defined(PETSC_USE_REAL_SINGLE)
    cublasSswap(bn,VecCUSPCastToRawPtr(*xarray),one,VecCUSPCastToRawPtr(*yarray),one);
#else
    cublasDswap(bn,VecCUSPCastToRawPtr(*xarray),one,VecCUSPCastToRawPtr(*yarray),one);
#endif
    ierr = cublasGetError();CHKERRCUSP(ierr);
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
#undef __FUNCT__
#define __FUNCT__ "VecAXPBY_SeqCUSP"
PetscErrorCode VecAXPBY_SeqCUSP(Vec yin,PetscScalar alpha,PetscScalar beta,Vec xin)
{
  PetscErrorCode    ierr;
  PetscInt          n = yin->map->n;
  PetscScalar       a = alpha,b = beta;
  CUSPARRAY         *xarray,*yarray;

  PetscFunctionBegin;
  if (a == 0.0) {
    ierr = VecScale_SeqCUSP(yin,beta);CHKERRQ(ierr);
  } else if (b == 1.0) {
    ierr = VecAXPY_SeqCUSP(yin,alpha,xin);CHKERRQ(ierr);
  } else if (a == 1.0) {
    ierr = VecAYPX_SeqCUSP(yin,beta,xin);CHKERRQ(ierr);
  } else if (b == 0.0) {
    ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = VecCUSPRestoreArrayReadWrite(yin,&yarray);CHKERRQ(ierr);
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
	VecCUSPAX());
    } catch(char* ex) {
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
    } catch(char* ex) {
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

#undef __FUNCT__
#define __FUNCT__ "VecAXPBYPCZ_SeqCUSP"
PetscErrorCode VecAXPBYPCZ_SeqCUSP(Vec zin,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec xin,Vec yin)
{
  PetscErrorCode     ierr;
  PetscInt           n = zin->map->n;
  CUSPARRAY          *xarray,*yarray,*zarray;

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
	VecCUSPXPBYPCZ());
    } catch(char* ex) {
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
	VecCUSPAXPBYPZ());
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = PetscLogFlops(4.0*n);CHKERRQ(ierr);
  } else {
    try {
      cusp::blas::axpbypcz(*xarray,*yarray,*zarray,*zarray,alpha,beta,gamma);
    } catch(char* ex) {
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

#undef __FUNCT__
#define __FUNCT__ "VecPointwiseMult_SeqCUSP"
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
  } catch(char* ex) {
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

#undef __FUNCT__
#define __FUNCT__ "VecNorm_SeqCUSP"
PetscErrorCode VecNorm_SeqCUSP(Vec xin,NormType type,PetscReal* z)
{
  const PetscScalar *xx;
  PetscErrorCode    ierr;
  PetscInt          n = xin->map->n;
  PetscBLASInt      one = 1, bn = PetscBLASIntCast(n);
  CUSPARRAY         *xarray;

  PetscFunctionBegin;
  if (type == NORM_2 || type == NORM_FROBENIUS) {
    ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
    try {
      *z = cusp::blas::nrm2(*xarray);
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ierr = WaitForGPU();CHKERRCUSP(ierr);
    ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
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
    ierr = VecCUSPGetArrayRead(xin,&xarray);CHKERRQ(ierr);
#if defined(PETSC_USE_REAL_SINGLE)
    *z = cublasSasum(bn,VecCUSPCastToRawPtr(*xarray),one);
#else
    *z = cublasDasum(bn,VecCUSPCastToRawPtr(*xarray),one);
#endif
    ierr = cublasGetError();CHKERRCUSP(ierr);
    ierr = VecCUSPRestoreArrayRead(xin,&xarray);CHKERRQ(ierr);
    ierr = WaitForGPU();CHKERRCUSP(ierr);
    ierr = PetscLogFlops(PetscMax(n-1.0,0.0));CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    ierr = VecNorm_SeqCUSP(xin,NORM_1,z);CHKERRQ(ierr);
    ierr = VecNorm_SeqCUSP(xin,NORM_2,z+1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*the following few functions should be modified to actually work with the GPU so they don't force unneccesary allocation of CPU memory */

#undef __FUNCT__
#define __FUNCT__ "VecSetRandom_SeqCUSP"
PetscErrorCode VecSetRandom_SeqCUSP(Vec xin,PetscRandom r)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecSetRandom_Seq(xin,r);CHKERRQ(ierr);
  if (xin->valid_GPU_array != PETSC_CUSP_UNALLOCATED){
    xin->valid_GPU_array = PETSC_CUSP_CPU;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecResetArray_SeqCUSP"
PetscErrorCode VecResetArray_SeqCUSP(Vec vin)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecCUSPCopyFromGPU(vin);CHKERRCUSP(ierr);
  ierr = VecResetArray_Seq(vin);CHKERRQ(ierr);
  if (vin->valid_GPU_array != PETSC_CUSP_UNALLOCATED){
    vin->valid_GPU_array = PETSC_CUSP_CPU;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPlaceArray_SeqCUSP"
PetscErrorCode VecPlaceArray_SeqCUSP(Vec vin,const PetscScalar *a)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecCUSPCopyFromGPU(vin);CHKERRCUSP(ierr);
  ierr = VecPlaceArray_Seq(vin,a);CHKERRQ(ierr);
  if (vin->valid_GPU_array != PETSC_CUSP_UNALLOCATED){
    vin->valid_GPU_array = PETSC_CUSP_CPU;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecReplaceArray_SeqCUSP"
PetscErrorCode VecReplaceArray_SeqCUSP(Vec vin,const PetscScalar *a)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecCUSPCopyFromGPU(vin);CHKERRCUSP(ierr);
  ierr = VecReplaceArray_Seq(vin,a);CHKERRQ(ierr);
  if (vin->valid_GPU_array != PETSC_CUSP_UNALLOCATED){
    vin->valid_GPU_array = PETSC_CUSP_CPU;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecCreateSeqCUSP"
/*@
   VecCreateSeqCUSP - Creates a standard, sequential array-style vector.

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
PetscErrorCode  VecCreateSeqCUSP(MPI_Comm comm,PetscInt n,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,n);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECSEQCUSP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*The following template functions are for VecDotNorm2_SeqCUSP.  Note that there is no complex support as currently written*/
template <typename T>
struct cuspdotnormcalculate : thrust::unary_function<T,T>
{
	__host__ __device__
	T operator()(T x)
	{
		return thrust::make_tuple(thrust::get<0>(x)*thrust::get<1>(x),thrust::get<1>(x)*thrust::get<1>(x));
	}
};

template <typename T>
struct cuspdotnormreduce : thrust::binary_function<T,T,T>
{
	__host__ __device__
	T operator()(T x,T y)
	{
		return thrust::make_tuple(thrust::get<0>(x)+thrust::get<0>(y),thrust::get<1>(x)+thrust::get<1>(y));
	}
};

#undef __FUNCT__
#define __FUNCT__ "VecDotNorm2_SeqCUSP"
PetscErrorCode VecDotNorm2_SeqCUSP(Vec s, Vec t, PetscScalar *dp, PetscScalar *nm)
{
  PetscErrorCode                         ierr;
  PetscScalar                            zero = 0.0,n=s->map->n;
  thrust::tuple<PetscScalar,PetscScalar> result;
  CUSPARRAY                              *sarray,*tarray;

  PetscFunctionBegin;
  /*ierr = VecCUSPCopyToGPU(s);CHKERRQ(ierr);
   ierr = VecCUSPCopyToGPU(t);CHKERRQ(ierr);*/
  ierr = VecCUSPGetArrayRead(s,&sarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(t,&tarray);CHKERRQ(ierr);
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
		  cuspdotnormcalculate<thrust::tuple<PetscScalar,PetscScalar> >(),
		  thrust::make_tuple(zero,zero), /*init */
		  cuspdotnormreduce<thrust::tuple<PetscScalar, PetscScalar> >()); /* binary function */
    *dp = thrust::get<0>(result);
    *nm = thrust::get<1>(result);
  } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  ierr = VecCUSPRestoreArrayRead(s,&sarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayRead(t,&tarray);CHKERRQ(ierr);
  ierr = WaitForGPU();CHKERRCUSP(ierr);
  ierr = PetscLogFlops(4.0*n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDuplicate_SeqCUSP"
PetscErrorCode VecDuplicate_SeqCUSP(Vec win,Vec *V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateSeqCUSP(((PetscObject)win)->comm,win->map->n,V);CHKERRQ(ierr);
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
#define __FUNCT__ "VecDestroy_SeqCUSP"
PetscErrorCode VecDestroy_SeqCUSP(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  try {
    if (v->spptr) {
      delete ((Vec_CUSP *)v->spptr)->GPUarray;
      delete (Vec_CUSP *)v->spptr;
    }
  } catch(char* ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  ierr = VecDestroy_Seq(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "VecCreate_SeqCUSP"
PetscErrorCode  VecCreate_SeqCUSP(Vec V)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)V)->comm,&size);CHKERRQ(ierr);
  if  (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQCUSP on more than one process");
  ierr = VecCreate_Seq_Private(V,0);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)V,VECSEQCUSP);CHKERRQ(ierr);
  V->ops->dot             = VecDot_SeqCUSP;
  V->ops->norm            = VecNorm_SeqCUSP;
  V->ops->tdot            = VecTDot_SeqCUSP;
  V->ops->scale           = VecScale_SeqCUSP;
  V->ops->copy            = VecCopy_SeqCUSP;
  V->ops->set             = VecSet_SeqCUSP;
  V->ops->swap            = VecSwap_SeqCUSP;
  V->ops->axpy            = VecAXPY_SeqCUSP;
  V->ops->axpby           = VecAXPBY_SeqCUSP;
  V->ops->axpbypcz        = VecAXPBYPCZ_SeqCUSP;
  V->ops->pointwisemult   = VecPointwiseMult_SeqCUSP;
  V->ops->pointwisedivide = VecPointwiseDivide_SeqCUSP;
  V->ops->setrandom       = VecSetRandom_SeqCUSP;
  V->ops->dot_local       = VecDot_SeqCUSP;
  V->ops->tdot_local      = VecTDot_SeqCUSP;
  V->ops->norm_local      = VecNorm_SeqCUSP;
  V->ops->maxpy           = VecMAXPY_SeqCUSP;
  V->ops->mdot            = VecMDot_SeqCUSP;
  V->ops->aypx            = VecAYPX_SeqCUSP;
  V->ops->waxpy           = VecWAXPY_SeqCUSP;
  V->ops->dotnorm2        = VecDotNorm2_SeqCUSP;
  V->ops->placearray      = VecPlaceArray_SeqCUSP;
  V->ops->replacearray    = VecReplaceArray_SeqCUSP;
  V->ops->resetarray      = VecResetArray_SeqCUSP;
  V->ops->destroy         = VecDestroy_SeqCUSP;
  V->ops->duplicate       = VecDuplicate_SeqCUSP;
  V->valid_GPU_array      = PETSC_CUSP_UNALLOCATED;
  PetscFunctionReturn(0);
}
EXTERN_C_END
