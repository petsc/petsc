/*
   Implements the sequential ViennaCL vectors.
*/

#include <petscconf.h>
#include <petsc/private/vecimpl.h>          /*I "petscvec.h" I*/
#include <../src/vec/vec/impls/dvecimpl.h>
#include <../src/vec/vec/impls/seq/seqviennacl/viennaclvecimpl.h>

#include <vector>

#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"
#include "viennacl/ocl/backend.hpp"


#undef __FUNCT__
#define __FUNCT__ "VecViennaCLGetArrayReadWrite"
PETSC_EXTERN PetscErrorCode VecViennaCLGetArrayReadWrite(Vec v, ViennaCLVector **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *a   = 0;
  ierr = VecViennaCLCopyToGPU(v);CHKERRQ(ierr);
  *a   = ((Vec_ViennaCL*)v->spptr)->GPUarray;
  ViennaCLWaitForGPU();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecViennaCLRestoreArrayReadWrite"
PETSC_EXTERN PetscErrorCode VecViennaCLRestoreArrayReadWrite(Vec v, ViennaCLVector **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  v->valid_GPU_array = PETSC_VIENNACL_GPU;

  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecViennaCLGetArrayRead"
PETSC_EXTERN PetscErrorCode VecViennaCLGetArrayRead(Vec v, const ViennaCLVector **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *a   = 0;
  ierr = VecViennaCLCopyToGPU(v);CHKERRQ(ierr);
  *a   = ((Vec_ViennaCL*)v->spptr)->GPUarray;
  ViennaCLWaitForGPU();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecViennaCLRestoreArrayRead"
PETSC_EXTERN PetscErrorCode VecViennaCLRestoreArrayRead(Vec v, const ViennaCLVector **a)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecViennaCLGetArrayWrite"
PETSC_EXTERN PetscErrorCode VecViennaCLGetArrayWrite(Vec v, ViennaCLVector **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *a   = 0;
  ierr = VecViennaCLAllocateCheck(v);CHKERRQ(ierr);
  *a   = ((Vec_ViennaCL*)v->spptr)->GPUarray;
  ViennaCLWaitForGPU();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecViennaCLRestoreArrayWrite"
PETSC_EXTERN PetscErrorCode VecViennaCLRestoreArrayWrite(Vec v, ViennaCLVector **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  v->valid_GPU_array = PETSC_VIENNACL_GPU;

  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "PetscObjectViennaCLSetFromOptions"
PETSC_EXTERN PetscErrorCode PetscObjectViennaCLSetFromOptions(PetscObject obj)
{
  PetscErrorCode       ierr;
  PetscBool            flg;

  PetscFunctionBegin;
  ierr = PetscObjectOptionsBegin(obj);

  ierr = PetscOptionsHasName(NULL,NULL,"-viennacl_device_cpu",&flg);CHKERRQ(ierr);
  if (flg) {
    try {
      viennacl::ocl::set_context_device_type(0, CL_DEVICE_TYPE_CPU);
    } catch (std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
  }
  ierr = PetscOptionsHasName(NULL,NULL,"-viennacl_device_gpu",&flg);CHKERRQ(ierr);
  if (flg) {
    try {
      viennacl::ocl::set_context_device_type(0, CL_DEVICE_TYPE_GPU);
    } catch (std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
  }
  ierr = PetscOptionsHasName(NULL,NULL,"-viennacl_device_accelerator",&flg);CHKERRQ(ierr);
  if (flg) {
    try {
      viennacl::ocl::set_context_device_type(0, CL_DEVICE_TYPE_ACCELERATOR);
    } catch (std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
  }

  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecViennaCLAllocateCheckHost"
/*
    Allocates space for the vector array on the Host if it does not exist.
    Does NOT change the PetscViennaCLFlag for the vector
    Does NOT zero the ViennaCL array
 */
PETSC_EXTERN PetscErrorCode VecViennaCLAllocateCheckHost(Vec v)
{
  PetscErrorCode ierr;
  PetscScalar    *array;
  Vec_Seq        *s;
  PetscInt       n = v->map->n;

  PetscFunctionBegin;
  s    = (Vec_Seq*)v->data;
  ierr = VecViennaCLAllocateCheck(v);CHKERRQ(ierr);
  if (s->array == 0) {
    ierr               = PetscMalloc1(n,&array);CHKERRQ(ierr);
    ierr               = PetscLogObjectMemory((PetscObject)v,n*sizeof(PetscScalar));CHKERRQ(ierr);
    s->array           = array;
    s->array_allocated = array;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecViennaCLAllocateCheck"
/*
    Allocates space for the vector array on the GPU if it does not exist.
    Does NOT change the PetscViennaCLFlag for the vector
    Does NOT zero the ViennaCL array

 */
PetscErrorCode VecViennaCLAllocateCheck(Vec v)
{
  PetscErrorCode ierr;
  int            rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  // First allocate memory on the GPU if needed
  if (!v->spptr) {
    try {
      ierr = PetscObjectViennaCLSetFromOptions((PetscObject)v);CHKERRQ(ierr);
      v->spptr                            = new Vec_ViennaCL;
      ((Vec_ViennaCL*)v->spptr)->GPUarray = new ViennaCLVector((PetscBLASInt)v->map->n);

    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecViennaCLCopyToGPU"
/* Copies a vector from the CPU to the GPU unless we already have an up-to-date copy on the GPU */
PetscErrorCode VecViennaCLCopyToGPU(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecViennaCLAllocateCheck(v);CHKERRQ(ierr);
  if (v->map->n > 0) {
    if (v->valid_GPU_array == PETSC_VIENNACL_CPU) {
      ierr = PetscLogEventBegin(VEC_ViennaCLCopyToGPU,v,0,0,0);CHKERRQ(ierr);
      try {
        ViennaCLVector *vec = ((Vec_ViennaCL*)v->spptr)->GPUarray;
        viennacl::fast_copy(*(PetscScalar**)v->data, *(PetscScalar**)v->data + v->map->n, vec->begin());
        ViennaCLWaitForGPU();
      } catch(std::exception const & ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
      }
      ierr = PetscLogEventEnd(VEC_ViennaCLCopyToGPU,v,0,0,0);CHKERRQ(ierr);
      v->valid_GPU_array = PETSC_VIENNACL_BOTH;
    }
  }
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "VecViennaCLCopyFromGPU"
/*
     VecViennaCLCopyFromGPU - Copies a vector from the GPU to the CPU unless we already have an up-to-date copy on the CPU
*/
PetscErrorCode VecViennaCLCopyFromGPU(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecViennaCLAllocateCheckHost(v);CHKERRQ(ierr);
  if (v->valid_GPU_array == PETSC_VIENNACL_GPU) {
    ierr = PetscLogEventBegin(VEC_ViennaCLCopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    try {
      ViennaCLVector *vec = ((Vec_ViennaCL*)v->spptr)->GPUarray;
      viennacl::fast_copy(vec->begin(),vec->end(),*(PetscScalar**)v->data);
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = PetscLogEventEnd(VEC_ViennaCLCopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    v->valid_GPU_array = PETSC_VIENNACL_BOTH;
  }
  PetscFunctionReturn(0);
}


/* Copy on CPU */
#undef __FUNCT__
#define __FUNCT__ "VecCopy_SeqViennaCL_Private"
static PetscErrorCode VecCopy_SeqViennaCL_Private(Vec xin,Vec yin)
{
  PetscScalar       *ya;
  const PetscScalar *xa;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecViennaCLAllocateCheckHost(xin);
  ierr = VecViennaCLAllocateCheckHost(yin);
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
#define __FUNCT__ "VecSetRandom_SeqViennaCL_Private"
static PetscErrorCode VecSetRandom_SeqViennaCL_Private(Vec xin,PetscRandom r)
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
#define __FUNCT__ "VecDestroy_SeqViennaCL_Private"
static PetscErrorCode VecDestroy_SeqViennaCL_Private(Vec v)
{
  Vec_Seq        *vs = (Vec_Seq*)v->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectSAWsViewOff(v);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%D",v->map->n);
#endif
  if (vs->array_allocated) ierr = PetscFree(vs->array_allocated);CHKERRQ(ierr);
  ierr = PetscFree(vs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecResetArray_SeqViennaCL_Private"
static PetscErrorCode VecResetArray_SeqViennaCL_Private(Vec vin)
{
  Vec_Seq *v = (Vec_Seq*)vin->data;

  PetscFunctionBegin;
  v->array         = v->unplacedarray;
  v->unplacedarray = 0;
  PetscFunctionReturn(0);
}


/*MC
   VECSEQVIENNACL - VECSEQVIENNACL = "seqviennacl" - The basic sequential vector, modified to use ViennaCL

   Options Database Keys:
. -vec_type seqviennacl - sets the vector type to VECSEQVIENNACL during a call to VecSetFromOptions()

  Level: beginner

.seealso: VecCreate(), VecSetType(), VecSetFromOptions(), VecCreateSeqWithArray(), VECMPI, VecType, VecCreateMPI(), VecCreateSeq()
M*/


#undef __FUNCT__
#define __FUNCT__ "VecAYPX_SeqViennaCL"
PetscErrorCode VecAYPX_SeqViennaCL(Vec yin, PetscScalar alpha, Vec xin)
{
  const ViennaCLVector  *xgpu;
  ViennaCLVector        *ygpu;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
  ierr = VecViennaCLGetArrayReadWrite(yin,&ygpu);CHKERRQ(ierr);
  try {
    if (alpha != 0.0 && xin->map->n > 0) {
      *ygpu = *xgpu + alpha * *ygpu;
      ierr = PetscLogFlops(2.0*yin->map->n);CHKERRQ(ierr);
    } else {
      *ygpu = *xgpu;
    }
    ViennaCLWaitForGPU();
  } catch(std::exception const & ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
  }
  ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
  ierr = VecViennaCLRestoreArrayReadWrite(yin,&ygpu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecAXPY_SeqViennaCL"
PetscErrorCode VecAXPY_SeqViennaCL(Vec yin,PetscScalar alpha,Vec xin)
{
  const ViennaCLVector  *xgpu;
  ViennaCLVector        *ygpu;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (alpha != 0.0 && xin->map->n > 0) {
    ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLGetArrayReadWrite(yin,&ygpu);CHKERRQ(ierr);
    try {
      *ygpu += alpha * *xgpu;
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayReadWrite(yin,&ygpu);CHKERRQ(ierr);
    ierr = PetscLogFlops(2.0*yin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecPointwiseDivide_SeqViennaCL"
PetscErrorCode VecPointwiseDivide_SeqViennaCL(Vec win, Vec xin, Vec yin)
{
  const ViennaCLVector  *xgpu,*ygpu;
  ViennaCLVector        *wgpu;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (xin->map->n > 0) {
    ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLGetArrayRead(yin,&ygpu);CHKERRQ(ierr);
    ierr = VecViennaCLGetArrayWrite(win,&wgpu);CHKERRQ(ierr);
    try {
      *wgpu = viennacl::linalg::element_div(*xgpu, *ygpu);
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = PetscLogFlops(win->map->n);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayRead(yin,&ygpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayWrite(win,&wgpu);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecWAXPY_SeqViennaCL"
PetscErrorCode VecWAXPY_SeqViennaCL(Vec win,PetscScalar alpha,Vec xin, Vec yin)
{
  const ViennaCLVector  *xgpu,*ygpu;
  ViennaCLVector        *wgpu;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (alpha == 0.0 && xin->map->n > 0) {
    ierr = VecCopy_SeqViennaCL(yin,win);CHKERRQ(ierr);
  } else {
    ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLGetArrayRead(yin,&ygpu);CHKERRQ(ierr);
    ierr = VecViennaCLGetArrayWrite(win,&wgpu);CHKERRQ(ierr);
    if (alpha == 1.0) {
      try {
        *wgpu = *ygpu + *xgpu;
      } catch(std::exception const & ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
      }
      ierr = PetscLogFlops(win->map->n);CHKERRQ(ierr);
    } else if (alpha == -1.0) {
      try {
        *wgpu = *ygpu - *xgpu;
      } catch(std::exception const & ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
      }
      ierr = PetscLogFlops(win->map->n);CHKERRQ(ierr);
    } else {
      try {
        *wgpu = *ygpu + alpha * *xgpu;
      } catch(std::exception const & ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
      }
      ierr = PetscLogFlops(2*win->map->n);CHKERRQ(ierr);
    }
    ViennaCLWaitForGPU();
    ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayRead(yin,&ygpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayWrite(win,&wgpu);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*
 * Operation x = x + sum_i alpha_i * y_i for vectors x, y_i and scalars alpha_i
 *
 * ViennaCL supports a fast evaluation of x += alpha * y and x += alpha * y + beta * z,
 * hence there is an iterated application of these until the final result is obtained
 */
#undef __FUNCT__
#define __FUNCT__ "VecMAXPY_SeqViennaCL"
PetscErrorCode VecMAXPY_SeqViennaCL(Vec xin, PetscInt nv,const PetscScalar *alpha,Vec *y)
{
  PetscErrorCode ierr;
  PetscInt       j;

  PetscFunctionBegin;
  for (j = 0; j < nv; ++j) {
    if (j+1 < nv) {
      VecAXPBYPCZ_SeqViennaCL(xin,alpha[j],alpha[j+1],1.0,y[j],y[j+1]);
      ++j;
    } else {
      ierr = VecAXPY_SeqViennaCL(xin,alpha[j],y[j]);CHKERRQ(ierr);
    }
  }
  ViennaCLWaitForGPU();
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecDot_SeqViennaCL"
PetscErrorCode VecDot_SeqViennaCL(Vec xin,Vec yin,PetscScalar *z)
{
  const ViennaCLVector  *xgpu,*ygpu;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (xin->map->n > 0) {
    ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLGetArrayRead(yin,&ygpu);CHKERRQ(ierr);
    try {
      *z = viennacl::linalg::inner_prod(*xgpu,*ygpu);
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    if (xin->map->n >0) {
      ierr = PetscLogFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
    }
    ViennaCLWaitForGPU();
    ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayRead(yin,&ygpu);CHKERRQ(ierr);
  } else *z = 0.0;
  PetscFunctionReturn(0);
}



/*
 * Operation z[j] = dot(x, y[j])
 *
 * We use an iterated application of dot() for each j. For small ranges of j this is still faster than an allocation of extra memory in order to use gemv().
 */
#undef __FUNCT__
#define __FUNCT__ "VecMDot_SeqViennaCL"
PetscErrorCode VecMDot_SeqViennaCL(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode       ierr;
  PetscInt             n = xin->map->n,i;
  const ViennaCLVector *xgpu,*ygpu;
  Vec                  *yyin = (Vec*)yin;
  std::vector<viennacl::vector_base<PetscScalar> const *> ygpu_array(nv);

  PetscFunctionBegin;
  if (xin->map->n > 0) {
    ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
    for (i=0; i<nv; i++) {
      ierr = VecViennaCLGetArrayRead(yyin[i],&ygpu);CHKERRQ(ierr);
      ygpu_array[i] = ygpu;
    }

    viennacl::vector_tuple<PetscScalar> y_tuple(ygpu_array);
    ViennaCLVector result = viennacl::linalg::inner_prod(*xgpu, y_tuple);

    for (i=0; i<nv; i++) {
      viennacl::copy(result.begin(), result.end(), z);
      ierr = VecViennaCLRestoreArrayRead(yyin[i],&ygpu);CHKERRQ(ierr);
    }

    ViennaCLWaitForGPU();
    ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = PetscLogFlops(PetscMax(nv*(2.0*n-1),0.0));CHKERRQ(ierr);
  } else {
    for (i=0; i<nv; i++) z[i] = 0.0;
  }
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "VecSet_SeqViennaCL"
PetscErrorCode VecSet_SeqViennaCL(Vec xin,PetscScalar alpha)
{
  ViennaCLVector *xgpu;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (xin->map->n > 0) {
    ierr = VecViennaCLGetArrayWrite(xin,&xgpu);CHKERRQ(ierr);
    try {
      *xgpu = viennacl::scalar_vector<PetscScalar>(xgpu->size(), alpha);
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = VecViennaCLRestoreArrayWrite(xin,&xgpu);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecScale_SeqViennaCL"
PetscErrorCode VecScale_SeqViennaCL(Vec xin, PetscScalar alpha)
{
  ViennaCLVector *xgpu;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == 0.0 && xin->map->n > 0) {
    ierr = VecSet_SeqViennaCL(xin,alpha);CHKERRQ(ierr);
    ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
  } else if (alpha != 1.0 && xin->map->n > 0) {
    ierr = VecViennaCLGetArrayReadWrite(xin,&xgpu);CHKERRQ(ierr);
    try {
      *xgpu *= alpha;
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = VecViennaCLRestoreArrayReadWrite(xin,&xgpu);CHKERRQ(ierr);
    ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecTDot_SeqViennaCL"
PetscErrorCode VecTDot_SeqViennaCL(Vec xin,Vec yin,PetscScalar *z)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Since complex case is not supported at the moment, this is the same as VecDot_SeqViennaCL */
  ierr = VecDot_SeqViennaCL(xin, yin, z);CHKERRQ(ierr);
  ViennaCLWaitForGPU();
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecCopy_SeqViennaCL"
PetscErrorCode VecCopy_SeqViennaCL(Vec xin,Vec yin)
{
  const ViennaCLVector *xgpu;
  ViennaCLVector       *ygpu;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  if (xin != yin && xin->map->n > 0) {
    if (xin->valid_GPU_array == PETSC_VIENNACL_GPU) {
      ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
      ierr = VecViennaCLGetArrayWrite(yin,&ygpu);CHKERRQ(ierr);
      try {
        *ygpu = *xgpu;
        ViennaCLWaitForGPU();
      } catch(std::exception const & ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
      }
      ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
      ierr = VecViennaCLRestoreArrayWrite(yin,&ygpu);CHKERRQ(ierr);

    } else if (xin->valid_GPU_array == PETSC_VIENNACL_CPU) {
      /* copy in CPU if we are on the CPU*/
      ierr = VecCopy_SeqViennaCL_Private(xin,yin);CHKERRQ(ierr);
      ViennaCLWaitForGPU();
    } else if (xin->valid_GPU_array == PETSC_VIENNACL_BOTH) {
      /* if xin is valid in both places, see where yin is and copy there (because it's probably where we'll want to next use it) */
      if (yin->valid_GPU_array == PETSC_VIENNACL_CPU) {
        /* copy in CPU */
        ierr = VecCopy_SeqViennaCL_Private(xin,yin);CHKERRQ(ierr);
        ViennaCLWaitForGPU();
      } else if (yin->valid_GPU_array == PETSC_VIENNACL_GPU) {
        /* copy in GPU */
        ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
        ierr = VecViennaCLGetArrayWrite(yin,&ygpu);CHKERRQ(ierr);
        try {
          *ygpu = *xgpu;
          ViennaCLWaitForGPU();
        } catch(std::exception const & ex) {
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
        }
        ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
        ierr = VecViennaCLRestoreArrayWrite(yin,&ygpu);CHKERRQ(ierr);
      } else if (yin->valid_GPU_array == PETSC_VIENNACL_BOTH) {
        /* xin and yin are both valid in both places (or yin was unallocated before the earlier call to allocatecheck
           default to copy in GPU (this is an arbitrary choice) */
        ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
        ierr = VecViennaCLGetArrayWrite(yin,&ygpu);CHKERRQ(ierr);
        try {
          *ygpu = *xgpu;
          ViennaCLWaitForGPU();
        } catch(std::exception const & ex) {
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
        }
        ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
        ierr = VecViennaCLRestoreArrayWrite(yin,&ygpu);CHKERRQ(ierr);
      } else {
        ierr = VecCopy_SeqViennaCL_Private(xin,yin);CHKERRQ(ierr);
        ViennaCLWaitForGPU();
      }
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecSwap_SeqViennaCL"
PetscErrorCode VecSwap_SeqViennaCL(Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  ViennaCLVector *xgpu,*ygpu;

  PetscFunctionBegin;
  if (xin != yin && xin->map->n > 0) {
    ierr = VecViennaCLGetArrayReadWrite(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLGetArrayReadWrite(yin,&ygpu);CHKERRQ(ierr);

    try {
      viennacl::swap(*xgpu, *ygpu);
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = VecViennaCLRestoreArrayReadWrite(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayReadWrite(yin,&ygpu);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


// y = alpha * x + beta * y
#undef __FUNCT__
#define __FUNCT__ "VecAXPBY_SeqViennaCL"
PetscErrorCode VecAXPBY_SeqViennaCL(Vec yin,PetscScalar alpha,PetscScalar beta,Vec xin)
{
  PetscErrorCode       ierr;
  PetscScalar          a = alpha,b = beta;
  const ViennaCLVector *xgpu;
  ViennaCLVector       *ygpu;

  PetscFunctionBegin;
  if (a == 0.0 && xin->map->n > 0) {
    ierr = VecScale_SeqViennaCL(yin,beta);CHKERRQ(ierr);
  } else if (b == 1.0 && xin->map->n > 0) {
    ierr = VecAXPY_SeqViennaCL(yin,alpha,xin);CHKERRQ(ierr);
  } else if (a == 1.0 && xin->map->n > 0) {
    ierr = VecAYPX_SeqViennaCL(yin,beta,xin);CHKERRQ(ierr);
  } else if (b == 0.0 && xin->map->n > 0) {
    ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLGetArrayReadWrite(yin,&ygpu);CHKERRQ(ierr);
    try {
      *ygpu = *xgpu * alpha;
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = PetscLogFlops(xin->map->n);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayReadWrite(yin,&ygpu);CHKERRQ(ierr);
  } else if (xin->map->n > 0) {
    ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLGetArrayReadWrite(yin,&ygpu);CHKERRQ(ierr);
    try {
      *ygpu = *xgpu * alpha + *ygpu * beta;
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayReadWrite(yin,&ygpu);CHKERRQ(ierr);
    ierr = PetscLogFlops(3.0*xin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/* operation  z = alpha * x + beta *y + gamma *z*/
#undef __FUNCT__
#define __FUNCT__ "VecAXPBYPCZ_SeqViennaCL"
PetscErrorCode VecAXPBYPCZ_SeqViennaCL(Vec zin,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec xin,Vec yin)
{
  PetscErrorCode       ierr;
  PetscInt             n = zin->map->n;
  const ViennaCLVector *xgpu,*ygpu;
  ViennaCLVector       *zgpu;

  PetscFunctionBegin;
  ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
  ierr = VecViennaCLGetArrayRead(yin,&ygpu);CHKERRQ(ierr);
  ierr = VecViennaCLGetArrayReadWrite(zin,&zgpu);CHKERRQ(ierr);
  if (alpha == 0.0 && xin->map->n > 0) {
    try {
      if (beta == 0.0) {
        *zgpu = gamma * *zgpu;
        ViennaCLWaitForGPU();
        ierr = PetscLogFlops(1.0*n);CHKERRQ(ierr);
      } else if (gamma == 0.0) {
        *zgpu = beta * *ygpu;
        ViennaCLWaitForGPU();
        ierr = PetscLogFlops(1.0*n);CHKERRQ(ierr);
      } else {
        *zgpu = beta * *ygpu + gamma * *zgpu;
        ViennaCLWaitForGPU();
        ierr = PetscLogFlops(3.0*n);CHKERRQ(ierr);
      }
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = PetscLogFlops(3.0*n);CHKERRQ(ierr);
  } else if (beta == 0.0 && xin->map->n > 0) {
    try {
      if (gamma == 0.0) {
        *zgpu = alpha * *xgpu;
        ViennaCLWaitForGPU();
        ierr = PetscLogFlops(1.0*n);CHKERRQ(ierr);
      } else {
        *zgpu = alpha * *xgpu + gamma * *zgpu;
        ViennaCLWaitForGPU();
        ierr = PetscLogFlops(3.0*n);CHKERRQ(ierr);
      }
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
  } else if (gamma == 0.0 && xin->map->n > 0) {
    try {
      *zgpu = alpha * *xgpu + beta * *ygpu;
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = PetscLogFlops(3.0*n);CHKERRQ(ierr);
  } else if (xin->map->n > 0) {
    try {
      /* Split operation into two steps. This is not completely ideal, but avoids temporaries (which are far worse) */
      if (gamma != 1.0)
        *zgpu *= gamma;
      *zgpu += alpha * *xgpu + beta * *ygpu;
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = VecViennaCLRestoreArrayReadWrite(zin,&zgpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayRead(yin,&ygpu);CHKERRQ(ierr);
    ierr = PetscLogFlops(5.0*n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPointwiseMult_SeqViennaCL"
PetscErrorCode VecPointwiseMult_SeqViennaCL(Vec win,Vec xin,Vec yin)
{
  PetscErrorCode       ierr;
  PetscInt             n = win->map->n;
  const ViennaCLVector *xgpu,*ygpu;
  ViennaCLVector       *wgpu;

  PetscFunctionBegin;
  if (xin->map->n > 0) {
    ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLGetArrayRead(yin,&ygpu);CHKERRQ(ierr);
    ierr = VecViennaCLGetArrayReadWrite(win,&wgpu);CHKERRQ(ierr);
    try {
      *wgpu = viennacl::linalg::element_prod(*xgpu, *ygpu);
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayRead(yin,&ygpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayReadWrite(win,&wgpu);CHKERRQ(ierr);
    ierr = PetscLogFlops(n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecNorm_SeqViennaCL"
PetscErrorCode VecNorm_SeqViennaCL(Vec xin,NormType type,PetscReal *z)
{
  PetscErrorCode       ierr;
  PetscInt             n = xin->map->n;
  PetscBLASInt         bn;
  const ViennaCLVector *xgpu;

  PetscFunctionBegin;
  if (xin->map->n > 0) {
    ierr = PetscBLASIntCast(n,&bn);CHKERRQ(ierr);
    ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
    if (type == NORM_2 || type == NORM_FROBENIUS) {
      try {
        *z = viennacl::linalg::norm_2(*xgpu);
        ViennaCLWaitForGPU();
      } catch(std::exception const & ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
      }
      ierr = PetscLogFlops(PetscMax(2.0*n-1,0.0));CHKERRQ(ierr);
    } else if (type == NORM_INFINITY) {
      ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
      try {
        *z = viennacl::linalg::norm_inf(*xgpu);
        ViennaCLWaitForGPU();
      } catch(std::exception const & ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
      }
      ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
    } else if (type == NORM_1) {
      try {
        *z = viennacl::linalg::norm_1(*xgpu);
        ViennaCLWaitForGPU();
      } catch(std::exception const & ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
      }
      ierr = PetscLogFlops(PetscMax(n-1.0,0.0));CHKERRQ(ierr);
    } else if (type == NORM_1_AND_2) {
      try {
        *z     = viennacl::linalg::norm_1(*xgpu);
        *(z+1) = viennacl::linalg::norm_2(*xgpu);
        ViennaCLWaitForGPU();
      } catch(std::exception const & ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
      }
      ierr = PetscLogFlops(PetscMax(2.0*n-1,0.0));CHKERRQ(ierr);
      ierr = PetscLogFlops(PetscMax(n-1.0,0.0));CHKERRQ(ierr);
    }
    ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    *z      = 0.0;
    *(z+1)  = 0.0;
  } else *z = 0.0;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecSetRandom_SeqViennaCL"
PetscErrorCode VecSetRandom_SeqViennaCL(Vec xin,PetscRandom r)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSetRandom_SeqViennaCL_Private(xin,r);CHKERRQ(ierr);
  xin->valid_GPU_array = PETSC_VIENNACL_CPU;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecResetArray_SeqViennaCL"
PetscErrorCode VecResetArray_SeqViennaCL(Vec vin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecViennaCLCopyFromGPU(vin);CHKERRQ(ierr);
  ierr = VecResetArray_SeqViennaCL_Private(vin);CHKERRQ(ierr);
  vin->valid_GPU_array = PETSC_VIENNACL_CPU;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecPlaceArray_SeqViennaCL"
PetscErrorCode VecPlaceArray_SeqViennaCL(Vec vin,const PetscScalar *a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecViennaCLCopyFromGPU(vin);CHKERRQ(ierr);
  ierr = VecPlaceArray_Seq(vin,a);CHKERRQ(ierr);
  vin->valid_GPU_array = PETSC_VIENNACL_CPU;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecReplaceArray_SeqViennaCL"
PetscErrorCode VecReplaceArray_SeqViennaCL(Vec vin,const PetscScalar *a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecViennaCLCopyFromGPU(vin);CHKERRQ(ierr);
  ierr = VecReplaceArray_Seq(vin,a);CHKERRQ(ierr);
  vin->valid_GPU_array = PETSC_VIENNACL_CPU;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecCreateSeqViennaCL"
/*@
   VecCreateSeqViennaCL - Creates a standard, sequential array-style vector.

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
PetscErrorCode  VecCreateSeqViennaCL(MPI_Comm comm,PetscInt n,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,n);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECSEQVIENNACL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*  VecDotNorm2 - computes the inner product of two vectors and the 2-norm squared of the second vector
 *
 *  Simply reuses VecDot() and VecNorm(). Performance improvement through custom kernel (kernel generator) possible.
 */
#undef __FUNCT__
#define __FUNCT__ "VecDotNorm2_SeqViennaCL"
PetscErrorCode VecDotNorm2_SeqViennaCL(Vec s, Vec t, PetscScalar *dp, PetscScalar *nm)
{
  PetscErrorCode                         ierr;

  PetscFunctionBegin;
  ierr = VecDot_SeqViennaCL(s,t,dp);CHKERRQ(ierr);
  ierr = VecNorm_SeqViennaCL(t,NORM_2,nm);CHKERRQ(ierr);
  *nm *= *nm; //squared norm required
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDuplicate_SeqViennaCL"
PetscErrorCode VecDuplicate_SeqViennaCL(Vec win,Vec *V)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreateSeqViennaCL(PetscObjectComm((PetscObject)win),win->map->n,V);CHKERRQ(ierr);
  ierr = PetscLayoutReference(win->map,&(*V)->map);CHKERRQ(ierr);
  ierr = PetscObjectListDuplicate(((PetscObject)win)->olist,&((PetscObject)(*V))->olist);CHKERRQ(ierr);
  ierr = PetscFunctionListDuplicate(((PetscObject)win)->qlist,&((PetscObject)(*V))->qlist);CHKERRQ(ierr);
  (*V)->stash.ignorenegidx = win->stash.ignorenegidx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecDestroy_SeqViennaCL"
PetscErrorCode VecDestroy_SeqViennaCL(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  try {
    if (v->spptr) {
      delete ((Vec_ViennaCL*)v->spptr)->GPUarray;
      delete (Vec_ViennaCL*) v->spptr;
    }
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex);
  }
  ierr = VecDestroy_SeqViennaCL_Private(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "VecCreate_SeqViennaCL"
PETSC_EXTERN PetscErrorCode VecCreate_SeqViennaCL(Vec V)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)V),&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQVIENNACL on more than one process");
  ierr = VecCreate_Seq_Private(V,0);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)V,VECSEQVIENNACL);CHKERRQ(ierr);

  V->ops->dot             = VecDot_SeqViennaCL;
  V->ops->norm            = VecNorm_SeqViennaCL;
  V->ops->tdot            = VecTDot_SeqViennaCL;
  V->ops->scale           = VecScale_SeqViennaCL;
  V->ops->copy            = VecCopy_SeqViennaCL;
  V->ops->set             = VecSet_SeqViennaCL;
  V->ops->swap            = VecSwap_SeqViennaCL;
  V->ops->axpy            = VecAXPY_SeqViennaCL;
  V->ops->axpby           = VecAXPBY_SeqViennaCL;
  V->ops->axpbypcz        = VecAXPBYPCZ_SeqViennaCL;
  V->ops->pointwisemult   = VecPointwiseMult_SeqViennaCL;
  V->ops->pointwisedivide = VecPointwiseDivide_SeqViennaCL;
  V->ops->setrandom       = VecSetRandom_SeqViennaCL;
  V->ops->dot_local       = VecDot_SeqViennaCL;
  V->ops->tdot_local      = VecTDot_SeqViennaCL;
  V->ops->norm_local      = VecNorm_SeqViennaCL;
  V->ops->mdot_local      = VecMDot_SeqViennaCL;
  V->ops->maxpy           = VecMAXPY_SeqViennaCL;
  V->ops->mdot            = VecMDot_SeqViennaCL;
  V->ops->aypx            = VecAYPX_SeqViennaCL;
  V->ops->waxpy           = VecWAXPY_SeqViennaCL;
  V->ops->dotnorm2        = VecDotNorm2_SeqViennaCL;
  V->ops->placearray      = VecPlaceArray_SeqViennaCL;
  V->ops->replacearray    = VecReplaceArray_SeqViennaCL;
  V->ops->resetarray      = VecResetArray_SeqViennaCL;
  V->ops->destroy         = VecDestroy_SeqViennaCL;
  V->ops->duplicate       = VecDuplicate_SeqViennaCL;

  ierr = VecViennaCLAllocateCheck(V);CHKERRQ(ierr);
  V->valid_GPU_array      = PETSC_VIENNACL_GPU;
  ierr = VecSet(V,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

