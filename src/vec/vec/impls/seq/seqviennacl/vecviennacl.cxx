/*
   Implements the sequential ViennaCL vectors.
*/

#include <petscconf.h>
#include <petsc/private/vecimpl.h>          /*I "petscvec.h" I*/
#include <../src/vec/vec/impls/dvecimpl.h>
#include <../src/vec/vec/impls/seq/seqviennacl/viennaclvecimpl.h>

#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_1.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_inf.hpp"

#ifdef VIENNACL_WITH_OPENCL
#include "viennacl/ocl/backend.hpp"
#endif

PETSC_EXTERN PetscErrorCode VecViennaCLGetArray(Vec v, ViennaCLVector **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQVIENNACL,VECMPIVIENNACL);
  *a   = 0;
  ierr = VecViennaCLCopyToGPU(v);CHKERRQ(ierr);
  *a   = ((Vec_ViennaCL*)v->spptr)->GPUarray;
  ViennaCLWaitForGPU();
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecViennaCLRestoreArray(Vec v, ViennaCLVector **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQVIENNACL,VECMPIVIENNACL);
  v->offloadmask = PETSC_OFFLOAD_GPU;

  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecViennaCLGetArrayRead(Vec v, const ViennaCLVector **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQVIENNACL,VECMPIVIENNACL);
  *a   = 0;
  ierr = VecViennaCLCopyToGPU(v);CHKERRQ(ierr);
  *a   = ((Vec_ViennaCL*)v->spptr)->GPUarray;
  ViennaCLWaitForGPU();
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecViennaCLRestoreArrayRead(Vec v, const ViennaCLVector **a)
{
  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQVIENNACL,VECMPIVIENNACL);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecViennaCLGetArrayWrite(Vec v, ViennaCLVector **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQVIENNACL,VECMPIVIENNACL);
  *a   = 0;
  ierr = VecViennaCLAllocateCheck(v);CHKERRQ(ierr);
  *a   = ((Vec_ViennaCL*)v->spptr)->GPUarray;
  ViennaCLWaitForGPU();
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecViennaCLRestoreArrayWrite(Vec v, ViennaCLVector **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQVIENNACL,VECMPIVIENNACL);
  v->offloadmask = PETSC_OFFLOAD_GPU;

  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode PetscViennaCLInit()
{
  PetscErrorCode       ierr;
  char                 string[20];
  PetscBool            flg,flg_cuda,flg_opencl,flg_openmp;

  PetscFunctionBegin;
  /* ViennaCL backend selection: CUDA, OpenCL, or OpenMP */
  ierr = PetscOptionsGetString(NULL,NULL,"-viennacl_backend",string,sizeof(string),&flg);CHKERRQ(ierr);
  if (flg) {
    try {
      ierr = PetscStrcasecmp(string,"cuda",&flg_cuda);CHKERRQ(ierr);
      ierr = PetscStrcasecmp(string,"opencl",&flg_opencl);CHKERRQ(ierr);
      ierr = PetscStrcasecmp(string,"openmp",&flg_openmp);CHKERRQ(ierr);

      /* A default (sequential) CPU backend is always available - even if OpenMP is not enabled. */
      if (flg_openmp) viennacl::backend::default_memory_type(viennacl::MAIN_MEMORY);
#if defined(PETSC_HAVE_CUDA)
      else if (flg_cuda) viennacl::backend::default_memory_type(viennacl::CUDA_MEMORY);
#endif
#if defined(PETSC_HAVE_OPENCL)
      else if (flg_opencl) viennacl::backend::default_memory_type(viennacl::OPENCL_MEMORY);
#endif
      else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: Backend not recognized or available: %s.\n Pass -viennacl_view to see available backends for ViennaCL.\n", string);
    } catch (std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
  }

  if (PetscDefined(HAVE_CUDA)) {
    /* For CUDA event timers */
    ierr = PetscDeviceInitialize(PETSC_DEVICE_CUDA);CHKERRQ(ierr);
  }

#if defined(PETSC_HAVE_OPENCL)
  /* ViennaCL OpenCL device type configuration */
  ierr = PetscOptionsGetString(NULL,NULL,"-viennacl_opencl_device_type",string,sizeof(string),&flg);CHKERRQ(ierr);
  if (flg) {
    try {
      ierr = PetscStrcasecmp(string,"cpu",&flg);CHKERRQ(ierr);
      if (flg) viennacl::ocl::set_context_device_type(0, CL_DEVICE_TYPE_CPU);

      ierr = PetscStrcasecmp(string,"gpu",&flg);CHKERRQ(ierr);
      if (flg) viennacl::ocl::set_context_device_type(0, CL_DEVICE_TYPE_GPU);

      ierr = PetscStrcasecmp(string,"accelerator",&flg);CHKERRQ(ierr);
      if (flg) viennacl::ocl::set_context_device_type(0, CL_DEVICE_TYPE_ACCELERATOR);
    } catch (std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
  }
#endif

  /* Print available backends */
  ierr = PetscOptionsHasName(NULL,NULL,"-viennacl_view",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "ViennaCL backends available: ");CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
    ierr = PetscPrintf(PETSC_COMM_WORLD, "CUDA, ");CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_OPENCL)
    ierr = PetscPrintf(PETSC_COMM_WORLD, "OpenCL, ");CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_OPENMP)
    ierr = PetscPrintf(PETSC_COMM_WORLD, "OpenMP ");CHKERRQ(ierr);
#else
    ierr = PetscPrintf(PETSC_COMM_WORLD, "OpenMP (1 thread) ");CHKERRQ(ierr);
#endif
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);

    /* Print selected backends */
    ierr = PetscPrintf(PETSC_COMM_WORLD, "ViennaCL backend  selected: ");CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
    if (viennacl::backend::default_memory_type() == viennacl::CUDA_MEMORY) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "CUDA ");CHKERRQ(ierr);
    }
#endif
#if defined(PETSC_HAVE_OPENCL)
    if (viennacl::backend::default_memory_type() == viennacl::OPENCL_MEMORY) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "OpenCL ");CHKERRQ(ierr);
    }
#endif
#if defined(PETSC_HAVE_OPENMP)
    if (viennacl::backend::default_memory_type() == viennacl::MAIN_MEMORY) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "OpenMP ");CHKERRQ(ierr);
    }
#else
    if (viennacl::backend::default_memory_type() == viennacl::MAIN_MEMORY) {
      ierr = PetscPrintf(PETSC_COMM_WORLD, "OpenMP (sequential - consider reconfiguration: --with-openmp=1) ");CHKERRQ(ierr);
    }
#endif
    ierr = PetscPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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

/*
    Allocates space for the vector array on the GPU if it does not exist.
    Does NOT change the PetscViennaCLFlag for the vector
    Does NOT zero the ViennaCL array

 */
PetscErrorCode VecViennaCLAllocateCheck(Vec v)
{
  PetscFunctionBegin;
  if (!v->spptr) {
    try {
      v->spptr                            = new Vec_ViennaCL;
      ((Vec_ViennaCL*)v->spptr)->GPUarray_allocated = new ViennaCLVector((PetscBLASInt)v->map->n);
      ((Vec_ViennaCL*)v->spptr)->GPUarray = ((Vec_ViennaCL*)v->spptr)->GPUarray_allocated;

    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
  }
  PetscFunctionReturn(0);
}

/* Copies a vector from the CPU to the GPU unless we already have an up-to-date copy on the GPU */
PetscErrorCode VecViennaCLCopyToGPU(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQVIENNACL,VECMPIVIENNACL);
  ierr = VecViennaCLAllocateCheck(v);CHKERRQ(ierr);
  if (v->map->n > 0) {
    if (v->offloadmask == PETSC_OFFLOAD_CPU) {
      ierr = PetscLogEventBegin(VEC_ViennaCLCopyToGPU,v,0,0,0);CHKERRQ(ierr);
      try {
        ViennaCLVector *vec = ((Vec_ViennaCL*)v->spptr)->GPUarray;
        viennacl::fast_copy(*(PetscScalar**)v->data, *(PetscScalar**)v->data + v->map->n, vec->begin());
        ViennaCLWaitForGPU();
      } catch(std::exception const & ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
      }
      ierr = PetscLogCpuToGpu((v->map->n)*sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = PetscLogEventEnd(VEC_ViennaCLCopyToGPU,v,0,0,0);CHKERRQ(ierr);
      v->offloadmask = PETSC_OFFLOAD_BOTH;
    }
  }
  PetscFunctionReturn(0);
}

/*
     VecViennaCLCopyFromGPU - Copies a vector from the GPU to the CPU unless we already have an up-to-date copy on the CPU
*/
PetscErrorCode VecViennaCLCopyFromGPU(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(v,VECSEQVIENNACL,VECMPIVIENNACL);
  ierr = VecViennaCLAllocateCheckHost(v);CHKERRQ(ierr);
  if (v->offloadmask == PETSC_OFFLOAD_GPU) {
    ierr = PetscLogEventBegin(VEC_ViennaCLCopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    try {
      ViennaCLVector *vec = ((Vec_ViennaCL*)v->spptr)->GPUarray;
      viennacl::fast_copy(vec->begin(),vec->end(),*(PetscScalar**)v->data);
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = PetscLogGpuToCpu((v->map->n)*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscLogEventEnd(VEC_ViennaCLCopyFromGPU,v,0,0,0);CHKERRQ(ierr);
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  }
  PetscFunctionReturn(0);
}

/* Copy on CPU */
static PetscErrorCode VecCopy_SeqViennaCL_Private(Vec xin,Vec yin)
{
  PetscScalar       *ya;
  const PetscScalar *xa;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecViennaCLAllocateCheckHost(xin);CHKERRQ(ierr);
  ierr = VecViennaCLAllocateCheckHost(yin);CHKERRQ(ierr);
  if (xin != yin) {
    ierr = VecGetArrayRead(xin,&xa);CHKERRQ(ierr);
    ierr = VecGetArray(yin,&ya);CHKERRQ(ierr);
    ierr = PetscArraycpy(ya,xa,xin->map->n);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(xin,&xa);CHKERRQ(ierr);
    ierr = VecRestoreArray(yin,&ya);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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

static PetscErrorCode VecDestroy_SeqViennaCL_Private(Vec v)
{
  Vec_Seq        *vs = (Vec_Seq*)v->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectSAWsViewOff(v);CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)v,"Length=%D",v->map->n);
#endif
  if (vs->array_allocated) { ierr = PetscFree(vs->array_allocated);CHKERRQ(ierr); }
  ierr = PetscFree(vs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

PetscErrorCode VecAYPX_SeqViennaCL(Vec yin, PetscScalar alpha, Vec xin)
{
  const ViennaCLVector  *xgpu;
  ViennaCLVector        *ygpu;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
  ierr = VecViennaCLGetArray(yin,&ygpu);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  try {
    if (alpha != 0.0 && xin->map->n > 0) {
      *ygpu = *xgpu + alpha * *ygpu;
      ierr = PetscLogGpuFlops(2.0*yin->map->n);CHKERRQ(ierr);
    } else {
      *ygpu = *xgpu;
    }
    ViennaCLWaitForGPU();
  } catch(std::exception const & ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
  ierr = VecViennaCLRestoreArray(yin,&ygpu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecAXPY_SeqViennaCL(Vec yin,PetscScalar alpha,Vec xin)
{
  const ViennaCLVector  *xgpu;
  ViennaCLVector        *ygpu;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (alpha != 0.0 && xin->map->n > 0) {
    ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLGetArray(yin,&ygpu);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      *ygpu += alpha * *xgpu;
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArray(yin,&ygpu);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(2.0*yin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      *wgpu = viennacl::linalg::element_div(*xgpu, *ygpu);
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(win->map->n);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayRead(yin,&ygpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayWrite(win,&wgpu);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    if (alpha == 1.0) {
      try {
        *wgpu = *ygpu + *xgpu;
      } catch(std::exception const & ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
      }
      ierr = PetscLogGpuFlops(win->map->n);CHKERRQ(ierr);
    } else if (alpha == -1.0) {
      try {
        *wgpu = *ygpu - *xgpu;
      } catch(std::exception const & ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
      }
      ierr = PetscLogGpuFlops(win->map->n);CHKERRQ(ierr);
    } else {
      try {
        *wgpu = *ygpu + alpha * *xgpu;
      } catch(std::exception const & ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
      }
      ierr = PetscLogGpuFlops(2*win->map->n);CHKERRQ(ierr);
    }
    ViennaCLWaitForGPU();
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
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

PetscErrorCode VecDot_SeqViennaCL(Vec xin,Vec yin,PetscScalar *z)
{
  const ViennaCLVector  *xgpu,*ygpu;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (xin->map->n > 0) {
    ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLGetArrayRead(yin,&ygpu);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      *z = viennacl::linalg::inner_prod(*xgpu,*ygpu);
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ViennaCLWaitForGPU();
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    if (xin->map->n >0) {
      ierr = PetscLogGpuFlops(2.0*xin->map->n-1);CHKERRQ(ierr);
    }
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
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    viennacl::vector_tuple<PetscScalar> y_tuple(ygpu_array);
    ViennaCLVector result = viennacl::linalg::inner_prod(*xgpu, y_tuple);
    viennacl::copy(result.begin(), result.end(), z);
    for (i=0; i<nv; i++) {
      ierr = VecViennaCLRestoreArrayRead(yyin[i],&ygpu);CHKERRQ(ierr);
    }
    ViennaCLWaitForGPU();
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(PetscMax(nv*(2.0*n-1),0.0));CHKERRQ(ierr);
  } else {
    for (i=0; i<nv; i++) z[i] = 0.0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecMTDot_SeqViennaCL(Vec xin,PetscInt nv,const Vec yin[],PetscScalar *z)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Since complex case is not supported at the moment, this is the same as VecMDot_SeqViennaCL */
  ierr = VecMDot_SeqViennaCL(xin,nv,yin,z);CHKERRQ(ierr);
  ViennaCLWaitForGPU();
  PetscFunctionReturn(0);
}

PetscErrorCode VecSet_SeqViennaCL(Vec xin,PetscScalar alpha)
{
  ViennaCLVector *xgpu;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (xin->map->n > 0) {
    ierr = VecViennaCLGetArrayWrite(xin,&xgpu);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      *xgpu = viennacl::scalar_vector<PetscScalar>(xgpu->size(), alpha);
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayWrite(xin,&xgpu);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecScale_SeqViennaCL(Vec xin, PetscScalar alpha)
{
  ViennaCLVector *xgpu;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (alpha == 0.0 && xin->map->n > 0) {
    ierr = VecSet_SeqViennaCL(xin,alpha);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(xin->map->n);CHKERRQ(ierr);
  } else if (alpha != 1.0 && xin->map->n > 0) {
    ierr = VecViennaCLGetArray(xin,&xgpu);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      *xgpu *= alpha;
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArray(xin,&xgpu);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(xin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode VecTDot_SeqViennaCL(Vec xin,Vec yin,PetscScalar *z)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Since complex case is not supported at the moment, this is the same as VecDot_SeqViennaCL */
  ierr = VecDot_SeqViennaCL(xin, yin, z);CHKERRQ(ierr);
  ViennaCLWaitForGPU();
  PetscFunctionReturn(0);
}

PetscErrorCode VecCopy_SeqViennaCL(Vec xin,Vec yin)
{
  const ViennaCLVector *xgpu;
  ViennaCLVector       *ygpu;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  if (xin != yin && xin->map->n > 0) {
    if (xin->offloadmask == PETSC_OFFLOAD_GPU) {
      ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
      ierr = VecViennaCLGetArrayWrite(yin,&ygpu);CHKERRQ(ierr);
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      try {
        *ygpu = *xgpu;
        ViennaCLWaitForGPU();
      } catch(std::exception const & ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
      }
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
      ierr = VecViennaCLRestoreArrayWrite(yin,&ygpu);CHKERRQ(ierr);

    } else if (xin->offloadmask == PETSC_OFFLOAD_CPU) {
      /* copy in CPU if we are on the CPU*/
      ierr = VecCopy_SeqViennaCL_Private(xin,yin);CHKERRQ(ierr);
      ViennaCLWaitForGPU();
    } else if (xin->offloadmask == PETSC_OFFLOAD_BOTH) {
      /* if xin is valid in both places, see where yin is and copy there (because it's probably where we'll want to next use it) */
      if (yin->offloadmask == PETSC_OFFLOAD_CPU) {
        /* copy in CPU */
        ierr = VecCopy_SeqViennaCL_Private(xin,yin);CHKERRQ(ierr);
        ViennaCLWaitForGPU();
      } else if (yin->offloadmask == PETSC_OFFLOAD_GPU) {
        /* copy in GPU */
        ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
        ierr = VecViennaCLGetArrayWrite(yin,&ygpu);CHKERRQ(ierr);
        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
        try {
          *ygpu = *xgpu;
          ViennaCLWaitForGPU();
        } catch(std::exception const & ex) {
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
        }
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
        ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
        ierr = VecViennaCLRestoreArrayWrite(yin,&ygpu);CHKERRQ(ierr);
      } else if (yin->offloadmask == PETSC_OFFLOAD_BOTH) {
        /* xin and yin are both valid in both places (or yin was unallocated before the earlier call to allocatecheck
           default to copy in GPU (this is an arbitrary choice) */
        ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
        ierr = VecViennaCLGetArrayWrite(yin,&ygpu);CHKERRQ(ierr);
        ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
        try {
          *ygpu = *xgpu;
          ViennaCLWaitForGPU();
        } catch(std::exception const & ex) {
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
        }
        ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
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

PetscErrorCode VecSwap_SeqViennaCL(Vec xin,Vec yin)
{
  PetscErrorCode ierr;
  ViennaCLVector *xgpu,*ygpu;

  PetscFunctionBegin;
  if (xin != yin && xin->map->n > 0) {
    ierr = VecViennaCLGetArray(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLGetArray(yin,&ygpu);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      viennacl::swap(*xgpu, *ygpu);
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArray(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArray(yin,&ygpu);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

// y = alpha * x + beta * y
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
    ierr = VecViennaCLGetArray(yin,&ygpu);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      *ygpu = *xgpu * alpha;
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(xin->map->n);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArray(yin,&ygpu);CHKERRQ(ierr);
  } else if (xin->map->n > 0) {
    ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLGetArray(yin,&ygpu);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      *ygpu = *xgpu * alpha + *ygpu * beta;
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArray(yin,&ygpu);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(3.0*xin->map->n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* operation  z = alpha * x + beta *y + gamma *z*/
PetscErrorCode VecAXPBYPCZ_SeqViennaCL(Vec zin,PetscScalar alpha,PetscScalar beta,PetscScalar gamma,Vec xin,Vec yin)
{
  PetscErrorCode       ierr;
  PetscInt             n = zin->map->n;
  const ViennaCLVector *xgpu,*ygpu;
  ViennaCLVector       *zgpu;

  PetscFunctionBegin;
  ierr = VecViennaCLGetArrayRead(xin,&xgpu);CHKERRQ(ierr);
  ierr = VecViennaCLGetArrayRead(yin,&ygpu);CHKERRQ(ierr);
  ierr = VecViennaCLGetArray(zin,&zgpu);CHKERRQ(ierr);
  if (alpha == 0.0 && xin->map->n > 0) {
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      if (beta == 0.0) {
        *zgpu = gamma * *zgpu;
        ViennaCLWaitForGPU();
        ierr = PetscLogGpuFlops(1.0*n);CHKERRQ(ierr);
      } else if (gamma == 0.0) {
        *zgpu = beta * *ygpu;
        ViennaCLWaitForGPU();
        ierr = PetscLogGpuFlops(1.0*n);CHKERRQ(ierr);
      } else {
        *zgpu = beta * *ygpu + gamma * *zgpu;
        ViennaCLWaitForGPU();
        ierr = PetscLogGpuFlops(3.0*n);CHKERRQ(ierr);
      }
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(3.0*n);CHKERRQ(ierr);
  } else if (beta == 0.0 && xin->map->n > 0) {
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      if (gamma == 0.0) {
        *zgpu = alpha * *xgpu;
        ViennaCLWaitForGPU();
        ierr = PetscLogGpuFlops(1.0*n);CHKERRQ(ierr);
      } else {
        *zgpu = alpha * *xgpu + gamma * *zgpu;
        ViennaCLWaitForGPU();
        ierr = PetscLogGpuFlops(3.0*n);CHKERRQ(ierr);
      }
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  } else if (gamma == 0.0 && xin->map->n > 0) {
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      *zgpu = alpha * *xgpu + beta * *ygpu;
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(3.0*n);CHKERRQ(ierr);
  } else if (xin->map->n > 0) {
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      /* Split operation into two steps. This is not completely ideal, but avoids temporaries (which are far worse) */
      if (gamma != 1.0)
        *zgpu *= gamma;
      *zgpu += alpha * *xgpu + beta * *ygpu;
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArray(zin,&zgpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayRead(yin,&ygpu);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(5.0*n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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
    ierr = VecViennaCLGetArray(win,&wgpu);CHKERRQ(ierr);
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    try {
      *wgpu = viennacl::linalg::element_prod(*xgpu, *ygpu);
      ViennaCLWaitForGPU();
    } catch(std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArrayRead(yin,&ygpu);CHKERRQ(ierr);
    ierr = VecViennaCLRestoreArray(win,&wgpu);CHKERRQ(ierr);
    ierr = PetscLogGpuFlops(n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      try {
        *z = viennacl::linalg::norm_2(*xgpu);
        ViennaCLWaitForGPU();
      } catch(std::exception const & ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
      }
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      ierr = PetscLogGpuFlops(PetscMax(2.0*n-1,0.0));CHKERRQ(ierr);
    } else if (type == NORM_INFINITY) {
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      try {
        *z = viennacl::linalg::norm_inf(*xgpu);
        ViennaCLWaitForGPU();
      } catch(std::exception const & ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
      }
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    } else if (type == NORM_1) {
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      try {
        *z = viennacl::linalg::norm_1(*xgpu);
        ViennaCLWaitForGPU();
      } catch(std::exception const & ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
      }
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      ierr = PetscLogGpuFlops(PetscMax(n-1.0,0.0));CHKERRQ(ierr);
    } else if (type == NORM_1_AND_2) {
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      try {
        *z     = viennacl::linalg::norm_1(*xgpu);
        *(z+1) = viennacl::linalg::norm_2(*xgpu);
        ViennaCLWaitForGPU();
      } catch(std::exception const & ex) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
      }
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
      ierr = PetscLogGpuFlops(PetscMax(2.0*n-1,0.0));CHKERRQ(ierr);
      ierr = PetscLogGpuFlops(PetscMax(n-1.0,0.0));CHKERRQ(ierr);
    }
    ierr = VecViennaCLRestoreArrayRead(xin,&xgpu);CHKERRQ(ierr);
  } else if (type == NORM_1_AND_2) {
    *z      = 0.0;
    *(z+1)  = 0.0;
  } else *z = 0.0;
  PetscFunctionReturn(0);
}

PetscErrorCode VecSetRandom_SeqViennaCL(Vec xin,PetscRandom r)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSetRandom_SeqViennaCL_Private(xin,r);CHKERRQ(ierr);
  xin->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode VecResetArray_SeqViennaCL(Vec vin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(vin,VECSEQVIENNACL,VECMPIVIENNACL);
  ierr = VecViennaCLCopyFromGPU(vin);CHKERRQ(ierr);
  ierr = VecResetArray_SeqViennaCL_Private(vin);CHKERRQ(ierr);
  vin->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode VecPlaceArray_SeqViennaCL(Vec vin,const PetscScalar *a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(vin,VECSEQVIENNACL,VECMPIVIENNACL);
  ierr = VecViennaCLCopyFromGPU(vin);CHKERRQ(ierr);
  ierr = VecPlaceArray_Seq(vin,a);CHKERRQ(ierr);
  vin->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode VecReplaceArray_SeqViennaCL(Vec vin,const PetscScalar *a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(vin,VECSEQVIENNACL,VECMPIVIENNACL);
  ierr = VecViennaCLCopyFromGPU(vin);CHKERRQ(ierr);
  ierr = VecReplaceArray_Seq(vin,a);CHKERRQ(ierr);
  vin->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

/*@C
   VecCreateSeqViennaCL - Creates a standard, sequential array-style vector.

   Collective

   Input Parameters:
+  comm - the communicator, should be PETSC_COMM_SELF
-  n - the vector length

   Output Parameter:
.  V - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   Level: intermediate

.seealso: VecCreateMPI(), VecCreate(), VecDuplicate(), VecDuplicateVecs(), VecCreateGhost()
@*/
PetscErrorCode VecCreateSeqViennaCL(MPI_Comm comm,PetscInt n,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCreate(comm,v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v,n,n);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECSEQVIENNACL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCreateSeqViennaCLWithArray - Creates a viennacl sequential array-style vector,
   where the user provides the array space to store the vector values.

   Collective

   Input Parameters:
+  comm - the communicator, should be PETSC_COMM_SELF
.  bs - the block size
.  n - the vector length
-  array - viennacl array where the vector elements are to be stored.

   Output Parameter:
.  V - the vector

   Notes:
   Use VecDuplicate() or VecDuplicateVecs() to form additional vectors of the
   same type as an existing vector.

   If the user-provided array is NULL, then VecViennaCLPlaceArray() can be used
   at a later stage to SET the array for storing the vector values.

   PETSc does NOT free the array when the vector is destroyed via VecDestroy().
   The user should not free the array until the vector is destroyed.

   Level: intermediate

.seealso: VecCreateMPIViennaCLWithArray(), VecCreate(), VecDuplicate(), VecDuplicateVecs(),
          VecCreateGhost(), VecCreateSeq(), VecCUDAPlaceArray(), VecCreateSeqWithArray(),
          VecCreateMPIWithArray()
@*/
PETSC_EXTERN PetscErrorCode  VecCreateSeqViennaCLWithArray(MPI_Comm comm,PetscInt bs,PetscInt n,const ViennaCLVector* array,Vec *V)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = VecCreate(comm,V);CHKERRQ(ierr);
  ierr = VecSetSizes(*V,n,n);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*V,bs);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQ on more than one process");
  ierr = VecCreate_SeqViennaCL_Private(*V,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecCreateSeqViennaCLWithArrays - Creates a ViennaCL sequential vector, where
   the user provides the array space to store the vector values.

   Collective

   Input Parameters:
+  comm - the communicator, should be PETSC_COMM_SELF
.  bs - the block size
.  n - the vector length
-  cpuarray - CPU memory where the vector elements are to be stored.
-  viennaclvec - ViennaCL vector where the Vec entries are to be stored on the device.

   Output Parameter:
.  V - the vector

   Notes:
   If both cpuarray and viennaclvec are provided, the caller must ensure that
   the provided arrays have identical values.

   PETSc does NOT free the provided arrays when the vector is destroyed via
   VecDestroy(). The user should not free the array until the vector is
   destroyed.

   Level: intermediate

.seealso: VecCreateMPIViennaCLWithArrays(), VecCreate(), VecCreateSeqWithArray(),
          VecViennaCLPlaceArray(), VecPlaceArray(), VecCreateSeqCUDAWithArrays(),
          VecViennaCLAllocateCheckHost()
@*/
PetscErrorCode  VecCreateSeqViennaCLWithArrays(MPI_Comm comm,PetscInt bs,PetscInt n,const PetscScalar cpuarray[],const ViennaCLVector* viennaclvec,Vec *V)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;

  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQ on more than one process");

  // set V's viennaclvec to be viennaclvec, do not allocate memory on host yet.
  ierr = VecCreateSeqViennaCLWithArray(comm,bs,n,viennaclvec,V);CHKERRQ(ierr);

  if (cpuarray && viennaclvec) {
    Vec_Seq *s = (Vec_Seq*)((*V)->data);
    s->array = (PetscScalar*)cpuarray;
    (*V)->offloadmask = PETSC_OFFLOAD_BOTH;
  } else if (cpuarray) {
    Vec_Seq *s = (Vec_Seq*)((*V)->data);
    s->array = (PetscScalar*)cpuarray;
    (*V)->offloadmask = PETSC_OFFLOAD_CPU;
  } else if (viennaclvec) {
    (*V)->offloadmask = PETSC_OFFLOAD_GPU;
  } else {
    (*V)->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  }

  PetscFunctionReturn(0);
}

/*@C
   VecViennaCLPlaceArray - Replace the viennacl vector in a Vec with
   the one provided by the user. This is useful to avoid a copy.

   Not Collective

   Input Parameters:
+  vec - the vector
-  array - the ViennaCL vector

   Notes:
   You can return to the original viennacl vector with a call to
   VecViennaCLResetArray() It is not possible to use VecViennaCLPlaceArray()
   and VecPlaceArray() at the same time on the same vector.

   Level: intermediate

.seealso: VecPlaceArray(), VecSetValues(), VecViennaCLResetArray(),
          VecCUDAPlaceArray(),

@*/
PETSC_EXTERN PetscErrorCode VecViennaCLPlaceArray(Vec vin,const ViennaCLVector* a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(vin,VECSEQVIENNACL,VECMPIVIENNACL);
  ierr = VecViennaCLCopyToGPU(vin);CHKERRQ(ierr);
  if (((Vec_Seq*)vin->data)->unplacedarray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"VecViennaCLPlaceArray()/VecPlaceArray() was already called on this vector, without a call to VecViennaCLResetArray()/VecResetArray()");
  ((Vec_Seq*)vin->data)->unplacedarray  = (PetscScalar *) ((Vec_ViennaCL*)vin->spptr)->GPUarray; /* save previous GPU array so reset can bring it back */
  ((Vec_ViennaCL*)vin->spptr)->GPUarray = (ViennaCLVector*)a;
  vin->offloadmask = PETSC_OFFLOAD_GPU;
  ierr = PetscObjectStateIncrease((PetscObject)vin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   VecViennaCLResetArray - Resets a vector to use its default memory. Call this
   after the use of VecViennaCLPlaceArray().

   Not Collective

   Input Parameters:
.  vec - the vector

   Level: developer

.seealso: VecViennaCLPlaceArray(), VecResetArray(), VecCUDAResetArray(), VecPlaceArray()
@*/
PETSC_EXTERN PetscErrorCode VecViennaCLResetArray(Vec vin)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckTypeNames(vin,VECSEQVIENNACL,VECMPIVIENNACL);
  ierr = VecViennaCLCopyToGPU(vin);CHKERRQ(ierr);
  ((Vec_ViennaCL*)vin->spptr)->GPUarray = (ViennaCLVector *) ((Vec_Seq*)vin->data)->unplacedarray;
  ((Vec_Seq*)vin->data)->unplacedarray = 0;
  vin->offloadmask = PETSC_OFFLOAD_GPU;
  ierr = PetscObjectStateIncrease((PetscObject)vin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*  VecDotNorm2 - computes the inner product of two vectors and the 2-norm squared of the second vector
 *
 *  Simply reuses VecDot() and VecNorm(). Performance improvement through custom kernel (kernel generator) possible.
 */
PetscErrorCode VecDotNorm2_SeqViennaCL(Vec s, Vec t, PetscScalar *dp, PetscScalar *nm)
{
  PetscErrorCode                         ierr;

  PetscFunctionBegin;
  ierr = VecDot_SeqViennaCL(s,t,dp);CHKERRQ(ierr);
  ierr = VecNorm_SeqViennaCL(t,NORM_2,nm);CHKERRQ(ierr);
  *nm *= *nm; //squared norm required
  PetscFunctionReturn(0);
}

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

PetscErrorCode VecDestroy_SeqViennaCL(Vec v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  try {
    if (v->spptr) {
      delete ((Vec_ViennaCL*)v->spptr)->GPUarray_allocated;
      delete (Vec_ViennaCL*) v->spptr;
    }
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex);
  }
  ierr = VecDestroy_SeqViennaCL_Private(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetArray_SeqViennaCL(Vec v,PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (v->offloadmask == PETSC_OFFLOAD_GPU) {
    ierr = VecViennaCLCopyFromGPU(v);CHKERRQ(ierr);
  } else {
    ierr = VecViennaCLAllocateCheckHost(v);CHKERRQ(ierr);
  }
  *a = *((PetscScalar**)v->data);
  PetscFunctionReturn(0);
}

PetscErrorCode VecRestoreArray_SeqViennaCL(Vec v,PetscScalar **a)
{
  PetscFunctionBegin;
  v->offloadmask = PETSC_OFFLOAD_CPU;
  PetscFunctionReturn(0);
}

PetscErrorCode VecGetArrayWrite_SeqViennaCL(Vec v,PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecViennaCLAllocateCheckHost(v);CHKERRQ(ierr);
  *a   = *((PetscScalar**)v->data);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecBindToCPU_SeqAIJViennaCL(Vec V,PetscBool flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  V->boundtocpu = flg;
  if (flg) {
    ierr = VecViennaCLCopyFromGPU(V);CHKERRQ(ierr);
    V->offloadmask = PETSC_OFFLOAD_CPU; /* since the CPU code will likely change values in the vector */
    V->ops->dot             = VecDot_Seq;
    V->ops->norm            = VecNorm_Seq;
    V->ops->tdot            = VecTDot_Seq;
    V->ops->scale           = VecScale_Seq;
    V->ops->copy            = VecCopy_Seq;
    V->ops->set             = VecSet_Seq;
    V->ops->swap            = VecSwap_Seq;
    V->ops->axpy            = VecAXPY_Seq;
    V->ops->axpby           = VecAXPBY_Seq;
    V->ops->axpbypcz        = VecAXPBYPCZ_Seq;
    V->ops->pointwisemult   = VecPointwiseMult_Seq;
    V->ops->pointwisedivide = VecPointwiseDivide_Seq;
    V->ops->setrandom       = VecSetRandom_Seq;
    V->ops->dot_local       = VecDot_Seq;
    V->ops->tdot_local      = VecTDot_Seq;
    V->ops->norm_local      = VecNorm_Seq;
    V->ops->mdot_local      = VecMDot_Seq;
    V->ops->mtdot_local     = VecMTDot_Seq;
    V->ops->maxpy           = VecMAXPY_Seq;
    V->ops->mdot            = VecMDot_Seq;
    V->ops->mtdot           = VecMTDot_Seq;
    V->ops->aypx            = VecAYPX_Seq;
    V->ops->waxpy           = VecWAXPY_Seq;
    V->ops->dotnorm2        = NULL;
    V->ops->placearray      = VecPlaceArray_Seq;
    V->ops->replacearray    = VecReplaceArray_Seq;
    V->ops->resetarray      = VecResetArray_Seq;
    V->ops->duplicate       = VecDuplicate_Seq;
  } else {
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
    V->ops->mtdot_local     = VecMTDot_SeqViennaCL;
    V->ops->maxpy           = VecMAXPY_SeqViennaCL;
    V->ops->mdot            = VecMDot_SeqViennaCL;
    V->ops->mtdot           = VecMTDot_SeqViennaCL;
    V->ops->aypx            = VecAYPX_SeqViennaCL;
    V->ops->waxpy           = VecWAXPY_SeqViennaCL;
    V->ops->dotnorm2        = VecDotNorm2_SeqViennaCL;
    V->ops->placearray      = VecPlaceArray_SeqViennaCL;
    V->ops->replacearray    = VecReplaceArray_SeqViennaCL;
    V->ops->resetarray      = VecResetArray_SeqViennaCL;
    V->ops->destroy         = VecDestroy_SeqViennaCL;
    V->ops->duplicate       = VecDuplicate_SeqViennaCL;
    V->ops->getarraywrite   = VecGetArrayWrite_SeqViennaCL;
    V->ops->getarray        = VecGetArray_SeqViennaCL;
    V->ops->restorearray    = VecRestoreArray_SeqViennaCL;
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecCreate_SeqViennaCL(Vec V)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)V),&size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQVIENNACL on more than one process");
  ierr = VecCreate_Seq_Private(V,0);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)V,VECSEQVIENNACL);CHKERRQ(ierr);

  ierr = VecBindToCPU_SeqAIJViennaCL(V,PETSC_FALSE);CHKERRQ(ierr);
  V->ops->bindtocpu = VecBindToCPU_SeqAIJViennaCL;

  ierr = VecViennaCLAllocateCheck(V);CHKERRQ(ierr);
  ierr = VecSet_SeqViennaCL(V,0.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  VecViennaCLGetCLContext - Get the OpenCL context in which the Vec resides.

  Caller should cast (*ctx) to (const cl_context). Caller is responsible for
  invoking clReleaseContext().

  Input Parameters:
.  v    - the vector

  Output Parameters:
.  ctx - pointer to the underlying CL context

  Level: intermediate

.seealso: VecViennaCLGetCLQueue(), VecViennaCLGetCLMemRead()
@*/
PETSC_EXTERN PetscErrorCode VecViennaCLGetCLContext(Vec v, PETSC_UINTPTR_T* ctx)
{
#if !defined(PETSC_HAVE_OPENCL)
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"PETSc must be configured with --with-opencl to get the associated cl_context.");
#else

  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);

  PetscErrorCode ierr;
  const ViennaCLVector *v_vcl;
  ierr = VecViennaCLGetArrayRead(v, &v_vcl);CHKERRQ(ierr);
  try{
    viennacl::ocl::context vcl_ctx = v_vcl->handle().opencl_handle().context();
    const cl_context ocl_ctx = vcl_ctx.handle().get();
    clRetainContext(ocl_ctx);
    *ctx = (PETSC_UINTPTR_T)(ocl_ctx);
  } catch (std::exception const & ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
  }

  PetscFunctionReturn(0);
#endif
}

/*@C
  VecViennaCLGetCLQueue - Get the OpenCL command queue to which all
  operations of the Vec are enqueued.

  Caller should cast (*queue) to (const cl_command_queue). Caller is
  responsible for invoking clReleaseCommandQueue().

  Input Parameters:
.  v    - the vector

  Output Parameters:
.  ctx - pointer to the CL command queue

  Level: intermediate

.seealso: VecViennaCLGetCLContext(), VecViennaCLGetCLMemRead()
@*/
PETSC_EXTERN PetscErrorCode VecViennaCLGetCLQueue(Vec v, PETSC_UINTPTR_T* queue)
{
#if !defined(PETSC_HAVE_OPENCL)
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"PETSc must be configured with --with-opencl to get the associated cl_command_queue.");
#else
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);

  PetscErrorCode ierr;
  const ViennaCLVector *v_vcl;
  ierr = VecViennaCLGetArrayRead(v, &v_vcl);CHKERRQ(ierr);
  try{
    viennacl::ocl::context vcl_ctx = v_vcl->handle().opencl_handle().context();
    const viennacl::ocl::command_queue& vcl_queue = vcl_ctx.current_queue();
    const cl_command_queue ocl_queue = vcl_queue.handle().get();
    clRetainCommandQueue(ocl_queue);
    *queue = (PETSC_UINTPTR_T)(ocl_queue);
  } catch (std::exception const & ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
  }

  PetscFunctionReturn(0);
#endif
}

/*@C
  VecViennaCLGetCLMemRead - Provides access to the the CL buffer inside a Vec.

  Caller should cast (*mem) to (const cl_mem). Caller is responsible for
  invoking clReleaseMemObject().

  Input Parameters:
.  v    - the vector

  Output Parameters:
.  mem - pointer to the device buffer

  Level: intermediate

.seealso: VecViennaCLGetCLContext(), VecViennaCLGetCLMemWrite()
@*/
PETSC_EXTERN PetscErrorCode VecViennaCLGetCLMemRead(Vec v, PETSC_UINTPTR_T* mem)
{
#if !defined(PETSC_HAVE_OPENCL)
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"PETSc must be configured with --with-opencl to get a Vec's cl_mem");
#else
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);

  PetscErrorCode ierr;
  const ViennaCLVector *v_vcl;
  ierr = VecViennaCLGetArrayRead(v, &v_vcl);CHKERRQ(ierr);
  try{
    const cl_mem ocl_mem = v_vcl->handle().opencl_handle().get();
    clRetainMemObject(ocl_mem);
    *mem = (PETSC_UINTPTR_T)(ocl_mem);
  } catch (std::exception const & ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
  }
  PetscFunctionReturn(0);
#endif
}

/*@C
  VecViennaCLGetCLMemWrite - Provides access to the the CL buffer inside a Vec.

  Caller should cast (*mem) to (const cl_mem). Caller is responsible for
  invoking clReleaseMemObject().

  The device pointer has to be released by calling
  VecViennaCLRestoreCLMemWrite().  Upon restoring the vector data the data on
  the host will be marked as out of date.  A subsequent access of the host data
  will thus incur a data transfer from the device to the host.

  Input Parameters:
.  v    - the vector

  Output Parameters:
.  mem - pointer to the device buffer

  Level: intermediate

.seealso: VecViennaCLGetCLContext(), VecViennaCLRestoreCLMemWrite()
@*/
PETSC_EXTERN PetscErrorCode VecViennaCLGetCLMemWrite(Vec v, PETSC_UINTPTR_T* mem)
{
#if !defined(PETSC_HAVE_OPENCL)
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"PETSc must be configured with --with-opencl to get a Vec's cl_mem");
#else
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);

  PetscErrorCode ierr;
  ViennaCLVector *v_vcl;
  ierr = VecViennaCLGetArrayWrite(v, &v_vcl);CHKERRQ(ierr);
  try{
    const cl_mem ocl_mem = v_vcl->handle().opencl_handle().get();
    clRetainMemObject(ocl_mem);
    *mem = (PETSC_UINTPTR_T)(ocl_mem);
  } catch (std::exception const & ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
  }

  PetscFunctionReturn(0);
#endif
}

/*@C
  VecViennaCLRestoreCLMemWrite - Restores a CL buffer pointer previously
  acquired with VecViennaCLGetCLMemWrite().

   This marks the host data as out of date.  Subsequent access to the
   vector data on the host side with for instance VecGetArray() incurs a
   data transfer.

  Input Parameters:
.  v    - the vector

  Level: intermediate

.seealso: VecViennaCLGetCLContext(), VecViennaCLGetCLMemWrite()
@*/
PETSC_EXTERN PetscErrorCode VecViennaCLRestoreCLMemWrite(Vec v)
{
#if !defined(PETSC_HAVE_OPENCL)
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"PETSc must be configured with --with-opencl to restore a Vec's cl_mem");
#else
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);

  PetscErrorCode ierr;
  ierr = VecViennaCLRestoreArrayWrite(v, PETSC_NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
#endif
}

/*@C
  VecViennaCLGetCLMem - Provides access to the the CL buffer inside a Vec.

  Caller should cast (*mem) to (const cl_mem). Caller is responsible for
  invoking clReleaseMemObject().

  The device pointer has to be released by calling VecViennaCLRestoreCLMem().
  Upon restoring the vector data the data on the host will be marked as out of
  date.  A subsequent access of the host data will thus incur a data transfer
  from the device to the host.

  Input Parameters:
.  v    - the vector

  Output Parameters:
.  mem - pointer to the device buffer

  Level: intermediate

.seealso: VecViennaCLGetCLContext(), VecViennaCLRestoreCLMem()
@*/
PETSC_EXTERN PetscErrorCode VecViennaCLGetCLMem(Vec v, PETSC_UINTPTR_T* mem)
{
#if !defined(PETSC_HAVE_OPENCL)
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"PETSc must be configured with --with-opencl to get a Vec's cl_mem");
#else
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);

  PetscErrorCode ierr;
  ViennaCLVector *v_vcl;
  ierr = VecViennaCLGetArray(v, &v_vcl);CHKERRQ(ierr);
  try{
    const cl_mem ocl_mem = v_vcl->handle().opencl_handle().get();
    clRetainMemObject(ocl_mem);
    *mem = (PETSC_UINTPTR_T)(ocl_mem);
  } catch (std::exception const & ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
  }

  PetscFunctionReturn(0);
#endif
}

/*@C
  VecViennaCLRestoreCLMem - Restores a CL buffer pointer previously
  acquired with VecViennaCLGetCLMem().

   This marks the host data as out of date. Subsequent access to the vector
   data on the host side with for instance VecGetArray() incurs a data
   transfer.

  Input Parameters:
.  v    - the vector

  Level: intermediate

.seealso: VecViennaCLGetCLContext(), VecViennaCLGetCLMem()
@*/
PETSC_EXTERN PetscErrorCode VecViennaCLRestoreCLMem(Vec v)
{
#if !defined(PETSC_HAVE_OPENCL)
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"PETSc must be configured with --with-opencl to restore a Vec's cl_mem");
#else
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQVIENNACL, VECMPIVIENNACL);

  PetscErrorCode ierr;
  ierr = VecViennaCLRestoreArray(v, PETSC_NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
#endif
}

PetscErrorCode VecCreate_SeqViennaCL_Private(Vec V,const ViennaCLVector *array)
{
  PetscErrorCode ierr;
  Vec_ViennaCL   *vecviennacl;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)V),&size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Cannot create VECSEQVIENNACL on more than one process");
  ierr = VecCreate_Seq_Private(V,0);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)V,VECSEQVIENNACL);CHKERRQ(ierr);
  ierr = VecBindToCPU_SeqAIJViennaCL(V,PETSC_FALSE);CHKERRQ(ierr);
  V->ops->bindtocpu = VecBindToCPU_SeqAIJViennaCL;

  if (array) {
    if (!V->spptr)
      V->spptr = new Vec_ViennaCL;
    vecviennacl = (Vec_ViennaCL*)V->spptr;
    vecviennacl->GPUarray_allocated = 0;
    vecviennacl->GPUarray           = (ViennaCLVector*)array;
    V->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
  }

  PetscFunctionReturn(0);
}
