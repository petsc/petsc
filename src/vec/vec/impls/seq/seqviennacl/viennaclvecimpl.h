#if !defined(__VIENNACLVECIMPL)
#define __VIENNACLVECIMPL

#include <petsc-private/vecimpl.h>

#include <algorithm>
#include <vector>
#include <string>
#include <exception>

#define VIENNACL_WITH_OPENCL

#include "viennacl/vector.hpp"

#define ViennaCLWaitForGPU() if (PetscViennaCLSynchronize) viennacl::backend::finish();

typedef viennacl::vector<PetscScalar>    ViennaCLVector;


PETSC_INTERN PetscErrorCode VecDotNorm2_SeqViennaCL(Vec,Vec,PetscScalar*, PetscScalar*);
PETSC_INTERN PetscErrorCode VecPointwiseDivide_SeqViennaCL(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecWAXPY_SeqViennaCL(Vec,PetscScalar,Vec,Vec);
PETSC_INTERN PetscErrorCode VecMDot_SeqViennaCL(Vec,PetscInt,const Vec[],PetscScalar*);
PETSC_INTERN PetscErrorCode VecSet_SeqViennaCL(Vec,PetscScalar);
PETSC_INTERN PetscErrorCode VecMAXPY_SeqViennaCL(Vec,PetscInt,const PetscScalar*,Vec*);
PETSC_INTERN PetscErrorCode VecAXPBYPCZ_SeqViennaCL(Vec,PetscScalar,PetscScalar,PetscScalar,Vec,Vec);
PETSC_INTERN PetscErrorCode VecPointwiseMult_SeqViennaCL(Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode VecPlaceArray_SeqViennaCL(Vec,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecResetArray_SeqViennaCL(Vec);
PETSC_INTERN PetscErrorCode VecReplaceArray_SeqViennaCL(Vec,const PetscScalar*);
PETSC_INTERN PetscErrorCode VecDot_SeqViennaCL(Vec,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecTDot_SeqViennaCL(Vec,Vec,PetscScalar*);
PETSC_INTERN PetscErrorCode VecScale_SeqViennaCL(Vec,PetscScalar);
PETSC_INTERN PetscErrorCode VecCopy_SeqViennaCL(Vec,Vec);
PETSC_INTERN PetscErrorCode VecSwap_SeqViennaCL(Vec,Vec);
PETSC_INTERN PetscErrorCode VecAXPY_SeqViennaCL(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecAXPBY_SeqViennaCL(Vec,PetscScalar,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecDuplicate_SeqViennaCL(Vec,Vec*);
PETSC_INTERN PetscErrorCode VecNorm_SeqViennaCL(Vec,NormType,PetscReal*);
PETSC_INTERN PetscErrorCode VecViennaCLCopyToGPU(Vec);
PETSC_INTERN PetscErrorCode VecViennaCLAllocateCheck(Vec);
PETSC_INTERN PetscErrorCode VecViennaCLAllocateCheckHost(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_SeqViennaCL(Vec);
PETSC_INTERN PetscErrorCode VecView_Seq(Vec,PetscViewer);
PETSC_INTERN PetscErrorCode VecDestroy_SeqViennaCL(Vec);
PETSC_INTERN PetscErrorCode VecAYPX_SeqViennaCL(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecSetRandom_SeqViennaCL(Vec,PetscRandom);

PETSC_INTERN PetscErrorCode VecViennaCLCopyToGPU_Public(Vec);
PETSC_INTERN PetscErrorCode VecViennaCLAllocateCheck_Public(Vec);

struct Vec_ViennaCL {
  viennacl::vector<PetscScalar> *GPUarray;        // this always holds the GPU data
};


#undef __FUNCT__
#define __FUNCT__ "ViennaCLSetFromOptions"
static PetscErrorCode ViennaCLSetFromOptions(PetscObject obj)
{
  PetscErrorCode       ierr;
  PetscBool            flg;

  PetscFunctionBegin;
  ierr = PetscObjectOptionsBegin(obj);

  ierr = PetscOptionsHasName(NULL,"-viennacl_device_cpu",&flg);CHKERRQ(ierr);
  if (flg) {
    try {
      viennacl::ocl::set_context_device_type(0, CL_DEVICE_TYPE_CPU);
    } catch (std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
  }
  ierr = PetscOptionsHasName(NULL,"-viennacl_device_gpu",&flg);CHKERRQ(ierr);
  if (flg) {
    try {
      viennacl::ocl::set_context_device_type(0, CL_DEVICE_TYPE_GPU);
    } catch (std::exception const & ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"ViennaCL error: %s", ex.what());
    }
  }
  ierr = PetscOptionsHasName(NULL,"-viennacl_device_accelerator",&flg);CHKERRQ(ierr);
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
#define __FUNCT__ "VecSetFromOptions_SeqViennaCL"
static PetscErrorCode VecSetFromOptions_SeqViennaCL(Vec v)
{
  PetscErrorCode       ierr;
  PetscBool            flg;

  PetscFunctionBegin;
  ViennaCLSetFromOptions((PetscObject)v);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecViennaCLGetArrayReadWrite"
PETSC_STATIC_INLINE PetscErrorCode VecViennaCLGetArrayReadWrite(Vec v, ViennaCLVector **a)
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
PETSC_STATIC_INLINE PetscErrorCode VecViennaCLRestoreArrayReadWrite(Vec v, ViennaCLVector **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  v->valid_GPU_array = PETSC_VIENNACL_GPU;

  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecViennaCLGetArrayRead"
PETSC_STATIC_INLINE PetscErrorCode VecViennaCLGetArrayRead(Vec v, const ViennaCLVector **a)
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
PETSC_STATIC_INLINE PetscErrorCode VecViennaCLRestoreArrayRead(Vec v, const ViennaCLVector **a)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecViennaCLGetArrayWrite"
PETSC_STATIC_INLINE PetscErrorCode VecViennaCLGetArrayWrite(Vec v, ViennaCLVector **a)
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
PETSC_STATIC_INLINE PetscErrorCode VecViennaCLRestoreArrayWrite(Vec v, ViennaCLVector **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  v->valid_GPU_array = PETSC_VIENNACL_GPU;

  ierr = PetscObjectStateIncrease((PetscObject)v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#endif
