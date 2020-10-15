#if !defined(__VIENNACLVECIMPL)
#define __VIENNACLVECIMPL

#include <petscviennacl.h>
#include <petsc/private/vecimpl.h>

#include <algorithm>
#include <vector>
#include <string>
#include <exception>

#include "viennacl/vector.hpp"

#define ViennaCLWaitForGPU() if (PetscViennaCLSynchronize) viennacl::backend::finish();

typedef viennacl::vector<PetscScalar>    ViennaCLVector;

PETSC_EXTERN PetscErrorCode PetscViennaCLInit();

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
PETSC_EXTERN PetscErrorCode VecViennaCLAllocateCheckHost(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_SeqViennaCL(Vec);
PETSC_INTERN PetscErrorCode VecCreate_SeqViennaCL_Private(Vec,const ViennaCLVector *);
PETSC_EXTERN PetscErrorCode VecView_Seq(Vec,PetscViewer);
PETSC_INTERN PetscErrorCode VecDestroy_SeqViennaCL(Vec);
PETSC_INTERN PetscErrorCode VecAYPX_SeqViennaCL(Vec,PetscScalar,Vec);
PETSC_INTERN PetscErrorCode VecSetRandom_SeqViennaCL(Vec,PetscRandom);
PETSC_INTERN PetscErrorCode VecGetArrayWrite_SeqViennaCL(Vec,PetscScalar**);
PETSC_INTERN PetscErrorCode VecGetArray_SeqViennaCL(Vec,PetscScalar**);
PETSC_INTERN PetscErrorCode VecRestoreArray_SeqViennaCL(Vec,PetscScalar**);

PETSC_INTERN PetscErrorCode VecCreate_MPIViennaCL_Private(Vec,PetscBool,PetscInt,const ViennaCLVector *);

PETSC_INTERN PetscErrorCode VecViennaCLCopyToGPU_Public(Vec);
PETSC_INTERN PetscErrorCode VecViennaCLAllocateCheck_Public(Vec);

struct Vec_ViennaCL {
  viennacl::vector<PetscScalar> *GPUarray;           // this always holds the GPU data
  viennacl::vector<PetscScalar> *GPUarray_allocated; // if the array was allocated by PETSc this is its pointer
};



#endif
