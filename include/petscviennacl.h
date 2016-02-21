#if !defined(__PETSCVIENNACL_H)
#define __PETSCVIENNACL_H


#define VIENNACL_WITH_OPENCL

#include <petscvec.h>
#include <viennacl/forwards.h>
#include <viennacl/vector_proxy.hpp>
#include <viennacl/vector.hpp>

PETSC_EXTERN PetscErrorCode VecViennaCLGetArrayReadWrite(Vec v, viennacl::vector<PetscScalar> **a);
PETSC_EXTERN PetscErrorCode VecViennaCLRestoreArrayReadWrite(Vec v, viennacl::vector<PetscScalar> **a);

PETSC_EXTERN PetscErrorCode VecViennaCLGetArrayRead(Vec v, const viennacl::vector<PetscScalar> **a);
PETSC_EXTERN PetscErrorCode VecViennaCLRestoreArrayRead(Vec v, const viennacl::vector<PetscScalar> **a);

PETSC_EXTERN PetscErrorCode VecViennaCLGetArrayWrite(Vec v, viennacl::vector<PetscScalar> **a);
PETSC_EXTERN PetscErrorCode VecViennaCLRestoreArrayWrite(Vec v, viennacl::vector<PetscScalar> **a);



#endif
