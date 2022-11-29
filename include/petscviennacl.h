#ifndef PETSCVIENNACL_H
#define PETSCVIENNACL_H

#include <petscvec.h>

#if defined(PETSC_HAVE_CUDA)
  #define VIENNACL_WITH_CUDA
#endif

#if defined(PETSC_HAVE_OPENCL)
  #define VIENNACL_WITH_OPENCL
#endif

#if defined(PETSC_HAVE_OPENMP)
  #define VIENNACL_WITH_OPENMP
#endif

#include <viennacl/forwards.h>
#include <viennacl/vector_proxy.hpp>
#include <viennacl/vector.hpp>

PETSC_EXTERN PetscErrorCode VecViennaCLGetArray(Vec v, viennacl::vector<PetscScalar> **a);
PETSC_EXTERN PetscErrorCode VecViennaCLRestoreArray(Vec v, viennacl::vector<PetscScalar> **a);

PETSC_EXTERN PetscErrorCode VecViennaCLGetArrayRead(Vec v, const viennacl::vector<PetscScalar> **a);
PETSC_EXTERN PetscErrorCode VecViennaCLRestoreArrayRead(Vec v, const viennacl::vector<PetscScalar> **a);

PETSC_EXTERN PetscErrorCode VecViennaCLGetArrayWrite(Vec v, viennacl::vector<PetscScalar> **a);
PETSC_EXTERN PetscErrorCode VecViennaCLRestoreArrayWrite(Vec v, viennacl::vector<PetscScalar> **a);

PETSC_EXTERN PetscErrorCode VecCreateSeqViennaCLWithArray(MPI_Comm, PetscInt, PetscInt, const viennacl::vector<PetscScalar> *, Vec *);
PETSC_EXTERN PetscErrorCode VecCreateSeqViennaCLWithArrays(MPI_Comm, PetscInt, PetscInt, const PetscScalar *, const viennacl::vector<PetscScalar> *, Vec *);
PETSC_EXTERN PetscErrorCode VecCreateMPIViennaCLWithArray(MPI_Comm, PetscInt, PetscInt, PetscInt, const viennacl::vector<PetscScalar> *, Vec *);
PETSC_EXTERN PetscErrorCode VecCreateMPIViennaCLWithArrays(MPI_Comm, PetscInt, PetscInt, PetscInt, const PetscScalar *, const viennacl::vector<PetscScalar> *, Vec *);

PETSC_EXTERN PetscErrorCode VecViennaCLPlaceArray(Vec, const viennacl::vector<PetscScalar> *);
PETSC_EXTERN PetscErrorCode VecViennaCLResetArray(Vec);

#endif
