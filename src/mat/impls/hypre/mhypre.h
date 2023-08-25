#pragma once

#include <petscsys.h>
#include <petscmat.h>
#include <../src/vec/vec/impls/hypre/vhyp.h>
#include <HYPRE_IJ_mv.h>

typedef struct {
  HYPRE_IJMatrix    ij;
  VecHYPRE_IJVector x;
  VecHYPRE_IJVector b;
  MPI_Comm          comm;
  PetscBool         inner_free;

  /* MatGetArray_HYPRE */
  void     *array;
  PetscInt  array_size;
  PetscBool array_available;

  /* MatSetOption_ support */
  PetscBool donotstash;

  /* An agent matrix which does the MatSetValuesCOO() job for IJMatrix */
  Mat       cooMat;
  PetscBool cooMatAttached;

  /* helper array storing row ids on device, used in MatZeroRows */
  PetscInt *rows_d;
} Mat_HYPRE;

PETSC_INTERN PetscErrorCode MatZeroRows_CUDA(PetscInt n, const PetscInt *rows, const HYPRE_Int *i, const HYPRE_Int *j, HYPRE_Complex *a, HYPRE_Complex diag);
PETSC_INTERN PetscErrorCode MatZeroRows_HIP(PetscInt n, const PetscInt *rows, const HYPRE_Int *i, const HYPRE_Int *j, HYPRE_Complex *a, HYPRE_Complex diag);
PETSC_INTERN PetscErrorCode MatZeroRows_Kokkos(PetscInt n, const PetscInt *rows, const HYPRE_Int *i, const HYPRE_Int *j, HYPRE_Complex *a, HYPRE_Complex diag);
