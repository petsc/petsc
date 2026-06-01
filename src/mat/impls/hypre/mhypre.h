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

PETSC_INTERN PetscErrorCode MatZeroRows_CUDA(PetscInt, const PetscInt *, const HYPRE_Int *, const HYPRE_Int *, HYPRE_Complex *, HYPRE_Complex);
PETSC_INTERN PetscErrorCode MatZeroRows_HIP(PetscInt, const PetscInt *, const HYPRE_Int *, const HYPRE_Int *, HYPRE_Complex *, HYPRE_Complex);
PETSC_INTERN PetscErrorCode MatZeroRows_Kokkos(PetscInt, const PetscInt *, const HYPRE_Int *, const HYPRE_Int *, HYPRE_Complex *, HYPRE_Complex);

// Cast entries in PetscInt a[] to HYPRE_Int b[] on device
PETSC_INTERN PetscErrorCode PetscHypreIntCastArray_CUDA(PetscInt, const PetscInt *, HYPRE_Int *);
PETSC_INTERN PetscErrorCode PetscHypreIntCastArray_HIP(PetscInt, const PetscInt *, HYPRE_Int *);
PETSC_INTERN PetscErrorCode PetscHypreIntCastArray_Kokkos(PetscInt, const PetscInt *, HYPRE_Int *);

PETSC_INTERN PetscErrorCode MatHypreDeviceMalloc_CUDA(size_t, void **);
PETSC_INTERN PetscErrorCode MatHypreDeviceMalloc_HIP(size_t, void **tr);
PETSC_INTERN PetscErrorCode MatHypreDeviceMalloc_Kokkos(size_t, void **);

PETSC_INTERN PetscErrorCode MatHypreDeviceFree_CUDA(void *);
PETSC_INTERN PetscErrorCode MatHypreDeviceFree_HIP(void *);
PETSC_INTERN PetscErrorCode MatHypreDeviceFree_Kokkos(void *);
