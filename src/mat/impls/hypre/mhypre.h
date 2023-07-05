#ifndef _MHYPRE_H
#define _MHYPRE_H

#include <petscsys.h>
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
} Mat_HYPRE;

#endif
