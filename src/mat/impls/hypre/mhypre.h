#if !defined(_MHYPRE_H)
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
  void              *array;
  PetscInt          size;
  PetscBool         available;
  PetscBool         sorted_full;
} Mat_HYPRE;

#endif
