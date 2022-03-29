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

  Mat               cooMat; /* An agent matrix which does the MatSetValuesCOO() job for IJMatrix */
  HYPRE_Int         *diagJ,*offdJ; /* Allocated by hypre, but we take the ownership away, so we need to free them on our own */
  PetscInt          *diag; /* Diagonal pointers (i.e., SeqAIJ->diag[]) on device, allocated by hypre_TAlloc(). */
  HYPRE_MemoryLocation memType;
} Mat_HYPRE;

#endif
