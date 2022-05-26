#if !defined(_MHYPRE_H)
#define _MHYPRE_H

#include <petscsys.h>
#include <../src/vec/vec/impls/hypre/vhyp.h>
#include <HYPRE_IJ_mv.h>
#include <../src/mat/impls/aij/seq/aij.h>          /*I "petscmat.h" I*/

typedef struct {
  SEQAIJHEADER(MatScalar);
  Mat_SeqAIJ_Inode inode;

  PetscCount  coo_n;  /* Number of entries in MatSetPreallocationCOO() */
  PetscCount  Atot;   /* Total number of valid (i.e., w/ non-negative indices) entries in the COO array */
  PetscInt  *jmap;  /* perm[jmap[i]..jmap[i+1]) give indices of entries in v[] associated with i-th nonzero of the matrix */
  PetscInt  *perm;  /* The permutation array in sorting (i,j) by row and then by col */
} Mat_SeqAIJ_info;

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
  Mat_SeqAIJ_info     *saij;
} Mat_HYPRE;

#endif
