
#if !defined(__BJACOBI_H)
#define __BJACOBI_H
/*
    Private data for block Jacobi and block Gauss-Seidel preconditioner.
*/
#include "petscksp.h"
#include "private/pcimpl.h"

/*
       This data is general for all implementations
*/
typedef struct {
  PetscInt   n,n_local;        /* number of blocks (global, local) */
  PetscInt   first_local;       /* number of first block on processor */
  PetscTruth use_true_local;    /* use block from true matrix, not preconditioner matrix for local MatMult() */
  KSP        *ksp;             /* KSP contexts for blocks */
  void       *data;             /* implementation-specific data */
  PetscTruth same_local_solves; /* flag indicating whether all local solvers are same (used for PCView()) */
  PetscInt   *l_lens;           /* lens of each block */
  PetscInt   *g_lens;
  Mat        tp_mat,tp_pmat;    /* diagonal block of matrix for this processor */
} PC_BJacobi;

/*
       This data is specific for certain implementations
*/

/*  This is for multiple blocks per processor */

typedef struct {
  Vec              *x,*y;             /* work vectors for solves on each block */
  PetscInt         *starts;           /* starting point of each block */
  Mat              *mat,*pmat;        /* submatrices for each block */
  IS               *is;               /* for gathering the submatrices */
} PC_BJacobi_Multiblock;

/*  This is for a single block per processor */
typedef struct {
  Vec  x,y;
} PC_BJacobi_Singleblock;

#endif


