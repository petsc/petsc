
#if !defined(__BJACOBI_H)
#define __BJACOBI_H
/*
    Private data for block Jacobi and block Gauss-Seidel preconditioner.
*/
#include <petscksp.h>
#include <petsc-private/pcimpl.h>

/*
       This data is general for all implementations
*/
typedef struct {
  PetscInt   n;                 /* number of global blocks */
  PetscInt   n_local;           /* number of blocks in this subcommunicator or in this process */
  PetscInt   first_local;       /* number of first block on processor */
  PetscBool  use_true_local;    /* use block from true matrix, not preconditioner matrix for local MatMult() */
  KSP        *ksp;              /* KSP contexts for blocks or for subcommunicator */
  void       *data;             /* implementation-specific data */
  PetscBool  same_local_solves; /* flag indicating whether all local solvers are same (used for PCView()) */
  PetscInt   *l_lens;           /* lens of each block */
  PetscInt   *g_lens;
  PetscSubcomm psubcomm;        /* for multiple processors per block */
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

/*  This is for multiple processors per block */
typedef struct {
  PC           pc;                 /* preconditioner used on each subcommunicator */
  Vec          xsub,ysub;          /* vectors of a subcommunicator to hold parallel vectors of ((PetscObject)pc)->comm */
  Mat          submats;            /* matrix and optional preconditioner matrix belong to a subcommunicator */
  PetscSubcomm psubcomm;
} PC_BJacobi_Multiproc;
#endif


