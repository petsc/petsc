/* $Id: bjacobi.h,v 1.21 1999/01/31 16:08:12 bsmith Exp bsmith $ */

#if !defined(__BJACOBI_H)
#define __BJACOBI_H
/*
    Private data for block Jacobi and block Gauss-Seidel preconditioner.
*/
#include "sles.h"
#include "src/sles/pc/pcimpl.h"

/*
       This data is general for all implementations
*/
typedef struct {
  int       n,n_local;        /* number of blocks (global, local) */
  int       first_local;       /* number of first block on processor */
  int       use_true_local;    /* use block from true matrix, not preconditioner matrix
                                  for local MatMult(). */
  SLES      *sles;             /* SLES contexts for blocks */
  void      *data;             /* implementation-specific data */
  int       same_local_solves; /* flag indicating whether all local solvers are same */
  int       *l_lens;           /* lens of each block */
  int       *g_lens;
  Mat       tp_mat,tp_pmat;    /* diagonal block of matrix for this processor */
} PC_BJacobi;

/*
       This data is specific for certain implementations
*/

/*  This is for multiple blocks per processor */

typedef struct {
  Vec              *x,*y;             /* work vectors for solves on each block */
  int              *starts;           /* starting point of each block */
  Mat              *mat,*pmat;        /* submatrices for each block */
  IS               *is;               /* for gathering the submatrices */
} PC_BJacobi_Multiblock;

/*  This is for a single block per processor */
typedef struct {
  Vec  x,y;
} PC_BJacobi_Singleblock;

#endif


