#if !defined(__BJACOBI)
#define __BJACOBI

#include "sles.h"
/* Base include file for block Jacobi */

typedef struct {
  int  n, n_local;           /* number of blocks (global, local) */
  int  first_local;          /* number of first block on processor */
  int  use_true_local;       /* use true local matrix, not precond matrix */
  SLES *sles;                /* SLES contexts for blocks */
  void *data;                /* implementation-specific data */
  int  same_local_solves;    /* flag indicating whether all local solvers
                                are the same */
} PC_BJacobi;

#endif
