if !defined(__BJACOBI)
#define __BJACOBI

#include "sles.h"
/* Base include file for block Jacobi */

typedef struct {
  int  n, n_local;           /* number of blocks (global, local) */
  int  usetruelocal;         /* use true local matrix, not precond matrix */
  SLES *sles;                /* SLES contexts for blocks */
  void *data;                /* implementation-specific data */
} PC_BJacobi;

#endif
