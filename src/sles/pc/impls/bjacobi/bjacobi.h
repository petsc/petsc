
#include "sles.h"
/* Base include file for block Jacobi */

typedef struct {
  int  n,n_local;                   /* number of blocks */
  int  usetruelocal;                /* use true local matrix, not precond */
  SLES *sles;
  void *data;
} PC_BJacobi;

