/* $Id: dvec2.c,v 1.20 1995/11/09 22:26:41 bsmith Exp bsmith $ */

#if !defined(__BJACOBI_H)
#define __BJACOBI_H
/*
    Private data for Block Jacobi preconditioner.
*/
#include "sles.h"

typedef struct {
  int  n, n_local;          /* number of blocks (global, local) */
  int  first_local;         /* number of first block on processor */
  int  use_true_local;      /* use true local matrix, not precond matrix */
  SLES *sles;               /* SLES contexts for blocks */
  void *data;               /* implementation-specific data */
  int  same_local_solves;   /* flag indicating whether all local solvers are same */
  int  *l_lens,*g_lens;     /* lens of each block */
} PC_BJacobi;

#endif
