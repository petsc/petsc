/* $Id: bjacobi.h,v 1.17 1996/03/04 05:15:29 bsmith Exp bsmith $ */

#if !defined(__BJACOBI_H)
#define __BJACOBI_H
/*
    Private data for block Jacobi and block Gauss-Seidel preconditioner.
*/
#include "sles.h"

typedef struct {
  int       n, n_local;        /* number of blocks (global, local) */
  int       first_local;       /* number of first block on processor */
  int       use_true_local;    /* use block from true matrix, not preconditioner matrix
                                  for local MatMult(). */
  SLES      *sles;             /* SLES contexts for blocks */
  void      *data;             /* implementation-specific data */
  int       same_local_solves; /* flag indicating whether all local solvers are same */
  int       *l_lens;           /* lens of each block */
  int       *g_lens;
  /* -----------------Options related to Gauss-Seidel ----------------------------*/
  int       gs;                /* flag indicating we are using Gauss-Seidel */
  PCBGSType gstype;
} PC_BJacobi;

#endif
