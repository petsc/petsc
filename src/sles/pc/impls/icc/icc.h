
/* static char vcid[] = "$Id: icc.c,v 1.42 1997/02/22 02:24:19 bsmith Exp bsmith $ "; */
#include "src/pc/pcimpl.h"          
#include "src/mat/matimpl.h"

/* Incomplete Cholesky factorization context */

typedef struct {
  Mat  fact;
  int  ordering;
  int  levels;
  void *implctx;
  int   bs_iter;        /* flag - use of BlockSolve iterative solvers */
} PC_ICC;

/* BlockSolve implementation interface */

typedef struct {
  int    blocksize;    /* number of systems to solve */
  int    pre_option;   /* preconditioner, one of PRE_DIAG,
                          PRE_STICCG, PRE_SSOR, PRE_BJACOBI */
  double rtol;
  int    max_it;
  double rnorm;
  int    guess_zero;
} PCiBS;
