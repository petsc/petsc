
/* Incomplete Cholesky factorization context */

typedef struct {
  Mat  fact;
  int  ordering;
  int  levels;
  int  (*ImplCreate)(PC);
  int  (*ImplDestroy)(PC);
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
