
/* Incomplete Cholesky factorization context */

typedef struct {
  Mat  fact;
  int  ordering;
  int  levels;
  int  (*ImplCreate)(PC);
  int  (*ImplDestroy)(PC);
  void *implctx;
} PC_ICC;

/* BlockSolve implementation interface */

typedef struct {
  Vec vwork;           /* all vector work space */
} PCiBS;
