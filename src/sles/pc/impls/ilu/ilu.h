
/* Incomplete LU factorization context */

typedef struct {
  Mat         fact;      /* factored matrix */
  MatOrdering ordering;  /* matrix reordering */
  int         levels;    /* levels of fill */
  IS          row, col;  /* row and column permutations for reordering */
  void        *implctx;  /* private implementation context */
  int         bs_iter;   /* flag - use of BlockSolve iterative solvers */
  int         inplace;   /* in-place ILU factorization */
} PC_ILU;

