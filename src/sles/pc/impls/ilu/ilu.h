/* $Id: dvec2.c,v 1.20 1995/11/09 22:26:41 bsmith Exp bsmith $ */

/* 
   Private data structure for ILU preconditioner.
*/

typedef struct {
  Mat         fact;      /* factored matrix */
  MatOrdering ordering;  /* matrix reordering */
  int         levels;    /* levels of fill */
  IS          row, col;  /* row and column permutations for reordering */
  void        *implctx;  /* private implementation context */
  int         bs_iter;   /* flag - use of BlockSolve iterative solvers */
  int         inplace;   /* in-place ILU factorization */
} PC_ILU;

