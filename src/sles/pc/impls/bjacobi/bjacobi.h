
/* Base include file for block Jacobi */

typedef struct {
  int  n;                           /* number of blocks */
  int  usetruelocal;                /* use true local matrix, not precond */
  void *data;
  int  (*view)(PetscObject,Viewer);
} PC_BJacobi;

