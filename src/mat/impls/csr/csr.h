#if !defined(__CSR_H)
#define __CSR_H

#include "src/mat/matimpl.h"

/* Pure virtual base class upon which AIJ matrices are derived */
/* Info about PETSc's csr data structure */
#define MAT_CSR_HEADER                                                             \
  PetscInt    nz;             /* nonzeros */                                       \
  PetscInt    *i;             /* pointer to beginning of each row */               \
  PetscInt    *j;             /* column values: j + i[k] - 1 is start of row k */  \
  PetscInt    *diag;          /* pointers to diagonal elements */                  \
  PetscScalar *a;             /* nonzero elements */                               \
  PetscScalar *solve_work;    /* work space used in MatSolve */                    \
  IS          row, col, icol  /* index sets, used for reorderings */               

typedef struct {
  MAT_CSR_HEADER;
} Mat_csr;

#endif
