
#include <../src/mat/impls/aij/seq/aij.h>

typedef struct {
  PetscInt     nz;
  PetscInt     m;     /* number of rows */
  PetscInt     rmax;  /* maximum number of columns in a row */
  PetscInt     ncols; /* number of columns in each row */
  PetscInt    *icols; /* columns of nonzeros, stored one column at a time */
  PetscScalar *acols; /* values of nonzeros, stored as icols */
  /* these are only needed for the parallel case */
  Vec          xwork, fwork;
  VecScatter   xscat; /* gathers the locally needed part of global vector */
  PetscScalar *array; /* array used to create xwork */
} Mat_AIJCRL;
