
#include "../src/mat/impls/sbaij/seq/sbaij.h"

typedef struct {
  PetscInt    nz;
  PetscInt    rbs;
  PetscInt    cbs;
  PetscInt    m;        /* number of rows */
  MatScalar   *as;      /* values of nonzeros, stored as icols */
  PetscInt    *asi, *asj;

} Mat_SeqSBSTRM;
