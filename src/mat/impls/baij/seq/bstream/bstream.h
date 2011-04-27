
#include "../src/mat/impls/baij/seq/baij.h"

typedef struct {
  PetscInt    nz;
  PetscInt    rbs;
  PetscInt    cbs;
  PetscInt    m;        /* number of rows */
  MatScalar   *as;      /* values of nonzeros, stored as icols */
  PetscInt    *asi, *asj;

  PetscErrorCode (*MatLUFactorSymbolic)(Mat B,Mat A,IS r,IS c,const MatFactorInfo *info);
  PetscErrorCode (*MatLUFactorNumeric)(Mat F,Mat A,const MatFactorInfo *info);
  PetscErrorCode (*AssemblyEnd)(Mat,MatAssemblyType);
  PetscErrorCode (*MatDestroy)(Mat);
  PetscErrorCode (*MatDuplicate)(Mat,MatDuplicateOption,Mat*);

} Mat_SeqBSTRM;

