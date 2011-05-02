
#include "../src/mat/impls/sbaij/seq/sbaij.h"

typedef struct {
  PetscInt    nz;
  PetscInt    rbs;
  PetscInt    cbs;
  PetscInt    m;        /* number of rows */
  MatScalar   *as;      /* values of nonzeros, stored as icols */
  PetscInt    *asi, *asj;

  PetscErrorCode (*MatICCFactorSymbolic)(Mat B,Mat A,IS perm ,const MatFactorInfo *info);
  PetscErrorCode (*MatCholeskyFactorSymbolic)(Mat B,Mat A,IS perm ,const MatFactorInfo *info);
  PetscErrorCode (*MatCholeskyFactorNumeric) (Mat F,Mat A,const MatFactorInfo *info);
  PetscErrorCode (*AssemblyEnd)(Mat,MatAssemblyType);
  PetscErrorCode (*MatDestroy)(Mat);
  PetscErrorCode (*MatDuplicate)(Mat,MatDuplicateOption,Mat*);

} Mat_SeqSBSTRM;

