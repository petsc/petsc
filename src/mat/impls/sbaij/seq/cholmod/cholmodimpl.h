#if !defined(CHOLMODIMPL_H_)
#define CHOLMODIMPL_H_

#include <petscsys.h>

#if defined(PETSC_USE_COMPLEX)
#  define CHOLMOD_SCALAR_TYPE       CHOLMOD_COMPLEX
#else
#  define CHOLMOD_SCALAR_TYPE       CHOLMOD_REAL
#endif

#if defined(PETSC_USE_64BIT_INDICES)
#  define CHOLMOD_INT_TYPE                CHOLMOD_LONG
#  define cholmod_X_start                 cholmod_l_start
#  define cholmod_X_analyze               cholmod_l_analyze
/* the type casts are needed because PetscInt is long long while SuiteSparse_long is long and compilers warn even when they are identical */
#  define cholmod_X_analyze_p(a,b,c,d,f)  cholmod_l_analyze_p(a,(SuiteSparse_long *)b,(SuiteSparse_long *)c,d,f)
#  define cholmod_X_copy                  cholmod_l_copy
#  define cholmod_X_factorize             cholmod_l_factorize
#  define cholmod_X_finish                cholmod_l_finish
#  define cholmod_X_free_factor           cholmod_l_free_factor
#  define cholmod_X_free_dense            cholmod_l_free_dense
#  define cholmod_X_resymbol(a,b,c,d,f,e) cholmod_l_resymbol(a,(SuiteSparse_long *)b,c,d,f,e)
#  define cholmod_X_solve                 cholmod_l_solve
#  define cholmod_X_solve2                cholmod_l_solve2
#  define cholmod_X_check_sparse          cholmod_l_check_sparse
#else
#  define CHOLMOD_INT_TYPE          CHOLMOD_INT
#  define cholmod_X_start           cholmod_start
#  define cholmod_X_analyze         cholmod_analyze
#  define cholmod_X_analyze_p       cholmod_analyze_p
#  define cholmod_X_copy            cholmod_copy
#  define cholmod_X_factorize       cholmod_factorize
#  define cholmod_X_finish          cholmod_finish
#  define cholmod_X_free_factor     cholmod_free_factor
#  define cholmod_X_free_dense      cholmod_free_dense
#  define cholmod_X_resymbol        cholmod_resymbol
#  define cholmod_X_solve           cholmod_solve
#  define cholmod_X_solve2          cholmod_solve2
#  define cholmod_X_check_sparse    cholmod_check_sparse
#endif

EXTERN_C_BEGIN
#include <cholmod.h>
#include <SuiteSparseQR_C.h>
EXTERN_C_END

typedef struct {
  PetscErrorCode               (*Wrap)(Mat,PetscBool,cholmod_sparse*,PetscBool*,PetscBool*);
  cholmod_sparse               *matrix;
  cholmod_factor               *factor;
  cholmod_common               *common;
  SuiteSparseQR_C_factorization *spqrfact;
  PetscBool                    pack;
  PetscBool                    normal;
} Mat_CHOLMOD;

PETSC_INTERN PetscErrorCode CholmodStart(Mat);
PETSC_INTERN PetscErrorCode MatView_CHOLMOD(Mat,PetscViewer);
PETSC_INTERN PetscErrorCode MatCholeskyFactorSymbolic_CHOLMOD(Mat,Mat,IS,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatGetInfo_CHOLMOD(Mat,MatInfoType,MatInfo*);
PETSC_INTERN PetscErrorCode MatDestroy_CHOLMOD(Mat);

PETSC_INTERN PetscErrorCode VecWrapCholmod(Vec, PetscInt, cholmod_dense *);
PETSC_INTERN PetscErrorCode VecUnWrapCholmod(Vec, PetscInt, cholmod_dense *);
PETSC_INTERN PetscErrorCode MatDenseWrapCholmod(Mat, PetscInt, cholmod_dense *);
PETSC_INTERN PetscErrorCode MatDenseUnWrapCholmod(Mat, PetscInt, cholmod_dense *);
#endif /* CHOLMODIMPL_H_ */
