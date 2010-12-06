#ifndef CHOLMODIMPL_H_
#define CHOLMODIMPL_H_

#include <petscsys.h>

#if defined(PETSC_USE_COMPLEX)
#  define CHOLMOD_SCALAR_TYPE       CHOLMOD_COMPLEX
#else
#  define CHOLMOD_SCALAR_TYPE       CHOLMOD_REAL
#endif

#if defined(PETSC_USE_64BIT_INDICES)
#  define CHOLMOD_INT_TYPE          CHOLMOD_LONG
#  define cholmod_X_start           cholmod_l_start
#  define cholmod_X_analyze         cholmod_l_analyze
#  define cholmod_X_analyze_p       cholmod_l_analyze_p
#  define cholmod_X_copy            cholmod_l_copy
#  define cholmod_X_factorize       cholmod_l_factorize
#  define cholmod_X_finish          cholmod_l_finish
#  define cholmod_X_free_factor     cholmod_l_free_factor
#  define cholmod_X_free_dense      cholmod_l_free_dense
#  define cholmod_X_resymbol        cholmod_l_resymbol
#  define cholmod_X_solve           cholmod_l_solve
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
#endif

#define UF_long long long
#define UF_long_max LONG_LONG_MAX
#define UF_long_id "%lld"
#undef I  /* complex.h defines I=_Complex_I, but cholmod_core.h uses I as a field member */

EXTERN_C_BEGIN
#include <cholmod.h>
EXTERN_C_END

typedef struct {
  PetscErrorCode (*Wrap)(Mat,PetscBool ,cholmod_sparse*,PetscBool *);
  PetscErrorCode (*Destroy)(Mat);
  cholmod_sparse *matrix;
  cholmod_factor *factor;
  cholmod_common *common;
  PetscBool      pack;
} Mat_CHOLMOD;

extern PetscErrorCode  CholmodStart(Mat);
extern PetscErrorCode  MatView_CHOLMOD(Mat,PetscViewer);
extern PetscErrorCode  MatCholeskyFactorSymbolic_CHOLMOD(Mat,Mat,IS,const MatFactorInfo*);
extern PetscErrorCode  MatDestroy_CHOLMOD(Mat);

#endif /* CHOLMODIMPL_H_ */
