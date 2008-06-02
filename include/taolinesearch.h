#ifndef __TAOLINESEARCH_H
#define __TAOLINESEARCH_H
#include "petscvec.h"

#include "taosolver.h"
PETSC_EXTERN_CXX_BEGIN

typedef enum {
    TAOLINESEARCH_CONTINUE_ITERATING = 0,
    TAOLINESEARCH_FAILED_EPS = -1,
    TAOLINESEARCH_FAILED_MAXFCN = -2,
    TAOLINESEARCH_FAILED_DOMAIN = -3,
    TAOLINESEARCH_FAILED_USER = -4,
    TAOLINESEARCH_SUCCESS = 1,
    TAOLINESEARCH_SUCCESS_USER = 2,
} TaoLineSearchTerminationReason;
	      
typedef struct _p_TaoLineSearch* TaoLineSearch;
#define TaoLineSearchType const char*
#define TAOLINESEARCH_UNIT "unit"
extern PetscCookie TAOLINESEARCH_DLLEXPORT TAOLINESEARCH_COOKIE;

EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchCreate(MPI_Comm,TaoLineSearch*);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetFromOptions(TaoLineSearch);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetUp(TaoLineSearch);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchDestroy(TaoLineSearch);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchView(TaoLineSearch,PetscViewer);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetOptionsPrefix(TaoLineSearch);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchApply(TaoLineSearch,Vec,PetscReal,Vec,Vec);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchGetStepLength(TaoLineSearch, PetscReal*);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetStepLength(TaoLineSearch, PetscReal);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchGetSolution(TaoLineSearch, Vec, PetscReal*, Vec, TaoLineSearchTerminationReason*);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchGetType(TaoLineSearch, TaoLineSearchType *);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetType(TaoLineSearch, TaoLineSearchType);

EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchUseTaoSolverRoutines(TaoLineSearch, TaoSolver);

EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetObjective(TaoLineSearch, PetscErrorCode(*)(TaoLineSearch, Vec, PetscReal*,void*), void*);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetGradient(TaoLineSearch, PetscErrorCode(*)(TaoLineSearch, Vec, Vec, void*), void*);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetObjectiveGradient(TaoLineSearch, PetscErrorCode(*)(TaoLineSearch, Vec, PetscReal*, Vec, void*), void*);

EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchComputeObjective(TaoLineSearch, Vec, PetscReal*);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchComputeGradient(TaoLineSearch, Vec, Vec);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchComputeObjectiveGradient(TaoLineSearch, Vec, PetscReal*, Vec);


EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchRegisterAll(const char path[]);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchRegister(const char[], const char[], const char[], PetscErrorCode (*)(TaoLineSearch));

PETSC_EXTERN_CXX_END
#endif
