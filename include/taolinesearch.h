#ifndef __TAOLINESEARCH_H
#define __TAOLINESEARCH_H
#include "petscvec.h"

#include "taosolver.h"
PETSC_EXTERN_CXX_BEGIN

typedef enum {
    TAOLINESEARCH_CONTINUE_ITERATING = 0,
    TAOLINESEARCH_FAILED_OTHER = -1,
    TAOLINESEARCH_FAILED_MAXFCN = -2,
    TAOLINESEARCH_FAILED_DOMAIN = -3,
    TAOLINESEARCH_FAILED_INFORNAN = -4,
    TAOLINESEARCH_FAILED_ASCENT = -5,
    TAOLINESEARCH_FAILED_BADPARAMETER = -6,
    TAOLINESEARCH_FAILED_UPPERBOUND = -7,
    TAOLINESEARCH_FAILED_LOWERBOUND = -8,
    TAOLINESEARCH_FAILED_RTOL = -9,
    TAOLINESEARCH_FAILED_USER = -10,
    TAOLINESEARCH_SUCCESS = 1,
    TAOLINESEARCH_SUCCESS_USER = 2,
} TaoLineSearchTerminationReason;
	      
typedef struct _p_TaoLineSearch* TaoLineSearch;

#define TaoLineSearchType  char*
#define TAOLINESEARCH_UNIT "unit"
#define TAOLINESEARCH_MT "more-thuente"
#define TAOLINESEARCH_GPCG "gpcg"

extern PetscClassId TAOLINESEARCH_DLLEXPORT TAOLINESEARCH_CLASSID;

EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchCreate(MPI_Comm,TaoLineSearch*);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetFromOptions(TaoLineSearch);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetUp(TaoLineSearch);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchDestroy(TaoLineSearch);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchView(TaoLineSearch,PetscViewer);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetOptionsPrefix(TaoLineSearch);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchApply(TaoLineSearch,Vec,PetscReal *,Vec,Vec, PetscReal *, TaoLineSearchTerminationReason *);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchGetStepLength(TaoLineSearch, PetscReal*);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetInitialStepLength(TaoLineSearch, PetscReal);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchGetSolution(TaoLineSearch, Vec, PetscReal*, Vec, PetscReal*, TaoLineSearchTerminationReason*);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchGetFullStepObjective(TaoLineSearch, PetscReal*);

EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchGetType(TaoLineSearch, const TaoLineSearchType *);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetType(TaoLineSearch, const TaoLineSearchType);

EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchUseTaoSolverRoutines(TaoLineSearch, TaoSolver);

EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetObjective(TaoLineSearch, PetscErrorCode(*)(TaoLineSearch, Vec, PetscReal*,void*), void*);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetGradient(TaoLineSearch, PetscErrorCode(*)(TaoLineSearch, Vec, Vec, void*), void*);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetObjectiveAndGradient(TaoLineSearch, PetscErrorCode(*)(TaoLineSearch, Vec, PetscReal*, Vec, void*), void*);

EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchComputeObjective(TaoLineSearch, Vec, PetscReal*);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchComputeGradient(TaoLineSearch, Vec, Vec);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchComputeObjectiveAndGradient(TaoLineSearch, Vec, PetscReal*, Vec);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchSetVariableBounds(TaoLineSearch, Vec, Vec);

EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchInitializePackage(const char path[]); 

EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchRegisterAll(const char path[]);
EXTERN PetscErrorCode TAOLINESEARCH_DLLEXPORT TaoLineSearchRegister(const char[], const char[], const char[], PetscErrorCode (*)(TaoLineSearch));

PETSC_EXTERN_CXX_END
#endif
