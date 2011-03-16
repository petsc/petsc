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

extern PetscClassId TAOLINESEARCH_CLASSID;

extern PetscErrorCode TaoLineSearchCreate(MPI_Comm,TaoLineSearch*);
extern PetscErrorCode TaoLineSearchSetFromOptions(TaoLineSearch);
extern PetscErrorCode TaoLineSearchSetUp(TaoLineSearch);
extern PetscErrorCode TaoLineSearchDestroy_(TaoLineSearch);
#define TaoLineSearchDestroy(a) (TaoLineSearchDestroy_(a) || (((a)=0),0))
extern PetscErrorCode TaoLineSearchView(TaoLineSearch,PetscViewer);
extern PetscErrorCode TaoLineSearchSetOptionsPrefix(TaoLineSearch);
extern PetscErrorCode TaoLineSearchApply(TaoLineSearch,Vec,PetscReal *,Vec,Vec, PetscReal *, TaoLineSearchTerminationReason *);
extern PetscErrorCode TaoLineSearchGetStepLength(TaoLineSearch, PetscReal*);
extern PetscErrorCode TaoLineSearchSetInitialStepLength(TaoLineSearch, PetscReal);
extern PetscErrorCode TaoLineSearchGetSolution(TaoLineSearch, Vec, PetscReal*, Vec, PetscReal*, TaoLineSearchTerminationReason*);
extern PetscErrorCode TaoLineSearchGetFullStepObjective(TaoLineSearch, PetscReal*);
extern PetscErrorCode TaoLineSearchGetNumberFunctionEvals(TaoLineSearch, PetscInt*);
extern PetscErrorCode TaoLineSearchGetType(TaoLineSearch, const TaoLineSearchType *);
extern PetscErrorCode TaoLineSearchSetType(TaoLineSearch, const TaoLineSearchType);

extern PetscErrorCode TaoLineSearchUseTaoSolverRoutines(TaoLineSearch, TaoSolver);

extern PetscErrorCode TaoLineSearchSetObjective(TaoLineSearch, PetscErrorCode(*)(TaoLineSearch, Vec, PetscReal*,void*), void*);
extern PetscErrorCode TaoLineSearchSetGradient(TaoLineSearch, PetscErrorCode(*)(TaoLineSearch, Vec, Vec, void*), void*);
extern PetscErrorCode TaoLineSearchSetObjectiveAndGradient(TaoLineSearch, PetscErrorCode(*)(TaoLineSearch, Vec, PetscReal*, Vec, void*), void*);

extern PetscErrorCode TaoLineSearchComputeObjective(TaoLineSearch, Vec, PetscReal*);
extern PetscErrorCode TaoLineSearchComputeGradient(TaoLineSearch, Vec, Vec);
extern PetscErrorCode TaoLineSearchComputeObjectiveAndGradient(TaoLineSearch, Vec, PetscReal*, Vec);
extern PetscErrorCode TaoLineSearchSetVariableBounds(TaoLineSearch, Vec, Vec);

extern PetscErrorCode TaoLineSearchInitializePackage(const char path[]); 

extern PetscErrorCode TaoLineSearchRegister(const char[], const char[], const char[], PetscErrorCode (*)(TaoLineSearch));

PETSC_EXTERN_CXX_END
#endif
