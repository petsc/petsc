#ifndef PETSCTAOLINESEARCH_H
#define PETSCTAOLINESEARCH_H

/* SUBMANSEC = Tao */

/*S
   TaoLineSearch - PETSc object that manages line searches for the `Tao` optimization solves

   Level: intermediate

.seealso: [](chapter_tao), `TaoLineSearchType`, `Tao`, `TaoCreate()`, `TaoDestroy()`, `TaoSetType()`, `TaoType`
S*/
typedef struct _p_TaoLineSearch *TaoLineSearch;

#include <petsctao.h>

/*E
    TaoLineSearchConvergedReason - reason a `TaoLineSearch` completed

   Values:
+ `TAOLINESEARCH_FAILED_ASCENT` - initial line search step * g is not descent direction
. `TAOLINESEARCH_FAILED_INFORNAN` - function evaluation gives `Inf` or `Nan` value
. `TAOLINESEARCH_FAILED_BADPARAMETER` - negative value set as parameter
. `TAOLINESEARCH_HALTED_MAXFCN` - maximum number of function evaluation reached
. `TAOLINESEARCH_HALTED_UPPERBOUND` - step is at upper bound
. `TAOLINESEARCH_HALTED_LOWERBOUND` - step is at lower bound
. `TAOLINESEARCH_HALTED_RTOL` - range of uncertainty is smaller than given tolerance
. `TAOLINESEARCH_HALTED_USER` - user can set this reason to stop line search
. `TAOLINESEARCH_HALTED_OTHER` - any other reason
- `TAOLINESEARCH_SUCCESS` - successful line search

   Level: beginner

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoSolve()`, `TaoGetConvergedReason()`, `KSPConvergedReason`, `SNESConvergedReason`
E*/
typedef enum {
  TAOLINESEARCH_FAILED_INFORNAN     = -1,
  TAOLINESEARCH_FAILED_BADPARAMETER = -2,
  TAOLINESEARCH_FAILED_ASCENT       = -3,
  TAOLINESEARCH_CONTINUE_ITERATING  = 0,
  TAOLINESEARCH_SUCCESS             = 1,
  TAOLINESEARCH_SUCCESS_USER        = 2,
  TAOLINESEARCH_HALTED_OTHER        = 3,
  TAOLINESEARCH_HALTED_MAXFCN       = 4,
  TAOLINESEARCH_HALTED_UPPERBOUND   = 5,
  TAOLINESEARCH_HALTED_LOWERBOUND   = 6,
  TAOLINESEARCH_HALTED_RTOL         = 7,
  TAOLINESEARCH_HALTED_USER         = 8
} TaoLineSearchConvergedReason;

/*J
        TaoLineSearchType - String with the name of a `TaoLineSearch` method

   Values:
+   `TAOLINESEARCHUNIT` -  "unit" do not perform a line search and always accept unit step length
.   `TAOLINESEARCHMT` - "more-thuente" line search with a cubic model enforcing the strong Wolfe/curvature condition
.   `TAOLINESEARCHGPCG` - "gpcg"
.   `TAOLINESEARCHARMIJO` - "armijo" simple backtracking line search enforcing only the sufficient decrease condition
.   `TAOLINESEARCHOWARMIJO` - "owarmijo"
-   `TAOLINESEARCHIPM` - "ipm"

  Options Database Key:
.  -tao_ls_type <type> - select which method Tao should use at runtime

  Level: beginner

.seealso: [](chapter_tao), `Tao`, `TaoLineSearch`, `TaoLineSearchSetType()`, `TaoCreate()`, `TaoSetType()`
J*/
typedef const char *TaoLineSearchType;
#define TAOLINESEARCHUNIT     "unit"
#define TAOLINESEARCHMT       "more-thuente"
#define TAOLINESEARCHGPCG     "gpcg"
#define TAOLINESEARCHARMIJO   "armijo"
#define TAOLINESEARCHOWARMIJO "owarmijo"
#define TAOLINESEARCHIPM      "ipm"

PETSC_EXTERN PetscClassId      TAOLINESEARCH_CLASSID;
PETSC_EXTERN PetscFunctionList TaoLineSearchList;

PETSC_EXTERN PetscErrorCode TaoLineSearchCreate(MPI_Comm, TaoLineSearch *);
PETSC_EXTERN PetscErrorCode TaoLineSearchSetFromOptions(TaoLineSearch);
PETSC_EXTERN PetscErrorCode TaoLineSearchSetUp(TaoLineSearch);
PETSC_EXTERN PetscErrorCode TaoLineSearchDestroy(TaoLineSearch *);
PETSC_EXTERN PetscErrorCode TaoLineSearchMonitor(TaoLineSearch, PetscInt, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TaoLineSearchView(TaoLineSearch, PetscViewer);
PETSC_EXTERN PetscErrorCode TaoLineSearchViewFromOptions(TaoLineSearch, PetscObject, const char[]);

PETSC_EXTERN PetscErrorCode TaoLineSearchSetOptionsPrefix(TaoLineSearch, const char prefix[]);
PETSC_EXTERN PetscErrorCode TaoLineSearchReset(TaoLineSearch);
PETSC_EXTERN PetscErrorCode TaoLineSearchAppendOptionsPrefix(TaoLineSearch, const char[]);
PETSC_EXTERN PetscErrorCode TaoLineSearchGetOptionsPrefix(TaoLineSearch, const char *[]);
PETSC_EXTERN PetscErrorCode TaoLineSearchApply(TaoLineSearch, Vec, PetscReal *, Vec, Vec, PetscReal *, TaoLineSearchConvergedReason *);
PETSC_EXTERN PetscErrorCode TaoLineSearchGetStepLength(TaoLineSearch, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoLineSearchGetStartingVector(TaoLineSearch, Vec *);
PETSC_EXTERN PetscErrorCode TaoLineSearchGetStepDirection(TaoLineSearch, Vec *);
PETSC_EXTERN PetscErrorCode TaoLineSearchSetInitialStepLength(TaoLineSearch, PetscReal);
PETSC_EXTERN PetscErrorCode TaoLineSearchGetSolution(TaoLineSearch, Vec, PetscReal *, Vec, PetscReal *, TaoLineSearchConvergedReason *);
PETSC_EXTERN PetscErrorCode TaoLineSearchGetFullStepObjective(TaoLineSearch, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoLineSearchGetNumberFunctionEvaluations(TaoLineSearch, PetscInt *, PetscInt *, PetscInt *);

PETSC_EXTERN PetscErrorCode TaoLineSearchGetType(TaoLineSearch, TaoLineSearchType *);
PETSC_EXTERN PetscErrorCode TaoLineSearchSetType(TaoLineSearch, TaoLineSearchType);

PETSC_EXTERN PetscErrorCode TaoLineSearchIsUsingTaoRoutines(TaoLineSearch, PetscBool *);
PETSC_EXTERN PetscErrorCode TaoLineSearchSetObjectiveAndGTSRoutine(TaoLineSearch, PetscErrorCode (*)(TaoLineSearch, Vec, Vec, PetscReal *, PetscReal *, void *), void *);
PETSC_EXTERN PetscErrorCode TaoLineSearchSetObjectiveRoutine(TaoLineSearch, PetscErrorCode (*)(TaoLineSearch, Vec, PetscReal *, void *), void *);
PETSC_EXTERN PetscErrorCode TaoLineSearchSetGradientRoutine(TaoLineSearch, PetscErrorCode (*)(TaoLineSearch, Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode TaoLineSearchSetObjectiveAndGradientRoutine(TaoLineSearch, PetscErrorCode (*)(TaoLineSearch, Vec, PetscReal *, Vec, void *), void *);

PETSC_EXTERN PetscErrorCode TaoLineSearchComputeObjective(TaoLineSearch, Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoLineSearchComputeGradient(TaoLineSearch, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoLineSearchComputeObjectiveAndGradient(TaoLineSearch, Vec, PetscReal *, Vec);
PETSC_EXTERN PetscErrorCode TaoLineSearchComputeObjectiveAndGTS(TaoLineSearch, Vec, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TaoLineSearchSetVariableBounds(TaoLineSearch, Vec, Vec);

PETSC_EXTERN PetscErrorCode TaoLineSearchInitializePackage(void);
PETSC_EXTERN PetscErrorCode TaoLineSearchFinalizePackage(void);

PETSC_EXTERN PetscErrorCode TaoLineSearchRegister(const char[], PetscErrorCode (*)(TaoLineSearch));
PETSC_EXTERN PetscErrorCode TaoLineSearchUseTaoRoutines(TaoLineSearch, Tao);

#endif
